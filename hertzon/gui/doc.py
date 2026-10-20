"""Document: project state, undo/redo, selection and the display geometry cache."""
from __future__ import annotations

import time
from dataclasses import dataclass

import gmsh
import numpy as np
from PySide6.QtCore import QObject, Signal

from ..model.design import Project, Design, project_from_dict
from ..kernel import occ as K


@dataclass
class Pick:
    obj: str
    index: int = -1          # face / edge / vertex index; -1 = whole object


@dataclass
class CacheEntry:
    key: str
    tess: K.Tess | None
    error: str | None = None


class Document(QObject):
    changed = Signal(str)               # what: "model", "boundaries", "excitations", "analysis", "results", ...
    geometryChanged = Signal()
    selectionChanged = Signal()
    message = Signal(str, str)          # level, text
    titleChanged = Signal()

    def __init__(self, project: Project | None = None):
        super().__init__()
        self.project = project or Project()
        self.design_name = self.project.designs[0].name
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.dirty = False
        self.selection_mode = "object"    # object | face | edge | vertex
        self.selection: list[Pick] = []
        self.cache: dict[str, CacheEntry] = {}
        self.design_caches: dict[str, tuple] = {}   # other designs stay built (keeps them open)
        self.tess: dict[str, K.Tess] = {}
        self.errors: dict[str, str] = {}
        self.last_build_time = 0.0
        self.display_quality = "normal"

    # ---- design access --------------------------------------------------------------------------
    @property
    def design(self) -> Design:
        return self.project.design(self.design_name)

    def evaluator(self):
        return self.design.evaluator(self.project)

    def set_design(self, name: str) -> None:
        if name != self.design_name:
            self.design_caches[self.design_name] = (self.cache, self.errors)
            self.design_name = name
            self.selection = []
            # only objects that changed since the design was shown last are rebuilt
            self.cache, self.errors = self.design_caches.pop(name, ({}, {}))
            self.rebuild()
            self.changed.emit("all")

    # ---- undo / redo -----------------------------------------------------------------------------
    def checkpoint(self) -> None:
        """Call before modifying the project."""
        self.undo_stack.append(self.project.to_dict())
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def commit(self, what: str = "model", geometry: bool = True) -> None:
        """Call after modifying the project."""
        self.dirty = True
        self.titleChanged.emit()
        if geometry:
            self.rebuild()
        self.changed.emit(what)

    def _restore(self, d: dict) -> None:
        path = self.project.path
        self.project = project_from_dict(d)
        self.project.path = path
        if not any(x.name == self.design_name for x in self.project.designs):
            self.design_name = self.project.designs[0].name
        self.selection = [s for s in self.selection if self.design.has_obj(s.obj)]
        self.dirty = True
        self.rebuild()
        self.changed.emit("all")
        self.titleChanged.emit()

    def undo(self) -> None:
        if self.undo_stack:
            self.redo_stack.append(self.project.to_dict())
            self._restore(self.undo_stack.pop())

    def redo(self) -> None:
        if self.redo_stack:
            self.undo_stack.append(self.project.to_dict())
            self._restore(self.redo_stack.pop())

    # ---- selection --------------------------------------------------------------------------------
    def set_selection_mode(self, mode: str) -> None:
        if mode != self.selection_mode:
            self.selection_mode = mode
            self.selection = []
            self.selectionChanged.emit()

    def select(self, picks: list[Pick], add: bool = False) -> None:
        if add:
            keys = {(p.obj, p.index) for p in self.selection}
            for p in picks:
                if (p.obj, p.index) in keys:
                    self.selection = [s for s in self.selection if (s.obj, s.index) != (p.obj, p.index)]
                else:
                    self.selection.append(p)
        else:
            self.selection = list(picks)
        self.selectionChanged.emit()

    def selected_objects(self) -> list[str]:
        out = []
        for p in self.selection:
            if p.obj not in out:
                out.append(p.obj)
        return out

    # ---- geometry --------------------------------------------------------------------------------
    # pure gmsh/numpy work (build + tessellation); the window replaces this with a runner that keeps
    # the GUI responsive while a large geometry loads
    run_heavy = staticmethod(lambda fn: fn())

    def rebuild(self) -> None:
        """Rebuild display tessellation for objects whose definition changed."""
        t0 = time.time()
        des = self.design
        ev = self.evaluator()
        live = des.live_objects()
        keys = {o.name: K.object_key(des, o, ev) for o in live}
        regions = [o.name for o in live if o.history[0].kind == "CreateRegion"]
        todo = [n for n, k in keys.items() if n not in regions and (n not in self.cache or self.cache[n].key != k)]
        K.ensure_gmsh("hertzon_gui")
        def heavy() -> None:
            build = list(todo)
            files = {}
            for n in todo:
                f = K._tess_cache_file(des, des.obj(n), ev, self.display_quality)
                t = K.load_tess(f) if f is not None and f.exists() else None
                if t is not None:
                    self.cache[n] = CacheEntry(keys[n], t, None)
                    build.remove(n)
                elif f is not None:
                    files[n] = f
            if build:
                gmsh.model.remove()
                gmsh.model.add("hertzon_gui")
                ctx = K.build_design(des, ev, names=build)
                tess = K.tessellate(ctx, self.display_quality) if ctx.built else {}
                for n in build:
                    self.cache[n] = CacheEntry(keys[n], tess.get(n), ctx.errors.get(n))
                    if n in files and tess.get(n) is not None and not ctx.errors.get(n):
                        K.save_tess(files[n], tess[n])
            if regions:
                # regions depend on the bounding box of all model objects
                boxes = [self.cache[o.name].tess.bbox for o in live
                         if o.name not in regions and o.model and o.name in self.cache and self.cache[o.name].tess]
                if boxes:
                    b = np.array(boxes)
                    bb = (*b[:, :3].min(0), *b[:, 3:].max(0))
                else:
                    bb = None
                gmsh.model.remove()
                gmsh.model.add("hertzon_gui")
                ctx = K.build_design(des, ev, names=regions, extra_bbox=bb)
                tess = K.tessellate(ctx, self.display_quality) if ctx.built else {}
                for n in regions:
                    self.cache[n] = CacheEntry(keys[n] + str(bb), tess.get(n), ctx.errors.get(n))

        self.run_heavy(heavy)
        # drop stale entries
        for n in list(self.cache):
            if n not in keys:
                del self.cache[n]
        self.tess = {n: e.tess for n, e in self.cache.items() if e.tess is not None}
        from ..model.facetrack import track_faces
        for msg in track_faces(des, self.tess):
            self.message.emit("warning" if "no longer" in msg else "info", msg)
        new_errors = {n: e.error for n, e in self.cache.items() if e.error}
        for n, e in new_errors.items():
            if self.errors.get(n) != e:
                self.message.emit("error", f"{n}: {e}")
        self.errors = new_errors
        self.last_build_time = time.time() - t0
        self.geometryChanged.emit()

    def model_bbox(self):
        boxes = [t.bbox for n, t in self.tess.items()]
        if not boxes:
            return (-1e-3, -1e-3, -1e-3, 1e-3, 1e-3, 1e-3)
        b = np.array(boxes)
        return (*b[:, :3].min(0), *b[:, 3:].max(0))


def result_dir(doc, setup: str):
    """Results of a setup for the current design: the nominal folder, or the solved parametric variation
    the design variables are set to (shown without solving again)."""
    from ..solver.results import results_dir
    if not doc.project.path:
        return results_dir(doc.project.path, doc.design_name, setup)
    from ..solver.parametric import solution_dir
    return solution_dir(doc.project, doc.design, setup)
