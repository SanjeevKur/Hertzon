"""Edit Sources: magnitude and phase per excitation used for post processing (no re-solve), and
for incident waves which field is shown (total, scattered or incident)."""
from __future__ import annotations

import json

FIELD_TYPES = ("Total", "Scattered", "Incident")


def defaults(design) -> dict:
    out = {}
    for i, e in enumerate(design.ports()):
        out[e.name] = {"mag": 1.0 if i == 0 else 0.0, "phase": 0.0}
    for e in design.excitations:
        if e.kind == "Incident Plane Wave" and e.enabled:
            out[e.name] = {"mag": 1.0, "phase": 0.0}
    return out


def get(design) -> dict | None:
    """Edited sources, or None while the user never opened Edit Sources (plots then show one port)."""
    raw = design.settings.get("Sources", "")
    if not raw:
        return None
    try:
        saved = json.loads(raw)
    except ValueError:
        return None
    out = defaults(design)
    for k, v in saved.items():
        if k in out:
            out[k] = {"mag": float(v.get("mag", 0)), "phase": float(v.get("phase", 0))}
    return out


def set_(design, sources: dict, field_type: str) -> None:
    design.settings["Sources"] = json.dumps(sources)
    design.settings["FieldType"] = field_type


def field_type(design) -> str:
    t = design.settings.get("FieldType", "Total")
    return t if t in FIELD_TYPES else "Total"
