# Plan: Lokale (verschachtelte) Netzverfeinerung wie bei kommerziellen Solvern

## Problem
Remesh erzeugt in jedem Pass ein komplett neues Netz. ΔS vergleicht dann zwei unabhängige Netze;
ab ca. Pass 7 ist dieses „Netzrauschen“ so groß wie die Schwelle → ΔS springt.
Kommerzielle Solver verfeinern dagegen **lokal und verschachtelt**: alte Tets bleiben, nur die schlechtesten
(~30 %) werden geteilt. Dann misst ΔS nur noch den echten Fortschritt.

## Recherche: welche Methoden gibt es?
| Methode | Verschachtelt | Konform | Aufwand | Bewertung |
|---|---|---|---|---|
| Remesh (gmsh, heute) | nein | ja | – | Rauschen, Ursache des Problems |
| Mmg (Metrik-Adaption) | nein (lokale Operationen, verschiebt Knoten) | ja | getestet | rauschig, langsam – verworfen |
| gmsh `refine()` | ja | ja | klein | nur **uniform** (jedes Tet ×8) – zu teuer |
| **Bisektion (Newest-Vertex / Arnold–Mukherjee–Pouly)** | **ja** | **ja** | mittel | **Standard der adaptiven FEM, Empfehlung** |
| Rot-Grün-Verfeinerung (1→8 + Abschluss) | ja | ja | mittel | Qualität sinkt bei Wiederholung |
| Nichtkonforme Verfeinerung (Palace/MFEM) | ja | nein (hängende Knoten) | groß | braucht Constraint-Unterstützung im Solver – nicht in EMerge |

Quellen: Arnold/Mukherjee/Pouly „Locally adapted tetrahedral meshes using bisection“
(https://www-users.cse.umn.edu/~arnold/papers/bistetima.pdf), Palace AMR
(https://arxiv.org/html/2311.06693v2), Bisektions-Bibliotheken
(https://www.math.uci.edu/~chenlong/bib/Bisection.html), Conformal n-D Bisection
(https://imtech.upc.edu/?p=1922).

## Gewählter Ansatz: eigene konforme Bisektion direkt auf dem EMerge-Netz
1. **Markieren**: wie heute die schlechtesten Tets nach Fehlerschätzer (RefinementPercent).
2. **Bisektion mit Abschluss** (numba): markierte Tets an ihrer Referenzkante (längste Kante,
   konsistent gewählt) halbieren; Nachbarn, die eine geteilte Kante haben, werden mitgeteilt
   (Closure), bis das Netz wieder konform ist. Garantiert beschränkte Formqualität
   (nur endlich viele Ähnlichkeitsklassen) – kein Qualitätsverfall über viele Passes.
3. **Geometrietreue**: neue Kantenmittelpunkte auf gekrümmten CAD-Flächen/-Kanten mit
   `gmsh.model.getClosestPoint` auf die echte Geometrie projizieren (wie bei kommerziellen Solvern).
4. **Zuordnung erhalten**: jedes Kind-Tet erbt Volumen-Tag (Material), jede Kind-Dreiecksfläche
   ihren Flächen-Tag (Ports, PEC, Randbedingungen).
5. **Zurück an EMerge**: Netz in das bestehende gmsh-Modell schreiben
   (`gmsh.model.mesh.clear` + `addNodes`/`addElementsByType` je CAD-Entität), dann der normale
   EMerge-Einlesepfad (`_pre_update`). Dadurch bleibt der Rest (Ports, Fernfeld, Plots) unverändert.
6. **Option** „Local (bisection)“ neben Remesh; nach Test als Standard.

## Schritte / Prüfung
1. Prototyp der Bisektion an einem Würfel: Konformität, Qualität über 10 Passes.
2. Rückschreiben ins gmsh-Modell, EMerge löst darauf (Patch-Antenne).
3. Projektion auf gekrümmte Flächen (Kugel, Zylinder).
4. Vergleich am Patch mit Remesh / Remesh (stable): ΔS-Verlauf, Tets, Zeit.
5. Einbau als Option, Tests, Sync.

## Risiken
- EMerge könnte beim Einlesen eines „von außen“ gesetzten Netzes etwas erwarten, das nur
  `generate()` erzeugt (z. B. periodische Ränder) → in Schritt 2 früh prüfen.
- Sehr flache Anfangs-Tets bleiben flach (Bisektion verbessert sie nicht) → Startnetz gut halten.
