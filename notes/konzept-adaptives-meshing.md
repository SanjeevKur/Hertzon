# Konzept: Adaptives Meshing wie bei kommerziellen Solvern

## Was wir gelernt haben (gemessen am Patch, Second Order)
| Verfahren | Netze verschachtelt? | Qualität | Ergebnis |
|---|---|---|---|
| Remesh (gmsh komplett neu) | nein | gut | ΔS springt bei hohen Passes (Netzrauschen) |
| Bisektion | ja | wird schlechter, schlechte Elemente bleiben | glatt, aber zu viele Tets und komische Bereiche |
| Hybrid (Bisektion + gmsh-Optimierer) | fast | Volumen ok, **Oberflächen nicht repariert** | glatt, ΔS 0,006 bei 22k Tets |
| Mmg global | nein (verändert alles) | gut | rauschig, langsam |
| TetGen (MeshPy) `-r -q` | **nein**: nur ~20 % der alten Knoten bleiben | gut | Wachstum nicht steuerbar (×4–5 pro Pass), AGPL |
| gmsh-Delaunay mit festen alten Knoten | ja | bricht ein (q5 % 0,53 → 0,01) | unbrauchbar |

**Kernproblem:** Wir brauchen zwei Dinge gleichzeitig, die kein Werkzeug „von der Stange“ zusammen liefert:
1. **Außerhalb** der Fehlerzone bleibt das Netz **exakt** gleich (sonst Rauschen in ΔS).
2. **Innerhalb** der Zone wird nicht nur geteilt, sondern **neu und gut** vernetzt (auch Oberflächen).

## Das Konzept: „Zonen-Remesh“ (lokales Neuvernetzen mit eingefrorenem Rest)
So arbeiten im Prinzip auch kommerzielle Solver: Änderungen bleiben auf die Umgebung der schlechtesten Elemente beschränkt.

Pro adaptivem Pass:
1. **Markieren:** die Tets mit dem größten Fehler, so viele, dass das Netz um ca. 30 % wächst (wie heute).
2. **Zone bilden:** markierte Tets + 1–2 Nachbarschichten. Alles außerhalb wird **eingefroren**.
3. **Zone neu vernetzen mit Mmg** (LGPL, bereits im Projekt angebunden):
   - eingefrorene Tets, Dreiecke und Knoten als *Required* übergeben (Medit-Schlüsselwörter
     `RequiredTetrahedra`, `RequiredTriangles`, `RequiredVertices`) → Mmg darf sie nicht anfassen;
   - in der Zone Zielgröße = halbe aktuelle Kantenlänge; Mmg teilt, tauscht Kanten/Flächen,
     glättet Knoten und repariert dabei auch **Oberflächendreiecke** (auf der echten Geometrie, da
     Mmg Flächen/Kanten als Referenzen erhält; gekrümmte Flächen projizieren wir zusätzlich).
4. **Kontrolle:** Wachstum gegen Ziel-% prüfen (Zonengröße anpassen), Qualität prüfen.
5. **Notbremse:** fällt die globale Qualität trotzdem zu stark, einmal komplett mit den aktuellen
   Elementgrößen neu vernetzen (wie heute in Hybrid).
6. **Startphase:** die ersten 3–4 Passes normales Remesh (grob, billig, saubere Basis).

**Warum das die Lücke schließt:**
- Außerhalb der Zone identisch → keine Sprünge in ΔS (wie Bisektion).
- In der Zone echtes Neuvernetzen mit Qualitätsoptimierung → keine schlechten Elemente (wie Remesh).
- Kein AGPL, keine neue Abhängigkeit (Mmg ist schon drin), läuft auf Windows.

## Risiken / offene Punkte
- Mmg muss die *Required*-Elemente wirklich unverändert lassen → erster Test.
- Übergang Zone ↔ eingefroren: Mmg kann an der Grenze nicht beliebig gut werden → 1–2 Pufferschichten.
- Laufzeit von Mmg auf großen Netzen (Datei-Ein-/Ausgabe) → messen.

## Umsetzungsschritte
1. Machbarkeit: Mmg mit eingefrorenen Elementen am Würfel/Kugel (bleibt außen exakt gleich? Qualität innen?).
2. Einbau als Verfeinerung „Zone“ (ersetzt Hybrid-Schritt 1+2), Patch 12 Passes vs. Remesh/Hybrid.
3. Wenn gut: Standard; Remesh, Hybrid als Alternativen behalten.
