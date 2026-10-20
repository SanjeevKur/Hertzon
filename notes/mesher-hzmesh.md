# hzmesh: lokale Delaunay-Verfeinerung für EMerge

Neue Verfeinerungsmethode **„Local (Delaunay, experimental)“** in Hertzon (Einstellung
*RefinementMethod*). Sie ersetzt im adaptiven Lauf das Neuvernetzen durch gmsh. Das Ausgangsnetz
eines Passes bleibt vollständig erhalten. Neue Knoten kommen nur dort hinzu, wo der Fehlerschätzer
es verlangt, wie bei kommerziellen Solvern. Konzept und Probleme stammen aus [lokales-meshing-plan.md](lokales-meshing-plan.md)
und [konzept-adaptives-meshing.md](konzept-adaptives-meshing.md).

## Warum nicht Bisektion?

Der Plan sah Bisektion vor. Sie ist verschachtelt und konform, hat aber zwei Nachteile:

- Flache Start-Tets bleiben flach. Gerade das Patch-Substrat (0.16 mm dick) besteht daraus.
- Die Kinder entstehen nur durch Halbieren, die Netzqualität kann also nie besser werden.

Delaunay-Einfügen mit Hohlraum (Bowyer–Watson) behält ebenfalls alle alten Knoten, ist also
ebenfalls verschachtelt. Es verbindet die Nachbarschaft des neuen Punktes aber neu. Dadurch
werden schlechte Elemente beim Verfeinern repariert statt vererbt.

## Algorithmus (C++17, `hertzon/mesher/src/hzmesh.cpp`)

1. **Eingabe** aus gmsh:
   - Knoten mit Klassifikation (Punkt, Kurve, Fläche, Volumen und Entitäts-Tag);
   - Tets mit Volumen-Tag (Material);
   - Randdreiecke mit Flächen-Tag (Ports, PEC, Randbedingungen) und Kurvensegmente.
2. **Größenfeld** (Python, `adapt.py`):
   - Ist-Größe h je Knoten: Mittel der EMerge-Elementgrößen der umliegenden Tets.
   - Die Knoten der k schlechtesten Tets erhalten das Ziel 0.7·h.
   - Das Feld wird mit der Growth-Rate zum Rest des Netzes abgestuft (Dijkstra in C++).
   - Zone = Knoten, deren Ziel unter der Ist-Größe liegt. Außerhalb der Zone ändert sich **nichts**.
3. **Verfeinern** in der Reihenfolge Kurvensegmente → Randdreiecke → Tets:
   - Größe: Ist die **längste Kante** länger als h, wird ihr Mittelpunkt eingefügt (auf der
     Kurve, Fläche oder im Volumen, zu der die Kante gehört).
   - Form: Elemente mit schlechtem Umkugel-Kanten-Verhältnis (Tet > 2, Dreieck > 1.3) bekommen
     ihren Umkugel-Mittelpunkt, aber nur innerhalb der Zone.
   - Einfügen erfolgt per Bowyer–Watson-Hohlraum. Er ist durch CAD-Flächen und -Kanten begrenzt
     (eingeschränkte Delaunay), also wird keine Material- oder Port-Grenze durchschnitten.
   - Ein Randdreieck oder Segment, das vom neuen Punkt „bedrängt“ wird (encroachment), wird
     zuerst geteilt.
   - Gekrümmte CAD-Flächen und -Kurven (Kugel, Zylinder, Spline) werden über
     `gmsh.model.getClosestPoint` auf die echte Geometrie projiziert.
4. **Verbessern**, nur neue Elemente: Flips 2-3 und 3-2, Kantenentfernung (optimale
   Triangulierung über 3 bis 7 Tets) und Glätten der neuen Knoten (Flächenknoten bleiben auf
   der Fläche). Alte Knoten bewegen sich nie.
5. **Wachstum steuern**: Die Zahl k der markierten Tets wird so gewählt, dass das Netz um
   RefinementPercent wächst (0.8–1.25-fach). Ein Probelauf auf einer Kopie kostet nur
   Millisekunden; es gibt höchstens 6 Versuche.
6. **Zurück an EMerge**:
   - Das Netz wird in das gmsh-Modell geschrieben (alte Knoten-Tags bleiben).
   - Dann folgt der normale EMerge-Import. Ports, Materialien und Fernfeld bleiben dadurch
     unverändert.

## Robustheit

- Shewchuks exakte Prädikate (orient3d, insphere), kompiliert mit `-ffp-contract=off`. Linux-
  und Windows-DLL liefern bitgleiche Netze (unter wine geprüft).
- Jeder Hohlraum wird vor dem Ersetzen geprüft: geschlossene Hülle, keine doppelt benutzten
  Flächen, keine Löcher, alle Grenzflächen getaggt und Volumen erhalten bei ebenen Entitäten.
  Ist die Prüfung negativ, wird der Punkt verworfen, und das Netz bleibt gültig.
- Nach jedem Pass prüft `hzm_check` die ganze Topologie. Bei einem Fehler oder einer Ausnahme
  fällt Hertzon automatisch auf „Remesh (stable)“ zurück und schreibt eine Warnung ins Log.
- Hohlraum um freie Kurven (17.10., `hzmesh_src/hzmesh.cpp`): Ein Hohlraum kann sich um eine
  freie Kurve legen, etwa die Kante eines Blechs oder eine eingebettete Kurve. Dann gingen deren
  Segmente verloren. Jetzt muss jedes Kurvensegment im Hohlraum auf dessen neuem Rand liegen, und
  kein Randdreieck, das nicht neu trianguliert wird, darf innen landen. Sonst wird das betroffene
  Dreieck mitgeteilt oder der Punkt verworfen. Diese Fassung liegt noch nicht in
  `hertzon/mesher/src/`, die DLL in `_bin/` ist aber vom 17.10.
- Stresstests auf Box, Kugel, Zylinder, Koax und Patch über viele Passes: immer gültig, alte
  Knoten unbewegt, Volumen und Flächen erhalten.

## Parallelisierung (Stand 17.10.: nicht umgesetzt, läuft seriell)

hzmesh läuft heute vollständig **seriell** auf einem Kern:

- Im C++-Kern gibt es keine Threads, kein OpenMP und keine Thread-Pools. Die `thread_local`-
  Puffer in `hzmesh.cpp` sind nur wiederverwendete Arbeitsspeicher, keine Parallelität.
- `build.sh` und `build.cmd` bauen ohne `-fopenmp` oder `-pthread`.
- Python ruft den Kern pro Pass einmal auf (`adapt.refine_model`). Auch die bis zu 6 Probeläufe
  für die Wachstumssteuerung laufen nacheinander, jeweils auf einer Kopie (`hzm_clone`).
- Parallel rechnet nur der Löser (EMerge, PARDISO, numba-Kernel der gekrümmten Elemente).

Der Kern braucht bei 570k Tets 0.15–0.3 s pro Pass, das ist <1 % der Lösungszeit.
Parallelisieren lohnt daher bisher nicht. Falls doch nötig, gibt es zwei Konzepte:

- Die Punkte eines Durchgangs werden in Gruppen eingefügt, deren Hohlräume sich nicht
  berühren (Graph-Färbung auf den Hohlraum-Tets).
- Alternativ wird die Zone in räumlich getrennte Cluster zerlegt, jeder Cluster mit einem
  Thread. Die Ränder der Cluster sind gesperrt und werden am Ende seriell erledigt.
- Beides ändert das Ergebnis nicht, solange die Einfügereihenfolge deterministisch bleibt.
- Einfacher wäre es, die Probeläufe der Wachstumssteuerung gleichzeitig auf eigenen Kopien zu
  rechnen. Das spart aber nur Millisekunden.

## EMerge-Import beschleunigt (`emerge_patches.py`)

Nach dem Mesher war `Mesh3D._pre_update` der teuerste Teil. Es baut Kanten, Dreiecke und alle
Index-Tabellen in Python-Schleifen pro Tet. Die Schleifen sind durch numpy-Sortieren und
-Suchen ersetzt:

| Netz | Original | neu |
|---|---|---|
| 35k Tets | 1.9 s | 0.3 s |
| 74k Tets | 5.2 s | 1.3 s |
| 296k Tets | 44–60 s | 3.7 s |

- Alle Arrays und Dicts sind **bitgleich** mit dem Original (inklusive Reihenfolge), getestet
  auf 6 Netzen.
- Jeder Block wird nur ersetzt, wenn der EMerge-Quelltext exakt passt.
- Bei einer anderen EMerge-Version oder einem Fehler läuft das Original.
- `HERTZON_SLOW_IMPORT=1` schaltet die Beschleunigung ab.

## Gekrümmte Elemente (`solver/curvilinear.py`)

EMerge rechnet jeden Tet affin, also mit geraden Kanten und ebenen Flächen. Ein runder Pfosten,
Stab oder Resonator wird dadurch zum Vieleck. Dieser Facettenfehler bestimmt bei stark
reflektierenden gekrümmten Objekten die S-Parameter. Der Fehlerschätzer sieht ihn nicht, weil er
nur das Feld im Netz bewertet und nicht die Abweichung des Netzes von der Geometrie.

Lösung: Tets an gekrümmten Flächen bekommen quadratische Kanten (isoparametrisch, 10 Knoten).

- Der Mittelpunkt jeder Netzkante auf einer gekrümmten CAD-Fläche oder -Kurve wird mit
  `gmsh.model.getClosestPoint` auf die echte Geometrie verschoben (Verschiebung d_e).
- Abbildung: x = y + Σ d_e φ_e(y) mit φ_e = 4 λi λj auf dem geraden Tet y, Jacobi-Matrix
  D = I + Σ d_e ⊗ ∇φ_e.
- Die kovariante Piola-Transformation lässt sich als gerader Tet mit punktweisem Material
  schreiben:
  - Rotation: μ⁻¹ → Dᵀ μ⁻¹ D / det D
  - Masse: ε → det D · D⁻¹ ε D⁻ᵀ
- Deshalb bleiben die Basisfunktionen von EMerge unverändert. Nur die Elementmatrizen der
  gekrümmten Tets werden nach EMerges Assemblierung neu berechnet: numba, parallel, 64-Punkt-
  Quadratur (Grad 7).
- Konform: Eine Fläche ohne gekrümmte Kante bleibt exakt eben, und ihre Tangentialspur ändert
  sich nicht. Jeder Tet, der eine gekrümmte Kante enthält, ist selbst gekrümmt und nutzt dieselbe
  Verschiebung dieser Kante.
- Gültigkeit: det D ≥ 0.3 an 79 Prüfpunkten je Tet, sonst wird die Verschiebung der betroffenen
  Kanten halbiert, notfalls auf null gesetzt (danach wird erneut geprüft, bis alles gültig ist).
- Eine unabhängige Referenz (Feld per finiten Differenzen der Abbildung, anisotrope komplexe
  Materialien, beide Tet-Orientierungen) bestätigt die Elementmatrizen: Abweichung 10⁻¹⁰ bei kleiner
  Krümmung, höchstens 3·10⁻⁵ an der Grenze det D ≈ 0.5.
- Gekrümmt werden PEC-, Metall- (Finite Conductivity), Symmetrie- und Materialflächen, auch
  Ellipsen, Splines, Rohre und Lofts. Bei Metall bleibt nur der sehr kleine Impedanzterm auf der
  ebenen Facette.
- Kanten auf Flächen mit eigenen Oberflächenintegralen bleiben gerade: Ports, Absorber,
  Impedanz- und RLC-Ränder, dünne Leiter und periodische Ränder.
- Eine gekrümmte Fläche, die einen Port berührt, bleibt vollständig gerade. Sonst sähe die gerade
  Port-Mode eine gekrümmte Leitung dahinter, also einen Impedanzsprung (Koax, runder Hohlleiter).
- Geprüft: Mit Verschiebung null sind die Matrizen für alle 7 Basistypen von EMerge bitgenau
  gleich (relative Abweichung 5·10⁻¹⁶). Das Pfostenvolumen stimmt auf 0.09 % statt 4.7 %.
- Einstellung *Curved elements on curved surfaces* (Reiter Options, Standard: an).
  `HERTZON_CURVED=0` oder `=1` überschreibt sie für Experimente.

## Sehnen-Regel im Mesher (Standard: aus)

Vor den gekrümmten Elementen hat hzmesh Kanten auf gekrümmten Flächen nachgeteilt, wenn ihre
Sehne zu weit von der Geometrie abweicht (Parameter `sag`, Anteil der Zielgröße). Mit gekrümmten
Elementen ist das unnötig und kostet nur Tets, deshalb ist `sag` standardmäßig 0.

- `HZMESH_SAG=0.02` schaltet die Regel für Experimente ein.
- Sie misst die Abweichung relativ zu den Endpunkten und stoppt bei Kanten unter 2 % der
  Zielgröße. Ein Knoten, der neben der Fläche eingefügt werden musste, kann so keine endlose
  Teilung mehr auslösen.

## Ergebnisse

Prüfstein ist ein WR90-Hohlleiter (10 GHz) mit einem zentrierten Pfosten über die volle Höhe. Das
Feld ist dort exakt z-unabhängig, also gibt es eine genaue 2D-Referenz (isoparametrische P2-FEM,
`tests/ref_wgpost.py`, auf 6 Stellen konvergiert). Fehler = |S11 − S11,ref|. Alle Läufe: zweite
Ordnung, höchstens 12 Passes, Delta S ≤ 0.0005 in 2 Passes hintereinander, 30 % Verfeinerung.

Metallpfosten r = 1.5 mm, Referenz S11 = 0.18037 + 0.93542j (79.09°):

| Methode | gerade Elemente | gekrümmte Elemente |
|---|---|---|
| Local (Delaunay) | 0.013 bei 23.3k Tets, driftet weg | **0.0006** bei 16.9k Tets, konvergiert (81 s) |
| Remesh | 0.0030 bei 34.7k Tets, nicht konvergiert | **0.0011** bei 7.7k Tets, konvergiert (70 s) |
| Remesh (stable) | 0.0074 bei 15.8k Tets, nicht konvergiert | **0.0007** bei 9.7k Tets, konvergiert (86 s) |

Dielektrischer Pfosten r = 2 mm, εr 9.8, tan δ 10⁻⁴, Referenz S11 = −0.33672 + 0.77887j:

| Methode | gerade Elemente | gekrümmte Elemente |
|---|---|---|
| Local (Delaunay) | 0.0044 bei 23.5k Tets, nicht konvergiert | **0.0003** bei 23.5k Tets, konvergiert (178 s) |
| Remesh | 0.0021 bei 29.1k Tets, nicht konvergiert | **0.0006** bei 15.8k Tets, konvergiert (143 s) |

- Mit geraden Elementen bleibt ein Fehler von 0.002–0.013, den keine Verfeinerung schnell
  beseitigt. Bei Local ohne Krümmung wächst er sogar: Die frühen Passes treffen zufällig, weil sich
  Facetten- und Feldfehler aufheben, danach bleibt nur der Facettenfehler.
- Mit gekrümmten Elementen fällt der Fehler bei beiden Methoden glatt mit der Netzgröße, und
  das Konvergenzkriterium stoppt nahe am richtigen Wert. Bei gleicher Tet-Zahl ist der Fehler
  etwa 5- bis 15-mal kleiner.
- Die Laufzeiten mit gekrümmten Elementen enthalten noch die einmalige Kompilierung von etwa
  40 s. Mit gespeicherten Kerneln braucht Remesh am Metallpfosten 33 s statt 70 s, bei
  identischem Ergebnis.
- Local und Remesh liegen mit gekrümmten Elementen gleichauf. Local braucht pro Pass etwa 0.2 s
  statt 0.6–5 s für gmsh, die Lösung dominiert in beiden Fällen die Laufzeit.

Kupfer statt PEC (gleicher Pfosten, festes Netz mit 2.9k Tets, ohne Adaption): gerade 0.012,
gekrümmt 0.0023, also dieselbe Verbesserung wie bei PEC. Der Unterschied zu PEC (0.0002) ist der
Verlust im Kupfer.

Ebene Welle (Metallkugel r = 25 cm, 1 GHz, Local): Der Lauf funktioniert, die gekrümmten
Elemente sind aktiv, und die Absorberflächen bleiben gerade. Die Kugel im Beispiel *NATO-Sphere*
ist dagegen facettiert (800 Ecken) und bekommt deshalb zu Recht keine gekrümmten Elemente. Für
einen Vergleich mit der Mie-Lösung reichen 18k bis 40k Tets dort noch nicht: Der RCS liegt 1.3 bis
6 dB daneben, mit und ohne Krümmung.

Patch-Antenne (keine gekrümmten Flächen, keine exakte Referenz): Local endet bei
−0.2167+0.0034j (28.5k Tets), Remesh bei −0.1788+0.0448j (18.5k), Remesh (stable) bei
−0.1413+0.0482j (22.7k). Local verfeinert nur lokal, deshalb kostet das Netz 3.9 s gegenüber
16 s bzw. 50 s bei gmsh.


## Benutzung

- In den Setup-Einstellungen *RefinementMethod* = „Local (Delaunay, experimental)“ wählen.
  RefinementPercent und GrowthRate wirken wie bei Remesh.
- Schalter für Experimente (Umgebungsvariablen):
  - `HZMESH_SHRINK`: Zielgröße der markierten Elemente, Standard 0.7;
  - `HZMESH_GRADE=0`: keine Abstufung;
  - `HZMESH_DUMP=<ordner>`: speichert die Eingabe jedes Passes;
  - `HZMESH_LIB=<pfad>`: andere Bibliothek laden;
  - `HZMESH_SAG=<anteil>`: Sehnen-Regel einschalten (siehe oben);
  - `HERTZON_CURVED=0|1`: gekrümmte Elemente erzwingen oder abschalten;
  - `HERTZON_CURVED_CHECK=1`: Volumen gerade/gekrümmt ins Log schreiben.
- Bauen: `hertzon/mesher/src/build.sh` (Linux, mit mingw zusätzlich die Windows-DLL).
  Die fertigen Bibliotheken liegen in `hertzon/mesher/_bin/`.

## Offene Punkte

- Gekrümmte Port-Flächen sind nicht umgesetzt: Dafür müssten der 2D-Modenlöser und die
  Port-Integrale ebenfalls gekrümmt rechnen. Koax- und Rundhohlleiter-Ports bleiben deshalb bei
  geraden Elementen an den angrenzenden Leiterflächen.
- Facettierte Grundkörper (NumberOfSegments oder NumberOfVertices > 0) sind eben und bekommen
  keine gekrümmten Elemente. Für runde Körper die echte Geometrie verwenden (Wert 0).
- Mit eingeschalteter Sehnen-Regel (`HZMESH_SAG`) geht die Information über Knoten neben der
  Fläche zwischen zwei Passes verloren. Die Tiefenbegrenzung hält die Folgen klein.
- Bei Modellen mit sehr vielen CAD-Flächen braucht das Zurückschreiben des Netzes in gmsh länger,
  weil gmsh je Fläche und Kante einen Aufruf braucht. Bei der facettierten Kugel mit 1600 Flächen
  sind das 2 s pro Pass, bei glatten Modellen 0.2 s.
- Der Fehlerschätzer rechnet weiter auf den geraden Tets. Das beeinflusst nur, welche Tets
  markiert werden, nicht das Ergebnis.
- Die gekrümmten Kernel werden beim ersten Lauf von numba kompiliert (etwa 40 s im ersten Pass)
  und dann im `__pycache__` neben `curvilinear.py` gespeichert. Danach entfällt das.

- Vereinzelt bleiben Tets mit 3–8° Diederwinkel. Sie stammen fast alle aus dem gmsh-Startnetz
  im dünnen Substrat. Ein Kantenkollaps oder das Verschieben alter Knoten könnte sie entfernen,
  dann wären die Netze aber nicht mehr streng verschachtelt.
- Der Fehlerschätzer und die Markierung stammen aus EMerge. Die langsame Konvergenz am Patch
  (S11 driftet noch bei 20–40k Tets) ist eine Eigenschaft des Problems: Remesh zeigt die Drift
  wegen des Netzrauschens nur nicht.
