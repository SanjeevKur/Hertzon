import json, sys
for d in sys.argv[1:]:
    j = json.load(open(d + "/convergence.json"))
    rows = j["passes"] if isinstance(j, dict) else j
    print(d)
    for r in rows:
        print("  pass %2d tets %6d dS %s  t %.1f" % (r["pass"], r["tets"], ("%.4f" % r["delta_s"]) if "delta_s" in r else "  -   ", r.get("time", 0)))
