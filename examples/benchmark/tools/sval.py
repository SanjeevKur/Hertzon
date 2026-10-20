import json, cmath, sys
for d in sys.argv[1:]:
    j = json.load(open(d + "/convergence.json"))
    rows = j["passes"] if isinstance(j, dict) else j
    print(d)
    for r in rows:
        S = r["S"]
        s = [complex(*x) for x in S[0]] if isinstance(S[0][0], list) else [complex(*S[0])]
        v = s[0]
        print("  pass %2d tets %6d  S=%+.4f%+.4fj |S|=%.4f ang=%7.2f" % (r["pass"], r["tets"], v.real, v.imag, abs(v), cmath.phase(v) * 57.2958))
