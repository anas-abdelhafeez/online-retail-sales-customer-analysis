"""Builds index.html from data/summary.json + src/template.html (adds a few derived series)."""
import json, os
b = os.path.dirname(__file__)
D = json.load(open(f"{b}/../data/summary.json"))
M = D["monthly"]; x = M["x"]
rev = dict(zip(x, M["revenue"]))
D["y2010"] = [rev[f"2010-{m:02d}"] for m in range(1, 12)]
D["y2011"] = [rev[f"2011-{m:02d}"] for m in range(1, 12)]
D["yoy_pct"] = (sum(D["y2011"][:10]) / sum(D["y2010"][:10]) - 1) * 100
q4 = sum(v for k, v in rev.items() if k[5:] in ("09", "10", "11") and k < "2011-12")
yr = sum(v for k, v in rev.items() if "2010-12" > k >= "2010-01") + 0
D["q4share"] = (rev["2010-09"] + rev["2010-10"] + rev["2010-11"]) / sum(rev[f"2010-{m:02d}"] for m in range(1, 13)) * 100
ob = D["order_buckets"]; tot_o = sum(r[1] for r in ob); tot_r = sum(r[2] for r in ob)
D["big_orders_pct"] = sum(r[1] for r in ob if r[0] in ("1k-2.5k", "2.5k+")) / tot_o * 100
D["big_rev_pct"] = sum(r[2] for r in ob if r[0] in ("1k-2.5k", "2.5k+")) / tot_r * 100
mret = []
co = D["cohort"]; ncoh = len(co["y"]); curve = []
vals = {}
for j, i, v in co["v"]:
    if v is not None and j > 0: vals.setdefault(j, []).append(v)
D["curve"] = [round(sum(vals[j]) / len(vals[j]), 1) for j in range(1, 13)]

D["hmax"] = max(r[2] for r in D["heat"])
open(f"{b}/../index.html", "w").write(open(f"{b}/template.html").read().replace("__DATA__", json.dumps(D)))
print("ok", round(D["yoy_pct"], 1), round(D["q4share"], 1), round(D["big_orders_pct"], 1), round(D["big_rev_pct"], 1))
