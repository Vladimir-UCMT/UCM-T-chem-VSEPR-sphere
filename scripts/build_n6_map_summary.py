import json
import csv
from pathlib import Path
import re

root = Path(r"C:\UCM\chem")
results_dir = root / "results"
out_csv = root / "paper_v1" / "data" / "n6_map_summary.csv"

rows = []

for p in sorted(results_dir.glob("MAP_N6_NB2_NL4_*_results_global.json")):
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)

    alpha = d.get("alpha")
    seed = d.get("diag", {}).get("seed")
    energy = d.get("energy")

    bb = d.get("angles_stats_by_type", {}).get("BB", {})
    bl = d.get("angles_stats_by_type", {}).get("BL", {})
    ll = d.get("angles_stats_by_type", {}).get("LL", {})

    rows.append({
        "file": p.name,
        "alpha": alpha,
        "seed": seed,
        "energy": energy,
        "BB_mean_deg": bb.get("mean_deg"),
        "BB_min_deg": bb.get("min_deg"),
        "BB_max_deg": bb.get("max_deg"),
        "BL_p50_deg": bl.get("p50_deg"),
        "BL_mean_deg": bl.get("mean_deg"),
        "LL_p50_deg": ll.get("p50_deg"),
        "LL_mean_deg": ll.get("mean_deg"),
    })

out_csv.parent.mkdir(parents=True, exist_ok=True)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"written: {out_csv}")
print(f"rows: {len(rows)}")
print("first 5 rows:")
for r in rows[:5]:
    print(r)