"""Print the paired comparisons ours - arm from the results JSON files.

    python -m bench.summarize [dataset ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).parent / "results"
OURS = "ours_centred_cv"


def curve(path: Path) -> None:
    r = json.loads(path.read_text())
    print(f"\n== {r['dataset']} (source {r['n_source_groups']} groups, test {r['n_test_groups']} groups)")
    for n, c in r["sizes"].items():
        lam = c["chosen_lam"].get(OURS, [])
        print(f"-- n_target_groups={c['n_target_groups']}  ours log-loss {c['arms'][OURS]['log_loss']:.4f}"
              f"  chosen lam {sorted(set(lam))}")
        for a, m in c["arms"].items():
            if a == OURS:
                continue
            p = m["paired_ours_minus_arm"]
            flag = "WIN" if p["upper95_total"] < 0 else ("LOSS" if p["mean_diff"] - 1.96 * p["se_total"] > 0 else "tie")
            print(f"   {a:34s} {m['log_loss']:.4f}  diff {p['mean_diff']:+.5f}  se_block {p['se_block']:.5f}"
                  f"  se_total {p['se_total']:.5f}  {flag}")


def synthetic(path: Path) -> None:
    r = json.loads(path.read_text())
    print(f"\n== synthetic ({r['n_seeds']} draws per cell)")
    for k, c in r["cells"].items():
        a = c["arms"]
        row = " ".join(f"{n}={a[n]['paired_ours_minus_arm']['mean_diff']:+.4f}({a[n]['paired_ours_minus_arm']['se']:.4f})"
                       for n in ("target_only_cv", "source_only", "pooled_indicator", "logreg_target_only"))
        print(f"{k:18s} ours {a[OURS]['log_loss']:.4f} oracle {a['oracle']['log_loss']:.4f}  {row}")


if __name__ == "__main__":
    names = sys.argv[1:] or [p.stem for p in sorted(RES.glob("*.json"))]
    for nm in names:
        p = RES / f"{nm}.json"
        (synthetic if nm.startswith("synthetic") else curve)(p)
