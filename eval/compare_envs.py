#!/usr/bin/env python
"""比较两套环境（如 cu117 与 cu128）对同一批 checkpoint 的 navtest 评估结果：逐 token PDMS 与子分数差异。
用法: python eval/compare_envs.py --a results/eval/ltf_seed{0,1,2}_navtest.csv --b results/eval/cu128/ltf_seed{0,1,2}_navtest.csv --labels cu117 cu128
"""
import argparse
from pathlib import Path
import pandas as pd

COLS = ["score", "ego_progress", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "comfort"]


def load(p):
    d = pd.read_csv(p, index_col=0); d = d[d.token.notna() & (d.token.astype(str) != "nan")]
    return d.set_index("token")[COLS]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", nargs="+", type=Path, required=True); ap.add_argument("--b", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs=2, default=["A", "B"]); ap.add_argument("--tol", type=float, default=1e-3)
    x = ap.parse_args(); la, lb = x.labels
    print(f"{'seed':>4s} {'n':>6s} {'PDMS '+la:>12s} {'PDMS '+lb:>12s} {'max|Δ| token':>13s} {'tokens |Δ|>tol':>15s}  子分数最大差")
    for i, (pa, pb) in enumerate(zip(x.a, x.b)):
        a, b = load(pa), load(pb); j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        d = (j.score_a - j.score_b).abs()
        sub = {c: float((j[c + "_a"] - j[c + "_b"]).abs().max()) for c in COLS[1:]}
        print(f"{i:4d} {len(j):6d} {100*j.score_a.mean():12.3f} {100*j.score_b.mean():12.3f} {d.max():13.5f} {int((d > x.tol).sum()):15d}  {max(sub.values()):.5f}")


if __name__ == "__main__":
    main()
