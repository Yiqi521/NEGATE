#!/usr/bin/env python
"""合并分片输出并打印汇总。用法: python scripts/merge_shards.py results/negatives_navtrain_v1_shard*.parquet --out results/negatives_navtrain_v1.parquet"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("shards", nargs="+", type=Path); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
d = pd.concat([pd.read_parquet(p) for p in a.shards], ignore_index=True)
d.to_parquet(a.out, index=False)
sc = d.groupby("token").first(); sc["cls"] = np.where(sc.unprotected_turn, "turn", np.where(sc.merging, "merging", "crossing"))
d = d.merge(sc[["cls"]], left_on="token", right_index=True); neg = d[d.is_negative]
print(f"merged {len(a.shards)} shards -> {a.out}: scenes={len(sc)} candidates={len(d)} negatives={len(neg)} covered={neg.token.nunique()}")
print("C3 scene-level:", round(sc.c3_pass.mean(), 3), "| expert_conservative:", int((sc.exp_ep < 0.7).sum()))
for c, g in sc.groupby("cls"):
    n = neg[neg.cls == c]
    print(f"  {c:9s} scenes={len(g):5d} covered={n.token.nunique():5d} negs={len(n):6d} per_scene={len(n)/max(1,n.token.nunique()):.2f} ops={n.operator.value_counts().to_dict()}")
