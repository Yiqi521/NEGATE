#!/usr/bin/env python
"""分层 PDMS 评估：读取 run_pdm_score 的 CSV（可多种子），按场景子集与专家保守性分层，给出均值、种子间标准差与帧级 bootstrap 95% CI；
可选与基线 CSV 做配对差异并执行拒绝准则（任一安全子项 CI 下界低于基线均值 → 标红）。

用法:
  python eval/stratified_pdms.py --runs exp/ltf_s0.csv exp/ltf_s1.csv --flags results/scene_flags_navtest.parquet \
      [--negatives results/negatives_navtest_v5.parquet] [--baseline base_s0.csv base_s1.csv] [--out results/eval_ltf.csv]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]
SAFETY = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "driving_direction_compliance"]


def load_runs(paths):
    dfs = []
    for i, p in enumerate(paths):
        d = pd.read_csv(p, index_col=0)
        d = d[d.token.notna() & (d.token.astype(str) != "nan")]  # 去掉末行均值
        d = d[d["valid"] == True] if "valid" in d else d
        d["seed"] = i; dfs.append(d[["token", "seed"] + SUB])
    return pd.concat(dfs, ignore_index=True)


def strata(flags: pd.DataFrame, negatives: pd.DataFrame = None):
    s = {"all": flags.token, "unprotected_turn": flags[flags.unprotected_turn].token, "merging": flags[flags.merging].token}
    if "unprotected_crossing" in flags:
        s["unprotected_crossing"] = flags[flags.unprotected_crossing & ~flags.unprotected_turn].token
        s["interact"] = flags[flags.unprotected_turn | flags.merging | flags.unprotected_crossing].token
    else:
        s["interact"] = flags[flags.unprotected_turn | flags.merging].token
    if negatives is not None:
        sc = negatives.groupby("token").first()
        s["expert_conservative"] = pd.Series(sc[sc.exp_ep < 0.7].index)
        s["has_negative"] = pd.Series(negatives[negatives.is_negative].token.unique())
    return {k: set(v) for k, v in s.items()}


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, dtype=float)
    if len(x) == 0: return (np.nan, np.nan)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(runs, strat, name):
    rows = []
    for sname, toks in strat.items():
        r = runs[runs.token.isin(toks)]
        if r.empty: continue
        per_seed = r.groupby("seed")[SUB].mean()
        frame_mean = r.groupby("token")[SUB].mean()  # 先对种子平均再做帧级 bootstrap
        row = {"run": name, "stratum": sname, "n_scenes": frame_mean.shape[0], "n_seeds": per_seed.shape[0]}
        for m in SUB:
            row[m] = 100 * per_seed[m].mean(); row[m + "_seed_std"] = 100 * per_seed[m].std(ddof=0)
            lo, hi = boot_ci(frame_mean[m].values); row[m + "_ci_lo"], row[m + "_ci_hi"] = 100 * lo, 100 * hi
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--baseline", nargs="*", type=Path, default=None)
    ap.add_argument("--flags", type=Path, required=True)
    ap.add_argument("--negatives", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    flags = pd.read_parquet(a.flags); negs = pd.read_parquet(a.negatives) if a.negatives else None
    strat = strata(flags, negs)
    runs = load_runs(a.runs); out = summarize(runs, strat, "run")
    if a.baseline:
        base = load_runs(a.baseline); ob = summarize(base, strat, "baseline"); out = pd.concat([ob, out], ignore_index=True)
        # 配对差异（同 token，先按种子平均）
        rm = runs.groupby("token")[SUB].mean(); bm = base.groupby("token")[SUB].mean(); common = rm.index.intersection(bm.index)
        diff = (rm.loc[common] - bm.loc[common])
        print("\n== 配对差异 run − baseline（百分点，帧级 bootstrap 95% CI）==")
        for sname, toks in strat.items():
            d = diff[diff.index.isin(toks)]
            if d.empty: continue
            line = f"{sname:22s} n={len(d):5d} "
            reject = False
            for m in ["ego_progress", "score"] + SAFETY:
                lo, hi = boot_ci(d[m].values); line += f"| {m[:12]} {100*d[m].mean():+5.2f} [{100*lo:+5.2f},{100*hi:+5.2f}] "
                if m in SAFETY and hi < 0: reject = True
            print(line + ("  <-- 安全子项显著劣化，拒绝" if reject else ""))
    pd.set_option("display.width", 200)
    cols = ["run", "stratum", "n_scenes", "n_seeds"] + [c for c in SUB]
    print("\n== 分层均值（%）==\n", out[cols].round(2).to_string(index=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.out, index=False); print("->", a.out)


if __name__ == "__main__":
    main()
