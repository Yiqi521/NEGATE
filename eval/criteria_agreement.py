#!/usr/bin/env python
"""判据间一致性：把每个客观判据视为独立"算法标注者"，在场景级 / 候选级计算两两 Cohen κ 与 Fleiss κ。

场景级标注者（"该场景中放慢/等待是否无正当理由"）：
  A_replay = c3_replay（非反应式回放 + 恒速外推间隙）
  A_gap    = O6（HCM 间隙接受：无冲突 ∨ t0 有 ≥ t_c 间隙 ∨ 专家果断且 PET 达标）
候选级标注者（"该候选是否安全"）：
  S_replay = c1（NC=DAC=TTC=1）       S_pet = PET ≥ τ       S_rss = RSS 满足
用法: python eval/criteria_agreement.py --negatives results/negatives_navtest_v7.parquet [--out results/criteria_agreement_v7.csv]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score


def fleiss_kappa(M: np.ndarray) -> float:
    """M: (N items, k categories) 计数矩阵。"""
    N, k = M.shape; n = M.sum(axis=1)[0]
    p = M.sum(axis=0) / (N * n); P = ((M * M).sum(axis=1) - n) / (n * (n - 1))
    Pbar, Pe = P.mean(), (p * p).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--negatives", type=Path, required=True); ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pet", type=float, default=1.5); a = ap.parse_args()
    d = pd.read_parquet(a.negatives); rows = []
    # ---- 场景级 ----
    sc = d.groupby("token").first()
    A = pd.DataFrame({"replay": sc.c3_replay.astype(bool), "gap_HCM": sc.c3_pass.astype(bool)})
    if "expert_category" in sc:
        A["expert_not_justified"] = sc.expert_category.isin(["no_conflict", "decisive", "over_conservative"])
    print(f"== 场景级（n={len(A)}）：'无正当理由' 标注者两两 κ ==")
    for i in A.columns:
        for j in A.columns:
            if i < j:
                kap = cohen_kappa_score(A[i], A[j]); agr = (A[i] == A[j]).mean()
                print(f"  {i:22s} vs {j:22s} κ={kap:6.3f} agreement={agr:.3f}  ({i} 真={A[i].mean():.2f}, {j} 真={A[j].mean():.2f})")
                rows.append(dict(level="scene", a=i, b=j, kappa=kap, agreement=agr))
    M = np.stack([A.sum(axis=1).values, (A.shape[1] - A.sum(axis=1)).values], axis=1)
    print(f"  Fleiss κ（{A.shape[1]} 标注者）= {fleiss_kappa(M):.3f}")
    # ---- 候选级 ----
    k = d[d.kin_ok & d.c1.notna()].copy()
    if "pet_candidate_s" in k:
        S = pd.DataFrame({"replay_NC_DAC_TTC": k.c1.astype(bool),
                          "PET": (k.pet_candidate_s.isna() | (k.pet_candidate_s >= a.pet)),
                          "RSS": (k.rss_rel_ok if "rss_rel_ok" in k else k.rss_ok_candidate).astype(bool)})
        print(f"\n== 候选级（n={len(S)}）：'候选安全' 标注者两两 κ ==")
        for i in S.columns:
            for j in S.columns:
                if i < j:
                    kap = cohen_kappa_score(S[i], S[j]); agr = (S[i] == S[j]).mean()
                    print(f"  {i:18s} vs {j:6s} κ={kap:6.3f} agreement={agr:.3f}  ({i} 真={S[i].mean():.3f}, {j} 真={S[j].mean():.3f})")
                    rows.append(dict(level="candidate", a=i, b=j, kappa=kap, agreement=agr))
        M2 = np.stack([S.sum(axis=1).values, (S.shape[1] - S.sum(axis=1)).values], axis=1)
        print(f"  Fleiss κ（3 标注者）= {fleiss_kappa(M2):.3f}")
        print("  注：安全判据的'真'比例都接近 1，κ 会因基率效应偏低；应同时报告分歧的绝对数量。")
        dis = S[~(S.replay_NC_DAC_TTC & S.PET & S.RSS)]
        print("  分歧/不安全候选数:", len(dis), "| 仅 PET 否:", int((S.replay_NC_DAC_TTC & ~S.PET & S.RSS).sum()), "| 仅 RSS(相对) 否:", int((S.replay_NC_DAC_TTC & S.PET & ~S.RSS).sum()), "| 仅回放否:", int((~S.replay_NC_DAC_TTC & S.PET & S.RSS).sum()))
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False); print("->", a.out)


if __name__ == "__main__":
    main()
