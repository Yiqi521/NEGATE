#!/usr/bin/env python
"""客观标签：用 10 s 未来窗口（20 帧）为交互场景计算间隙接受 / PET / RSS / 起步延迟特征，并给出规则标签。
阈值默认值（待文献核实后在 docs/objective_criteria.md 固定）：t_c 临界间隙、t_startup 起步损失时间、PET 安全阈值。

用法: python scripts/objective_labels.py --split navtest --flags results/scene_flags_navtest.parquet --out results/objective_navtest.parquet
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from conservative_negatives.definition.objective_criteria import (agent_tracks, gap_analysis, pet_of_candidate,
                                                                   startup_delay, rss_follow_ok)
from conservative_negatives.generators.operators import g2_speed_scale, g1_delayed_departure

SPLIT_DIR = {"navtest": "test", "navtrain": "trainval"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest"); ap.add_argument("--flags", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--future-frames", type=int, default=20)
    ap.add_argument("--t-c", type=float, default=6.5); ap.add_argument("--t-startup", type=float, default=2.0)
    ap.add_argument("--pet-safe", type=float, default=1.5); ap.add_argument("--max-scenes", type=int, default=None)
    a = ap.parse_args()
    df = pd.read_parquet(a.flags)
    if "unprotected_crossing" not in df: df["unprotected_crossing"] = False
    sel = df[df.unprotected_turn | df.merging | df.unprotected_crossing]
    if a.max_scenes: sel = sel.head(a.max_scenes)
    root = Path(os.environ["OPENSCENE_DATA_ROOT"]); dev = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    sf = instantiate(OmegaConf.load(dev / f"navsim/planning/script/config/common/train_test_split/scene_filter/{a.split}.yaml"))
    sf.tokens = sel.token.tolist(); sf.log_names = sel.log_name.unique().tolist(); sf.num_future_frames = a.future_frames
    loader = SceneLoader(root / "navsim_logs" / SPLIT_DIR[a.split], root / "sensor_blobs" / SPLIT_DIR[a.split], sf, SensorConfig.build_no_sensors())
    print(f"{len(sel)} interact scenes; {len(loader)} loadable with {a.future_frames} future frames ({a.future_frames*0.5:.0f} s)")
    flags = sel.set_index("token"); rows = []; t = time.time()
    for i, tok in enumerate(loader.tokens):
        try:
            sc = loader.get_scene_from_token(tok)
            exp_long = sc.get_future_trajectory(num_trajectory_frames=a.future_frames).poses.astype(float)
            exp4 = exp_long[:8]
            v0 = float(np.linalg.norm(sc.frames[3].ego_status.ego_velocity[:2]))
            tracks = agent_tracks(sc)
            g = gap_analysis(exp_long, tracks, t_c=a.t_c)
            cand = g1_delayed_departure(exp4, 1.5) if v0 < 0.5 else g2_speed_scale(exp4, 0.65)   # 代表性负样本
            rss_e, rr_e = rss_follow_ok(sc, exp_long); rss_c, rr_c = rss_follow_ok(sc, cand)
            sd = startup_delay(exp_long)
            r = dict(token=tok, log_name=sc.scene_metadata.log_name, cls=("turn" if flags.loc[tok, "unprotected_turn"] else ("merging" if flags.loc[tok, "merging"] else "crossing")),
                     ego_speed_t0=v0, startup_delay_expert_s=sd, rss_ok_expert=rss_e, rss_ratio_expert=rr_e,
                     pet_candidate_s=pet_of_candidate(cand, tracks), rss_ok_candidate=rss_c, rss_ratio_candidate=rr_c,
                     n_gaps=len(g["gaps_s"]), gaps_s=str(g["gaps_s"]), gaps_censored=str(g["gaps_censored"]))
            for k in ("conflict", "n_conflict_agents", "expert_accepted_gap_s", "expert_reached_conflict", "expert_arrival_s",
                      "max_rejected_gap_s", "pet_expert_s", "first_gap_ge_tc_s"):
                r[k] = g[k]
            r["n_rejected"] = len(g["expert_rejected_gaps_s"])
            # 专家四分类：no_conflict / decisive（未拒绝任何间隙）/ justified_wait（拒绝的都 < t_c）/ over_conservative（拒绝过 ≥ t_c）
            if not g["conflict"]:
                r["expert_category"] = "no_conflict"
            elif r["n_rejected"] == 0:
                r["expert_category"] = "decisive"
            elif g["max_rejected_gap_s"] is not None and g["max_rejected_gap_s"] >= a.t_c:
                r["expert_category"] = "over_conservative"
            else:
                r["expert_category"] = "justified_wait"
            # ---- 规则标签 ----
            # 专家过度保守（HCM 间隙接受）：拒绝过 ≥ t_c 的间隙，或 t0 即有 ≥ t_c 间隙却起步延迟 > t_startup
            r["L_expert_rejected_tc_gap"] = bool(g["max_rejected_gap_s"] is not None and g["max_rejected_gap_s"] >= a.t_c)
            r["L_expert_late_start"] = bool(g["first_gap_ge_tc_s"] == 0.0 and v0 < 0.5 and sd > a.t_startup)
            # 专家谨慎有理由：存在冲突流且所有被拒间隙 < t_c（没有可接受的间隙）
            r["L_expert_justified"] = bool(g["conflict"] and (g["max_rejected_gap_s"] is None or g["max_rejected_gap_s"] < a.t_c) and not r["L_expert_late_start"])
            # 候选负样本：安全（PET ≥ 阈值 或 无冲突）且 RSS 满足
            r["L_cand_safe"] = bool((r["pet_candidate_s"] is None or r["pet_candidate_s"] >= a.pet_safe) and rss_c)
            # 候选"无理由"：无冲突流，或 t0 时已有 ≥ t_c 的间隙
            r["L_cand_unjustified"] = bool((not g["conflict"]) or g["first_gap_ge_tc_s"] == 0.0)
            rows.append(r)
        except Exception as e:
            rows.append(dict(token=tok, error=f"{type(e).__name__}: {e}"))
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(loader)} {time.time()-t:.0f}s", flush=True)
    out = pd.DataFrame(rows); a.out.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(a.out, index=False)
    ok = out[out.get("error").isna()] if "error" in out else out
    print(f"\nscenes={len(out)} errors={len(out)-len(ok)} time={time.time()-t:.0f}s")
    print("conflict rate:", round(ok.conflict.mean(), 3), "| expert reached conflict within window:", round(ok.expert_reached_conflict.dropna().mean(), 3))
    for col in ["L_expert_rejected_tc_gap", "L_expert_late_start", "L_expert_justified", "L_cand_safe", "L_cand_unjustified"]:
        print(f"  {col:26s} {int(ok[col].sum()):5d} ({100*ok[col].mean():.1f}%)")
    print("expert_category by class:\n", pd.crosstab(ok.cls, ok.expert_category).to_string())
    print("by class:\n", ok.groupby("cls")[["conflict", "L_expert_rejected_tc_gap", "L_expert_late_start", "L_expert_justified", "L_cand_safe", "L_cand_unjustified"]].mean().round(3).to_string())
    print("PET candidate quantiles:", ok.pet_candidate_s.dropna().quantile([.05, .25, .5]).round(2).to_dict(), "| n with PET:", int(ok.pet_candidate_s.notna().sum()))
    print("max rejected gap quantiles:", ok.max_rejected_gap_s.dropna().quantile([.5, .75, .9]).round(2).to_dict())


if __name__ == "__main__":
    main()
