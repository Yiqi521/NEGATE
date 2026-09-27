#!/usr/bin/env python
"""行为指标（研究计划 §7.3 的三个自定义指标）：不必要停车率、起步延迟、间隙接受率。

与 PDMS 互补：PDMS 的 EP 只说明"慢"，这三个指标说明"为什么慢"，且判定全部基于客观判据
（HCM 临界间隙、PET、遮挡可达性），不依赖人工标注。

对每个 token 运行规划器推理得到 4 s 轨迹，然后计算：
  1. unnecessary_stop      : 规划器停车（≥1 s 近零速）而专家未停，且客观判据认为该停车无正当理由
                             （无冲突车流 或 拒绝了 ≥ t_c 的可接受间隙）且不受遮挡辩护
  2. startup_delay_delta_s : 规划器起步延迟 − 专家起步延迟（仅在专家 t0 静止的场景）
  3. gap_accepted          : 在"存在可接受间隙（≥ t_c）"的冲突场景中，规划器是否在 4 s 内通过冲突点

用法:
  python eval/behavior_metrics.py --agent-config configs/agent/ltf_camera_only.yaml \\
      --checkpoint $NAVSIM_EXP_ROOT/checkpoints/ltf/ltf_seed_0.ckpt \\
      --negatives results/negatives_navtest_v13.parquet --out results/eval/behavior_ltf_seed0.parquet
  # 参照组（人类专家自身）：加 --human，不需要 agent/checkpoint
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from conservative_negatives.definition.objective_criteria import (agent_tracks, gap_analysis, conflict_geometry,
                                                                   candidate_gap_verdict, startup_delay, _path_times)
from conservative_negatives.definition.occlusion import occlusion_verdict, ego_route_lane_ids
from conservative_negatives.tracking import Run

SPLIT_DIR = {"navtest": "test", "navtrain": "trainval"}
STOP_SPEED = 0.5          # m/s，判定"停车"的速度阈值
STOP_MIN_STEPS = 2        # 连续 ≥ 2 步（1 s）近零速才算停车


def speed_profile(poses: np.ndarray) -> np.ndarray:
    pts, _ = _path_times(poses)
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) / 0.5


def compute_trajectory_on_device(agent, agent_input, device: str) -> np.ndarray:
    """等价于 AbstractAgent.compute_trajectory，但把特征搬到 agent 所在设备（官方实现固定在 CPU）。"""
    feats = {}
    for builder in agent.get_feature_builders():
        feats.update(builder.compute_features(agent_input))
    feats = {k: v.unsqueeze(0).to(device) for k, v in feats.items()}
    with torch.no_grad():
        pred = agent.forward(feats)
    return pred["trajectory"].squeeze(0).detach().cpu().numpy().astype(np.float64)


def is_stopped(poses: np.ndarray) -> bool:
    v = speed_profile(poses)
    run = best = 0
    for x in v:
        run = run + 1 if x < STOP_SPEED else 0
        best = max(best, run)
    return bool(best >= STOP_MIN_STEPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--negatives", type=Path, required=True, help="提供 t_c_used / expert_category 等场景级字段")
    ap.add_argument("--agent-config", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--human", action="store_true", help="用人类专家轨迹作为被测对象（参照组）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pet-safe", type=float, default=1.5)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--track", action="store_true", help="记录到 results/runs（实验追踪）")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()

    neg = pd.read_parquet(a.negatives)
    sc = neg.groupby("token").first()
    tokens = list(sc.index)[: a.max_scenes] if a.max_scenes else list(sc.index)

    agent = None
    if not a.human:
        assert a.agent_config and a.checkpoint, "非 --human 模式需要 --agent-config 与 --checkpoint"
        cfg = OmegaConf.load(a.agent_config)
        cfg.checkpoint_path = str(a.checkpoint)
        agent = instantiate(cfg)
        agent.initialize()
        agent.eval().to(a.device)
        sensor_config = agent.get_sensor_config()
    else:
        sensor_config = SensorConfig.build_no_sensors()

    root = Path(os.environ["OPENSCENE_DATA_ROOT"]); dev = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    sf = instantiate(OmegaConf.load(dev / f"navsim/planning/script/config/common/train_test_split/scene_filter/{a.split}.yaml"))
    sf.tokens = tokens
    sf.log_names = sc.loc[tokens].log_name.unique().tolist()
    sf.num_future_frames = 20                      # 间隙分析需要 10 s 窗口
    loader = SceneLoader(root / "navsim_logs" / SPLIT_DIR[a.split], root / "sensor_blobs" / SPLIT_DIR[a.split],
                         sf, sensor_config)
    print(f"{len(loader)} scenes | subject = {'human expert' if a.human else 'planner ' + str(a.checkpoint.name)}")

    rows, t0, n_err = [], time.time(), 0
    for i, tok in enumerate(loader.tokens):
        try:
            scene = loader.get_scene_from_token(tok)
            exp_long = scene.get_future_trajectory(num_trajectory_frames=20).poses.astype(np.float64)
            exp4 = exp_long[:8]
            if a.human:
                pred = exp4
            else:
                pred = compute_trajectory_on_device(agent, loader.get_agent_input_from_token(tok), a.device)
            v0 = float(np.linalg.norm(scene.frames[3].ego_status.ego_velocity[:2]))
            t_c = float(sc.loc[tok, "t_c_used"]) if "t_c_used" in sc.columns else 6.5

            tracks = agent_tracks(scene)
            g = gap_analysis(exp_long, tracks, t_c=t_c)
            cg = conflict_geometry(exp_long, tracks)
            verdict = candidate_gap_verdict(pred, cg, g["gaps_intervals"], t_c, a.pet_safe, g["expert_gap_interval"])
            exp_verdict = candidate_gap_verdict(exp4, cg, g["gaps_intervals"], t_c, a.pet_safe, g["expert_gap_interval"])
            occ = (bool(sc.loc[tok, "occ_justified"]) if "occ_justified" in sc.columns
                   else occlusion_verdict(scene, exp_long, ego_route_lane_ids(scene)).justified)

            pred_stop, exp_stop = is_stopped(pred), is_stopped(exp4)
            sd_pred, sd_exp = startup_delay(pred), startup_delay(exp4)
            acceptable_gap = bool(g["conflict"] and ((g["first_gap_ge_tc_s"] == 0.0) or
                                                     (g["expert_gap_interval"] is not None and
                                                      (g["expert_gap_interval"][1] - g["expert_gap_interval"][0]) >= t_c)))
            # 可比性：只统计"专家已在 4 s 内通过的可接受间隙"，否则冲突点在视界之外，任何轨迹都到不了
            gap_eligible = bool(acceptable_gap and exp_verdict["cand_reached"])
            rows.append(dict(
                token=tok, log_name=scene.scene_metadata.log_name, ego_speed_t0=v0, t_c=t_c,
                expert_category=sc.loc[tok, "expert_category"] if "expert_category" in sc.columns else None,
                pred_progress_m=float(np.linalg.norm(pred[-1, :2])), exp_progress_m=float(np.linalg.norm(exp4[-1, :2])),
                pred_stopped=pred_stop, expert_stopped=exp_stop,
                # 指标 1：不必要停车
                unnecessary_stop=bool(pred_stop and not exp_stop and not occ and
                                      verdict["cand_reason"] in ("no_conflict", "rejected_acceptable_gap")),
                stop_justified_by_gap=bool(pred_stop and verdict["cand_reason"] == "waiting_no_acceptable_gap"),
                stop_justified_by_occlusion=bool(pred_stop and occ),
                # 指标 2：起步延迟
                expert_starts_from_stop=bool(v0 < STOP_SPEED),
                startup_delay_pred_s=sd_pred, startup_delay_expert_s=sd_exp,
                startup_delay_delta_s=float(sd_pred - sd_exp) if v0 < STOP_SPEED else np.nan,
                # 指标 3：间隙接受
                acceptable_gap_exists=acceptable_gap, expert_reached_conflict_4s=bool(exp_verdict["cand_reached"]),
                gap_eligible=gap_eligible,
                gap_accepted=bool(verdict["cand_reached"]) if gap_eligible else np.nan,
                cand_reason=verdict["cand_reason"], cand_pet_s=verdict["cand_pet_s"], occ_justified=occ,
            ))
        except Exception as ex:
            n_err += 1
            if n_err <= 5:
                print(f"[warn] {tok}: {type(ex).__name__}: {ex}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(loader)} {time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit(f"所有 {n_err} 个场景都失败，未产生任何结果（见上方 [warn]）")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"\nscenes={len(out)} errors={n_err} time={time.time()-t0:.0f}s -> {a.out}")
    summary = summarize(out)
    if a.track:
        name = a.run_name or ("behavior_human" if a.human else f"behavior_{Path(a.checkpoint).stem}")
        with Run(name, config={"split": a.split, "negatives": str(a.negatives), "human": a.human,
                               "checkpoint": str(a.checkpoint) if a.checkpoint else None,
                               "pet_safe": a.pet_safe}, tags=["step5", "behavior"]) as run:
            run.log_artifact(a.out)
            run.finish({**summary, "n_scenes": len(out), "n_errors": n_err})


def summarize(out: pd.DataFrame) -> dict:
    n = len(out)
    print("\n== 行为指标 ==")
    print(f"  停车场景                 {int(out.pred_stopped.sum()):5d} / {n} ({100*out.pred_stopped.mean():.1f}%)"
          f"   [专家停车 {int(out.expert_stopped.sum())}]")
    print(f"  不必要停车率             {100*out.unnecessary_stop.mean():5.2f}%  ({int(out.unnecessary_stop.sum())} 场景)")
    print(f"    其中被间隙判据辩护     {int(out.stop_justified_by_gap.sum()):5d}")
    print(f"    其中被遮挡判据辩护     {int(out.stop_justified_by_occlusion.sum()):5d}")
    d = out[out.expert_starts_from_stop]
    if len(d):
        print(f"  起步延迟 Δ（vs 专家）    {d.startup_delay_delta_s.mean():+5.2f} s  (中位 {d.startup_delay_delta_s.median():+.2f}, n={len(d)})")
    e = out[out.gap_eligible]
    if len(e):
        print(f"  间隙接受率               {100*e.gap_accepted.mean():5.1f}%  ({int(e.gap_accepted.sum())}/{len(e)} 个专家已通过的可接受间隙)")
        print(f"    [存在可接受间隙的场景 {int(out.acceptable_gap_exists.sum())}，其中专家 4 s 内通过 {len(e)}]")
    if "expert_category" in out and out.expert_category.notna().any():
        print("\n  按专家四分类:")
        g = out.groupby("expert_category").agg(n=("token", "size"), unnec_stop=("unnecessary_stop", "mean"),
                                               delay=("startup_delay_delta_s", "mean"), gap=("gap_accepted", "mean"))
        print(g.round(3).to_string())
    return {
        "stop_rate": float(out.pred_stopped.mean()),
        "unnecessary_stop_rate": float(out.unnecessary_stop.mean()),
        "n_unnecessary_stop": int(out.unnecessary_stop.sum()),
        "startup_delay_delta_mean_s": float(d.startup_delay_delta_s.mean()) if len(d) else float("nan"),
        "startup_delay_delta_median_s": float(d.startup_delay_delta_s.median()) if len(d) else float("nan"),
        "n_startup_scenes": int(len(d)),
        "gap_acceptance_rate": float(e.gap_accepted.mean()) if len(e) else float("nan"),
        "n_gap_eligible": int(len(e)),
    }


if __name__ == "__main__":
    main()
