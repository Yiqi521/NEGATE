#!/usr/bin/env python
"""Step 3：为目标子集生成保守负样本候选并执行四级过滤（C1 安全、C2 低进度、C3 事后验证、C4 近专家）。

用法:
  source ~/navsim_workspace/env.sh; export PYTHONPATH=$PWD
  python scripts/build_negatives.py --split navtest --flags results/scene_flags_navtest.parquet \
      --subset interact --out results/negatives_navtest_v0.parquet [--only-cached] [--max-scenes N]
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from conservative_negatives.filters.pdm_eval import PDMEvaluator
from conservative_negatives.generators.operators import generate_candidates, kinematic_ok, g2_speed_scale, advance_departure, _arc_length
from conservative_negatives.definition.objective_criteria import (agent_tracks, gap_analysis, pet_of_candidate, rss_follow_ok,
                                                                   near_stop_control, critical_gap_for, RAGLAND_85, stop_line_types_near,
                                                                   conflict_geometry, candidate_gap_verdict)
from conservative_negatives.definition.occlusion import occlusion_verdict, ego_route_lane_ids

SPLIT_DIR = {"navtest": "test", "navtrain": "trainval"}
VEH = {"vehicle"}


def conflict_clear_time(scene, t0=3, horizon=8, radius=20.0):
    """他车（运动）最后一次出现在自车未来位姿 radius 内的时间（s）；从未出现返回 None。"""
    last = None
    for k in range(1, horizon + 1):
        idx = t0 + k
        if idx >= len(scene.frames):
            break
        f = scene.frames[idx]
        ego = f.ego_status.ego_pose
        c, s = np.cos(ego[2]), np.sin(ego[2])
        ann = f.annotations
        for box, name, vel in zip(ann.boxes, ann.names, ann.velocity_3d):
            if name in VEH and np.linalg.norm(vel[:2]) > 0.5 and np.hypot(box[0], box[1]) <= radius:
                last = k * 0.5
                break
    return last


def min_arrival_gap(scene, ref_poses, t0=3, radius=30.0, speed_min=0.5, cross_dist=2.5):
    """间隙代理：t0 时刻附近运动车辆恒速外推 4 s，与参考轨迹（自车系）路径最近点距离 < cross_dist 视为交汇；
    返回 |t_ego - t_agent| 的最小值（s）；无交汇返回 None。所有量在 t0 自车系中计算。"""
    f = scene.frames[t0]; ann = f.annotations
    ego_t = np.arange(1, len(ref_poses) + 1) * 0.5           # 参考轨迹各位姿的到达时刻
    ego_xy = ref_poses[:, :2]
    best = None
    for box, name, vel in zip(ann.boxes, ann.names, ann.velocity_3d):
        if name not in VEH or np.hypot(box[0], box[1]) > radius:
            continue
        # NAVSIM 的 velocity_3d 与 boxes 同为自车局部系（navsim_scenario_utils 中才旋转到全局），无需再旋转
        v = np.asarray(vel[:2], dtype=float)
        sp = np.linalg.norm(v)
        if sp < speed_min:
            continue
        ts = np.arange(0, 4.01, 0.1)
        ag_xy = np.asarray(box[:2], dtype=float)[None, :] + ts[:, None] * v[None, :]   # (T,2)
        d = np.linalg.norm(ag_xy[:, None, :] - ego_xy[None, :, :], axis=-1)              # (T, N)
        if d.min() > cross_dist:
            continue
        ti, ei = np.unravel_index(np.argmin(d), d.shape)
        gap = abs(ts[ti] - ego_t[ei])
        best = gap if best is None else min(best, gap)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--flags", type=Path, required=True)
    ap.add_argument("--subset", default="interact", choices=["interact", "unprotected_turn", "merging", "unprotected_crossing", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-cached", action="store_true")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--metric-cache", type=Path, default=None, help="metric cache 目录（默认 $NAVSIM_EXP_ROOT/metric_cache）")
    ap.add_argument("--shard", default=None, help="k/N：按日志名把子集切成 N 份，只处理第 k 份（0 起）")
    # 阈值（P1 校准前的默认值）
    ap.add_argument("--c2-ep-ratio", type=float, default=0.8)
    ap.add_argument("--c4-min-ep-ratio", type=float, default=0.3)
    ap.add_argument("--c3-fast-alpha", type=float, default=1.15)
    ap.add_argument("--c3-advance-s", type=float, default=1.0)
    ap.add_argument("--c3-gap-s", type=float, default=2.0, help="参考轨迹与他车到达时间差阈值；小于则谨慎有理由")
    ap.add_argument("--objective", action="store_true", help="C3 改用客观判据（HCM 间隙接受 + PET + RSS），场景用 20 帧未来窗口")
    ap.add_argument("--t-c", type=float, default=None, help="临界间隙 [s]；缺省按运动类型与停车控制取 HCM 基准值")
    ap.add_argument("--pet-safe", type=float, default=1.5, help="候选轨迹 PET 安全阈值 [s]")
    ap.add_argument("--future-frames", type=int, default=20)
    ap.add_argument("--no-occlusion", action="store_true", help="关闭 O7 遮挡可达性判据")
    ap.add_argument("--control", default=None, choices=["random", "safety"],
                    help="生成对照标签：random = 任意纵向扰动且不经 O1–O7 筛选；safety = 只保留 O1 回放失败（NC 或 TTC<1）的候选")
    ap.add_argument("--control-seed", type=int, default=0)
    ap.add_argument("--occ-rho", type=float, default=1.0, help="O7 响应时间 [s]")
    ap.add_argument("--occ-vlim-factor", type=float, default=1.1, help="O7 幻影车速度 = factor × 车道限速")
    a = ap.parse_args()

    df = pd.read_parquet(a.flags)
    if "unprotected_crossing" not in df.columns:
        df["unprotected_crossing"] = False
    if a.subset == "interact":
        sel = df[df.unprotected_turn | df.merging | df.unprotected_crossing]
    elif a.subset == "all":
        sel = df
    else:
        sel = df[df[a.subset]]
    ev = PDMEvaluator(metric_cache_path=a.metric_cache)
    if a.shard:
        k, n = map(int, a.shard.split("/"))
        logs = sorted(sel.log_name.unique())
        keep = set(logs[k::n])
        sel = sel[sel.log_name.isin(keep)]
        print(f"shard {k}/{n}: {len(keep)} logs, {len(sel)} scenes")
    tokens = sel.token.tolist()
    if a.only_cached:
        tokens = [t for t in tokens if ev.has_cache(t)]
    if a.max_scenes:
        tokens = tokens[: a.max_scenes]
    print(f"subset={a.subset}: {len(sel)} scenes, {len(tokens)} to process")

    root = Path(os.environ["OPENSCENE_DATA_ROOT"]); dev = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    sf = instantiate(OmegaConf.load(dev / f"navsim/planning/script/config/common/train_test_split/scene_filter/{a.split}.yaml"))
    sf.tokens = tokens
    sf.log_names = sel.set_index("token").loc[tokens].log_name.unique().tolist()
    if a.objective:
        sf.num_future_frames = a.future_frames   # 间隙分析需要 ≥ t_c 的观测窗口
    loader = SceneLoader(root / "navsim_logs" / SPLIT_DIR[a.split], root / "sensor_blobs" / SPLIT_DIR[a.split],
                         sf, SensorConfig.build_no_sensors())
    flags = sel.set_index("token")

    rows, t, n_err = [], time.time(), 0
    for i, tok in enumerate(loader.tokens):
        try:
            scene = loader.get_scene_from_token(tok)
            exp = scene.get_future_trajectory(num_trajectory_frames=8).poses.astype(np.float64)
            v0 = float(np.linalg.norm(scene.frames[3].ego_status.ego_velocity[:2]))
            e = ev.score(tok, exp)
            # C3 参考轨迹：行驶中的专家 → 速度 ×1.15；静止起步的专家 → 提前 advance_s 起步
            ref = advance_departure(exp, a.c3_advance_s) if v0 < 0.5 else g2_speed_scale(exp, a.c3_fast_alpha)
            fast = ev.score(tok, ref)
            ref_gap = min_arrival_gap(scene, ref)
            exp_gap = min_arrival_gap(scene, exp)
            # C3（回放版）：专家安全 ∧ 参考轨迹 NC=TTC=1（DAC 因外推伪影不查）∧ 参考轨迹与他车到达时间差 ≥ 阈值
            ref_safe = fast.nc >= 1 and fast.ttc >= 1
            gap_ok = (ref_gap is None) or (ref_gap >= a.c3_gap_s)
            c3_replay = bool(e.safe() and ref_safe and gap_ok)
            # C3（客观版）：场景级 —— 无冲突车流，或 t0 时已存在 ≥ t_c 的可接受间隙（HCM 间隙接受）
            obj = None
            if a.objective:
                tracks = agent_tracks(scene)
                exp_long = scene.get_future_trajectory(num_trajectory_frames=a.future_frames).poses.astype(np.float64)
                stop_ctrl = near_stop_control(scene)
                tt = flags.loc[tok, "connector_turn_type"] if "connector_turn_type" in flags.columns else None
                t_c = a.t_c if a.t_c is not None else critical_gap_for(tt, stop_ctrl)
                gsc = gap_analysis(exp_long, tracks, t_c=t_c)
                n_rej = len(gsc["expert_rejected_gaps_s"])
                cat = ("no_conflict" if not gsc["conflict"] else "decisive" if n_rej == 0
                       else "over_conservative" if (gsc["max_rejected_gap_s"] or 0) >= t_c else "justified_wait")
                strong_over = bool(gsc["conflict"] and (gsc["max_rejected_gap_s"] or 0) >= RAGLAND_85)
                # O6：无冲突流；或 t0 已有 ≥ t_c 的间隙；或专家未拒绝任何间隙且安全通过（PET ≥ 阈值）—— 以行为揭示的可接受间隙
                decisive_safe = (n_rej == 0) and gsc["conflict"] and (gsc["pet_expert_s"] is not None) and (gsc["pet_expert_s"] >= a.pet_safe)
                scene_unjustified = (not gsc["conflict"]) or (gsc["first_gap_ge_tc_s"] == 0.0) or decisive_safe
                rss_ok_e, rss_r_e = rss_follow_ok(scene, exp, expert_poses=exp)
                conf_geo = conflict_geometry(exp_long, tracks)
                # O7 遮挡可达性：沿专家（果断）轨迹检查地图冲突点的视距是否足够
                if a.no_occlusion:
                    occ = None
                else:
                    occ = occlusion_verdict(scene, exp_long, ego_route_lane_ids(scene),
                                            v_lim_factor=a.occ_vlim_factor, rho=a.occ_rho)
                obj = dict(expert_category=cat, scene_unjustified=scene_unjustified, first_gap_ge_tc_s=gsc["first_gap_ge_tc_s"],
                           conf_geo=conf_geo, gaps_intervals=gsc["gaps_intervals"], expert_gap_interval=gsc["expert_gap_interval"], t_c=t_c,
                           max_rejected_gap_s=gsc["max_rejected_gap_s"], pet_expert_s=gsc["pet_expert_s"], t_c_used=t_c,
                           stop_controlled=stop_ctrl, stop_line_types=str(stop_line_types_near(scene)), expert_strong_over_conservative=strong_over,
                           rss_ok_expert=rss_ok_e, rss_ratio_expert=rss_r_e,
                           occ_justified=bool(occ.justified) if occ else False,
                           occ_reason=occ.reason if occ else "disabled",
                           occ_d_occ_m=occ.d_occ_m if occ else None,
                           occ_t_phantom_s=occ.t_phantom_s if occ else None,
                           occ_t_clear_s=occ.t_clear_s if occ else None,
                           tracks=tracks)
                # 场景级值仅作记录；最终 O6 在候选级判定（见下）
                c3_pass = bool(e.safe() and scene_unjustified)
                expert_pet_ok = gsc["pet_expert_s"] is None or gsc["pet_expert_s"] >= a.pet_safe
            else:
                c3_pass = c3_replay
            ccs = conflict_clear_time(scene)
            base = dict(token=tok, log_name=scene.scene_metadata.log_name,
                        unprotected_turn=bool(flags.loc[tok, "unprotected_turn"]), merging=bool(flags.loc[tok, "merging"]),
                        unprotected_crossing=bool(flags.loc[tok, "unprotected_crossing"]),
                        ego_speed_t0=v0, expert_progress_m=float(_arc_length(exp)[-1]), conflict_clear_s=ccs,
                        exp_nc=e.nc, exp_dac=e.dac, exp_ttc=e.ttc, exp_ep=e.ep, exp_pdms=e.pdms,
                        fast_nc=fast.nc, fast_ttc=fast.ttc, fast_dac=fast.dac, c3_ref=("advance" if v0 < 0.5 else "scale"), ref_min_gap_s=ref_gap, exp_min_gap_s=exp_gap, c3_ref_safe=ref_safe, c3_gap_ok=gap_ok,
                        c3_replay=c3_replay, c3_pass=c3_pass,
                        **({k: v for k, v in obj.items() if k not in ("tracks", "conf_geo", "gaps_intervals", "expert_gap_interval", "t_c")} if obj else {}))
            cands = generate_candidates(exp, conflict_clear_s=ccs, ego_speed_t0=v0)
            if a.control == "random":
                # 随机对照：速度乘 U(0.3,1.5)（可快可慢）+ 随机起步延迟 U(0,2)s，与正式算子参数网格无交集
                rng = np.random.default_rng(a.control_seed + int(tok[:8], 16) % 100000)
                from conservative_negatives.generators.operators import Candidate, g1_delayed_departure
                cands = []
                for j in range(4):
                    alpha = float(rng.uniform(0.3, 1.5)); delay = float(rng.uniform(0.0, 2.0)) if v0 < 0.5 else 0.0
                    p = g2_speed_scale(exp, alpha)
                    if delay > 0: p = g1_delayed_departure(p, round(delay * 2) / 2)
                    cands.append(Candidate("RND", {"alpha": round(alpha, 3), "delay_s": round(delay, 2), "seed": a.control_seed}, p))
            for c in cands:
                r = dict(base, operator=c.operator, params=str(c.params))
                r["kin_ok"] = kinematic_ok(c.poses)
                if not r["kin_ok"]:
                    r.update(c1=False, c2=False, c4=False, is_negative=False, is_negative_strict=False); rows.append(r); continue
                s = ev.score(tok, c.poses)
                ep_ratio = s.ep / max(e.ep, 1e-6)
                ade = float(np.mean(np.linalg.norm(c.poses[:, :2] - exp[:, :2], axis=1)))
                fde = float(np.linalg.norm(c.poses[-1, :2] - exp[-1, :2]))
                r.update(nc=s.nc, dac=s.dac, ttc=s.ttc, ep=s.ep, comfort=s.comfort, pdms=s.pdms,
                         ep_ratio=ep_ratio, ade=ade, fde=fde,
                         c1=s.safe(), c2=ep_ratio <= a.c2_ep_ratio, c4=ep_ratio >= a.c4_min_ep_ratio)
                if obj is not None:
                    verdict = candidate_gap_verdict(c.poses, obj["conf_geo"], obj["gaps_intervals"], obj["t_c"], a.pet_safe, obj["expert_gap_interval"])
                    pet_c = verdict["cand_pet_s"]; rss_c, rss_r = rss_follow_ok(scene, c.poses, expert_poses=exp)
                    r.update(pet_candidate_s=pet_c, rss_ok_candidate=rss_c, rss_ratio_candidate=rss_r,
                             cand_reached=verdict["cand_reached"], cand_arrival_s=verdict["cand_arrival_s"], cand_reason=verdict["cand_reason"])
                    # RSS 相对判据：候选的 RSS 裕度不低于专家（排队等场景中专家自身也可能不满足绝对 RSS）
                    rss_rel_ok = bool(rss_c or rss_r >= min(1.0, obj["rss_ratio_expert"]) - 1e-6)
                    r["rss_rel_ok"] = rss_rel_ok
                    r["c1_obj"] = bool((pet_c is None or pet_c >= a.pet_safe) and rss_rel_ok)
                    # 候选级 O6：专家自身在冲突点安全（PET）∧ 候选的延迟无理由
                    r["c3_cand"] = bool(e.safe() and expert_pet_ok and verdict["cand_unjustified"])
                    # O7：遮挡使谨慎有理由 → 该场景不产生负样本
                    r["c7_occ_ok"] = bool(not obj["occ_justified"])
                    if a.control == "random":
                        # 不经 O1–O7 筛选，只要求运动学可行且与专家有差异（EP 比 ≠ 1）
                        r["is_negative"] = bool(abs(ep_ratio - 1.0) > 0.05)
                    elif a.control == "safety":
                        # BeyondDrive 式安全负样本：回放中 NC 或 TTC 失败（危险），其余判据不管
                        r["is_negative"] = bool(s.nc < 1.0 or s.ttc < 1.0)
                    else:
                        r["is_negative"] = bool(r["c1"] and r["c1_obj"] and r["c2"] and r["c3_cand"] and r["c4"] and r["c7_occ_ok"])
                    r["is_negative_strict"] = bool(r["is_negative"] and (c3_replay or a.control is not None))
                else:
                    r["is_negative"] = bool(r["c1"] and r["c2"] and c3_pass and r["c4"])
                rows.append(r)
        except Exception as ex:
            n_err += 1
            if n_err <= 5:
                print(f"[warn] {tok}: {type(ex).__name__}: {ex}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(loader.tokens)}  {time.time()-t:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"\nscenes={out.token.nunique()} candidates={len(out)} errors={n_err} time={time.time()-t:.0f}s")
    print("kinematic pass rate:", round(out.kin_ok.mean(), 3))
    k = out[out.kin_ok]
    print("C1 safe:", round(k.c1.mean(), 3), " C2 low-progress:", round(k.c2.mean(), 3),
          " C3 (scene-level):", round(out.groupby('token').c3_pass.first().mean(), 3), " C4:", round(k.c4.mean(), 3))
    print("negatives:", int(out.is_negative.sum()), "; scenes with >=1 negative:",
          int(out.groupby('token').is_negative.any().sum()), "/", out.token.nunique())
    print("per operator negative counts:", out[out.is_negative].operator.value_counts().to_dict())
    if "cand_reason" in out:
        print("candidate O6 reasons (kin ok):", out[out.kin_ok].cand_reason.value_counts().to_dict())
        sc_o = out.groupby("token").first()
        print("O7 occlusion: justified scenes", int(sc_o.occ_justified.sum()), f"({100*sc_o.occ_justified.mean():.1f}%)",
              "| reasons", sc_o.occ_reason.value_counts().to_dict())
        print("strict negatives:", int(out.is_negative_strict.sum()), "; scenes:", int(out.groupby('token').is_negative_strict.any().sum()))
    print("ep_ratio quantiles (kin ok):", k.ep_ratio.quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
    print("ade quantiles (negatives):", out[out.is_negative].ade.quantile([.25, .5, .75]).round(2).to_dict())


if __name__ == "__main__":
    main()
