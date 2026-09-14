"""客观判据（替代/校验人工标注）：基于日志中他车真实未来位置的间隙接受、PET、RSS 与起步延迟指标。

所有量在 t0 自车后轴系中计算；他车位置来自 scene.frames[t0+k].annotations（k = 0..8，0.5 s），先转换到 t0 自车系。
阈值以参数形式给出，默认值来源见 docs/objective_criteria.md（文献核实后填写）。

主要输出（每场景）：
  - conflict: 是否存在与自车路径交汇的他车流
  - gaps_s: 冲突点处他车到达时刻序列 → 可用间隙（相邻到达时刻之差，含 t0 到首车、末车到视界末端）
  - expert_accepted_gap_s / expert_rejected_gaps_s: 专家实际穿越冲突点时所处的间隙及其之前拒绝的间隙
  - pet_expert_s / pet_candidate_s: 专家 / 候选轨迹与最近他车在冲突点的 PET
  - startup_delay_s: 专家从 t0 到首次 v > 0.5 m/s 的时间；gap_open_at_s: 首个 ≥ t_c 的间隙开始时刻
  - rss_ok: 专家 / 候选沿路径相对前车的 RSS 纵向安全距离是否始终满足
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

DT = 0.5
VEH = {"vehicle"}


@dataclass
class RSSParams:
    rho: float = 1.0             # 响应时间 [s]
    a_max_accel: float = 3.5     # 后车最大加速度 [m/s²]
    a_min_brake: float = 4.0     # 后车最小制动减速度 [m/s²]
    a_max_brake: float = 8.0     # 前车最大制动减速度 [m/s²]


def rss_long_min_dist(v_rear: float, v_front: float, p: RSSParams) -> float:
    """RSS 纵向最小安全距离（Shalev-Shwartz et al. 2017, Lemma 2）。"""
    d = (v_rear * p.rho + 0.5 * p.a_max_accel * p.rho ** 2
         + (v_rear + p.rho * p.a_max_accel) ** 2 / (2 * p.a_min_brake)
         - v_front ** 2 / (2 * p.a_max_brake))
    return max(0.0, d)


# ---------- 坐标与轨迹工具 ----------

def _to_t0_frame(scene, k: int):
    """把 t0+k 帧的他车框（该帧自车系）转换到 t0 自车系；返回 (xy (N,2), vel (N,2), names)。"""
    t0 = scene.scene_metadata.num_history_frames - 1
    f0, fk = scene.frames[t0], scene.frames[t0 + k]
    p0, pk = f0.ego_status.ego_pose, fk.ego_status.ego_pose          # 全局位姿 (x, y, yaw)
    ann = fk.annotations
    if len(ann.boxes) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), []
    xy_k = np.asarray(ann.boxes)[:, :2].astype(float)
    c, s = np.cos(pk[2]), np.sin(pk[2])
    gx = pk[0] + c * xy_k[:, 0] - s * xy_k[:, 1]
    gy = pk[1] + s * xy_k[:, 0] + c * xy_k[:, 1]
    c0, s0 = np.cos(-p0[2]), np.sin(-p0[2])
    dx, dy = gx - p0[0], gy - p0[1]
    xy0 = np.stack([c0 * dx - s0 * dy, s0 * dx + c0 * dy], axis=1)
    vel = np.asarray(ann.velocity_3d)[:, :2].astype(float)         # 自车局部系速度（不旋转，见 navsim_scenario_utils）
    return xy0, vel, list(ann.names)


def agent_tracks(scene, horizon: Optional[int] = None) -> Dict[str, np.ndarray]:
    """track_token → (H+1, 2) 在 t0 自车系中的位置序列（缺帧为 nan）。"""
    t0 = scene.scene_metadata.num_history_frames - 1
    tracks: Dict[str, np.ndarray] = {}
    H = len(scene.frames) - 1 - t0 if horizon is None else min(horizon, len(scene.frames) - 1 - t0)
    for k in range(H + 1):
        xy, _, names = _to_t0_frame(scene, k)
        toks = scene.frames[t0 + k].annotations.track_tokens
        for i, tok in enumerate(toks):
            if names[i] not in VEH:
                continue
            if tok not in tracks:
                tracks[tok] = np.full((H + 1, 2), np.nan)
            tracks[tok][k] = xy[i]
    return tracks


def _path_times(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """自车路径点 (含原点) 与到达时刻。"""
    pts = np.concatenate([np.zeros((1, 2)), poses[:, :2]], axis=0)
    t = np.arange(len(pts)) * DT
    return pts, t


def _arrival_time_along_path(pts: np.ndarray, t: np.ndarray, point: np.ndarray) -> float:
    """自车到达路径上最接近 point 的位置的时刻（线性插值）。若自车从未到达该弧长（停车），返回 inf。"""
    d = np.linalg.norm(pts - point[None, :], axis=1)
    i = int(np.argmin(d))
    return float(t[i])


def crossing_conflicts(poses: np.ndarray, tracks: Dict[str, np.ndarray], cross_dist: float = 2.5,
                       ahead_min_m: float = 1.0) -> List[dict]:
    """找出其未来位置与自车路径交汇（距离 < cross_dist）的他车；返回每辆车的冲突点、他车到达时刻与自车到达时刻。"""
    pts, t = _path_times(poses)
    out = []
    for tok, tr in tracks.items():
        valid = ~np.isnan(tr[:, 0])
        if valid.sum() < 2:
            continue
        # 他车各时刻到自车路径的最近距离
        d = np.linalg.norm(tr[valid][:, None, :] - pts[None, :, :], axis=-1)   # (Tk, N)
        k_idx = np.where(valid)[0]
        kk, ii = np.unravel_index(np.argmin(d), d.shape)
        if d[kk, ii] > cross_dist or pts[ii, 0] < ahead_min_m:      # 不交汇或冲突点在自车后方
            continue
        # 排除与自车同向且始终在同一路径上的前车（跟驰，不属于穿越冲突）：他车在所有帧都靠近路径则视为同流
        same_stream = np.mean(d.min(axis=1) < cross_dist) > 0.8
        out.append(dict(track=tok, conflict_xy=pts[ii], agent_arrival_s=float(k_idx[kk] * DT),
                        ego_arrival_s=float(t[ii]), same_stream=bool(same_stream)))
    return out


def gap_analysis(poses_expert: np.ndarray, tracks: Dict[str, np.ndarray], t_c: float = 6.5,
                 horizon_s: Optional[float] = None) -> dict:
    """在冲突点处计算他车到达时刻序列、可用间隙（含左/右截断标记）、专家接受/拒绝的间隙与 PET。
    poses_expert 可为 10 s（20 帧）轨迹以覆盖临界间隙；horizon_s 默认 = 轨迹时长。"""
    horizon_s = horizon_s or len(poses_expert) * DT
    conf = [c for c in crossing_conflicts(poses_expert, tracks) if not c["same_stream"]]
    res = dict(conflict=bool(conf), n_conflict_agents=len(conf), gaps_s=[], gaps_censored=[], expert_accepted_gap_s=None,
               expert_reached_conflict=None, expert_arrival_s=None, expert_rejected_gaps_s=[], max_rejected_gap_s=None,
               pet_expert_s=None, gap_open_at_s=None, first_gap_ge_tc_s=None)
    if not conf:
        return res
    arrivals = sorted(set(round(c["agent_arrival_s"], 3) for c in conf))
    ego_t = min(c["ego_arrival_s"] for c in conf)
    res["expert_arrival_s"] = float(ego_t)
    res["expert_reached_conflict"] = bool(ego_t < horizon_s - 1e-6)
    # 区间边界：0（观测起点）, 各到达时刻, 视界；去掉零长度区间
    bounds = [0.0] + [a for a in arrivals if 0.0 < a < horizon_s] + [horizon_s]
    gaps = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 1e-6]
    cens = [("L" if a == 0.0 and (not arrivals or arrivals[0] > 0.0) else "") + ("R" if b == horizon_s else "") for a, b in gaps]
    res["gaps_s"] = [round(b - a, 2) for a, b in gaps]; res["gaps_censored"] = cens
    acc, rej = None, []
    for (a, b), cz in zip(gaps, cens):
        if a <= ego_t < b:
            acc = b - a
        elif b <= ego_t or not res["expert_reached_conflict"]:
            rej.append(b - a)          # 截断区间的长度是真实间隙的下界，用于 ≥ t_c 判定仍然成立
    res["expert_accepted_gap_s"] = acc
    res["expert_rejected_gaps_s"] = [round(g, 2) for g in rej]
    res["max_rejected_gap_s"] = max(rej) if rej else None
    res["pet_expert_s"] = float(min(abs(ego_t - a) for a in arrivals)) if res["expert_reached_conflict"] else None
    for (a, b) in gaps:
        if b - a >= t_c:
            res["first_gap_ge_tc_s"] = a; break
    res["gap_open_at_s"] = res["first_gap_ge_tc_s"]
    return res


def pet_of_candidate(poses_cand: np.ndarray, tracks: Dict[str, np.ndarray]) -> Optional[float]:
    conf = [c for c in crossing_conflicts(poses_cand, tracks) if not c["same_stream"]]
    if not conf:
        return None
    return float(min(abs(c["ego_arrival_s"] - c["agent_arrival_s"]) for c in conf))


def startup_delay(poses: np.ndarray, v_thresh: float = 0.5) -> float:
    pts, t = _path_times(poses)
    v = np.linalg.norm(np.diff(pts, axis=0), axis=1) / DT
    mv = np.where(v > v_thresh)[0]
    return float(mv[0] * DT) if len(mv) else float(t[-1])


def rss_follow_ok(scene, poses: np.ndarray, params: RSSParams = RSSParams(), lane_half_width: float = 1.8,
                  expert_poses: Optional[np.ndarray] = None, veh_len: float = 4.5) -> Tuple[bool, float]:
    """沿候选轨迹检查与同向前车的 RSS 纵向安全距离；返回 (始终满足?, 最小 距离/所需距离 比)。
    非反应式伪影处理：只把"在同一时刻位于专家前方"的车辆视为前车（跟在专家后面的车在候选变慢后会出现在候选前方，
    现实中它们会减速，不应计入）。expert_poses 缺省时退化为对所有前方车辆检查。"""
    t0 = scene.scene_metadata.num_history_frames - 1
    pts, t = _path_times(poses)
    v_ego = np.concatenate([[0.0], np.linalg.norm(np.diff(pts, axis=0), axis=1) / DT])
    epts = _path_times(expert_poses)[0] if expert_poses is not None else None
    worst = np.inf
    H = min(len(pts) - 1, len(scene.frames) - 1 - t0)
    for k in range(1, H + 1):
        xy, vel, names = _to_t0_frame(scene, k)
        if len(xy) == 0:
            continue
        ego_xy = pts[k]; heading = np.arctan2(pts[k, 1] - pts[k - 1, 1], pts[k, 0] - pts[k - 1, 0] + 1e-9)
        c, s = np.cos(-heading), np.sin(-heading)
        rel = xy - ego_xy[None, :]
        lon = c * rel[:, 0] - s * rel[:, 1]; lat = s * rel[:, 0] + c * rel[:, 1]
        if epts is not None and k < len(epts):
            rel_e = xy - epts[k][None, :]
            lon_e = c * rel_e[:, 0] - s * rel_e[:, 1]
        else:
            lon_e = lon
        for i in range(len(xy)):
            if names[i] in VEH and 0 < lon[i] < 60 and abs(lat[i]) < lane_half_width and lon_e[i] > 0:
                v_front = max(0.0, float(np.dot(vel[i], [np.cos(heading), np.sin(heading)])))
                need = rss_long_min_dist(v_ego[k], v_front, params)
                worst = min(worst, (lon[i] - veh_len) / max(need, 0.5))
    return bool(worst >= 1.0), float(worst)


HCM_TC = {  # HCM 6th ed. Ch. 20 基准临界间隙 [s]（两车道主路）
    "LT_major": 4.1, "RT_minor": 6.2, "TH_minor": 6.5, "LT_minor": 7.1,
}
RAGLAND_85 = 8.6   # Ragland et al. 2006：85% 人类驾驶员接受的对向间隙（无保护左转）


def near_stop_control(scene, radius: float = 12.0) -> bool:
    """t0 自车前方 radius 内是否有 STOP_SIGN / STOP_LINE / YIELD 地图要素（支路进口的标志）。"""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    t0 = scene.scene_metadata.num_history_frames - 1
    x, y, yaw = scene.frames[t0].ego_status.ego_pose
    layers = [SemanticMapLayer.STOP_SIGN, SemanticMapLayer.STOP_LINE, SemanticMapLayer.YIELD]
    try:
        objs = scene.map_api.get_proximal_map_objects(Point2D(x, y), radius, layers)
    except Exception:
        return False
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    for layer in layers:
        for o in objs.get(layer, []):
            try:
                cx, cy = o.polygon.centroid.x, o.polygon.centroid.y
            except Exception:
                continue
            lon = c * (cx - x) - s_ * (cy - y)
            if -3.0 <= lon <= radius:
                return True
    return False


def critical_gap_for(turn_type: Optional[str], stop_controlled: bool) -> float:
    """按运动类型与是否受停车/让行控制选择 HCM 基准临界间隙。"""
    if turn_type == "LEFT":
        return HCM_TC["LT_minor"] if stop_controlled else HCM_TC["LT_major"]
    if turn_type == "RIGHT":
        return HCM_TC["RT_minor"]
    return HCM_TC["TH_minor"]   # 直行穿越 / 未知


def scene_objective_features(scene, expert_poses: np.ndarray, cand_poses: Optional[np.ndarray] = None,
                             t_c: float = 6.5) -> dict:
    """expert_poses 建议传入长视界（如 20 帧 / 10 s）轨迹；cand_poses 为 4 s 候选。"""
    tracks = agent_tracks(scene)
    g = gap_analysis(expert_poses, tracks, t_c=t_c)
    out = dict(g)
    out["startup_delay_expert_s"] = startup_delay(expert_poses)
    out["rss_ok_expert"], out["rss_ratio_expert"] = rss_follow_ok(scene, expert_poses)
    if cand_poses is not None:
        out["pet_candidate_s"] = pet_of_candidate(cand_poses, tracks)
        out["startup_delay_candidate_s"] = startup_delay(cand_poses)
        out["rss_ok_candidate"], out["rss_ratio_candidate"] = rss_follow_ok(scene, cand_poses)
    return out
