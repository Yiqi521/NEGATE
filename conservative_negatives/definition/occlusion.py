"""遮挡可达性判据（O7）：判断"因看不见而减速/等待"是否有正当理由。

判据（Orzechowski 2018 / Yu 2019 / Park 2023 的确定性特例）：
    在与自车路径冲突的来车车道上，取自车视线所及的最远点作为幻影车位置；
    幻影车以 v_lim_factor × 车道限速 匀速行进，到达冲突点用时 t_phantom = d_occ / v_phantom。
    自车沿候选轨迹清空冲突区的时刻为 t_clear。
    若 t_phantom < t_clear + rho（rho 为响应时间），则该处存在无法排除的碰撞可能 → **谨慎有正当理由**。

与 O6（间隙接受）互补：O6 处理"看得见的车流"，O7 处理"看不见的车流"。
两者任一判为有理由，则该候选不作负样本。

依赖：shapely（随 nuplan-devkit 安装）。所有几何在 t0 自车后轴坐标系中计算。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon, Point

from conservative_negatives.definition.objective_criteria import DT, VEH, _path_times, _to_t0_frame

# nuPlan / NAVSIM 标注框索引：x, y, z, length, width, height, heading
BOX_X, BOX_Y, BOX_L, BOX_W, BOX_H = 0, 1, 3, 4, 6
EGO_LENGTH, EGO_WIDTH = 4.6, 2.3          # nuPlan 车辆参数（近似）
SENSOR_OFFSET_M = 1.7                      # 后轴 → 传感器塔的纵向偏移


@dataclass
class OcclusionVerdict:
    has_conflicting_lane: bool
    d_occ_m: Optional[float]               # 视线所及最远点到冲突点的弧长
    v_phantom_mps: Optional[float]
    t_phantom_s: Optional[float]
    t_clear_s: Optional[float]
    fully_visible: Optional[bool]          # 上游 max_range 内无遮挡
    justified: bool                        # True = 谨慎有正当理由（不可作负样本）
    reason: str


def vehicle_polygons_t0(scene, frame_offset: int = 0) -> List[Polygon]:
    """t0（或 t0+offset）帧的车辆框多边形，t0 自车系。"""
    t0 = scene.scene_metadata.num_history_frames - 1
    f = scene.frames[t0 + frame_offset]
    polys: List[Polygon] = []
    for box, name in zip(f.annotations.boxes, f.annotations.names):
        if name not in VEH:
            continue
        x, y, l, w, h = box[BOX_X], box[BOX_Y], box[BOX_L], box[BOX_W], box[BOX_H]
        c, s = np.cos(h), np.sin(h)
        dx, dy = l / 2.0, w / 2.0
        corners = [(x + c * sx * dx - s * sy * dy, y + s * sx * dx + c * sy * dy)
                   for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        polys.append(Polygon(corners))
    return polys


def _global_to_t0(scene, pts_global: np.ndarray) -> np.ndarray:
    """全局坐标 (N,2) → t0 自车系。"""
    t0 = scene.scene_metadata.num_history_frames - 1
    x0, y0, yaw0 = scene.frames[t0].ego_status.ego_pose
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    d = pts_global - np.array([x0, y0])[None, :]
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)


def ego_route_lane_ids(scene) -> set:
    """自车路线 roadblock 内的全部车道 id（含 connector）。"""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    t0 = scene.scene_metadata.num_history_frames - 1
    ids = set()
    for rb in scene.frames[t0].roadblock_ids:
        for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
            try:
                o = scene.map_api.get_map_object(rb, layer)
            except Exception:
                o = None
            if o is not None:
                try:
                    ids.update(e.id for e in o.interior_edges)
                except Exception:
                    pass
                break
    return ids


def map_conflict_points(scene, ego_poses: np.ndarray, route_lane_ids: set, query_step_m: float = 6.0,
                        query_radius_m: float = 18.0, min_cross_angle_deg: float = 25.0,
                        max_points: int = 4) -> List[Tuple[np.ndarray, object]]:
    """由**地图拓扑**找出与自车路径交叉的非路线车道及交叉点（t0 自车系）。

    与 O6 不同：不依赖是否观测到车辆，因此能覆盖"路口看起来空、但视线被挡"的情形。
    返回按沿自车路径的弧长排序的 [(conflict_xy_t0, lane_object)]。
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    t0 = scene.scene_metadata.num_history_frames - 1
    x0, y0, yaw0 = scene.frames[t0].ego_status.ego_pose
    c, s_ = np.cos(yaw0), np.sin(yaw0)
    pts_t0, _ = _path_times(ego_poses)
    gx = x0 + c * pts_t0[:, 0] - s_ * pts_t0[:, 1]
    gy = y0 + s_ * pts_t0[:, 0] + c * pts_t0[:, 1]
    ego_line = LineString(np.stack([gx, gy], axis=1))
    if ego_line.length < 1.0:
        return []

    seen, out = set(), []
    n_q = max(2, int(ego_line.length / query_step_m))
    for frac in np.linspace(0.0, 1.0, n_q):
        p = ego_line.interpolate(frac, normalized=True)
        try:
            objs = scene.map_api.get_proximal_map_objects(
                Point2D(p.x, p.y), query_radius_m, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
        except Exception:
            continue
        for lane in objs[SemanticMapLayer.LANE] + objs[SemanticMapLayer.LANE_CONNECTOR]:
            if lane.id in seen or lane.id in route_lane_ids:
                continue
            seen.add(lane.id)
            bl = np.array([q.point.array for q in lane.baseline_path.discrete_path])
            if len(bl) < 2:
                continue
            lane_line = LineString(bl)
            inter = ego_line.intersection(lane_line)
            if inter.is_empty:
                continue
            ipt = inter if inter.geom_type == "Point" else (
                inter.geoms[0] if hasattr(inter, "geoms") and len(inter.geoms) else inter.centroid)
            ix, iy = float(ipt.x), float(ipt.y)
            # 交叉角：排除近乎平行（同向/并行车道边界误判）
            j = int(np.argmin(np.linalg.norm(bl - np.array([ix, iy])[None, :], axis=1)))
            j0, j1 = max(0, j - 1), min(len(bl) - 1, j + 1)
            lv = bl[j1] - bl[j0]
            e = ego_line.interpolate(ego_line.project(ipt))
            e2 = ego_line.interpolate(min(ego_line.length, ego_line.project(ipt) + 2.0))
            ev = np.array([e2.x - e.x, e2.y - e.y])
            if np.linalg.norm(lv) < 1e-6 or np.linalg.norm(ev) < 1e-6:
                continue
            ang = np.degrees(np.arccos(np.clip(abs(np.dot(lv, ev)) / (np.linalg.norm(lv) * np.linalg.norm(ev)), -1, 1)))
            if ang < min_cross_angle_deg:
                continue
            xy_t0 = _global_to_t0(scene, np.array([[ix, iy]]))[0]
            out.append((xy_t0, lane, float(ego_line.project(ipt))))
    out.sort(key=lambda r: r[2])
    return [(xy, lane) for xy, lane, _ in out[:max_points]]


def conflicting_lane_upstream(scene, conflict_xy_t0: np.ndarray, ego_route_lane_ids: set,
                              max_range_m: float = 60.0, step_m: float = 2.0,
                              lane=None) -> Optional[Tuple[np.ndarray, float]]:
    """在冲突点附近找到与自车路径冲突的来车车道，返回其上游采样点（t0 系，由近及远）与限速。

    上游 = 沿车道基线反向（含 incoming_edges 递归），最多 max_range_m。
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    t0 = scene.scene_metadata.num_history_frames - 1
    x0, y0, yaw0 = scene.frames[t0].ego_status.ego_pose
    c, s = np.cos(yaw0), np.sin(yaw0)
    gx = x0 + c * conflict_xy_t0[0] - s * conflict_xy_t0[1]
    gy = y0 + s * conflict_xy_t0[0] + c * conflict_xy_t0[1]
    if lane is not None:
        pts_l = np.array([p.point.array for p in lane.baseline_path.discrete_path])
        best, best_idx = lane, int(np.argmin(np.linalg.norm(pts_l - np.array([gx, gy])[None, :], axis=1)))
        return _upstream_from(scene, best, best_idx, max_range_m, step_m)
    try:
        objs = scene.map_api.get_proximal_map_objects(
            Point2D(gx, gy), 6.0, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    except Exception:
        return None
    cands = objs[SemanticMapLayer.LANE] + objs[SemanticMapLayer.LANE_CONNECTOR]
    cands = [l for l in cands if l.id not in ego_route_lane_ids]
    if not cands:
        return None

    # 取基线最接近冲突点的一条
    best, best_d, best_idx = None, np.inf, 0
    for lane in cands:
        pts = np.array([p.point.array for p in lane.baseline_path.discrete_path])
        d = np.linalg.norm(pts - np.array([gx, gy])[None, :], axis=1)
        i = int(np.argmin(d))
        if d[i] < best_d:
            best, best_d, best_idx = lane, d[i], i
    if best is None or best_d > 6.0:
        return None

    return _upstream_from(scene, best, best_idx, max_range_m, step_m)


def _upstream_from(scene, best, best_idx: int, max_range_m: float, step_m: float) -> Optional[Tuple[np.ndarray, float]]:
    """沿基线反向收集上游点（含前驱车道），等间隔重采样后转到 t0 自车系。"""
    pts = np.array([p.point.array for p in best.baseline_path.discrete_path])
    up = list(pts[:best_idx + 1][::-1])                       # 由冲突点向上游
    lane, guard = best, 0
    while _poly_len(up) < max_range_m and guard < 6:
        guard += 1
        try:
            inc = lane.incoming_edges
        except Exception:
            break
        if not inc:
            break
        lane = inc[0]
        p2 = np.array([p.point.array for p in lane.baseline_path.discrete_path])
        up.extend(list(p2[::-1]))
    up_g = np.array(up)
    # 等间隔重采样
    seg = np.linalg.norm(np.diff(up_g, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(min(s_cum[-1], max_range_m) / step_m))
    s_new = np.linspace(0.0, min(s_cum[-1], max_range_m), n)
    xs = np.interp(s_new, s_cum, up_g[:, 0]); ys = np.interp(s_new, s_cum, up_g[:, 1])
    speed_limit = getattr(best, "speed_limit_mps", None) or 13.4     # 缺省 ≈ 30 mph
    return _global_to_t0(scene, np.stack([xs, ys], axis=1)), float(speed_limit)


def _poly_len(pts: List[np.ndarray]) -> float:
    if len(pts) < 2:
        return 0.0
    a = np.array(pts)
    return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))


def visibility_limit(upstream_pts_t0: np.ndarray, occluders: List[Polygon],
                     sensor_xy: Tuple[float, float] = (SENSOR_OFFSET_M, 0.0)) -> Tuple[float, bool]:
    """沿上游采样点做视线检测，返回 (视线所及最远点到冲突点的弧长 d_occ, 是否全程可见)。

    d_occ 小 = 只能看到很近的地方 = 遮挡严重。
    """
    origin = Point(sensor_xy)
    seg = np.linalg.norm(np.diff(upstream_pts_t0, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    for i, p in enumerate(upstream_pts_t0):
        ray = LineString([origin, Point(p[0], p[1])])
        if any(ray.intersects(o) for o in occluders):
            return float(s_cum[i]), False
    return float(s_cum[-1]), True


def ego_clear_time(cand_poses: np.ndarray, conflict_xy: np.ndarray,
                   clear_margin_m: float = EGO_LENGTH / 2 + 1.0) -> Optional[float]:
    """自车沿候选轨迹使车尾越过冲突点的时刻；4 s 内未清空返回 None（= 一直占用/未到达）。"""
    pts, t = _path_times(cand_poses)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    i = int(np.argmin(np.linalg.norm(pts - conflict_xy[None, :], axis=1)))
    s_target = s[i] + clear_margin_m
    if s[-1] < s_target:
        return None
    return float(np.interp(s_target, s, t))


def occlusion_verdict(scene, ref_poses: np.ndarray, route_lane_ids: Optional[set] = None,
                      v_lim_factor: float = 1.1, rho: float = 1.0, max_range_m: float = 60.0,
                      occluders: Optional[List[Polygon]] = None) -> OcclusionVerdict:
    """场景级判定：沿**参考轨迹**（专家，即"果断通过"的代表）检查每个地图冲突点。

    只要有一个冲突点满足 t_phantom < t_clear + rho，就判为"谨慎有正当理由"——
    即在该场景中放慢是合理的，不应作为保守负样本。
    """
    route_lane_ids = ego_route_lane_ids(scene) if route_lane_ids is None else route_lane_ids
    cps = map_conflict_points(scene, ref_poses, route_lane_ids)
    if not cps:
        return OcclusionVerdict(False, None, None, None, None, None, False, "no_conflicting_lane")
    occluders = vehicle_polygons_t0(scene) if occluders is None else occluders
    worst = None
    for conflict_xy, lane in cps:
        up = conflicting_lane_upstream(scene, conflict_xy, route_lane_ids, max_range_m=max_range_m, lane=lane)
        if up is None:
            continue
        up_pts, v_lim = up
        d_occ, fully_visible = visibility_limit(up_pts, occluders)
        v_ph = v_lim_factor * v_lim
        t_ph = d_occ / max(v_ph, 1e-3)
        t_clear = ego_clear_time(ref_poses, conflict_xy)
        if t_clear is None:
            continue                       # 参考轨迹在视界内未清空该冲突点
        # 关键：幻影车只存在于**被遮挡**的区域。若上游 max_range 内全程可见且无车（可见即已由 O6 处理），
        # 则不存在幻影车，遮挡不构成正当理由（Orzechowski 2018：隐藏障碍置于传感视场边界）。
        justified = bool((not fully_visible) and t_ph < t_clear + rho)
        v = OcclusionVerdict(True, d_occ, v_ph, t_ph, t_clear, fully_visible, justified,
                             "phantom_reaches_first" if justified else
                             ("fully_visible" if fully_visible else "enough_sight_distance"))
        if justified:
            return v                       # 任一冲突点有理由即返回
        if worst is None or (v.t_phantom_s - v.t_clear_s) < (worst.t_phantom_s - worst.t_clear_s):  # 记录最接近的冲突点
            worst = v
    return worst or OcclusionVerdict(True, None, None, None, None, None, False, "reference_never_clears")


def verdict_to_dict(v: OcclusionVerdict, prefix: str = "occ_") -> dict:
    return {prefix + k: val for k, val in asdict(v).items()}
