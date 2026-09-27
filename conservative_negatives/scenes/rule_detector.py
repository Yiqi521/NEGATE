"""Step 2（路径 C）：基于地图拓扑与自车运动的规则场景检测器。

只依赖 NAVSIM Scene（元数据 + 地图），不需要传感器数据。
输出每个场景的两组标志：
  - unprotected_turn: 自车未来 4 s 经过无信号灯控制的转向 lane connector，且航向变化 ≥ turn_heading_deg
  - merging（城市定义）: 自车未来 4 s 内换道（同一 roadblock 内 lane id 变化）且目标车道附近有车；
                         或从停止/上下客状态重新进入有车车流
以及辅助字段（冲突车辆数、专家是否停车、专家起步延迟等），供 Step 1 阈值校准使用。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.common.dataclasses import Scene

VEHICLE_NAMES = {"vehicle", "bicycle"}  # OpenScene gt_names 中的机动/非机动车类别（bicycle 可选）


@dataclass
class SceneFlags:
    token: str
    log_name: str
    map_name: str
    # 运动特征
    heading_change_deg: float
    ego_speed_t0: float
    ego_speed_t4: float
    expert_progress_m: float
    expert_stopped_t0: bool
    expert_start_delay_s: float  # 从 t0 起第一次速度 > 0.5 m/s 的时间；一直停则 = 4.0
    # 无保护转向
    passes_lane_connector: bool
    connector_has_traffic_light: Optional[bool]
    connector_turn_type: Optional[str]
    unprotected_turn: bool
    # 汇入
    lane_id_t0: Optional[str]
    lane_id_t4: Optional[str]
    same_roadblock: Optional[bool]
    lane_change: bool
    target_lane_vehicles_30m: int
    unprotected_crossing: bool
    near_pudo_t0: bool
    lateral_offset_t0: float
    reentry_from_stop: bool
    merging: bool
    # 交互
    n_vehicles_30m: int
    n_moving_vehicles_30m: int


def _global_poses(scene: Scene) -> np.ndarray:
    """(T,3) 全局位姿 x, y, yaw，按帧顺序。"""
    return np.array([f.ego_status.ego_pose for f in scene.frames], dtype=np.float64)


def _speeds(scene: Scene) -> np.ndarray:
    return np.array([np.linalg.norm(f.ego_status.ego_velocity[:2]) for f in scene.frames], dtype=np.float64)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _nearest_lane(map_api: AbstractMap, x: float, y: float, radius: float = 2.0) -> Optional[LaneGraphEdgeMapObject]:
    objs = map_api.get_proximal_map_objects(
        Point2D(x, y), radius, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    )
    cands = objs[SemanticMapLayer.LANE] + objs[SemanticMapLayer.LANE_CONNECTOR]
    if not cands:
        return None
    dists = []
    for lane in cands:
        pts = np.array([s.point.array for s in lane.baseline_path.discrete_path])
        dists.append(np.min(np.linalg.norm(pts - np.array([x, y]), axis=1)))
    return cands[int(np.argmin(dists))]


def _connectors_on_path(map_api: AbstractMap, poses: np.ndarray, radius: float = 1.5) -> List[LaneGraphEdgeMapObject]:
    seen: Dict[str, LaneGraphEdgeMapObject] = {}
    for x, y, _ in poses:
        objs = map_api.get_proximal_map_objects(Point2D(x, y), radius, [SemanticMapLayer.LANE_CONNECTOR])
        for c in objs[SemanticMapLayer.LANE_CONNECTOR]:
            if c.contains_point(Point2D(x, y)):
                seen[c.id] = c
    return list(seen.values())


def _vehicles_near(scene: Scene, frame_idx: int, center_xy: np.ndarray, radius: float) -> Tuple[int, int]:
    """返回 (radius 内车辆数, 其中运动车辆数)。boxes 为局部坐标（自车系）；这里用相对距离近似。"""
    ann = scene.frames[frame_idx].annotations
    n, n_mov = 0, 0
    ego = scene.frames[frame_idx].ego_status.ego_pose
    c, s = np.cos(ego[2]), np.sin(ego[2])
    for box, name, vel in zip(ann.boxes, ann.names, ann.velocity_3d):
        if name not in VEHICLE_NAMES:
            continue
        # 局部 -> 全局
        gx = ego[0] + c * box[0] - s * box[1]
        gy = ego[1] + s * box[0] + c * box[1]
        if np.linalg.norm(np.array([gx, gy]) - center_xy) <= radius:
            n += 1
            if np.linalg.norm(vel[:2]) > 0.5:
                n_mov += 1
    return n, n_mov


def _vehicles_in_lane(scene: Scene, frame_idx: int, lane: LaneGraphEdgeMapObject, max_dist: float = 30.0) -> int:
    ann = scene.frames[frame_idx].annotations
    ego = scene.frames[frame_idx].ego_status.ego_pose
    c, s = np.cos(ego[2]), np.sin(ego[2])
    n = 0
    for box, name in zip(ann.boxes, ann.names):
        if name not in VEHICLE_NAMES:
            continue
        gx = ego[0] + c * box[0] - s * box[1]
        gy = ego[1] + s * box[0] + c * box[1]
        if np.hypot(gx - ego[0], gy - ego[1]) > max_dist:
            continue
        if lane.contains_point(Point2D(gx, gy)):
            n += 1
    return n


def _near_pudo(map_api: AbstractMap, x: float, y: float, radius: float = 3.0) -> bool:
    layers = [SemanticMapLayer.PUDO, SemanticMapLayer.EXTENDED_PUDO, SemanticMapLayer.CARPARK_AREA]
    try:
        objs = map_api.get_proximal_map_objects(Point2D(x, y), radius, layers)
    except Exception:
        return False
    return any(len(objs.get(l, [])) > 0 for l in layers)


def _lateral_offset(lane: Optional[LaneGraphEdgeMapObject], x: float, y: float) -> float:
    if lane is None:
        return float("nan")
    pts = np.array([s.point.array for s in lane.baseline_path.discrete_path])
    return float(np.min(np.linalg.norm(pts - np.array([x, y]), axis=1)))


def detect(scene: Scene, turn_heading_deg: float = 45.0, horizon_frames: int = 8) -> SceneFlags:
    md = scene.scene_metadata
    t0 = md.num_history_frames - 1
    t4 = min(t0 + horizon_frames, len(scene.frames) - 1)
    poses = _global_poses(scene)
    speeds = _speeds(scene)
    fut = poses[t0 : t4 + 1]

    heading_change = abs(np.degrees(_wrap(fut[-1, 2] - fut[0, 2])))
    progress = float(np.sum(np.linalg.norm(np.diff(fut[:, :2], axis=0), axis=1)))
    stopped_t0 = bool(speeds[t0] < 0.5)
    moving = np.where(speeds[t0 : t4 + 1] > 0.5)[0]
    start_delay = float(moving[0] * 0.5) if len(moving) else 4.0

    # --- 无保护转向 ---
    connectors = _connectors_on_path(scene.map_api, fut)
    # 帧级信号灯数据：[(lane_connector_id, is_green), ...]；出现在列表中的 connector 视为有信号控制
    signalized_ids = set()
    for fi in range(max(0, t0 - 2), t4 + 1):
        for tl in scene.frames[fi].traffic_lights:
            signalized_ids.add(str(tl[0]))

    def _signalized(c) -> bool:
        return bool(c.has_traffic_lights()) or (str(c.id) in signalized_ids)
    def _tt(c, thresh_deg: float = 30.0) -> str:
        """nuPlan 的 turn_type() 未实现，改用 connector 基线路径首尾航向差推断。"""
        try:
            path = c.baseline_path.discrete_path
            d = np.degrees(_wrap(path[-1].heading - path[0].heading))
        except Exception:
            return "NA"
        if d > thresh_deg:
            return "LEFT"
        if d < -thresh_deg:
            return "RIGHT"
        return "STRAIGHT"

    turn_connectors = [c for c in connectors if _tt(c) in ("LEFT", "RIGHT")]
    passes = len(connectors) > 0
    has_tl: Optional[bool] = None
    turn_type: Optional[str] = None
    if turn_connectors:
        c0 = turn_connectors[0]
        has_tl = _signalized(c0)
        turn_type = _tt(c0)
    elif connectors:
        has_tl = _signalized(connectors[0])
        turn_type = _tt(connectors[0])
    unprotected_turn = bool(turn_connectors) and (has_tl is False) and heading_change >= turn_heading_deg

    # --- 候选第三类：无信号灯路口直行穿越（间隙接受）---
    straight_conn = [c for c in connectors if _tt(c) == "STRAIGHT"]
    unprotected_crossing = bool(straight_conn) and (not _signalized(straight_conn[0])) and stopped_t0 and heading_change < turn_heading_deg

    # --- 汇入（城市定义）---
    lane0 = _nearest_lane(scene.map_api, fut[0, 0], fut[0, 1])
    lane4 = _nearest_lane(scene.map_api, fut[-1, 0], fut[-1, 1])
    lane_id0 = lane0.id if lane0 else None
    lane_id4 = lane4.id if lane4 else None
    same_rb: Optional[bool] = None
    lane_change = False
    target_vehicles = 0
    if lane0 is not None and lane4 is not None:
        same_rb = lane0.get_roadblock_id() == lane4.get_roadblock_id()
        lane_change = bool(same_rb and lane_id0 != lane_id4 and heading_change < turn_heading_deg)
        target_vehicles = _vehicles_in_lane(scene, t0, lane4)
    n_veh, n_mov = _vehicles_near(scene, t0, fut[0, :2], 30.0)
    near_pudo = _near_pudo(scene.map_api, fut[0, 0], fut[0, 1])
    lat_off = _lateral_offset(lane0, fut[0, 0], fut[0, 1])
    # 收紧：排除堵车 / 红灯排队起步。必须“从车道外或车道边缘重新进入车流”：
    #   近上下客区/停车区，或横向偏离车道中心 > 1.2 m，或 4 s 内车道 id 变化
    #   注意：车道 id 变化必须发生在同一 roadblock 内（真正换道），否则直行穿越路口（lane -> connector）也会触发
    reentry_geom = near_pudo or (not np.isnan(lat_off) and lat_off > 1.2) or bool(same_rb and lane_id0 != lane_id4)
    reentry = bool(stopped_t0 and speeds[t4] > 3.0 and n_mov >= 1 and heading_change < turn_heading_deg and reentry_geom)
    merging = bool((lane_change and target_vehicles >= 1) or reentry)

    return SceneFlags(
        token=md.initial_token, log_name=md.log_name, map_name=md.map_name,
        heading_change_deg=float(heading_change), ego_speed_t0=float(speeds[t0]), ego_speed_t4=float(speeds[t4]),
        expert_progress_m=progress, expert_stopped_t0=stopped_t0, expert_start_delay_s=start_delay,
        passes_lane_connector=passes, connector_has_traffic_light=has_tl, connector_turn_type=turn_type,
        unprotected_turn=unprotected_turn,
        lane_id_t0=lane_id0, lane_id_t4=lane_id4, same_roadblock=same_rb, lane_change=lane_change,
        target_lane_vehicles_30m=target_vehicles, unprotected_crossing=unprotected_crossing and n_mov >= 1, near_pudo_t0=bool(near_pudo), lateral_offset_t0=lat_off,
        reentry_from_stop=reentry, merging=merging,
        n_vehicles_30m=n_veh, n_moving_vehicles_30m=n_mov,
    )


def flags_to_dict(f: SceneFlags) -> dict:
    return asdict(f)
