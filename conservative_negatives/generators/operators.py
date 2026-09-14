"""Step 3：保守负样本生成算子 G1–G4（作用于专家轨迹的局部坐标位姿序列）。

约定：poses 形状 (N,3) = (x, y, yaw)，自车后轴系，间隔 dt=0.5 s，N=8（NAVSIM 4 s）。
所有算子保持专家路径几何不变，只改变纵向速度剖面，因此天然满足“近专家”（C4）的路径一致性。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

DT = 0.5


ORIGIN = np.zeros((1, 3))


def _with_origin(poses: np.ndarray) -> np.ndarray:
    """在前面补上 t=0 的原点位姿（自车后轴系，yaw=0）。"""
    return np.concatenate([ORIGIN, poses], axis=0)


def _arc_length(poses: np.ndarray) -> np.ndarray:
    """(N,) 每个未来位姿相对原点的累计弧长（含原点→第一个位姿段）。"""
    p = _with_origin(poses)
    d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    return np.cumsum(d)


def _resample_along_path(poses: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    """按弧长 target_s 在专家路径上重采样；超出末端时沿末端航向线性外推（供更快的参考轨迹使用）。"""
    p = _with_origin(poses)
    s = np.concatenate([[0.0], _arc_length(poses)])
    # 去除零长度段避免插值退化
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    p, s = p[keep], s[keep]
    if len(s) == 1:  # 专家完全静止：所有目标点都沿 yaw=0 外推
        return np.stack([target_s, np.zeros_like(target_s), np.zeros_like(target_s)], axis=1)
    yaw_u = np.unwrap(p[:, 2])
    x = np.interp(target_s, s, p[:, 0]); y = np.interp(target_s, s, p[:, 1]); yaw = np.interp(target_s, s, yaw_u)
    over = target_s > s[-1]
    if np.any(over):
        h = yaw_u[-1]; ext = target_s[over] - s[-1]
        x[over] = p[-1, 0] + ext * np.cos(h); y[over] = p[-1, 1] + ext * np.sin(h); yaw[over] = h
    return np.stack([x, y, yaw], axis=1)


def _speed_profile(poses: np.ndarray) -> np.ndarray:
    """(N,) 每段平均速度，第 i 段为位姿 i-1（或原点）→ 位姿 i。"""
    s = _arc_length(poses)
    return np.diff(np.concatenate([[0.0], s])) / DT


def _from_speed(poses: np.ndarray, v: np.ndarray) -> np.ndarray:
    """由速度剖面 v (N,) 积分弧长并沿专家路径重采样（可外推）。"""
    return _resample_along_path(poses, np.cumsum(np.maximum(v, 0.0) * DT))


# ---------------- 算子 ----------------

def g1_delayed_departure(poses: np.ndarray, delay_s: float) -> np.ndarray:
    """G1 延迟起步：前 delay_s 静止（停在原点），之后按专家速度剖面行驶并截断。"""
    v = _speed_profile(poses)
    k = int(round(delay_s / DT))
    v_new = np.concatenate([np.zeros(k), v[: len(v) - k]]) if k < len(v) else np.zeros(len(v))
    return _from_speed(poses, v_new)


def advance_departure(poses: np.ndarray, advance_s: float) -> np.ndarray:
    """参考轨迹（C3 用）：把速度剖面提前 advance_s（更早起步），末端以最后速度延续。"""
    v = _speed_profile(poses)
    k = int(round(advance_s / DT))
    v_new = np.concatenate([v[k:], np.repeat(v[-1:], k)]) if k < len(v) else np.repeat(v[-1:], len(v))
    return _from_speed(poses, v_new)


def g2_speed_scale(poses: np.ndarray, alpha: float) -> np.ndarray:
    """G2 限速缩放：速度剖面乘 alpha (<1)。"""
    return _from_speed(poses, _speed_profile(poses) * alpha)


def g3_early_brake(poses: np.ndarray, brake_dist_m: float, decel: float) -> np.ndarray:
    """G3 提前制动：在专家停止点（或终点）前 brake_dist_m 处开始以 decel 匀减速至停止。"""
    v = _speed_profile(poses)
    s = _arc_length(poses)
    stop_idx = np.where(v < 0.3)[0]
    s_stop = s[stop_idx[0]] if len(stop_idx) else s[-1]
    s_brake = max(0.0, s_stop - brake_dist_m)
    v_new = v.copy()
    cur_s = 0.0
    for i in range(len(v)):
        if cur_s >= s_brake:
            v_allow = np.sqrt(max(0.0, 2 * decel * max(0.0, s_stop - cur_s)))
            v_new[i] = min(v_new[i], v_allow)
        cur_s += v_new[i] * DT
    return _from_speed(poses, v_new)


def g3b_brake_to_stop(poses: np.ndarray, decel: float, start_s: float = 0.0) -> np.ndarray:
    """G3b 无理由停车：从 start_s 起以 decel 匀减速至停止并保持（专家本身不停车的场景中的"过度制动"）。"""
    v = _speed_profile(poses)
    v_new = v.copy()
    k0 = int(round(start_s / DT))
    for i in range(k0, len(v)):
        v_new[i] = max(0.0, v_new[i - 1] - decel * DT) if i > 0 else max(0.0, v[0] - decel * DT)
    return _from_speed(poses, v_new)


def g4_large_gap(poses: np.ndarray, conflict_clear_s: Optional[float], extra_wait_s: float) -> np.ndarray:
    """G4 过大间隙：等待冲突车辆通过（conflict_clear_s，由日志他车未来计算）后再加 extra_wait_s 起步。
    无冲突车辆（None）时退化为 G1(extra_wait_s)。"""
    wait = (conflict_clear_s or 0.0) + extra_wait_s
    return g1_delayed_departure(poses, min(wait, (len(poses) - 1) * DT))


@dataclass
class Candidate:
    operator: str
    params: Dict[str, float]
    poses: np.ndarray  # (N,3)


DEFAULT_GRID = {
    "G1": [{"delay_s": d} for d in (1.0, 1.5, 2.0)],
    "G2": [{"alpha": a} for a in (0.5, 0.65, 0.8)],
    "G3": [{"brake_dist_m": d, "decel": b} for d in (5.0, 10.0) for b in (1.5, 2.5)],
    "G3b": [{"decel": 1.5, "start_s": 0.0}, {"decel": 1.5, "start_s": 1.0}, {"decel": 2.5, "start_s": 1.0}],
    "G4": [{"extra_wait_s": w} for w in (1.0, 2.0)],
}


def generate_candidates(expert_poses: np.ndarray, conflict_clear_s: Optional[float] = None,
                        grid: Dict[str, List[Dict[str, float]]] = DEFAULT_GRID,
                        ego_speed_t0: Optional[float] = None, stop_speed: float = 0.5) -> List[Candidate]:
    """适用规则：G1 / G4（延迟起步、过大间隙）只在专家 t0 静止时生成，否则会产生不真实的急停；
    G2 / G3 任何情况都生成，再交由运动学与 PDM 过滤。"""
    out: List[Candidate] = []
    stopped = ego_speed_t0 is None or ego_speed_t0 < stop_speed
    for p in (grid.get("G1", []) if stopped else []):
        out.append(Candidate("G1", p, g1_delayed_departure(expert_poses, **p)))
    for p in grid.get("G2", []):
        out.append(Candidate("G2", p, g2_speed_scale(expert_poses, **p)))
    for p in grid.get("G3", []):
        out.append(Candidate("G3", p, g3_early_brake(expert_poses, **p)))
    for p in (grid.get("G3b", []) if not stopped else []):   # 只对行驶中的专家生成
        out.append(Candidate("G3b", p, g3b_brake_to_stop(expert_poses, **p)))
    for p in (grid.get("G4", []) if stopped else []):
        out.append(Candidate("G4", {**p, "conflict_clear_s": conflict_clear_s or -1.0},
                             g4_large_gap(expert_poses, conflict_clear_s, **p)))
    return out


def kinematic_ok(poses: np.ndarray, a_min: float = -4.0, a_max: float = 2.5) -> bool:
    """运动学一级过滤：速度非负、加速度范围。"""
    v = _speed_profile(poses)
    if np.any(v < -1e-6):
        return False
    a = np.diff(v) / DT
    return bool(np.all(a >= a_min - 1e-6) and np.all(a <= a_max + 1e-6))
