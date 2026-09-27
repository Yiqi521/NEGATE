"""任意候选轨迹 → NAVSIM PDMS 子分数 的封装（复用官方 PDMSimulator / PDMScorer 与 metric cache）。

用于：C1 安全检查、C2 进度比、C3 事后验证（参考轨迹在日志回放中的安全性）。
"""
from __future__ import annotations

import lzma, os, pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache


@dataclass
class SubScores:
    nc: float
    dac: float
    ep: float
    ttc: float
    comfort: float
    ddc: float
    pdms: float

    def safe(self) -> bool:
        """C1：NC、DAC、TTC 全满分（DDC 在 v1 权重为 0，单独记录不作门控）。"""
        return self.nc >= 1.0 and self.dac >= 1.0 and self.ttc >= 1.0


class PDMEvaluator:
    def __init__(self, metric_cache_path: Optional[Path] = None, devkit_root: Optional[Path] = None):
        devkit_root = Path(devkit_root or os.environ["NAVSIM_DEVKIT_ROOT"])
        metric_cache_path = Path(metric_cache_path or Path(os.environ["NAVSIM_EXP_ROOT"]) / "metric_cache")
        cfg = OmegaConf.load(devkit_root / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
        self.simulator = instantiate(cfg.simulator)
        self.scorer = instantiate(cfg.scorer)
        self._loader: Optional[MetricCacheLoader] = None
        self._cache_root = metric_cache_path
        try:
            self._loader = MetricCacheLoader(metric_cache_path)
        except Exception:
            # 缓存尚未写 metadata csv 时，退化为按目录扫描
            self._loader = None
        self._cache: Dict[str, MetricCache] = {}

    # ---------- cache access ----------
    def cache_path(self, token: str) -> Optional[Path]:
        if self._loader is not None and token in self._loader.metric_cache_paths:
            return Path(self._loader.metric_cache_paths[token])
        hits = list(self._cache_root.rglob(f"{token}/metric_cache.pkl"))
        return hits[0] if hits else None

    def has_cache(self, token: str) -> bool:
        return self.cache_path(token) is not None

    def load(self, token: str) -> MetricCache:
        if token not in self._cache:
            p = self.cache_path(token)
            if p is None:
                raise KeyError(f"no metric cache for {token}")
            with lzma.open(p, "rb") as f:
                self._cache[token] = pickle.load(f)
        return self._cache[token]

    # ---------- scoring ----------
    def score(self, token: str, poses: np.ndarray) -> SubScores:
        """poses: (8,3) 局部坐标（后轴系，0.5 s 间隔）。"""
        traj = Trajectory(np.asarray(poses, dtype=np.float32))
        r = pdm_score(self.load(token), traj, self.simulator.proposal_sampling, self.simulator, self.scorer)
        return SubScores(
            nc=float(r.no_at_fault_collisions), dac=float(r.drivable_area_compliance), ep=float(r.ego_progress),
            ttc=float(r.time_to_collision_within_bound), comfort=float(r.comfort),
            ddc=float(r.driving_direction_compliance), pdms=float(r.score),
        )

    def score_dict(self, token: str, poses: np.ndarray) -> dict:
        return asdict(self.score(token, poses))
