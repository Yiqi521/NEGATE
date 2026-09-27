"""训练数据侧：NegativeBank —— token → (K, T, 3) 保守负样本张量 + (K,) 掩码。

从 build_negatives 输出的 parquet 读取 is_negative 行，按需重建负样本位姿（算子 + 参数 + 专家轨迹）。
为避免训练时重复计算，可先 `materialize(scene_loader)` 把位姿缓存到 npz。

接入方式（TransFuser TargetBuilder 内）：
    bank = NegativeBank.from_parquet("negatives_navtrain_v1.parquet", k=4)
    negs, mask = bank.get(token, expert_poses)   # torch.Tensor (K,8,3), torch.BoolTensor (K,)
    targets["negatives"], targets["neg_mask"] = negs, mask
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from conservative_negatives.generators.operators import (g1_delayed_departure, g2_speed_scale, g3_early_brake,
                                                          g3b_brake_to_stop, g4_large_gap)


def _rebuild(op: str, params: str, expert: np.ndarray) -> np.ndarray:
    p = ast.literal_eval(params)
    if op == "G1":
        return g1_delayed_departure(expert, p["delay_s"])
    if op == "G2":
        return g2_speed_scale(expert, p["alpha"])
    if op == "G3":
        return g3_early_brake(expert, p["brake_dist_m"], p["decel"])
    if op == "G3b":
        return g3b_brake_to_stop(expert, p["decel"], p.get("start_s", 0.0))
    if op == "G4":
        ccs = p.get("conflict_clear_s", -1.0)
        return g4_large_gap(expert, None if ccs is None or ccs < 0 else ccs, p["extra_wait_s"])
    raise ValueError(op)


class NegativeBank:
    def __init__(self, table: pd.DataFrame, k: int = 4, horizon: int = 8, label_col: str = "is_negative",
                 prefer_diverse_ops: bool = True):
        t = table[table[label_col].fillna(False)].copy()
        self.k, self.T = k, horizon
        self.prefer_diverse_ops = prefer_diverse_ops
        # 每个 token 的候选列表：按 (算子多样性, EP 比接近 0.6) 排序，选前 k
        t["rank_key"] = (t.ep_ratio - 0.6).abs()
        self._by_token: Dict[str, List[Tuple[str, str]]] = {}
        for tok, g in t.groupby("token"):
            g = g.sort_values("rank_key")
            chosen: List[Tuple[str, str]] = []
            if prefer_diverse_ops:
                for op, gg in g.groupby("operator", sort=False):
                    chosen.append((op, gg.iloc[0].params))
                rest = [(r.operator, r.params) for r in g.itertuples() if (r.operator, r.params) not in chosen]
                chosen = (chosen + rest)[:k]
            else:
                chosen = [(r.operator, r.params) for r in g.head(k).itertuples()]
            self._by_token[tok] = chosen
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def from_parquet(cls, path: Path, **kw) -> "NegativeBank":
        return cls(pd.read_parquet(path), **kw)

    @property
    def tokens(self) -> List[str]:
        return list(self._by_token.keys())

    def __contains__(self, token: str) -> bool:
        return token in self._by_token

    def get(self, token: str, expert_poses: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        negs = np.zeros((self.k, self.T, 3), dtype=np.float32)
        mask = np.zeros((self.k,), dtype=bool)
        for i, (op, params) in enumerate(self._by_token.get(token, [])[: self.k]):
            key = f"{token}|{op}|{params}"
            if key not in self._cache:
                self._cache[key] = _rebuild(op, params, np.asarray(expert_poses, dtype=np.float64))[: self.T].astype(np.float32)
            negs[i] = self._cache[key]
            mask[i] = True
        return torch.from_numpy(negs), torch.from_numpy(mask)

    def stats(self) -> Dict[str, float]:
        n = np.array([len(v) for v in self._by_token.values()])
        ops = pd.Series([op for v in self._by_token.values() for op, _ in v]).value_counts(normalize=True).round(3).to_dict()
        return {"tokens": int(len(n)), "mean_k": float(n.mean()), "frac_full_k": float((n >= self.k).mean()), "op_share": ops}
