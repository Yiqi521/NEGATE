"""Step 4：保守负样本分离损失（回归型规划器用）。

输入约定（batch）:
  pred:  (B, T, 3)  规划器输出轨迹（局部坐标 x, y, yaw）
  expert:(B, T, 3)  专家轨迹
  negs:  (B, K, T, 3) 每帧最多 K 条保守负样本（不足用 0 填充）
  mask:  (B, K) bool  有效负样本掩码（帧不在目标子集或候选被过滤 → False）
两种形式：
  - hinge:   L = mean_k max(0, m - (d(pred, neg_k) - d(pred, expert)))
  - infonce: L = -log( exp(-d(pred,expert)/T) / (exp(-d(pred,expert)/T) + Σ_k exp(-d(pred,neg_k)/T)) )
d 为 (x, y) 航点 L1 均值；yaw 不参与（负样本与专家共路径）。
无有效负样本的样本对损失贡献为 0；返回 (loss, stats)。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _traj_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(…, T, 3) x (…, T, 3) -> (…,)  xy 的 L1 均值。"""
    return (a[..., :2] - b[..., :2]).abs().mean(dim=(-1, -2))


def hinge_separation_loss(pred, expert, negs, mask, margin: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    d_pos = _traj_dist(pred, expert)                       # (B,)
    d_neg = _traj_dist(pred.unsqueeze(1), negs)            # (B,K)
    viol = F.relu(margin - (d_neg - d_pos.unsqueeze(1)))   # (B,K)
    viol = viol * mask.float()
    n_valid = mask.float().sum()
    loss = viol.sum() / n_valid.clamp(min=1.0)
    with torch.no_grad():
        active = ((viol > 0) & mask).float().sum() / n_valid.clamp(min=1.0)
    return loss, {"neg_active_frac": float(active), "n_valid_negs": float(n_valid), "d_pos": float(d_pos.mean())}


def infonce_separation_loss(pred, expert, negs, mask, temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    d_pos = _traj_dist(pred, expert)                       # (B,)
    d_neg = _traj_dist(pred.unsqueeze(1), negs)            # (B,K)
    logits_pos = -d_pos / temperature                      # (B,)
    logits_neg = (-d_neg / temperature).masked_fill(~mask, float("-inf"))
    logits = torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1)  # (B,1+K)
    has_neg = mask.any(dim=1)
    loss_per = -F.log_softmax(logits, dim=1)[:, 0]
    loss = (loss_per * has_neg.float()).sum() / has_neg.float().sum().clamp(min=1.0)
    return loss, {"frac_samples_with_negs": float(has_neg.float().mean()), "d_pos": float(d_pos.mean())}


class SeparationLoss(torch.nn.Module):
    """封装：kind ∈ {hinge, infonce}；weight 由外部 λ_neg 与 warm-up 调度控制。"""

    def __init__(self, kind: str = "hinge", margin: float = 1.0, temperature: float = 1.0):
        super().__init__()
        assert kind in ("hinge", "infonce")
        self.kind, self.margin, self.temperature = kind, margin, temperature

    def forward(self, pred, expert, negs, mask):
        if self.kind == "hinge":
            return hinge_separation_loss(pred, expert, negs, mask, self.margin)
        return infonce_separation_loss(pred, expert, negs, mask, self.temperature)


def warmup_weight(step: int, total_steps: int, lam: float, warmup_frac: float = 0.2) -> float:
    """线性 warm-up：前 warmup_frac 的训练步从 0 线性升到 lam。"""
    w = int(total_steps * warmup_frac)
    return lam * min(1.0, step / max(1, w))
