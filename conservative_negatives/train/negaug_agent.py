"""通用适配器：给任意 NAVSIM 智能体附加保守负样本监督（研究计划 §16.1 "可附加"主张的代码对应）。

    host  = 任意 AbstractAgent（TransFuser / LTF / 将来的 PLUTO 包装）
    agent = NegativeAugmentedAgent(host, ...)
        get_sensor_config / get_feature_builders / forward / get_training_callbacks  → 原样转发宿主
        get_target_builders  → 宿主的 + NegativeTargetBuilder（按 token 查 NegativeBank，输出 negatives/neg_mask）
        compute_loss         → 宿主损失 + λ(step) · SeparationLoss(pred, expert, negatives, neg_mask)
        get_optimizers       → 宿主参数，学习率可覆盖（微调用小学习率）
宿主代码一行不改。对照组：λ=0（无分离损失）、shuffle_seed（错位负样本）。

注意：负样本作为 target 会被 NAVSIM 缓存到 <cache>/<log>/<token>/negative_targets.gz，
标签文件或 K 改变后需 force_cache_computation=true 重算。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from conservative_negatives.losses.negative_bank import NegativeBank
from conservative_negatives.losses.separation_loss import SeparationLoss, warmup_weight


class NegativeTargetBuilder(AbstractTargetBuilder):
    """按 token 从 NegativeBank 取保守负样本；不在库中的 token 输出全零 + 全 False 掩码（损失为 0）。"""

    def __init__(self, label_path: Union[str, Path], k: int = 4, label_col: str = "is_negative_strict",
                 shuffle_seed: Optional[int] = None):
        self._bank = NegativeBank.from_parquet(Path(label_path), k=k, label_col=label_col)
        self._k = k
        self._shuffle: Optional[Dict[str, str]] = None
        if shuffle_seed is not None:                      # 错位对照：token → 另一个 token 的负样本
            toks = list(self._bank.tokens)
            perm = np.random.default_rng(shuffle_seed).permutation(len(toks))
            self._shuffle = {t: toks[j] for t, j in zip(toks, perm)}
        self._n_hit = self._n_total = 0

    def get_unique_name(self) -> str:
        return "negative_targets" if self._shuffle is None else "negative_targets_shuffled"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        token = scene.scene_metadata.initial_token
        expert = scene.get_future_trajectory(num_trajectory_frames=8).poses.astype(np.float64)
        self._n_total += 1
        src = self._shuffle.get(token, token) if self._shuffle is not None else token
        if src in self._bank:
            self._n_hit += 1
            negs, mask = self._bank.get(src, expert)
        else:
            negs = torch.zeros((self._k, 8, 3), dtype=torch.float32)
            mask = torch.zeros((self._k,), dtype=torch.bool)
        return {"negatives": negs, "neg_mask": mask}

    @property
    def hit_rate(self) -> float:
        return self._n_hit / max(1, self._n_total)


class NegStatsCallback(pl.Callback):
    """把适配器每步的分离损失统计写进 Lightning 日志（激活率 = 风险 R6 的监控量）。"""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        agent = getattr(pl_module, "agent", None)
        stats = getattr(agent, "last_neg_stats", None)
        if stats:
            pl_module.log_dict({f"train/{k}": float(v) for k, v in stats.items()}, on_step=True, on_epoch=True,
                               prog_bar=("neg_active_frac" in stats), sync_dist=False)


class NegativeAugmentedAgent(AbstractAgent):
    def __init__(self, host: AbstractAgent, label_path: str, lambda_neg: float = 0.1, loss_kind: str = "hinge",
                 margin: float = 1.0, temperature: float = 1.0, k: int = 4, label_col: str = "is_negative_strict",
                 shuffle_seed: Optional[int] = None, warmup_frac: float = 0.2, total_steps: int = 1,
                 host_checkpoint: Optional[str] = None, lr: Optional[float] = None,
                 trajectory_key: str = "trajectory", lambda_scale: Optional[float] = None,
                 checkpoint_path: Optional[str] = None):
        super().__init__(requires_scene=getattr(host, "requires_scene", False))
        self.host = host
        # λ 是相对宿主轨迹项权重的比例：有效权重 = lambda_neg × lambda_scale（默认取宿主 config.trajectory_weight，TransFuser 为 10）
        if lambda_scale is None:
            lambda_scale = float(getattr(getattr(host, "_config", None), "trajectory_weight", 1.0))
        self._lambda_scale = float(lambda_scale)
        self._checkpoint_path = checkpoint_path        # 推理时装载本适配器训练出的 Lightning ckpt
        self._neg_builder = NegativeTargetBuilder(label_path, k=k, label_col=label_col, shuffle_seed=shuffle_seed)
        self._sep = SeparationLoss(loss_kind, margin=margin, temperature=temperature)
        self._lambda, self._warmup_frac, self._total_steps = float(lambda_neg), warmup_frac, max(1, int(total_steps))
        self._lr, self._traj_key = lr, trajectory_key
        self._step = 0
        self._cum_valid_negs = 0.0
        self.last_neg_stats: Dict[str, float] = {}
        if host_checkpoint:                               # 微调：先把宿主权重装进去
            self._load_host_checkpoint(host_checkpoint)

    # ---------- 宿主转发 ----------
    def name(self) -> str:
        return f"NegAug[{self.host.name()}]"

    def initialize(self) -> None:
        """推理时由 run_pdm_score 调用：优先装本适配器训练出的 ckpt（键 agent.host.*），否则退回宿主自己的 checkpoint。"""
        if self._checkpoint_path:
            sd = torch.load(self._checkpoint_path, map_location="cpu")
            sd = sd.get("state_dict", sd)
            sd = {k[len("agent."):] if k.startswith("agent.") else k: v for k, v in sd.items()}
            self.load_state_dict(sd)
            print(f"[NegAug] initialized from {Path(self._checkpoint_path).name}")
        elif getattr(self.host, "_checkpoint_path", None):
            self.host.initialize()

    def get_sensor_config(self) -> SensorConfig:
        return self.host.get_sensor_config()

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return self.host.get_feature_builders()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return list(self.host.get_target_builders()) + [self._neg_builder]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.host.forward(features)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        return self.host.compute_trajectory(agent_input)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return list(self.host.get_training_callbacks()) + [NegStatsCallback()]

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        if self._lr is None:
            return self.host.get_optimizers()
        return torch.optim.Adam(self.host.parameters(), lr=self._lr)

    # ---------- 损失 ----------
    def compute_loss(self, features, targets, predictions) -> torch.Tensor:
        host_loss = self.host.compute_loss(features, targets, predictions)
        lam = warmup_weight(self._step, self._total_steps, self._lambda, self._warmup_frac) * self._lambda_scale
        self._step += 1
        if self._lambda <= 0 or "negatives" not in targets:
            self.last_neg_stats = {"loss_host": float(host_loss.detach()), "lambda_eff": lam, "loss_neg": 0.0,
                                   "neg_active_frac": 0.0, "n_valid_negs": 0.0}
            return host_loss
        pred = predictions[self._traj_key].float()
        expert = targets[self._traj_key].float()
        negs = targets["negatives"].float().to(pred.device)
        mask = targets["neg_mask"].bool().to(pred.device)
        loss_neg, stats = self._sep(pred, expert, negs, mask)
        total = host_loss + lam * loss_neg
        self._cum_valid_negs += float(stats.get("n_valid_negs", 0.0))
        if self._step == 50 and self._cum_valid_negs == 0.0:
            print("[NegAug][WARN] 50 步内没有任何有效负样本：请检查 label_path 是否与训练 split 匹配"
                  "（navtest 场景需 negatives_navtest.parquet，navtrain 需 negatives_navtrain.parquet）、"
                  "以及缓存是否用 force_cache_computation=true 重建。")
        self.last_neg_stats = {"loss_host": float(host_loss.detach()), "loss_neg": float(loss_neg.detach()),
                               "lambda_eff": lam, "loss_neg_weighted": float((lam * loss_neg).detach()),
                               **{k: float(v) for k, v in stats.items()}}
        return total

    # ---------- checkpoint ----------
    def _load_host_checkpoint(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        # Lightning 存的是 AgentLightningModule，键形如 "agent._transfuser_model.xxx"；去掉 "agent." 前缀装进宿主
        sd = {k[len("agent."):] if k.startswith("agent.") else k: v for k, v in sd.items()}
        missing, unexpected = self.host.load_state_dict(sd, strict=False)
        print(f"[NegAug] loaded host checkpoint {Path(path).name}: missing={len(missing)} unexpected={len(unexpected)}")

    def load_state_dict(self, state_dict, strict: bool = True):
        """让本适配器训练产生的 Lightning checkpoint（键为 agent.host.xxx）也能被 run_pdm_score 装载。"""
        sd = {k[len("host."):] if k.startswith("host.") else k: v for k, v in state_dict.items()}
        return self.host.load_state_dict(sd, strict=False)
