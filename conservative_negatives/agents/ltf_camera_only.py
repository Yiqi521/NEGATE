"""纯相机 Latent TransFuser（LTF）智能体：用于只下载了相机 blob 的本机评估。

在 latent=True 时，官方 TransfuserBackbone 用可学习的 lidar_latent 代替激光雷达特征，
但官方 TransfuserFeatureBuilder 仍会读取点云并计算直方图。本子类：
  1. get_sensor_config 只请求 cam_f0 / cam_l0 / cam_r0（当前帧）；
  2. 特征构建器的 lidar_feature 用同形状零张量占位。
权重与官方 ltf_seed_*.ckpt 完全兼容。
"""
from __future__ import annotations

from typing import List

import torch

from navsim.agents.abstract_agent import AbstractFeatureBuilder
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder


class CameraOnlyTransfuserFeatureBuilder(TransfuserFeatureBuilder):
    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:  # noqa: D401
        cfg: TransfuserConfig = self._config
        h = int((cfg.lidar_max_x - cfg.lidar_min_x) * int(cfg.pixels_per_meter))
        w = int((cfg.lidar_max_y - cfg.lidar_min_y) * int(cfg.pixels_per_meter))
        c = 2 if cfg.use_ground_plane else 1
        return torch.zeros((c, h, w), dtype=torch.float32)


class LTFCameraOnlyAgent(TransfuserAgent):
    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[3], cam_l0=[3], cam_l1=False, cam_l2=False,
            cam_r0=[3], cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=False,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [CameraOnlyTransfuserFeatureBuilder(config=self._config)]
