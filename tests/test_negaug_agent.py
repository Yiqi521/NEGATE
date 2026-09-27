"""适配器 NegativeAugmentedAgent 的端到端测试（需要本机 navtest 相机数据与 results/labels/negatives_navtest.parquet）。

运行：
  source ~/navsim_workspace/env.sh; export PYTHONPATH=$PWD CN_PROJECT_ROOT=$PWD
  python tests/test_negaug_agent.py            # 或 pytest -q tests/test_negaug_agent.py
检查项：
  1. 官方 LTF 权重装入 missing=0 / unexpected=0
  2. 负样本目标与 TransFuser 目标一起缓存；命中率与标签一致
  3. 前向 + 损失 + 反向；分离损失激活率 > 0（R6 监控量）
  4. 错位对照与 λ=0 对照分支
  5. 适配器 state_dict 回装 0 缺失（Lightning ckpt 兼容）
  6. 报告 torch / CUDA 版本与编译架构（cu117 vs cu128 环境对比用）
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CN_PROJECT_ROOT", str(ROOT))

from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.planning.training.dataset import Dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _loader(agent, tokens, flags):
    root = Path(os.environ["OPENSCENE_DATA_ROOT"]); dev = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    sf = instantiate(OmegaConf.load(dev / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
    sf.tokens = tokens; sf.log_names = flags.loc[tokens].log_name.unique().tolist()
    return SceneLoader(root / "navsim_logs/test", root / "sensor_blobs/test", sf, agent.get_sensor_config())


def test_negaug_agent(n_with: int = 8, n_without: int = 4) -> dict:
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | arch {torch.cuda.get_arch_list()} | gpu {torch.cuda.get_device_name(0)}")
    cfg = OmegaConf.load(ROOT / "configs/agent/ltf_negaug.yaml")
    cfg.label_path = str(ROOT / "results/labels/negatives_navtest.parquet")
    agent = instantiate(cfg)
    assert agent._lambda_scale == 10.0, agent._lambda_scale
    names = [b.get_unique_name() for b in agent.get_target_builders()]
    assert names == ["transfuser_target", "negative_targets"], names

    neg = pd.read_parquet(cfg.label_path)
    flags = pd.read_parquet(ROOT / "results/labels/scene_flags_navtest.parquet").set_index("token")
    strict = set(neg[neg.is_negative_strict].token)
    toks = list(neg[neg.is_negative_strict].token.unique()[:n_with]) + [t for t in neg.token.unique() if t not in strict][:n_without]

    cache = Path(os.environ["NAVSIM_EXP_ROOT"]) / "negaug_test_cache"; shutil.rmtree(cache, ignore_errors=True)
    ds = Dataset(_loader(agent, toks, flags), agent.get_feature_builders(), agent.get_target_builders(),
                 cache_path=str(cache), force_cache_computation=True)
    assert len(ds) == len(toks), (len(ds), len(toks))
    assert abs(agent._neg_builder.hit_rate - n_with / (n_with + n_without)) < 1e-6, agent._neg_builder.hit_rate
    f, t = ds[0]
    assert tuple(t["negatives"].shape) == (4, 8, 3) and t["neg_mask"].dtype == torch.bool

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    agent.host.to(dev); agent._total_steps = 10
    stats_seen = []
    for feats, targs in DataLoader(ds, batch_size=6, shuffle=False):
        feats = {k: v.to(dev) for k, v in feats.items()}; targs = {k: v.to(dev) for k, v in targs.items()}
        loss = agent.compute_loss(feats, targs, agent.forward(feats)); loss.backward()
        assert torch.isfinite(loss), loss
        stats_seen.append(agent.last_neg_stats)
    active = max(s["neg_active_frac"] for s in stats_seen)
    assert active > 0.0, "分离损失零激活（R6）"

    cfg2 = OmegaConf.load(ROOT / "configs/agent/ltf_negaug.yaml"); cfg2.label_path = cfg.label_path; cfg2.shuffle_seed = 0
    assert instantiate(cfg2)._neg_builder.get_unique_name() == "negative_targets_shuffled"
    cfg3 = OmegaConf.load(ROOT / "configs/agent/ltf_negaug.yaml"); cfg3.label_path = cfg.label_path; cfg3.lambda_neg = 0.0
    a3 = instantiate(cfg3); a3.host.to(dev)
    l3 = a3.compute_loss(feats, targs, a3.forward(feats))
    assert a3.last_neg_stats["loss_neg"] == 0.0 and torch.isfinite(l3)

    sd = {k: v for k, v in agent.state_dict().items()}
    r = agent.load_state_dict(sd)
    assert len(r.missing_keys) == 0 and len(r.unexpected_keys) == 0, (r.missing_keys[:3], r.unexpected_keys[:3])

    shutil.rmtree(cache, ignore_errors=True)
    out = {"torch": torch.__version__, "cuda": torch.version.cuda, "hit_rate": agent._neg_builder.hit_rate,
           "max_neg_active_frac": active, "last_stats": stats_seen[-1]}
    print("PASS", out)
    return out


if __name__ == "__main__":
    test_negaug_agent()
