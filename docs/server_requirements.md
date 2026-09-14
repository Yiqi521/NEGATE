# 服务器需求配置（用于申请）

依据：研究计划 §1 的实验规模、NAVSIM 官方数据规模、以及 2026-09-14 在本机（RTX 4060 8 GB / 22 线程 / 62 GB）的实测。

## 1. 一句话需求
**4 张 ≥ 24 GB 显存的 NVIDIA GPU（推荐 48 GB）、32 核以上 CPU、128 GB 内存、4 TB NVMe 存储、≥ 1 Gbps 外网，Linux，独占使用约 3 个月（2026-10 至 2027-01）。**

## 2. 配置表

| 项目 | 最低可行 | 推荐 | 依据 |
|---|---|---|---|
| GPU 数量 | 2 | **4** | 主线约 36 次 TransFuser 级训练（基线 3 + λ 网格 4×3 + 打分型 2×3 + 消融 ≈ 15），官方成本"1 GPU · 1 天/次"[9] → 36 GPU·天；4 卡并行 ≈ 9 天，2 卡 ≈ 18 天 |
| 单卡显存 | 24 GB（RTX 3090 / 4090 / A5000 / A10） | **48 GB（L40S / A6000 / A40）或 80 GB（A100 / H100）** | TransFuser batch 64 + 16-mixed 需 > 8 GB（本机无法训练）；Hydra-MDP 类打分型基线用 V2-99 骨干 + 4096–8192 词表，显存需求约为 TransFuser 的 2–3 倍；PLUTO 官方用 4×RTX 3090（24 GB） |
| CPU | 32 核 / 64 线程 | **64 核** | PDM 评分与指标缓存是 CPU 瓶颈：本机 22 线程缓存 navtest 12k 场景 25 min、评估 4 线程 40 min/seed；训练数据加载（JPEG 解码 + 特征缓存）每卡需 8–16 个 worker |
| 内存 | 64 GB | **128 GB（PLUTO 阶段 256 GB 更稳）** | NAVSIM 场景过滤会把整个 split 的日志 pickle 读入内存（navtrain 元数据 14 GB，展开后约 30 GB）；PLUTO 缓存 100 万帧用 40 线程，官方要求高内存 |
| 存储（NVMe） | 2 TB | **4 TB** | 见 §3 存储预算：NAVSIM 主线约 1.0 TB；nuPlan 闭环支线再加约 1.3 TB；checkpoint 与缓存约 0.3 TB |
| 外网带宽 | 300 Mbps | **≥ 1 Gbps** | 需下载约 700 GB（navtrain 445 + navtest 223 + 其它）；本机 HF 下载实测约 0.4 GB/min，1 Gbps 下约 2 h |
| 操作系统 / 驱动 | Ubuntu 20.04+，NVIDIA 驱动 ≥ 525（CUDA 11.7 兼容 torch 2.0.1） | Ubuntu 22.04，驱动 ≥ 535 | NAVSIM v1.1 锁定 torch 2.0.1 + cu117；PLUTO/PlanTF 用 nuPlan devkit（Python 3.9） |
| 软件 | conda、git、tmux/screen、rsync | + Docker、W&B/MLflow 出网权限 | 实验追踪与可复现性（研究计划 §13） |
| 使用期限 | 2 个月 | **3 个月（2026-10-01 → 2027-01-31）**，2 月保留 2 卡做补实验 | 对应时间表 P3–P5 |

## 3. 存储预算（GB）

| 内容 | 大小 | 阶段 |
|---|---|---|
| navtrain 传感器 blob（含历史帧） | 445（仅当前帧 300） | P0 |
| navtest 传感器 blob（相机 + 激光雷达） | 223 | P0 |
| OpenScene 元数据 + 地图 | 17 | P0 |
| TransFuser 训练特征缓存（navtrain，10.3 万样本） | 约 100–150（估算：拼接相机 1024×256×3 + 激光雷达 BEV 256×256） | P0 |
| metric cache（navtest + navtrain interact） | 6（实测） | P0 |
| checkpoint（36 次 × 0.7 GB × 保留 3 个/次） | 约 80 | P3–P4 |
| 实验日志 / 评估 CSV | < 10 | 全程 |
| **NAVSIM 主线小计** | **约 950–1,000** | |
| nuPlan DB（无传感器）train + val + test | 约 950 + 90 + 90（第三方统计，待核实） | P5 |
| PLUTO / PlanTF 特征缓存（100 万帧） | 约 200–300 | P5 |
| **含 nuPlan 闭环支线合计** | **约 2,300–2,400** | |
| 建议容量（含 30% 余量与临时解压空间） | **4 TB** | |

若存储紧张的降级方案：navtrain 只下当前帧（省 145 GB）；nuPlan 只下 val + test DB（PLUTO 用官方预训练权重微调而非从头训练，省约 950 GB）→ 主线 + 支线约 1.3 TB，2 TB 可行。

## 4. 算力预算（GPU·天）

| 实验 | 次数 | 单次 | 小计 |
|---|---|---|---|
| TransFuser / LTF 基线复现 | 3 seeds | 1 | 3 |
| 回归型 + 分离损失：λ ∈ {0.05, 0.1, 0.2, 0.3} × 3 seeds | 12 | 1 | 12 |
| 打分型（Hydra-MDP 类）基线 + 负样本版：2 配置 × 3 seeds | 6 | 2 | 12 |
| 消融（算子子集、过滤级数、损失形式、激活范围）≈ 5 维 × 3 seeds | 15 | 1 | 15 |
| PLUTO / PlanTF 迁移：基线 + 负样本版 × 3 seeds（4 卡 × 1–2 天/次） | 6 | 4–8 | 24–48 |
| 评估（navtest 12k 场景 × 每个 checkpoint，CPU 为主） | ~40 次 | 0.1 | 4 |
| **合计** | | | **70–94 GPU·天** |

4 卡：约 18–24 天纯计算；加上调试与排队，3 个月是合理申请期。2 卡：约 5–6 周纯计算，需砍掉消融的一半与 PLUTO 的 1 个 seed。

## 5. 替代方案：云 GPU
若无法申请到物理服务器，等价云配置为：4× A100 40/80 GB 或 4× L40S 实例 + 4 TB 块存储，累计约 500–600 GPU·小时（按 4 卡 × 24 天 × 部分利用率折算）。费用因供应商差异大，需按当时报价核算；数据下载流量（约 700 GB 入、少量出）通常免费或低价。注意 NAVSIM 数据需从 Hugging Face / AWS S3 拉取，云实例的出入网限制要提前确认。

## 6. 申请用简述

**中文**：本研究在 NAVSIM/nuPlan 基准上训练端到端自动驾驶规划器（TransFuser、Hydra-MDP 类打分型规划器与 PLUTO），共需约 36 次 GPU 级训练（官方单次成本 1 GPU·天）加 6 次多卡训练，总计约 70–94 GPU·天；数据集约 1 TB（含 nuPlan 闭环支线约 2.4 TB）。申请配置：4 张 48 GB 显存 GPU（或等价 24 GB × 4）、64 核 CPU、128 GB 内存、4 TB NVMe、1 Gbps 外网，使用期 2026 年 10 月至 2027 年 1 月。

**English**: The project trains end-to-end driving planners (TransFuser, Hydra-MDP-style scorers, PLUTO) on the NAVSIM/nuPlan benchmarks: about 36 single-GPU training runs (official cost ≈ 1 GPU-day each) plus 6 multi-GPU runs, totaling roughly 70–94 GPU-days. Datasets require ≈ 1 TB (≈ 2.4 TB including the nuPlan closed-loop branch). Requested: 4 × 48 GB GPUs (or 4 × 24 GB), 64 CPU cores, 128 GB RAM, 4 TB NVMe, ≥ 1 Gbps network, for October 2026 – January 2027.

**日本語**：本研究では NAVSIM / nuPlan ベンチマーク上でエンドツーエンド運転プランナ（TransFuser、Hydra-MDP 系スコアリング型、PLUTO）を学習します。単一 GPU 学習（公式コスト約 1 GPU・日）約 36 回とマルチ GPU 学習 6 回、合計約 70〜94 GPU・日を要し、データ容量は約 1 TB（nuPlan 閉ループ実験を含めると約 2.4 TB）です。希望構成：48 GB GPU × 4（または 24 GB × 4）、CPU 64 コア、メモリ 128 GB、NVMe 4 TB、1 Gbps 以上のネットワーク、利用期間 2026 年 10 月〜2027 年 1 月。
