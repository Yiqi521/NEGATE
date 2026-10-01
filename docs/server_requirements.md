# 服务器配置申请（2026-09-27 修订：分阶段申请）

> 面向：实验室导师与设备管理者。第 9 节为可直接提交的申请文（日 / 中 / 英）。
> 修订原因：改为**分阶段申请**。本次只申请第一阶段，内容是必要的基本实验（TransFuser 微调、λ 筛选、两个对照，每臂 2 个种子，第 3 个种子留到第二阶段补），不到 1 GPU·天。从头训练锚点、第二宿主 PLUTO 与消融等第一阶段结果出来后再设计、另行申请。旧版（2026-09-18）按全部实验一次申请 2 × 24 GB GPU、2 个月，已作废；旧版中的实验矩阵也与研究计划 §16.2 不一致（写的是 63 次，§16.2 为 40 次），本版以 §16.2 为准。

## 1. 一句话需求（第一阶段）

**1 张 24 GB 以上显存的 GPU（Ampere / Ada / Blackwell 均可）、32 核 CPU、128 GB 内存、1 TB 存储、≥ 1 Gbps 外网、Ubuntu 22.04，使用 4 周（2026-10-12 前后开始，日期可调）。GPU 实际占用约 17–22 小时，不需要独占，共享服务器上的配额即可。**

## 2. 分阶段总览

| 阶段 | 地点 | 内容 | 算力 | 时间 | 状态 |
|---|---|---|---|---|---|
| 0 最小验证实验 | 本机 RTX 5060 Ti 8 GB | LTF 单种子 4 次微调，只看方向（研究计划 §16.1，`docs/minimal_experiment_runbook.md`） | 本机约 9–13 h | 2026-10 上旬 | 待执行（需先装 WSL2） |
| **1 基本实验（本次申请）** | 服务器 | TransFuser 微调：λ 筛选 + 最优 λ、λ=0、错位负样本对照各 2 种子 | **约 17–22 GPU·小时（不到 1 GPU·天）** | **4 周** | 本次申请 |
| 2 扩展实验 | 服务器 | 按第一阶段结果设计（见第 7 节） | 参考约 12–15 GPU·天 | 届时另行申请 | — |

阶段之间有判定门：第 0 阶段四条标准全过才进第一阶段；第一阶段的判定结果决定第二阶段做什么、要多少资源。

## 3. 第一阶段实验内容

宿主选 **TransFuser（相机 + 激光雷达）**：它是 NAVSIM 官方基线（PDMS 84.0），也是第二阶段从头训练锚点的宿主，第一阶段选出的 λ 可以直接沿用；纯相机的 LTF 已由第 0 阶段在本机覆盖。如改用 LTF，GPU 时间不变，存储可省约 100 GB（navtest 不需要激光雷达）。

| 组 | 内容 | 次数 | 单次 | GPU 时间 | 为什么必要 |
|---|---|---|---|---|---|
| S0 验收 | 官方 TransFuser 权重 seed 0（即微调起点），在 cu128 环境下跑 navtest 全量 PDMS | 1 次评估 | 约 1 h | 约 1 h | 确认服务器流水线与官方一致（84.0 ± 1），作为本阶段的正式参照行 |
| S1 λ 筛选 | 交互富集子集约 3 万帧（navtrain 交互场景 9,738 + 2 倍背景帧），从官方权重微调 15 epoch；λ ∈ {0, 0.05, 0.1, 0.2, 0.3} 单种子 | 5 | 1–1.5 h | 5–7.5 h | 剂量方向；选出最优 λ |
| S2 补种子 | 最优 λ 与 λ=0 各补 1 个种子（合计各 2 种子） | 2 | 1–1.5 h | 2–3 h | 检查两个种子方向是否一致；λ=0 排除"多训几轮"的解释 |
| S3 错位对照 | 最优 λ + 错位负样本（打乱 token 配对），2 种子 | 2 | 1–1.5 h | 2–3 h | 排除"损失项自身的正则化"的解释 |
| S4 评估 | 9 个 checkpoint：navtest 全量 PDMS + 三个行为指标 + 分层配对比较 | 9 | 约 0.8 h | 约 7 h | 交互层看效果，全量看是否伤到其它场景 |
| | **合计** | **9 次训练** | | **约 17–22 h（不到 1 GPU·天）** | |

**种子策略**：第一阶段每臂 2 个种子，第 3 个种子留到第二阶段补齐。依据是各臂都从同一份官方权重出发，种子只影响数据顺序和 dropout，预计方差小于从头训练。作为参照，官方 LTF 三个从头训练的种子在 interact 层 PDMS 的标准差约 1.0 分（`docs/results_baseline.md`），与 1.5 分的效应量门槛接近；微调种子的方差尚未实测。因此加一条规则：若某一臂两个种子的效应方向相反，或种子间差异与效应量相当，当场补第 3 个种子（每臂 1–1.5 h），不等第二阶段。

**第一阶段判定**（沿用研究计划 §16.2 的显著性标准，只取与本阶段对应的部分）：
1. 工程：S0 落在官方值 ±1 以内；处理组的分离损失激活率在 warm-up 后 ≥ 1e-3。
2. 效果：交互层上最优 λ 相对 λ=0 的间隙接受率与 EP 提升，帧级 bootstrap 95% CI 不含零（先对种子平均），且两个种子各自方向一致；NC / DAC / DDC / TTC 的 CI 下界不低于对照均值。
3. 不伤全局：navtest 全量 PDMS 不低于 λ=0。
4. 机制：错位对照的效应量显著小于处理组。

**暂不做（留到第二阶段按结果决定）**：从头训练锚点、随机负样本与安全负样本两个对照、三维消融、PLUTO 迁移与 nuPlan 闭环。

**开工前需补的工程项**（本机即可完成，不占服务器时间）：
- `scripts/run_negaug_finetune.sh` 与 `scripts/eval_minimal_experiment.sh` 写死了 `agent=ltf_negaug`（行为指标的 `--agent-config` 也是），需加一个 `AGENT` 变量切到 `tf_negaug`。
- `scripts/setup/` 目前只下载 LTF 权重、navtest 相机包和 navtrain 第 1 包，需补：官方 TransFuser 权重、navtrain 全量当前帧、navtest 激光雷达。
- 在全量 navtrain 上生成微调子集：`build_finetune_subset.py --sensor-dir $OPENSCENE_DATA_ROOT/sensor_blobs/trainval --bg-ratio 2 --name navtrain_full_ft`。

## 4. 第一阶段推荐配置

| 项目 | 推荐 | 最低可行 | 依据 |
|---|---|---|---|
| GPU | **1 张，≥ 24 GB**：RTX 4090 / RTX 5090 / RTX 6000 Ada / L40S / A100 | 1 张 16 GB（batch 32 × 梯度累积 2） | TransFuser batch 64 约 13 GB（实测 batch 32 为 6.4 GB 后外推）。A5000、L4 可用，但估计慢约一倍，本阶段仍可在几天内跑完 |
| 驱动 | ≥ 535（Ampere / Ada）；Blackwell 需 ≥ 570 | 同 | 已准备并验证两套环境：cu117（到 sm_86）与 cu128（含 sm_120），见 `docs/env.md` |
| CPU | 32 核 | 16 核 | 特征缓存与 PDM 评分是 CPU 瓶颈；训练数据加载约 8 个 worker |
| 内存 | 128 GB | 64 GB | 按场景过滤会把整个 split 的日志读入内存（navtrain 元数据 14 GB，展开约 30 GB），另有多进程缓存 |
| 存储 | **1 TB** | 700 GB | 见第 5 节，约 0.63 TB |
| 外网 | ≥ 1 Gbps | 300 Mbps | 需从 Hugging Face / AWS 下载约 0.54 TB；1 Gbps 约半天，300 Mbps 约 4–8 小时起 |
| 系统 | Ubuntu 22.04 | Ubuntu 20.04 | |
| 账户 | 普通用户即可，不需要 sudo；可选 Docker | 同 | |
| 期限 | **4 周** | 2 周 | 纯计算约 2–3 天，其余为数据准备、分析与重跑余量（第 6 节） |

## 5. 第一阶段存储预算

| 内容 | 大小（GB） | 说明 |
|---|---|---|
| navtrain 传感器（仅当前帧，相机 + 激光雷达） | 300 | 压缩包按日志打包，无法只下交互场景 |
| navtest 传感器（相机 + 激光雷达） | 223 | 评估输入 |
| OpenScene 元数据 + 地图 | 17 | |
| 微调子集特征缓存（约 3 万帧，含负样本目标） | 30–45 | 按每帧约 1–1.5 MB 估算 |
| navtest metric cache | 3 | 服务器上重新生成约 0.5 h |
| checkpoint（9 次 × 约 0.7 GB） | 约 7 | |
| 环境（Docker 镜像或 conda）、官方权重、日志 | 约 30 | 镜像 22.2 GB |
| **合计** | **约 610–625** | 建议 1 TB，含解压临时空间与余量 |

## 6. 第一阶段时间表（1 张 24 GB 卡）

| 周 | 内容 |
|---|---|
| 第 1 周 | 环境（约 0.5 天）；下载约 0.54 TB（0.5–1 天）；navtest metric cache、全量 navtrain 微调子集与特征缓存（约 3 h）；S0 验收（约 1 h） |
| 第 2 周 | S1 λ 筛选 5 次（约 6 h）→ 选 λ；S2、S3 共 4 次（约 5 h）；S4 全部评估（约 7 h） |
| 第 3 周 | 分层配对比较与 bootstrap、按第 3 节四条判定、失效样例出图；写第二阶段设计与申请 |
| 第 4 周 | 余量：修 bug、重跑；若某臂两种子方向不一致，补第 3 个种子 |

若分到 A5000 / L4 级别的卡，第 2 周的计算量约翻倍，仍在 4 周内。

## 7. 第二阶段（参考，不在本次申请内）

第二阶段的内容取决于第一阶段的判定：

| 第一阶段结果 | 第二阶段做什么 | 资源估计 |
|---|---|---|
| 四条全过 | 第一阶段三臂（最优 λ、λ=0、错位）各补第 3 个种子（3 × 1–1.5 h）；从头训练锚点（基线与最优 λ 各 3 种子，6 × 约 30 h，7.5 GPU·天）；随机 / 安全负样本对照与三维消融（约 0.5 GPU·天）；PLUTO 迁移与 nuPlan Test14 闭环（9 × 8–16 h，3–6 GPU·天）；评估约 1.5 GPU·天 | 约 12–15 GPU·天；2 × 24 GB GPU、2 TB、6–8 周（新增 nuPlan 验证 / 测试库约 180 GB、PLUTO 缓存 100–200 GB、全量 TransFuser 特征缓存 100–150 GB） |
| 方向正确但不显著 | 每臂加 2–4 个种子，或扩大子集 / 延长 epoch | < 1 GPU·天，可沿用第一阶段资源 |
| 分离损失激活率持续 < 1e-3（风险 R6） | 启用打分型规划器基线 | 约 6 GPU·天，需 48 GB 显存 |
| 方向错误，或错位对照与处理组效应相当 | 不申请第二阶段，回本机修判据或损失设计 | — |

## 8. 数字来源

- 显存与步时：2026-09-18 在旧开发机 RTX 4060 Laptop（8 GB）上用随机张量实测 TransFuser 前向 + 反向 + Adam，batch 4/8/16/32 峰值 1.4/2.3/3.7/6.4 GB，batch 64 溢出；cu117 与 cu128 两套环境结果一致（`docs/env.md`）。
- 单次微调约 1 h 的推算：NAVSIM 论文写的是 TransFuser 在 navtrain 上 "1 GPU for 1 day"（100 epoch × 约 10.3 万样本，论文未注明卡型），折合约 43 万样本 / 小时；第一阶段单次微调 3 万帧 × 15 epoch ≈ 45 万样本，约 1 h，按 1–1.5 h 预算。
- 最小实验的对照值：7,000 帧 × 10 epoch 在 8 GB 卡上约 30–40 分钟（`docs/minimal_experiment_runbook.md`）。
- 数据规模：NAVSIM `docs/splits.md`（navtrain 仅当前帧 300 GB；navtest 223 GB）；旧开发机实测 navtest 相机部分 121 GB。
- CPU 瓶颈：旧开发机 metric caching 12,146 场景 25 min（22 线程）；LTF 全量 PDMS 评估 40 min / 种子（4 线程）。
- 本机约束：RTX 5060 Ti 8 GB，C 盘剩余 374 GB（`docs/env.md` 末节），放不下第一阶段约 0.6 TB 的数据。

## 9. 提交用申请文

### 日本語（提出用）

**件名**：修士研究用 GPU サーバ利用申請（第 1 段階）

**申請者**：SU Yiqi（37-255134）　**指導教員**：中野公彦 先生

**研究課題**：Machine Learning-Based Human-Like Automated Driving Vehicle Behavior ― 「安全だが過度に保守的な」軌道を負例として利用するエンドツーエンド運転プランナの学習フレームワーク

**段階的な申請について**：本申請は第 1 段階（基本検証）のみを対象とします。第 1 段階の結果を踏まえて後続の実験（ゼロからの学習、別プランナへの転移、アブレーション）を設計し、必要な計算資源はその時点で改めて申請します。

**第 1 段階の内容**：NAVSIM ベンチマーク上で、公開されている TransFuser の学習済み重みに提案手法の負例損失を付加してファインチューニングします。損失重みを 5 水準で比較したうえで、損失なし（λ=0）および負例の対応をシャッフルした対照と 2 シードずつ比較し（3 シード目は第 2 段階で追加）、交差点・合流などの相互作用場面で安全性を損なわずに過度な保守性が低減されるかを検証します。学習は計 9 回（各約 1 時間）です。データ準備・負例生成・評価ツールは完成済みですが、ノート PC（GPU 8 GB、空きディスク 374 GB）では全量データ（約 0.6 TB）を扱えません。

**希望構成**

| 項目 | 希望 | 最低 |
|---|---|---|
| GPU | 24 GB 以上 × 1 枚（RTX 4090 / RTX 5090 / RTX 6000 Ada / L40S / A100 など） | 16 GB × 1 枚 |
| CPU | 32 コア | 16 コア |
| メモリ | 128 GB | 64 GB |
| ストレージ | 1 TB | 700 GB |
| ネットワーク | 1 Gbps 以上 | 300 Mbps |
| OS | Ubuntu 22.04、NVIDIA ドライバ 535 以上（Blackwell 世代は 570 以上） | Ubuntu 20.04 |
| 利用期間 | 2026 年 10 月 12 日頃から 4 週間（開始日は調整可） | 2 週間 |

**所要計算量**：GPU 使用は約 17〜22 時間（1 GPU 日未満）で、専有は不要です。共有サーバ上での利用でも実施できます。

**備考**：Ampere / Ada / Blackwell のいずれの世代にも対応する実行環境を構築・検証済みで、sudo 権限は不要です。コードは研究室リポジトリ（NEGATE）で管理します。第 2 段階の概算（第 1 段階の結果次第）は GPU 24 GB × 2 枚、ストレージ 2 TB、6〜8 週間、約 12〜15 GPU 日です。

### 中文

申请一台 Linux 服务器用于硕士研究的第一阶段基本实验：1 张 24 GB 以上显存的 GPU（RTX 4090 / 5090 / RTX 6000 Ada / L40S / A100 等，Ampere、Ada、Blackwell 均可）、32 核 CPU、128 GB 内存、1 TB 存储、≥ 1 Gbps 外网，自 2026 年 10 月 12 日前后起使用 4 周。内容：在 NAVSIM 上给官方 TransFuser 权重接入保守负样本损失并微调，5 档 λ 筛选后，与 λ=0 和错位负样本两个对照各做 2 种子比较（第 3 个种子在第二阶段补），共 9 次训练、GPU 约 17–22 小时，不需要独占。数据准备、负样本生成与评估工具已完成，但笔记本（8 GB 显存、剩余 374 GB 磁盘）放不下约 0.6 TB 的全量数据。第二阶段（从头训练、PLUTO 迁移、消融）将根据第一阶段结果设计后另行申请，参考规模为 2 × 24 GB GPU、2 TB、6–8 周。

### English

Request (phase 1 of a staged plan): a Linux server with 1 × GPU of at least 24 GB (RTX 4090 / 5090 / RTX 6000 Ada / L40S / A100; Ampere, Ada or Blackwell), 32 CPU cores, 128 GB RAM, 1 TB storage and ≥ 1 Gbps network, for 4 weeks starting around 12 October 2026. Purpose: fine-tune the official NAVSIM TransFuser weights with the proposed conservative-negative loss, screen five loss weights, then compare the best weight against two controls (no loss, shuffled negatives) with 2 seeds each (a third seed is added in phase 2) — 9 training runs, about 17–22 GPU-hours in total, so a shared server is sufficient. Data preparation, negative generation and evaluation tooling are complete, but the full data (≈ 0.6 TB) does not fit on the laptop (8 GB GPU, 374 GB free disk). Phase 2 (training from scratch, transfer to PLUTO, ablations) will be designed from the phase 1 results and requested separately; the current estimate is 2 × 24 GB GPUs, 2 TB and 6–8 weeks.

## 10. 备选方案：云 GPU

第一阶段等价于约 22 GPU·小时（24 GB 卡）+ 1 TB 块存储 + 约 0.54 TB 入网流量，按供应商当时报价核算；入网流量通常免费。数据下载在整个阶段中耗时最长，云实例的出网与磁盘限制需提前确认。
