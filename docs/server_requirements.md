# 服务器配置申请（2026-09-18 修订版）

> 面向：实验室导师与设备管理者。第 6 节为可直接提交的一页申请文（日 / 中 / 英）。
> 修订原因：实验矩阵已从 42 次训练（70–94 GPU·天）精简为约 57 次小任务（15–18 GPU·天），配置需求相应下调；所有数字均来自 2026-09-14/15 在本机的实测或官方文档，来源见第 5 节。

## 1. 一句话需求

**2 张 24 GB 显存 GPU（Ampere / Ada 架构）、32 核 CPU、128 GB 内存、2 TB NVMe、1 Gbps 外网、Ubuntu 22.04，独占使用 2 个月（2026-10-01 → 2026-11-30），可选延长至 2027-01-31 做补实验。**

若实验室能提供 4 张同规格 GPU，纯训练时间从约 9 天缩到约 5 天；若只能提供 1 张，仍可完成但需约 3 周且无法并行做消融。

## 2. 推荐配置与依据

| 项目 | 推荐 | 最低可行 | 依据（实测 / 官方） |
|---|---|---|---|
| GPU 数量 | **2** | 1 | 精简后主线 15–18 GPU·天；λ 筛选 → 补种子存在串行依赖，2 卡约 8–9 天，4 卡约 4.5–5 天，1 卡约 3 周 |
| 单卡显存 | **24 GB**（RTX 4090 / RTX A5000 / L4 均可） | 16 GB | 本机实测 TransFuser batch 32 峰值 6.4 GB，官方 batch 64 外推约 13 GB；PLUTO 官方在 24 GB 卡上训练 |
| GPU 架构 | **Ampere 或 Ada（sm_80–sm_89）** | 同 | 项目锁定 torch 2.0.1 + CUDA 11.7，编译内核只到 sm_86；Ada 可向下兼容运行。**Blackwell（RTX 5090 / B 系列）需重建环境到 torch ≥ 2.7 + CUDA 12.8 并重跑基线验收**，见第 4 节 |
| CPU | **32 核 / 64 线程** | 16 核 | PDM 评分与特征缓存是 CPU 瓶颈：本机 22 线程缓存 12k 场景 25 分钟，4 线程评估 40 分钟/次；训练数据加载每卡需 8 个 worker |
| 内存 | **128 GB** | 64 GB | NAVSIM 场景过滤把整个 split 的日志读入内存（navtrain 元数据 14 GB，展开约 30 GB）；PLUTO 特征缓存阶段多进程占用高 |
| 存储 | **2 TB NVMe** | 1.5 TB | 见第 3 节：主线约 1.0 TB，PLUTO 支线约 0.5 TB，余量 0.5 TB |
| 外网 | **≥ 1 Gbps** | 300 Mbps | 需从 Hugging Face / AWS 拉取约 900 GB；本机实测 0.4 GB/min 下需 1.5 天，1 Gbps 下约 3 小时 |
| 系统与驱动 | Ubuntu 22.04，NVIDIA 驱动 ≥ 535 | Ubuntu 20.04，驱动 ≥ 525 | CUDA 11.7 运行时向后兼容新驱动 |
| 软件 | conda、git、tmux、rsync；可选 Docker | 同 | 环境定义已固化（`docs/env.md`） |
| 账户 | 普通用户 + 50 GB home；数据盘可写 | 同 | 不需要 sudo |
| 期限 | **2 个月** + 可选 2 个月延长 | 6 周 | 见第 7 节时间表 |

**不需要的东西**：48 GB 以上显存（打分型规划器已改为条件触发）、NVLink（本规模 DDP 走 PCIe 足够）、InfiniBand、多节点。

## 3. 存储预算

| 内容 | 大小（GB） | 说明 |
|---|---|---|
| NAVSIM navtrain 传感器（仅当前帧） | 300 | 训练输入；缓存后可删原始文件回收 |
| NAVSIM navtest 传感器（相机 + 激光雷达） | 223 | 评估输入；本机已有相机部分 121 GB 可 rsync 过去 |
| OpenScene 元数据 + 地图 | 17 | |
| TransFuser 特征缓存（10.3 万样本） | 100–150 | 估算：拼接相机 3×256×1024 + 激光雷达 BEV 256×256 |
| metric cache（navtest + navtrain 交互子集） | 6 | 本机已生成，可直接复制 |
| checkpoint（约 57 次 × 0.7 GB × 保留 2 个） | 80 | |
| **NAVSIM 主线小计** | **约 750–800** | |
| nuPlan 数据库：仅 val + test 分片（无传感器） | 约 180 | PLUTO 在 val 微调、Test14 闭环评估，不下 950 GB 的 train 分片 |
| PLUTO 特征缓存 | 100–200 | 仅交互场景 |
| **含 PLUTO 支线合计** | **约 1.1–1.2 TB** | |
| 建议容量 | **2 TB** | 含解压临时空间与 30% 余量 |

## 4. 算力预算（精简后的实验矩阵）

| 组别 | 次数 | 单次（24 GB 卡） | GPU·天 | 说明 |
|---|---|---|---|---|
| TransFuser 从头训：基线 + 最优 λ，各 3 种子 | 6 | 约 30 h | 7.5 | 证明效果不是微调假象 |
| TransFuser 微调：λ 四档单种子筛选 + 最优两档补种子 | 8 | 约 4.5 h | 1.5 | 剂量曲线 |
| TransFuser 微调：三维消融（有无 O7、hinge/InfoNCE、严格/非严格标签）+ 补种子 | 7 | 约 4.5 h | 1.3 | |
| PLUTO 微调：对照 + 处理组，各 3 种子 | 6 | 8–16 h | 3–4 | 跨规划器迁移（RQ3） |
| 评估：PDMS 约 30 次 + nuPlan 闭环 6 次 | 36 | CPU 为主 | 约 2 | 可与训练并行 |
| **合计** | **约 63** | | **15–18** | 约 400 GPU·小时 |

条件触发项（不计入申请）：若微调中分离损失激活率持续 < 1e-3（风险 R6），启用打分型规划器基线，追加约 6 GPU·天与 48 GB 显存需求，届时另行申请。

## 5. 数字来源

- 显存与步时：2026-09-18 本机 RTX 4060 Laptop（8 GB）用随机张量实测 TransFuser 前向 + 反向 + Adam，batch 4/8/16/32 峰值 1.4/2.3/3.7/6.4 GB，batch 64 溢出；步时 batch 32 约 1.0 s。24 GB 卡按算力比约 3.5 倍外推。
- 官方训练成本：NAVSIM 论文 [9] "TransFuser: 1 GPU for 1 day on navtrain"；PLUTO 论文：4 × RTX 3090，100 万帧 25 epoch，22 h（无 CIL）/ 45 h（含 CIL）。
- 数据规模：NAVSIM `docs/splits.md`（navtrain 445 GB 含历史帧 / 300 GB 仅当前帧；navtest 223 GB）；本机已下载 navtest 相机 121 GB 实测。
- CPU 瓶颈：本机 metric caching 12,146 场景 25 min（22 线程）；PDMS 评估 40 min/seed（4 线程）。
- 兼容性：`torch.cuda.get_arch_list()` 在锁定环境中返回 `['sm_37', …, 'sm_86']`；RTX 5090 为 sm_120。

## 6. 提交用申请文

### 日本語（提出用）

**件名**：修士研究用 GPU サーバ利用申請

**申請者**：SU Yiqi（37-255134）　**指導教員**：中野公彦 先生

**研究課題**：Machine Learning-Based Human-Like Automated Driving Vehicle Behavior ― 「安全だが過度に保守的な」軌道を負例として利用するエンドツーエンド運転プランナの学習フレームワーク

**目的**：NAVSIM / nuPlan ベンチマーク上で、既存プランナ（TransFuser、PLUTO）に提案する負例損失を付加して学習・評価し、進行性の向上と安全性の維持を 3 シード × 層別評価で検証する。データ準備・負例生成・評価ツールはすでにノート PC 上で完成しており、残るのは学習のみである（ノート PC は GPU 8 GB のため学習不可）。

**希望構成**

| 項目 | 希望 | 最低 |
|---|---|---|
| GPU | 24 GB × 2 枚（RTX 4090 / A5000 / L4 相当、Ampere または Ada 世代） | 24 GB × 1 枚 |
| CPU | 32 コア | 16 コア |
| メモリ | 128 GB | 64 GB |
| ストレージ | NVMe 2 TB | 1.5 TB |
| ネットワーク | 1 Gbps 以上 | 300 Mbps |
| OS | Ubuntu 22.04、NVIDIA ドライバ 535 以上 | Ubuntu 20.04 |
| 利用期間 | 2026 年 10 月 1 日 〜 11 月 30 日（延長の場合 2027 年 1 月末まで） | 6 週間 |

**根拠**：学習は計 63 回・約 15〜18 GPU 日（約 400 GPU 時間）。TransFuser の学習は公式で 1 GPU・日 / 回、VRAM 約 13 GB（実測外挿）。データは約 1.2 TB（NAVSIM 約 0.8 TB、nuPlan 検証・テスト分割 0.2 TB、特徴キャッシュ 0.2 TB）。GPU 2 枚で純学習約 9 日、評価・データ準備を含め約 2 週間半。

**備考**：本プロジェクトは torch 2.0.1 + CUDA 11.7 に固定されており、Blackwell 世代（RTX 5090 等）では環境の再構築が必要となるため、Ampere / Ada 世代を希望する。sudo 権限は不要。学習済みモデルとコードは研究室リポジトリ（NEGATE）で管理する。

### 中文

申请 2 张 24 GB 显存 GPU（RTX 4090 / A5000 / L4 级，Ampere 或 Ada 架构）、32 核 CPU、128 GB 内存、2 TB NVMe、1 Gbps 外网的 Linux 服务器，独占使用 2026 年 10 月至 11 月，可选延长至 2027 年 1 月。用途：在 NAVSIM / nuPlan 上为 TransFuser 与 PLUTO 附加保守负样本损失并训练评估，共约 63 次训练、15–18 GPU·天、数据约 1.2 TB。数据准备、负样本生成与评估工具已在笔记本上完成，仅训练无法在 8 GB 显存上进行。锁定的 torch 2.0.1 + CUDA 11.7 不支持 Blackwell 架构，故不申请 RTX 5090。

### English

Request: a Linux server with 2 × 24 GB GPUs (RTX 4090 / A5000 / L4 class, Ampere or Ada), 32 CPU cores, 128 GB RAM, 2 TB NVMe and ≥ 1 Gbps network, for exclusive use October–November 2026 (optional extension to January 2027). Purpose: attach the proposed conservative-negative loss to TransFuser and PLUTO on the NAVSIM / nuPlan benchmarks and evaluate with 3 seeds and stratified metrics — about 63 training runs, 15–18 GPU-days, ≈ 1.2 TB of data. Data preparation, negative generation and evaluation tooling are complete on a laptop; only training exceeds its 8 GB GPU. The pinned torch 2.0.1 + CUDA 11.7 stack does not support Blackwell GPUs, so RTX 5090-class cards are not requested.

## 7. 使用期间的时间表（2 卡）

| 周 | 内容 |
|---|---|
| 第 1 周 | 环境重建与验收（LTF 基线 PDMS 83.8 ± 1）、数据下载与特征缓存、λ 单种子筛选（4 次微调） |
| 第 2–3 周 | 从头训基线与最优 λ 各 3 种子；同时补种子与三维消融；PLUTO 数据准备与缓存 |
| 第 4–5 周 | PLUTO 微调 6 次与 Test14 闭环评估；全部 PDMS 与行为指标评估 |
| 第 6–8 周 | 失效分析、补实验、结果打包；释放 GPU |

## 8. 备选方案：云 GPU

等价工作量约 400 GPU·小时（24 GB 卡）+ 2 TB 块存储 + 约 900 GB 入网流量。按供应商当时报价核算；入网流量通常免费。注意 NAVSIM 数据需从 Hugging Face 与 AWS S3 拉取，云实例的出网限制需提前确认。
