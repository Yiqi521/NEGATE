# 服务器交接清单（训练与全量评估）

本机（8 GB GPU / 单盘）只完成 Step 0–3 与损失模块；以下工作需在实验室服务器（≥ 24 GB 显存，≥ 1.2 TB 磁盘）执行。

## 1. 需要复制到服务器的内容
| 内容 | 路径（本机） | 大小 | 说明 |
|---|---|---|---|
| 项目代码 | `~/E2E_planner/`（不含 `results/*.png`） | < 50 MB | 含 `conservative_negatives/`、`configs/`、`scripts/`、`docs/` |
| 负样本标签（navtest） | `results/labels/negatives_navtest.parquet`（用 `is_negative_strict` 列） | ~1 MB | 评估用（v13，含 O7） |
| 负样本标签（navtrain） | `results/labels/negatives_navtrain.parquet`（用 `is_negative_strict` 列） | ~5 MB | 训练用（v5：20,799 strict / 5,819 场景） |
| 场景标志表 | `results/labels/scene_flags_{navtest,navtrain}.parquet` | ~8 MB | 分层评估 |
| metric cache | `~/navsim_workspace/exp/metric_cache`（navtest 3.1 GB）、`metric_cache_navtrain_interact`（生成中） | 3–6 GB | 可在服务器重新生成（navtest 约 25 min / 22 线程） |
| 环境定义 | `~/navsim_workspace/env.sh`、navsim v1.1 commit 3e8291b | — | 见 `docs/env.md` |

## 2. 服务器上需要下载的数据
- navtrain 传感器 blob：445 GB（`download/download_navtrain.sh`，含历史帧）或 300 GB（仅当前帧）
- navtest 传感器 blob：223 GB（`download/download_test.sh`；本机只有相机部分）
- 地图 + 元数据：同本机（约 20 GB）

## 3. 服务器上的执行顺序
```bash
# 0) 环境（与本机一致）
git clone -b v1.1 https://github.com/autonomousvision/navsim.git && cd navsim && git checkout 3e8291bfa89ff247231e0227778840cd0a036896
conda env create --name navsim -f environment.yml && conda activate navsim && pip install -e .
# 1) 基线复现（3 seeds），验收 PDMS 84.0 ± 1.0（TransFuser）/ 83.8（LTF）
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py agent=transfuser_agent experiment_name=baseline_tf_s0 train_test_split=navtrain
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py train_test_split=navtest agent=transfuser_agent agent.checkpoint_path=$CKPT experiment_name=baseline_tf_s0_eval
# 2) 分层评估（用项目自带 scene_filter）
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py --config-dir $PROJ/configs train_test_split=navtest_unprotected_turn agent=transfuser_agent agent.checkpoint_path=$CKPT experiment_name=baseline_tf_s0_turn
# 3) 负样本训练（Step 4）：接入 conservative_negatives/losses/separation_loss.py，见第 4 节
```

## 4. 训练接入设计（待实现，服务器上）
- **数据侧**：`NegativeBank`（token → (K,8,3) 负样本 + mask），从 `results/labels/negatives_navtrain.parquet` 读取（`label_col="is_negative_strict"`）；K = 4，不足补零。在 TransFuser 的 `TargetBuilder` 中按 token 附加 `negatives`、`neg_mask` 两个 target。
- **损失侧**：`compute_loss` 中在官方 `transfuser_loss` 之外加 `λ_neg(step) · SeparationLoss(pred_traj, gt_traj, negatives, neg_mask)`，`λ_neg ∈ {0.05, 0.1, 0.2, 0.3}`，前 20% step 线性 warm-up；日志记录 `neg_active_frac`（< 1e-3 持续则触发 R6）。
- **样本激活**：只有 token ∈ navtrain_interact 且有通过 C1–C4 的负样本时 `neg_mask` 为真；其余帧损失为 0。
- **打分型平行基线**：GTRS 仓库中的 Hydra-MDP（V2-99）依赖 NAVSIM v2，需先确认能否在 v1 数据上运行；若不能，用 v1 分支自行实现 4096 词表 + 子分数头（工作量约 1 周）。

## 5. 验收（对应研究计划 §6.4 / §7.4）
- 至少一组 λ_neg 在 `navtest_interact` 子集上 EP 提升，且 NC / DAC / TTC 的 bootstrap 95% CI 下界 ≥ 基线均值。
- 3 种子；结果分层报告：全量 / turn / merging / crossing / `expert_conservative` 帧。

## 6. 实验追踪（Step 4 训练时使用）

`conservative_negatives/tracking.py` 零依赖，直接可用；若服务器已装 W&B 或 MLflow，传参即自动镜像，调用代码不变。

```python
from conservative_negatives.tracking import Run

with Run(f"tf_lambda{lam}_seed{seed}",
         config={"lambda_neg": lam, "seed": seed, "loss": "hinge", "label_col": "is_negative_strict"},
         tags=["step4", "transfuser"],
         wandb_project="negate") as run:        # wandb_project / mlflow_uri 可省略
    for step, batch in enumerate(loader):
        ...
        run.log({"loss": loss.item(), "loss_neg": l_neg.item(),
                 "neg_active_frac": stats["neg_active_frac"]}, step=step)
    run.finish({"pdms": pdms, "ep": ep, "unnecessary_stop_rate": usr})
```

每次运行落盘到 `results/runs/<run_id>/`（meta.json 含 git commit 与是否 dirty、conda 环境、GPU 型号、完整配置快照；metrics.jsonl 为逐步指标；summary.json 为最终指标），并在 `results/runs/index.csv` 追加一行便于横向比较。TensorBoard 曲线在 `results/runs/<run_id>/tb`。

**风险 R6 的自动检查**：训练日志中 `neg_active_frac` 若在 warm-up 结束后持续 < 1e-3，说明分离损失零激活，需改用提议打分型基线。

## 7. 评估流程（服务器上按序执行）

```bash
# 1) PDMS（官方）
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py --config-dir $PROJ/configs \
  train_test_split=navtest agent=<agent> agent.checkpoint_path=$CKPT worker.max_workers=<N> \
  experiment_name=<exp>
# 2) 行为指标（三个自定义指标 + 人类参照组）
python eval/behavior_metrics.py --negatives results/labels/negatives_navtest.parquet \
  --agent-config configs/agent/<agent>.yaml --checkpoint $CKPT --out results/eval/behavior_<exp>.parquet --track
# 3) 合并报告：分层 PDMS + bootstrap CI + 配对差异 + 拒绝准则 + 行为指标
python eval/stratified_pdms.py --runs results/eval/<exp>_seed{0,1,2}.csv --baseline <baseline csv...> \
  --flags results/labels/scene_flags_navtest.parquet --negatives results/labels/negatives_navtest.parquet \
  --behavior results/eval/behavior_<exp>_seed{0,1,2}.parquet \
  --behavior-reference results/eval/behavior_human_navtest.parquet --out results/eval/<exp>_stratified.csv
```
