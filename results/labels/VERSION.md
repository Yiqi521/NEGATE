# 标签版本沿革

本目录的文件名保持稳定，内容随判据迭代更新；每次更新在此记录。判据定义见 `docs/objective_criteria.md`。

## 当前版本：v13 / navtrain v5（2026-09-15）

判据：`is_negative` = O1 回放安全 ∧ O2 冲突点 PET ≥ 1.5 s ∧ O3 相对 RSS ∧ O4 EP 比 ≤ 0.8 ∧ O5 EP 比 ≥ 0.3 ∧ O6 候选级间隙接受 ∧ ¬O7 遮挡可达性。
**训练与评估用 `is_negative_strict`**（再与回放参考检查取交集）。

| 文件 | 场景 | strict 负样本 | 覆盖场景 | 来源 |
|---|---|---|---|---|
| `negatives_navtest.parquet` | 1,045 | 2,783 | 742 | navtest v13 |
| `negatives_navtrain.parquet` | 9,505 | 20,799 | 5,819 | navtrain v5（4 分片合并） |
| `negatives_navtest_control_random.parquet` | 1,045 | 3,258 | 1,003 | 对照组：任意纵向扰动（速度 ×U(0.3,1.5) + 随机起步延迟），**不经 O1–O7** |
| `negatives_navtest_control_safety.parquet` | 1,045 | 274 | 116 | 对照组：BeyondDrive 式安全负样本（复用 G1–G4/G3b 候选，只留回放 NC 或 TTC 失败者） |
| `negatives_navtrain_control_random.parquet` | 9,505 | 25,741 | 8,513 | 同上（navtrain） |
| `negatives_navtrain_control_safety.parquet` | 9,505 | 3,087 | 1,170 | 同上（navtrain） |
| `scene_flags_navtest.parquet` | 12,146 | — | — | 规则检测器 v4 |
| `scene_flags_navtrain.parquet` | 103,288 | — | — | 规则检测器 v4 |

## 迭代要点（详见 `docs/objective_criteria.md` §6 与 §9）

| 版本 | 关键变化 | navtest strict 负样本 |
|---|---|---|
| v5 | 纯回放判据（C1–C4 + 恒速外推间隙代理） | 3,689 |
| v9 | 改用 HCM 间隙接受 + PET + 相对 RSS；t_c 按运动类型 | 2,827 |
| v11 | 冲突检测改为"从路径外进入"几何规则（排除同向跟驰与相邻车道平行车） | 3,442 |
| v12 | O6 从场景级改为候选级（候选可在同一间隙内安全通过时，其额外延迟仍算无理由） | 3,463 |
| **v13** | **新增 O7 遮挡可达性判据（幻影车）** | **2,783** |

navtrain 对应版本：v1（纯回放）30,251 → v4（客观判据）28,666 → **v5（加 O7）20,799**，覆盖 5,819 / 9,505 场景。
O7 在 navtrain 上判为"谨慎有理由"的比例为 29.2%（直行穿越 55.3%、无保护转向 21.5%、汇入 9.9%）。

对照标签均由 `scripts/build_negatives.py ... --objective --control random|safety` 生成，`is_negative_strict` 列即标签；与正式标签在 (token, 算子, 参数) 上零重叠（已核验）。
