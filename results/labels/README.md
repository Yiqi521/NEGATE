# 稳定标签目录

下游脚本与服务器交接一律引用本目录的固定文件名，不再追版本号。
`results/` 下带版本号的中间文件（`negatives_*_v*.parquet` 等）已被 git 忽略，只作本地迭代用。

| 文件 | 内容 | 重建命令 |
|---|---|---|
| `negatives_navtest.parquet` | navtest 交互子集的保守负样本标签 | `python scripts/build_negatives.py --split navtest --flags results/labels/scene_flags_navtest.parquet --subset interact --objective --out <路径>` |
| `negatives_navtrain.parquet` | navtrain 训练标签（同配置，4 分片后合并） | 同上，加 `--metric-cache $NAVSIM_EXP_ROOT/metric_cache_navtrain_interact --shard k/4`，再 `scripts/merge_shards.py` |
| `negatives_{navtest,navtrain}_control_{random,safety}.parquet` | 完整实验对照组 D 的随机 / 安全负样本标签 | 同上加 `--control random` 或 `--control safety` |
| `scene_flags_navtest.parquet` | 场景类别标志（规则检测器输出） | `python scripts/mine_scenes.py --split navtest --out <路径>` |
| `scene_flags_navtrain.parquet` | 同上（navtrain） | 同上，`--split navtrain` |

**训练与评估请使用 `is_negative_strict` 列**（客观判据 ∧ 回放参考检查的交集）。
各列含义与判据定义见 `docs/objective_criteria.md`，版本沿革见 `VERSION.md`。
