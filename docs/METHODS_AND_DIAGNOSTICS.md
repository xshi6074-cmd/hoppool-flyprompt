# Prompt 方法与诊断入口

所有方法通过 `main.py --method ...` 选择。模型实现位于 `models/`，在线更新与合并时点位于 `methods/`；统计、快照和离线分析的计算位于 `utils/`，绘图及单独重算的命令位于 `scripts/`。

## 方法

| 方法 | 可训练适应模块 | 默认输出更新 |
| --- | --- | --- |
| `hfpool` | 由 `--hopfield_trainable` 选择 HFPool 部件 | 原有默认：Q 投影，完整参数更新 |
| `hfpool_lora` | HFPool 输出矩阵的低秩 A/B | 每个完整 stream mini-batch 更新结束后合并并重置 |
| `spatial_gelu` | CLS + 3×3 patch 平均池化 → GELU → 输出映射 | 完整输出矩阵；可选同一套 LoRA 更新方式 |

LoRA 在 prompt 生成器的输出投影上更新，骨干 ViT 保持冻结。`spatial_gelu` 仅压缩生成侧分支，主干继续保留原有图像 tokens。两种新方法均自动选择仅训练输出映射，沿用 `hfpool` 的插入位置、prompt 长度和 batch/mask 协议。

下面的命令在仓库根目录运行；数据目录布局见 [框架说明](FRAMEWORK_GUIDE.md)。训练需要可用的 CUDA 环境和本地预训练权重。

```bash
python main.py --method hfpool_lora \
  --dataset cifar100 --data_dir /path/to/CIFAR \
  --backbone vit_base_patch16_224 --backbone_path /path/to/ViT-B_16.npz \
  --gpu 0 --seeds 1 --n_tasks 5 --num_epochs 3 \
  --opt_name adam --lr 0.005 --batchsize 64 --online_iter 3 \
  --num_pooling_blocks 3 --prompt_length 10 \
  --hopfield_lora_rank 8 --hopfield_lora_alpha 8 --hopfield_lora_lr 0.005
```

替换为 `--method spatial_gelu` 即使用完整输出映射；附加 `--hopfield_o_mode lora_merge_batch` 可使用低秩输出更新。

### 更新周期与插层

- `--hopfield_o_mode lora`：持续训练同一对 A/B。
- `--hopfield_o_mode lora_merge_1024`：按全局 stream visits 计数，满 1024 后在完整 batch 结束时合并，跨 epoch 与 benchmark task 累计。
- `--hopfield_o_mode lora_merge_batch`：完成当前 batch 的全部 `online_iter` 次更新后合并。
- 合并后清理 A/B 的 Adam 状态，保留分类头的优化器状态；底座与 A/B 保持 FP32。
- `--hopfield_start_layer`：从指定的零基 block 生成/插入。
- 单个生成器复用到指定层：`--num_pooling_blocks 1 --hopfield_start_layer 0 --hopfield_prompt_blocks 2 3 4`。生成源固定在 block 0 之前，生成一次后复用同一 prompt。

原实验的 `hfpool --hopfield_trainable o --hopfield_o_mode ...` 与 `hfpool --hopfield_pooling_mode spatial_gelu --hopfield_trainable o` 入口继续保留。

## 分类头统计回放

冻结 ViT 系列方法可附加 `--classifier_replay`，包括 `baseline`、`shared_prompt`、`gate` 以及上述三个生成方法。类别统计来自分类头实际接收的最终 CLS，每个 stream batch 只记录一次；回放损失通过 detached 伪特征约束分类头。

```bash
# 附加到上面的训练命令
--classifier_replay --step_num 10 \
--statistic_type diagonal --statistics_space raw \
--distill_mode sample --distill_metric mse \
--replay_class_budget 8 --anchors_per_class 8 \
--classifier_replay_weight_mode fixed --distill_weight 1
```

`utils/feature_statistics.py` 提供 prompt-generator replay 与 classifier replay 共用的在线均值、方差及低秩协方差统计。`models/classifier_replay.py` 管理历史分类头和输出约束；训练器管理统计更新、延迟窗口和梯度比例。`--classifier_replay_weight_mode gradnorm` 或 `learned_gradnorm` 使用单 GPU 的梯度比例控制。

## 诊断：训练记录与离线计算

| 选项 / 产物 | 计算位置 | 记录口径 |
| --- | --- | --- |
| `--prompt_probe_every_batches 10` | `utils/prompt_probe.py` | 固定测试图像，只用于观察；记录同图跨时间与同时间跨输入的 prompt 相似度 |
| `--cls_probe` | `utils/cls_probe.py` | 首检查点实际已见训练图像组成固定平衡队列；追踪最终 CLS 的原始/归一化几何 |
| `--offline_probe` | `utils/offline_probe.py` | 每个检查点只使用截至当时实际已见的训练索引，冻结表示后重拟合所有已见类别的线性头 |
| `seed_*_class_history.npz` | 训练器记录，`scripts/plot_forgetting.py` 分析 | 保存 task-end 逐类准确率，与现有 `F_last` 的有效类别规则一致 |
| `--ema_diagnostics` | `utils/ema_diagnostics.py` | 单独记录在线头、EMA 头和集合输出；默认主指标仍来自在线头 |

CLS 与线性 probe 支持冻结 ViT 系列方法，每个 job 使用单 GPU。离线 probe 在数据流完成后运行；`--offline_probe_cache_features` 同时保存 FP32 特征缓存。训练器在每个评估时点保存适应模块、分类头、实际已见索引及配置；离线计算完成后恢复模型状态、训练模式和随机数状态。

```bash
# 附加到训练命令，按需要启用
--paired_training_rng \
--prompt_probe_every_batches 10 --prompt_probe_examples 8 \
--cls_probe --cls_probe_per_class 16 \
--offline_probe --offline_probe_cache_features
```

`--paired_training_rng` 在模型初始化后重置训练 RNG，用于配对比较不同方法。缓存重拟合与绘图可在 CPU 上独立执行：

```bash
python scripts/offline_linear_probe.py results/logs/DATASET/NOTE/seed_1_lp \
  --from-cache --device cpu --output-dir /path/to/refit

python scripts/plot_cls_probe.py results/logs --output-dir /path/to/cls_plots
python scripts/plot_prompt_similarity.py results/logs --output-dir /path/to/prompt_plots
python scripts/plot_forgetting.py results/logs --output-dir /path/to/class_plots
```

CLS 默认图沿用真实 stream visits；不同 seed 的访问时点不一致时分别画线。`--x-axis checkpoint` 显式按检查点序号计算跨 seed 均值及样本标准差。`--require-pairs` 要求每个数据集/骨干/seed 都有 shared-prompt 参考，并检查队列、标签、时间和 clean 特征完全匹配。

Prompt 图默认按方法独立绘制；`--reference-note NAME` 指定同数据集目录下的 shared-prompt 实验 note 后，会校验固定图像并在相同 visits 和插入层上配对。

## 验证

在已安装 `requirements.txt` 依赖的环境运行 CPU 回归：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*.py'
```

测试使用随机初始化的小型 ViT 和合成数据，覆盖方法注册、梯度、输出合并与状态清理、已见样本隔离、诊断状态恢复和绘图时间轴。
