# FlyGCL 方法扩展实验总结

## 总览

本仓库基于已有的 [FlyGCL](https://github.com/AnAppleCore/FlyGCL) 代码，尝试两条新方法线：

1. **HFPool**：使用 Hopfield pooling 生成 prompt，并通过模块消融分析有效部件。
2. **Generator**：使用轻量生成器产生 prompt，以旧类统计量和 hard teacher 进行函数蒸馏。

实验以三 seed 汇总为主，重点记录机制对照与消融结果。

## 1 HFPool

### Router 输入修复

早期实现让 analytic router 使用已经被 prompt 改变的表征，导致 router 的统计空间随方法更新。随后将 router 的统计与推理输入改为 clean frozen-backbone CLS feature，并分别评估 task-wise decoder 与 shared online FC。

修复后，实验路径拆分为 ViT + FC、shared prompt、HFPool、per-head gate 和 EMA decoder 等独立机制。

### 模块消融

HFPool 主要比较了以下可训练部件：

- pooling queries；
- Q / K projection；
- output projection (W_o)；
- (W_o) 与 (W_k)、pooling query 的组合。

仅训练 pooling queries、Q projection 或 K projection 时，结果基本重合。训练 (W_o) 后，CIFAR-100 / CUB-200 的 (A_{auc}) 相对 Q projection 分别提高约 **6.76pt / 1.39pt**。继续加入 (W_k) 或 pooling query，变化小于 seed 波动。当前实验中，output projection 是 HFPool 的主要有效部件。

最高 HFPool 配置为：仅训练 (W_o)，5 个 pooling blocks、1 个 pooling head、prompt length 10、浅层插入。

### Per-head gate

prompt 可能通过调制 ViT attention 发挥部分 gating 作用。为此增加了 per-head gate：在 ViT blocks 0–4 上学习 head-wise gain，并与 HFPool 和 ViT + FC 对照。

gate 在 CUB-200 上接近 HFPool，在 CIFAR-100 上低于 HFPool。结果见文末汇总表。

### 其他消融

- 1 / 2 / 3 epochs；
- pooling blocks、pooling heads、prompt length；
- prompt 起始层与 shallow / deep placement；
- local token input 与 cached clean full-ViT input；
- (W_o)、(W_k)、pooling query 的联合训练与 gradient clipping；
- shared prompt、EMA decoder 与 ViT + FC；
- 不同预训练 checkpoint。

相关实现：

- [HFPool model](models/hfpool.py)
- [HFPool trainer](methods/hfpool.py)
- [Per-head gate](models/gate.py)
- [Method split tests](tests/test_method_split.py)

## 2 Generator

### Loss 尺度处理

Generator 同时优化分类 CE 与 teacher distillation loss，两者数值尺度相差较大。实验中依次加入：

- fixed weight 与 distill-off；
- one-sided GradNorm-lite，以 generator 最后一层的梯度范数比例调节蒸馏权重；
- teacher snapshot 后的短暂 CE-only delay；
- zero / non-finite gradient、权重边界与实际梯度比例统计。

修复后训练目标与权重保持稳定，distillation active updates 约占全部 optimizer updates 的 79.3%（CIFAR-100）和 76.1%（CUB-200）。

### 消融类别

- **统计量**：diagonal variance、diagonal + low-rank covariance；
- **旧类 anchor / 目标**：统计分布采样、closed-form MSE expectation；
- **蒸馏距离**：raw MSE、cosine；
- **低秩设置**：rank-16 covariance、top-k 子空间；
- **权重策略**：fixed、distill-off、GradNorm-lite；
- **时间设置**：hard teacher snapshot、100 stream samples delay。

代表性组合包括：

- sampled anchors + diagonal statistics + raw MSE；
- sampled anchors + diagonal statistics + raw cosine；
- closed-form expectation + low-rank statistics + MSE；
- sampled anchors + low-rank statistics + top-k cosine。

CIFAR-100 上最高的 MLP 配置为 distill-off；CUB-200 上最高的是 sampled anchors + diagonal statistics + raw MSE。其余 replay / distillation 组合没有拉开超过 seed 波动的差距。cosine 分支更容易出现 zero distillation gradient。

相关实现：

- [MLP generator](models/mlp_generator.py)
- [Generator trainer](methods/mlp_generator.py)
- [GradNorm and delay gate](utils/distill_gradnorm.py)
- [GradNorm tests](tests/test_distill_gradnorm.py)
- [Generator replay design note](docs/generator_replay_cil_research.md)

## 3 结果汇总

以下按 (A_{auc}) 选择 HFPool、gate 和 MLP 的最高配置，并使用同为 3 epochs 的 ViT + FC 作为 baseline。数值为三 seed 的 mean ± sample standard deviation。

### CIFAR-100

| 方法 | 配置 | (A_{auc}) |
| --- | --- | ---: |
| ViT + FC baseline | frozen ViT + shared online FC + batch-masked CE | **0.834933 ± 0.012228** |
| HFPool | 仅训练 (W_o)；5 blocks；1 head；L=10；early | 0.830676 ± 0.011843 |
| Per-head gate | ViT blocks 0–4；head-wise gain | 0.797749 ± 0.015777 |
| MLP Generator | distill-off | 0.776301 ± 0.016731 |

### CUB-200

| 方法 | 配置 | (A_{auc}) |
| --- | --- | ---: |
| ViT + FC baseline | frozen ViT + shared online FC + batch-masked CE | **0.707734 ± 0.016038** |
| HFPool | 仅训练 (W_o)；5 blocks；1 head；L=10；early | 0.701196 ± 0.018786 |
| Per-head gate | ViT blocks 0–4；head-wise gain | 0.694427 ± 0.020673 |
| MLP Generator | GradNorm；sampled anchors；diagonal statistics；raw MSE | 0.684627 ± 0.023622 |

## 4 实验记录

[run_v2.sh](run_v2.sh) 用于多 GPU 批量实验，并保存 Git commit、展开后的命令、runner 快照、逐 job 日志和退出状态。
