# FlyGCL 方法扩展实验总结

## 研究范围

本仓库基于已有的 [FlyGCL](https://github.com/AnAppleCore/FlyGCL) 代码，尝试在其持续学习实验框架中验证新的机制。本文只总结新增探索及其证据，不重复介绍原方法、模型架构或任务设定，也不把目前结果包装成已经成立的新方法。

实验主要沿两条线展开：

1. **HFPool**：使用 Hopfield pooling 生成 prompt，并通过机制拆分判断性能究竟来自哪里。
2. **Generator**：以轻量生成器产生 prompt，并尝试用旧类统计量与 hard teacher 保持生成函数。

除特别说明外，本文只采用修复后的有效实验路径与多 seed 汇总。routing 修复前的结果仅用于追踪问题来源，不用于当前方法比较。

---

## 1 HFPool

### 1.1 Router 输入审计与修复

早期实现让 analytic router 使用已经被 prompt 改变的表征。这样做存在两个问题：

- router 的输入坐标会随待评估方法一同变化，旧统计量与当前表征可能不再一致；
- prompt、router 和 task-wise decoder 的贡献被混在一起，无法判断性能变化究竟来自哪个部分。

随后将 router 的统计与推理输入改为 **clean frozen-backbone CLS feature**，并把 task-wise EMA decoder bank 与 shared online FC 分开比较。这个修改首先是 correctness fix，不把它本身解释为性能提升。

修复后，实验被拆成 frozen ViT + FC、shared prompt、HFPool、per-head gate、EMA decoder 等独立机制，避免继续用一个带多组开关的方法名代表不同实验路径。

### 1.2 HFPool 模块级消融

核心问题是：HFPool 中究竟哪个可训练部件产生了有效变化。主要比较了 pooling queries、Q/K projection、output projection (W_o) 及其组合。

| 可训练部件 | CIFAR-100 (A_{auc}) | CUB-200 (A_{auc}) | 判断 |
| --- | ---: | ---: | --- |
| Q projection | 0.7564 ± 0.0180 | 0.6830 ± 0.0172 | 与其他非输出部件基本重合 |
| Pooling queries | 0.7566 ± 0.0154 | 0.6831 ± 0.0167 | 没有形成独立增益 |
| K projection | 0.7571 ± 0.0179 | 0.6829 ± 0.0165 | 没有形成独立增益 |
| Output projection (W_o) | 0.8240 ± 0.0155 | 0.6969 ± 0.0222 | 唯一强而一致的正向信号 |
| (W_o + W_k) | 0.8296 ± 0.0146 | 0.6977 ± 0.0207 | 相对仅训练 (W_o) 的差异小于 seed 波动 |

相对 Q projection，仅训练 (W_o) 时 CIFAR-100 / CUB-200 的 (A_{auc}) 分别提高约 **6.76pt / 1.39pt**。加入 (W_k)、pooling query 或同时训练三者，没有得到超过 seed 波动的稳定额外收益。

因此，当前证据支持的不是“Hopfield association 整体有效”，而是：**输出映射层在已测试配置中占据主导，早期只训练 association 内部部件低估了 HFPool 的可适配能力。**

### 1.3 Prompt 可能具有 gating 作用

模块消融引出了一个替代解释：prompt 的实际作用可能部分接近对 ViT attention 输出的调制，而不完全依赖可解释的“prompt 记忆”或检索机制。

为此增加了 **ViT per-head gate** 对照：在指定 blocks 上学习 (1+	anh(alpha)) 的 head-wise gain，不插入 prompt token。

| 方法 | CIFAR-100 (A_{auc}) | CUB-200 (A_{auc}) |
| --- | ---: | ---: |
| Per-head gate | 0.7977 ± 0.0158 | 0.6944 ± 0.0207 |
| HFPool，仅训练 (W_o) | 0.8240 ± 0.0155 | 0.6969 ± 0.0222 |

在 CUB-200 上，gate 与 HFPool-(W_o) 的结果接近；在 CIFAR-100 上仍有明显差距。这个实验说明简单 gating 可以解释一部分效果，但不足以证明二者等价，也不能据此把 prompt 的全部作用归结为 gating。

### 1.4 其他消融类别

此外还覆盖了以下类别：

- 训练充分性：1 / 2 / 3 epochs；
- pooling blocks 数量、pooling heads 数量、prompt length；
- prompt 插入的起始层与 shallow / deep placement；
- local token input 与 cached clean full-ViT input；
- (W_o)、(W_k)、pooling query 的联合训练与 gradient clipping；
- frozen ViT + FC、shared prompt、HFPool 与 EMA/task-wise decoder 的机制拆分；
- 不同预训练 checkpoint 下的 baseline 与 HFPool 对照。

多数结构旋钮的差异没有稳定超过 seed 波动；继续扩大超参数网格的优先级低于解释 (W_o) 的作用，以及验证 gating 假设。

相关实现：

- [HFPool model](models/hfpool.py)
- [HFPool trainer](methods/hfpool.py)
- [Per-head gate](models/gate.py)
- [Method split tests](tests/test_method_split.py)

---

## 2 Generator

### 2.1 早期 loss 尺度问题

Generator 路线尝试用旧类特征统计量产生 replay anchors，并让当前 generator 对齐上一阶段 hard teacher 的 prompt 输出。

早期实验首先暴露的是一个工程错误：分类 CE 与 distillation loss 相差多个数量级，而我在使用 AI 辅助实现自适应权重时，没有把正值约束、尺度目标和异常处理说明清楚。生成代码采用了不适合当前目标的 weighting 形式，出现权重失控以及负值或非有限训练目标。

这不是方法层面的负结果，而是一次实现与沟通失误。相关旧结果不用于判断 Generator 是否有效。

修复后的处理包括：

- 保留 fixed weight 与严格的 distill-off control；
- 改用 one-sided GradNorm-lite，以 generator 最后一层的梯度范数比例作为控制目标；
- 在 teacher snapshot 后设置短暂 CE-only delay；
- 显式处理 zero / non-finite gradient、权重上下界与日志统计；
- 为 delay gate、GradNorm 更新、异常输入和参数契约增加独立测试。

### 2.2 修复后的消融类别

loss 稳定后，主要比较了以下维度：

- **统计量形式**：每类 diagonal variance 与 diagonal + low-rank covariance；
- **旧类 anchor / 目标计算**：从统计分布采样，以及可闭式计算的 MSE 期望；
- **蒸馏距离**：raw MSE 与 cosine；
- **低秩设置**：rank-16 covariance 与 top-k 子空间；
- **权重策略**：fixed、distill-off、GradNorm-lite；
- **时间设置**：hard teacher snapshot 与 snapshot 后 100 个 stream samples 的 delay；
- **机制监控**：raw / weighted distillation loss、实际梯度比例、有效 GradNorm update、边界命中和 zero-gradient 原因。

代表性组合包括：

- sampled anchors + diagonal statistics + raw MSE；
- sampled anchors + diagonal statistics + raw cosine；
- closed-form expectation + low-rank statistics + MSE；
- sampled anchors + low-rank statistics + top-k cosine。

### 2.3 当前结论

修复后可以确认：

1. **数值稳定性问题已经解决。** 没有继续出现负值或非有限训练目标，权重也没有持续撞上下界。
2. **delay 没有让 teacher 分支持续关闭。** distillation active updates 约占全部 optimizer updates 的 79.3%（CIFAR-100）和 76.1%（CUB-200）。
3. **当前 teacher distillation 没有显示可辨识收益。** CIFAR-100 四组均值均低于现有 distill-off 参照，12 个同 seed 差值中有 11 个为负；CUB-200 四组 (A_{auc}) 落在 0.6822–0.6846，配置间差异远小于 seed 波动。
4. **没有证据支持某一种 replay 形式。** sampled / closed-form、diagonal / low-rank、MSE / cosine 均未稳定分离。
5. **cosine 是最可疑的分支。** 在 CIFAR-100 上其 loss 很快变小，并频繁出现 zero distillation gradient；low-rank / top-k cosine 的有效 GradNorm update 仅约占 active updates 的 27.3%。

第 3 点的 distill-off 参照与主批次跨 commit，因此当前严谨表述是“**稳定但未获益**”，而不是已经证明 distillation 必然有害。最小下一证据应是同一 commit、同一设置下补齐 distill-off，而不是继续扩展 replay 或 loss 网格。

相关实现：

- [MLP generator](models/mlp_generator.py)
- [Generator trainer](methods/mlp_generator.py)
- [GradNorm and delay gate](utils/distill_gradnorm.py)
- [GradNorm tests](tests/test_distill_gradnorm.py)
- [Generator replay design note](docs/generator_replay_cil_research.md)

---

## 3 可复现性说明

当前仓库将各机制拆成独立 method，并使用 [run_v2.sh](run_v2.sh) 进行多 GPU 批量实验。每批运行保存：

- 当前 Git commit；
- 展开后的完整命令；
- runner 与配置快照；
- 每个 task × seed 的独立日志；
- 启停事件、GPU 分配和退出状态。

本总结的作用是给出研究判断与代码入口。完整实验记录保留了旧路径、修复过程、逐 seed 数值及未成立的假设；对外引用时应以本文列出的有效路径与限定结论为准。
