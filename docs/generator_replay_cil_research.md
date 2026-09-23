# Generator replay 方案：CIL 调研、问题拆解与首轮实验草案

> 状态：讨论稿，不是实现规格。  
> 日期：2026-08-09  
> 当前分支：`generator`  
> 本轮范围：只读审计、成熟方法调研、实验设计；尚未修改模型或训练代码。

## 1. 先给结论

用户草图里的核心方向值得继续，但不建议原样落地成“保存二阶统计量 + 保存上一阶段 HFP + 新类 CE + attention 蒸馏”。目前至少有三类问题被混在了一起：

1. **生成器遗忘**：同一个旧类输入，更新后的生成器是否仍产生功能相近的 prompt；
2. **分类器偏置/遗忘**：旧类没有真实样本参与当前优化，FC 是否只向新类倾斜；
3. **表征坐标变化**：统计量保存在哪个特征空间，该空间随后是否仍保持不变。

首轮主方案应改写为一个可逐项消融的 **双空间 functional replay**：

- 在生成器之前的稳定中间特征空间保存每类统计量，用来重放生成器输入；
- 在生成器之后的最终 CLS 空间另存每类统计量，用来约束/校准分类器；
- 当前真实样本仍通过分类头给生成器提供任务梯度；
- 对旧类，优先蒸馏“生成出的 prompt”或下游功能，而不是默认蒸馏内部 attention map；
- 先在具有明确边界的标准 Split-CIL 场景验证机制，再移植到 FlyGCL 的 Si-Blurry 流。

这条路线和 APG 的 knowledge pool 非常接近。这不是坏事：它给了我们一个成熟参照和消融模板；但也意味着，如果不先解决当前 HFP 的输入接口，我们很可能只是做出一个无法真正重放旧类输入分布的半成品。

## 2. 当前代码真正实现了什么

### 2.1 现有前向路径

当前 `models/flyprompt.py` 中的 HFP 路径可以概括为：

```text
imagescm-history-item:d%3A%5Cworksapce%5CFlyprompt%5CFlyGCL?%7B%22repositoryId%22%3A%22scm0%22%2C%22historyItemId%22%3A%22435d33a3827146342f57af6ed23c8a5b76d4912c%22%2C%22historyItemParentId%22%3A%22d201753b658dad8b8ef55bdace789bdb498c697d%22%2C%22historyItemDisplayId%22%3A%22435d33a%22%7D
  -> frozen patch embedding / frozen ViT blocks
  -> full token sequence x_l: [B, 197, D]
  -> HopfieldPooling_l(x_l)
  -> generated prompt p_l: [B, L, D]
  -> insert p_l after CLS
  -> frozen ViT block_l
  -> remove prompt tokens
  -> repeat for later selected blocks
  -> final prompted CLS z_theta(x): [B, D]
  -> trainable FC
```

代码证据：

- `HopfieldPoolingPrompts.forward()` 在每个选定 block 上把完整 token 序列送入 Hopfield，再插入生成的 prompt：`models/flyprompt.py:109-159`；
- backbone 全冻结，`backbone.fc` 解冻：`models/flyprompt.py:349-364`；
- 当前 HFP 先冻结全部参数，再只让 `association_core.in_proj_*` 的 Q-like 前 1/3 切片获得梯度：`models/flyprompt.py:74-107`；
- `_extract_cls_features()` 当前确实走 HFP/Prompt 路径：`models/flyprompt.py:427-439`。

这里的“只训练 Q”指 **Hopfield association 内部的投影切片**，不是 ViT backbone block 的 attention `qkv`。后面讨论“开放 W_qkv”时必须把这两个层次分开。

### 2.2 哪个特征会漂移，哪个不会

令冻结 backbone 为 `F_0`，生成器为 `G_theta`。

- 若 `p_0` 指的是 prompt 介入前、由冻结 `F_0` 提取的干净中间 CLS，那么在相同输入和确定性预处理下，它**不会因为训练而漂移**。
- 若它指生成的 prompt `G_theta(h)`、prompt 介入后的最终 CLS `z_theta(x)`，或第二个及之后 HFP block 的输入 token，那么它会随着 `theta` 更新而漂移。

尤其需要注意：当前是逐层 HFP。第一个 HFP block 的输入可以位于纯冻结 backbone 坐标中；但它生成的 prompt 已改变该 block 输出，因此后续 HFP block 的输入已经依赖旧的 prompt 生成器。把所有层都叫作“pretrained feature”是不准确的。

所以用户提出的表征偏移担忧是对的，但必须精确到保存位置：

| 保存对象 | 训练后是否漂移 | 能否直接用于当前 HFP 重放 |
|---|---:|---:|
| 第一个 prompt block 之前的 clean CLS | 否 | 否，当前 HFP 需要完整 tokens |
| 第一个 prompt block 之前的 full tokens | 否 | 是，但统计/存储代价很高 |
| 后续 HFP block 的输入 tokens | 是 | 形式上能，语义会陈旧 |
| generated prompt | 是 | 可作为蒸馏目标，不能作为输入分布 |
| final prompted CLS | 是 | 可用于分类器重放，不能驱动生成器 |

### 2.3 当前训练协议还会造成两个混淆

1. 默认 `--no_batchmask=False` 时，CE 只在当前 batch 出现的类别之间竞争：`methods/flyprompt.py:76-90`。若 batch 恰好只有一个类，softmax 只剩一个有效类，分类梯度可能接近零。generator 方案若继续沿用这一点，分类器/生成器的结论会被 mask 机制污染。
2. FlyGCL 的 prompt 内部 step 按累计样本量推进，`online_after_task()` 不推进生成器 step；Si-Blurry sampler 又会把 blurry 样本跨 session 混合。因此“每过完一个类就封存 teacher/statistics”在当前 benchmark 中没有自然可观测的事件。

此外，仓库里的 `methods/slca.py` 只是顺序 CE slow-learning 基线；它没有论文里的类均值、协方差、伪特征采样和 classifier alignment。不能直接复用它来声称“已实现 SLCA alignment”。

## 3. 对原始方案的几次必要质疑

### 3.1 “上一阶段 teacher + 当前新类数据蒸馏”没有旧类覆盖

如果 teacher 只在当前新类图像上被调用，那么它只约束新类流形附近：

```text
G_theta_t(h_new) ~= G_theta_{t-1}(h_new)
```

这不推出：

```text
G_theta_t(h_old) ~= G_theta_{t-1}(h_old)
```

没有旧图像、旧特征样本或能覆盖旧类的生成分布时，蒸馏 loss 可能很低，旧类功能仍然大幅漂移。保存旧类输入统计量的真正意义，正是给 teacher/student 提供旧类覆盖，而不是只用来训练 FC。

### 3.2 蒸馏 attention map 不等于保持功能

可以考虑三类目标：

1. **prompt 输出**：`d(G_theta(h_old), G_teacher(h_old))`；最直接约束生成器函数。
2. **下游 CLS/logit**：更接近最终功能，但会把生成器和分类器问题纠缠在一起。
3. **HFP association/attention map**：有机制解释，但不同内部注意力模式可能产生近似相同的 prompt；反过来，近似相同的 attention map 也未必保持分类 margin。

因此首轮默认应使用 prompt 输出蒸馏，并同时记录下游 CLS/margin。attention-map 蒸馏应作为一个明确的 ablation，而不是先验认定的主损失。ViT exemplar-free regularization 研究也显示，attention distillation 更偏 rigidity，而 contextual embedding 往往有更好的整体准确率，且对称约束会损害 plasticity。

### 3.3 “训练 attention 时先不用 classifier”目前缺少学习目标

这是当前构想里我最不同意的一点。分类头不一定是最终最优 decoder，但 current-class CE 是生成器学习任务有用表示的直接梯度来源。拿掉它以后，我们必须证明替代损失与分类功能对齐；仅让 prompt 彼此相似、匹配 teacher 或形成好看的簇，并不保证它改变 backbone 到正确决策方向。

APG 的消融提供了一个很强的警告：在其 ImageNet-Subset、non-pretrained 10-task 设置中，仅当前分类 loss 为 19.37；加入旧类分类器统计重放后升到 71.84，再加入 APG 的旧类约束才到 73.46。数值不能直接搬到 FlyGCL，但它说明 **classifier 不是可以后补的边角模块**。

更合理的安排是：

- 保留 `fc_online`，让真实新类样本持续给生成器提供 CE 梯度；
- 用旧类 final-feature statistics 同时平衡它；
- 每个可用边界再从相同统计量得到 `fc_aligned`；
- 同时评估 online 与 aligned 两个头，判断问题来自 representation 还是 decoder。

### 3.4 “analytic 最没道理”这个判断把两件事混在了一起

analytic head 本身并不天然不合理。真正不合理的是：**把旧坐标系里的统计量当成当前坐标系的数据使用，却不验证或补偿漂移**。同一批陈旧伪特征无论用于闭式 ridge、SGD linear FC 还是 cosine FC，都会受影响。

而且 “analytic” 至少包含不同家族：

- RanPAC 式 random projection + ridge closed form；
- SimpleCIL/NCM 的原型距离；
- FeCAM 的 shrinkage covariance + Mahalanobis；
- LDA/QDA 式生成分类器。

FeCAM 的核心实验甚至报告：直接用 covariance-aware Mahalanobis 比从高斯采样再训练线性分类器更好。因此首轮不该争论哪个头“理论上最对”，而应把多个低成本 decoder 放在**同一个冻结 checkpoint、同一统计库**上做诊断。

### 3.5 全协方差不是免费午餐

以 ViT-B 的 `D=768`、float32 为例：

- full covariance：`768 x 768 x 4 bytes ~= 2.25 MiB/class/bank`；
- input/output 两个 bank，100 类约 450 MiB，200 类约 900 MiB；
- 当类样本数 `n_c <= D` 时，样本协方差必然奇异；即使 `n_c > D`，也可能病态。

建议首轮用在线 Welford 的 `count + mean + diagonal variance`。第二阶段只比较 `diag` 与 `low-rank(r=16/32)+diag`，不要一开始就上 full covariance。rank 32 时约 99 KiB/class/bank，两个 bank、100 类约 19 MiB。

## 4. 与成熟方法的对应关系

| 方法 | 对当前设计最有用的部分 | 不应机械照搬的部分 |
|---|---|---|
| APG | 中间特征统计重放 generator；最终特征统计重放 classifier；prompt centroid；明确消融 | 它的 generator 输入是单个中间 feature，和当前 full-token、逐层 HFP 不同 |
| SLCA | online 学 representation，边界后用类高斯伪特征平衡 classifier；细粒度数据上 decoder gap 更明显 | 当前仓库 `slca` 不是论文完整实现；其任务边界假设也不等于 Si-Blurry |
| FeCAM | 协方差异质性、shrinkage、Mahalanobis；适合作为统计库质量的后处理诊断头 | 冻结表示下的效果不能直接证明动态 HFP 输出统计量长期有效 |
| LUCIR/cosine classifier | 减少新旧类 feature/weight norm 偏置，成本低 | cosine 不能修复旧类流形覆盖或 generator drift |
| SDC | 用当前数据观测到的旧/新 encoder 差异估计 prototype drift | 从新类邻域外推旧类漂移是强假设，而且它主要补 mean |
| Consistent MoE Prompt Generator | 直接把“旧输入上 generator 函数不变”写成目标；通过旧输入/门控子空间投影限制更新 | 正交投影保证依赖线性 MoE 结构，不能无证明地套到非线性 HFP |
| ViT attention regularization | attention、contextual embedding、非对称蒸馏可形成有信息的 ablation | 当前样本上的 attention 一致性仍不提供旧类覆盖 |

最接近用户草图的是 [APG 论文](https://arxiv.org/abs/2308.10445) 与其[官方代码](https://github.com/TOM-tym/APG)。相关补充参照包括 [SLCA](https://arxiv.org/abs/2303.05118)、[FeCAM](https://arxiv.org/abs/2309.14062)、[SDC](https://arxiv.org/abs/2004.00440)、[Consistent MoE Prompt Generator](https://ojs.aaai.org/index.php/AAAI/article/view/34108)、[ViT attention/functional regularization](https://arxiv.org/abs/2203.13167) 与 [LUCIR](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)。

## 5. 最大架构分叉：当前 HFP 如何接收统计重放

### 路线 A：保留当前 full-token HFP，只在当前图像上蒸馏

优点：改动最小，可以快速确认 teacher loss 是否可优化。  
缺点：没有旧类覆盖，不能验证真正的 anti-forgetting；最多是一个 negative/control ablation。

这个版本可以做，但不应被命名为完整 generator replay。

### 路线 B：把 generator 改为稳定 intermediate CLS-conditioned（推荐主线）

在某个冻结 block `k` 取 clean CLS：

```text
h = CLS(F_0^{<=k}(x))          # stable [B, D]
p = G_theta(h)                 # [B, L, D] 或 [B, K, L, D]
z = CLS(F_0^{>k}([tokens, p]))
```

这让旧类的 `h` 可以从每类高斯/低秩分布采样后直接送入 generator。首版应只做**一个插入位置**，等因果链清楚后再扩展成多层 prompt。若一开始就让 `G(h)` 同时生成多层 prompt，也应一次性从同一个稳定 `h` 生成，避免后层 generator 输入再次漂移。

代价是：这已经不是给现有 HFP 加 loss，而是改变 HFP 的 conditioning interface。它仍可使用 Hopfield/cross-attention 作为内部生成器，但必须明确研究对象从“token-set pooling”变为“stable-feature-conditioned prompt generator”。

### 路线 C：保留 full-token HFP，并保存 token anchors/分布

理论上最忠实于当前 HFP，但不适合作为首轮：

- 单层 `[197, 768]` token tensor 每样本约 0.58 MiB（float32）；
- 对展平的 151,296 维向量做 full covariance 完全不可行；
- 多层输入中，第一层之后的 tokens 又受旧 generator 影响；
- 最终容易变成 token sketch、anchor replay、transport correction 的方法拼接。

除非路线 B 明确失败且我们确认 full-token conditioning 是关键，否则不建议走这条路。

## 6. 推荐的双空间统计库与损失

### 6.1 每类保存什么

对每个已见类 `c`：

```text
K_in[c]  = {n_c, mu_in_c, var_in_c}        # stable intermediate CLS h
K_out[c] = {n_c, mu_out_c, var_out_c}      # prompted final CLS z
P[c]     = {prompt centroid or compact teacher target}
```

`K_in` 与 `K_out` 不能合并：前者用于提醒 generator 如何处理旧输入，后者用于平衡 classifier。`K_out` 是否仍可信，要靠 generator consistency 指标验证；如果旧类 prompt 功能漂移很大，它也会陈旧。

### 6.2 首轮目标函数

真实当前数据：

```text
L_new = CE(C_online(z_theta(x_new)), y_new; seen-class logits)
```

旧类 generator replay：

```text
h_tilde_c ~ q(K_in[c])
p_target  = stopgrad(G_teacher(h_tilde_c))
         or stored prompt centroid P[c]
L_gen_old = distance(G_theta(h_tilde_c), p_target)
```

旧类 classifier replay：

```text
z_tilde_c ~ q(K_out[c])
L_cls_old = CE(C_online(z_tilde_c), c)
```

总损失先保持克制：

```text
L = L_new + lambda_g * L_gen_old + lambda_c * L_cls_old
```

首轮不要同时加入 triplet、attention-map KD、feature KD、drift compensation、QKV 解冻。它们只能在对应失败被测量到之后单独进入。

### 6.3 teacher 目标也要消融

- **上一 snapshot teacher**：保留 input-conditioned 函数，比单一 centroid 丰富；但会递归继承旧 teacher 的误差。
- **每类 prompt centroid**：非递归、内存小；但可能把类内多模态压扁，且过度限制 plasticity。
- **teacher + centroid**：可能更稳，也更像方法拼接；不应作为首版默认。

因此应先比较 teacher 与 centroid，而不是默认二者都开。

## 7. Prompt “语义”应该怎样操作化

“prompt 有语义”不能只理解成 PCA 图上能分任务。建议分成三层：

### 7.1 几何表征

- class mean prompt 的 between-class / within-class scatter ratio；
- class probe、task probe、input-identity probe，报告 chance-normalized 结果；
- 每层 prompt 的 SVD、effective rank、collapse ratio；
- 同类与异类 prompt cosine 分布，而不只看全局平均。

少量 task 本身就会让 task-mean 矩阵低秩，所以“PCA 低秩”不能单独证明 generator 学到了共享语义。

### 7.2 因果功能

这是比几何更重要的一层：

- **prompt swap**：在相同图像上把自身 prompt 换成另一类/另一 task 的 prompt，观察正确类 logit margin 与 CLS 的变化；
- **prompt interpolation**：沿两个类 prompt 插值，观察决策是否平滑、有方向性；
- **prompt ablation**：移除某层 prompt，量化每层边际贡献；
- **Jacobian/sensitivity**：记录 `||d z / d p||` 或一阶近似，判断“prompt 变了很多但 backbone 几乎不理它”的情况。

若 prompt 几何分得开但 swap 不影响预测，它更可能只是旁路编码；若几何不漂亮但 swap 有稳定的类条件功能，也不能判定 generator 无效。

### 7.3 连续学习功能

- 固定旧类 probe 上的 prompt drift、prompted CLS drift、logit margin drift；
- 新类初学准确率/损失与旧类遗忘分开；
- online head、aligned head、cosine head、FeCAM head 在同一 checkpoint 上比较；
- real feature 与 Gaussian sample 的 two-sample discriminator AUC，检查统计重放是否真的覆盖原分布。

因此，“task-wise prompt 比 average prompt 提升不大”只能说明当前 task ID 不是一个强干预变量，不能单独推出 generator 无效。FlyGCL 的 task 本来就缺少明确语义，这个判据尤其弱。

## 8. 实验矩阵：先定位，再叠加

### 8.1 Phase 0：不改变训练行为的诊断

| ID | 训练配置 | 目的 |
|---|---|---|
| B0 | frozen ViT + online FC | 严格有效 baseline |
| B1 | 当前 HFP，Q-like only + online FC | 当前 generator baseline |
| D1 | B1 + prompt/CLS/margin/gradient 日志 | 区分没学会与后来遗忘 |
| D2 | B1 checkpoint + online/cosine/SLCA-like heads | 区分 representation 与 classifier |
| D3 | B1 + prompt swap/interpolation | 检查 prompt 是否有因果功能 |

HFP 随 epoch 继续改善但未追上 baseline，说明“完全没学”不成立；同时早期差距与后期差距必须分开。Phase 0 应先补 gradient norm、真实 parameter delta、固定样本 feature delta 与 online-head 曲线，再决定是否扩大容量。

### 8.2 Phase 1：最小 generator replay

| ID | 变化 | 回答的问题 |
|---|---|---|
| G0 | 当前 HFP + current-input teacher KD | 仅作没有旧覆盖的对照 |
| G1 | stable-CLS generator + `L_new` | 新接口本身是否可学 |
| G2 | G1 + input-stat prompt-centroid replay | 旧 generator 功能是否更稳 |
| G3 | G1 + input-stat teacher replay | teacher 是否优于 centroid |
| C1 | **无 generator** + output-stat classifier replay/alignment | classifier-only 能解决多少 |
| GC | 最优 generator replay + 与 C1 相同的 classifier 协议 | 两类遗忘是否互补 |

必须同时跑 `batch-only mask` 与 `seen-class mask` 中至少一个明确对照；推荐主结果用 seen-class competition，把当前默认 batch mask 保留为 benchmark-compatibility 对照。

这里的 `C1` 必须真的移除 generator，不能写成“G1 + classifier replay”。否则 `GC-C1` 只能隔离 generator replay loss，不能隔离 generator 本身。令 `G*` 表示 G1/G2/G3 中存活的最佳 generator 配置，则形成严格二乘二：

| | 无 classifier replay/alignment | 有 classifier replay/alignment |
|---|---|---|
| 无 generator | B0 | C1 |
| 有 generator | G* | GC |

因此可以分别报告 `G*-B0`、`GC-C1`、`C1-B0`，以及交互项 `GC-G*-C1+B0`。

### 8.3 Phase 2：同一 checkpoint 的 classifier study

首轮只保留三个低复杂度选项：

1. online linear FC；
2. normalized cosine head；
3. SLCA-style pseudo-feature aligned FC。

FeCAM diagonal/shrinkage Mahalanobis 只在前三者暴露出明确的非球形类分布问题时进入；RanPAC/closed-form ridge 不进入首轮。classifier 很关键不等于 classifier study 应该复杂化。

这一步不重新训练 generator，避免把 head 和 representation 改动混在一起。

### 8.4 Phase 3：容量与优化

只在日志显示 HFP Q-like slice 梯度存在、实际更新发生、但容量/拟合仍不足时，依次比较：

1. HFP internal Q only；
2. HFP internal QK；
3. HFP internal QKV；
4. pooling states/prompt candidates 是否训练。

不建议此时解冻 backbone ViT `qkv`。一旦 backbone 也变，`K_in` 的稳定坐标假设会被破坏，需要另加 drift transport/SDC 类机制，研究问题会发生本质变化。

### 8.5 Phase 4：从标准 Split-CIL 迁回 FlyGCL

标准 Split-CIL 中可以在 task end 固化统计量和 teacher。FlyGCL 中则应改成：

- 每类统计用在线 Welford 持续更新，不宣称“类已结束”；
- teacher 按 `seen_samples`/internal step 周期 snapshot，而不是按类完成；
- snapshot 记录 outer session 与 internal step 两套时间坐标；
- replay 对 old classes 做 class-balanced sampling；
- blurry 类重复出现时明确是继续更新统计，还是使用衰减/版本化统计。

这不是实现细节，而是从 task-aware CIL 到 task-free/blurred stream 的协议变化。

## 9. 首轮任务场景

建议按以下顺序：

1. **Split CIFAR-100**：明确边界、成本低，用来验证 loss 和统计库闭环；
2. **Split CUB-200**：细粒度类别，专门检查 classifier alignment/covariance 是否比 CIFAR 更重要；
3. **FlyGCL Si-Blurry CIFAR-100/CUB-200**：验证在线统计和周期 teacher；
4. **ImageNet-R 或有明显 pretraining semantic gap 的场景**：只有在前三步机制成立后，用来测试 generator 的迁移价值。

跨场景的目的不是堆 benchmark，而是改变一个明确因素：边界清晰度、类内异质性、预训练语义差距或流式复现程度。

## 10. 防止“方法拼接”的准入规则

任何新组件加入前必须填写：

1. 它要修复哪个已测得的失败指标？
2. 不加它时，哪个最小对照会失败？
3. 加入后，哪个中间量应先改善，再带动最终指标？
4. 它是否破坏了当前统计库的坐标稳定假设？
5. 能否在至少两个任务场景复现同一机制，而不是只涨一个 A_auc？

例如：

- `QKV unfreeze` 只有在 Q-only 确实容量不足时准入；
- `classifier alignment` 只有在 aligned/oracle head 显著优于 online head 时准入；
- `drift compensation` 只有在 clean feature 稳定而 prompted output stats 明显漂移时准入；
- `attention KD` 只有在 prompt-output consistency 仍不能保持下游功能时准入。

## 11. 用户确认后再进入的实现顺序

1. 冻结实验协议：数据场景、mask、边界、可保存信息；
2. 只加 instrumentation，验证现有 B0/B1 与梯度/漂移指标；
3. 实现通用在线 class-stat bank，并做合并、采样、数值稳定单测；
4. 实现 stable-CLS generator 接口和等价性/shape/gradient 测试；
5. 实现 teacher snapshot 与 centroid 两种 target，保持互斥开关；
6. 实现 `L_gen_old`，先不动 classifier replay；
7. 实现 `L_cls_old` 与独立 aligned head；
8. 按 G/C/GC 矩阵做一批一批的审查；
9. 标准 Split-CIL 成立后，再写 FlyGCL 周期 snapshot/在线统计适配。

每一步都应先检查 forward/gradient connectivity、保存张量 shape、统计更新时点和 checkpoint 恢复，再跑完整实验。

## 12. 关键协议问题与当前决议

1. **已决：**generator 改读 frozen backbone 的 stable intermediate CLS；现有 full-token HFP 保留为诊断基线，而不是强行承担统计重放接口。
2. **已决：**允许保存一小组仅用于诊断、不参与梯度和模型选择的旧类固定 probe；正式训练仍保持 exemplar-free。
3. **已决：**先在 Split CIFAR-100/CUB-200 验证闭环，再迁回 Si-Blurry。
4. **已决：**classifier 不移除；generator 的训练与评价都必须经过一个明确的 classifier。主结果使用 seen-class competition，batch-only mask 仅作兼容性对照。
5. **已决：**容量消融只开放 HFP association 内部 Q/K/V，不开放 frozen ViT backbone 的 `qkv`。
6. **待详尽计划时冻结：**teacher 在标准 Split-CIL 使用 task-end snapshot；FlyGCL 使用 sample-count 或固定周期 snapshot，具体周期由基线曲线确定。

## 13. 项目内历史与外部来源

项目历史：

- [FlyGCL-memory 结果记录](https://app.notion.com/p/3b2aaacddef180a8adeed0b24565c361)
- [从 router 转向 generator 的一些理解](https://app.notion.com/p/3b2aaacddef1819fb88dcae4d0f63a4a)
- [Bayes 启发的 classifier 建模、简化算法与特征偏移后的更新](https://app.notion.com/p/3b2aaacddef181b7becfea359f7640d2)
- [从 PILOT 梳理的 CIL 脉络](https://app.notion.com/p/3b3aaacddef1802cbb49d45be56281fd)

主要论文与官方代码：

- Tang et al., *When Prompt-based Incremental Learning Does Not Meet Strong Pretraining* ([paper](https://arxiv.org/abs/2308.10445), [code](https://github.com/TOM-tym/APG))
- Zhang et al., *SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model* ([paper](https://arxiv.org/abs/2303.05118), [code](https://github.com/GengDavid/SLCA))
- Goswami et al., *FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning* ([paper](https://arxiv.org/abs/2309.14062), [code](https://github.com/dipamgoswami/FeCAM))
- Yu et al., *Semantic Drift Compensation for Class-Incremental Learning* ([paper](https://arxiv.org/abs/2004.00440), [code](https://github.com/yulu0724/SDC-IL))
- *Training Consistent Mixture-of-Experts-Based Prompt Generator for Continual Learning* ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/34108))
- Pelosin et al., *Towards Exemplar-Free Continual Learning in Vision Transformers* ([paper](https://arxiv.org/abs/2203.13167))
- Hou et al., *Learning a Unified Classifier Incrementally via Rebalancing* ([paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html))
- *Routing without Forgetting: Dynamic Hopfield Pooling for Task-Free Continual Learning* ([paper](https://arxiv.org/abs/2603.09576))
- *Is Prompt Selection Necessary for Task-Free Online Continual Learning?* ([paper](https://arxiv.org/abs/2604.04420), [code](https://github.com/efficient-learning-lab/SinglePrompt))
- Goswami et al., *Covariances for Free: Exploiting Mean Distributions for Federated Learning with Pre-trained Models* ([paper](https://arxiv.org/abs/2412.14326), [code](https://github.com/dipamgoswami/FedCOF))
- Ghashami et al., *Frequent Directions: Simple and Deterministic Matrix Sketching* ([paper](https://arxiv.org/abs/1501.01711))

## 14. 第二轮讨论决议：classifier 兼容性、RwF 审计与低内存统计

### 14.1 classifier 是必要接口，但不能替 generator 领功

classifier 有两个不同角色，必须分开：

1. **兼容性诊断：**generator 可能产生了有用但尺度、范数或分布不适配 online FC 的表征；cosine head 或一次轻量 alignment 可以检查这些信息是否被原 head 吞掉。
2. **独立方法贡献：**classifier replay/alignment 本身就可能带来大部分增益；这部分不能归因给 generator。

因此在每个简单 head `h` 下都做成对比较：

```text
Delta_gen(h) = metric(generator + h) - metric(no-generator + h)
```

并额外报告 `Delta_head`。只有满足下列之一，才认为 classifier 帮助揭示了 generator，而不只是替代了它：

- `Delta_gen(online)` 很小，但 `Delta_gen(cosine/aligned)` 在配对 seeds 上稳定为正，同时旧类 prompted-CLS drift 或 margin drift 也改善；
- generator 极廉价，虽然绝对增益不大，但在额外参数、每类内存和推理 FLOPs 都明确报告后，仍有稳定的正边际收益。

反例也要预先写清：若 `classifier-only` 提升，而 `generator + classifier - classifier-only` 落在配对 seed 波动内，就只能得出“classifier 有效，generator 未证实”。廉价性可以改变是否值得部署的门槛，不能改变因果归因。

### 14.2 对 RwF 的审计结论

RwF 与当前工作的真正交集是：在早期 ViT block 用少量 learnable queries 对完整 token 序列做 many-to-few pooling，再把得到的 prompts 暂时插回 frozen backbone。它给了两个值得测试的工程假设：早层提示可能更有效；冻结 K/V、只学 query-like 参数可能是一个廉价起点。

但目前不能用它证明“Hopfield/attention 天然抗遗忘”：

- 文中的 free-energy 严格凸性是对固定 `q,K` 下的 routing distribution 而言；它不约束 query 参数训练、classifier 漂移或整个 CIL 优化。论文也明确承认该更新在数学上等价于 softmax attention。
- “closed-form routing”只是一次 attention 前向的闭式归一化；learnable queries 仍靠梯度训练，不能据此推出 one-pass learning 或 without-forgetting。
- 主要表格缺少相同可训练参数量的 standard cross-attention、mean/attention pooling、MLP、static/random query 等因果对照。`k=0` sequential fine-tuning 对 `k>=1` HFP 只能说明加模块有效，不能说明 Hopfield 结构独有地有效。
- 所审 arXiv v1 正文没有交代 classifier 形式、seen-class/batch mask、CE 细节和学习率；若结果主要受 classifier 影响，这是关键缺口。
- 论文声称完整调参与若干消融在 supplementary material，但当前 13 页 PDF 末尾直接进入 references；截至 2026-08-09 的检索也未找到作者官方代码。这里只能说公开证据不足，不能反过来断言结果造假。
- 论文自己承认 smoothing 可能损伤 CUB 一类细粒度局部线索。这与“attention 主要发挥平滑作用”的备选解释是一致的，但仍需我们自己的新类学习/旧类漂移曲线验证。

结论：RwF 可以作为结构和消融的灵感，不作为结果可信度或理论正确性的前提。

### 14.3 先纠正“恢复压缩信息”的表述

RwF 压缩的是**当前样本仍然存在的 token**：attention 可以从这些 values 中选择和混合信息。我们的场景要从保存的类统计量生成旧类输入；已经没有保留、且不在已存 values 张成空间中的分量，attention 无法凭空恢复。

因此可检验的说法应是：

> attention/HF generator 能否从一个低内存的 associative sketch 中提取足够的下游判别信息？

而不是“从少量参数恢复原始二阶统计量”。前者可被等内存、等参数对照证伪；后者在一般情形下没有可识别性。

#### stable CLS 与当前 HFP 之间还有一个不能跳过的结构冲突

当前 `HopfieldPoolingPrompts` 是用 `L` 个 pooling queries 对完整 `N`-token 序列做注意力。如果把输入机械替换成单个 `h`，则 `N=1`，softmax 永远等于 1：query 不再决定路由，多个 prompt slot 取回的 input-conditioned 内容也会相同。对标准 attention 而言，score 路径上的 Q/K 梯度会退化为零。这不能被称为“保留 HFP，只把输入换成 stable CLS”。

stable coordinate 仍然是当前最合适的重放接口，但首轮要明确区分三种生成器：

1. **`G_mlp(h)`：**最小闭环和必要的非 attention 对照；
2. **`G_codebook(h)`：**用 `h` 作 query，在少量全局 learned prompt atoms 上做 attention/Hopfield retrieval；它是非退化且容易实现的廉价候选，但本质仍接近 APG 式 soft routing，不能包装成信息恢复；
3. **`G_set(H)`：**让 HFP attend 一个稳定的小集合，例如若干 frozen blocks 的 clean CLS。它更保留 many-to-few pooling 语义，但需要 clean feature pass、第二次 prompted pass，并把统计库扩展到多层，暂不作为首个实现。

所以 Phase 1 推荐 `G_mlp` 与 parameter-matched `G_codebook`；当前 full-token HFP 继续作为历史/诊断基线。只有前两者证明 stable-feature generator 有边际价值后，才值得为 `G_set` 付出架构和存储复杂度。

### 14.4 低内存 `K_in` 实验：只改变 generator 输入库

为了不让 classifier 再次成为混杂因素，固定 `K_out` 的定义、对角存储预算、更新时点和 classifier replay/alignment 算法；每个配置仍收集它自己产生的 final CLS statistics。实验变量只改变 generator 输入端 `K_in`：

| `K_in` 表示 | 约每类存储量 | 它保留什么 | 首轮地位 |
|---|---:|---|---|
| mean only | `D` | 中心 | 最低内存基线 |
| mean + diagonal variance | `2D` | 独立维度尺度 | 标准统计基线 |
| two streaming prototypes + weight | `2D+1` | 两个局部 mode | 等内存的多模态对照 |
| fixed projected mean + diagonal variance | `2r` | 固定低维子空间中的中心与尺度 | 主压缩曲线，`r=D,128,32` |
| per-class Frequent-Directions sketch | 约 `(l+1)D` | 低秩协方差方向 | 前四项遇到明确上限后才进入 |

推荐主线是固定随机正交投影：

```text
s = R^T normalize(h),  R fixed
K_in[c] = {mu_s[c], diag_var_s[c]}
p = G(s)
```

固定 `R` 避免每阶段 PCA 坐标漂移；从 `K_in[c]` 采样 `s` 后直接喂给 generator。`r` 的 sweep 给出内存—性能曲线，而不是只报告一个涨点。

two-prototype 版本先用 streaming k-means/farthest update，不做端到端 learned memory，防止把压缩器本身变成第二个大方法。FedCOF 的“由多个子集均值估计协方差”可以作为旁证，但单一流里每类只有一个总均值时不能识别协方差；若最终仍保存重建后的 `D x D` 矩阵，也没有解决本地存储问题。因此它最多用于一次 classifier 初始化后丢弃，不作为首轮 generator replay 方案。

最关键的结构对照是：在相同 `K_in`、近似参数量和训练 loss 下比较 `G_codebook` 与 linear/MLP decoder。若未来进入 `G_set`，再以同一个稳定 token set 比较 HFP、mean pooling 与 DeepSets/MLP。若不做这些对照，即使压缩后仍涨点，也只能说明“generator 可用”，不能说明 attention/HF 的独特优势。

### 14.5 保持原 Phase 1/2/3 顺序，但增加硬门槛

1. **Phase 1：**先完成 stable-CLS generator 的 `G1/G2/G3/C1/GC`，其中 `G1` 至少包含 `G_mlp` 与 parameter-matched `G_codebook`。generator 在有/无 classifier replay 下的主贡献分别是 `GC-C1` 与 `G*-B0`，不是笼统的 `GC-B0`。
2. **Phase 2A：**只比较 online linear、cosine、SLCA-style aligned FC；用相同 checkpoint 做 classifier 兼容性诊断。
3. **Phase 2B：**仅对 Phase 1 存活配置做 `mean / diagonal / two prototypes / projected-r` 内存曲线，并做 attention 对 parameter-matched MLP。
4. **Phase 3：**只有 generator 已有边际价值但表现出容量不足时，才做 internal Q、QK、QKV；否则不再投入。

止损规则：

- 一次 smoke seed 只检查闭环、shape、梯度与数值稳定；仅存活配置进入三个配对 seeds。
- 若 `C1` 有效但 `GC-C1` 落在配对波动内，且旧类 prompt/CLS/margin drift 没有一致改善，停止 generator 主线。
- 若压缩表示不能在明显更低内存下接近 diagonal 输入，停止“廉价统计”故事；不追加更复杂的 learned compressor。
- 若 attention/HF 不优于等参数 MLP/linear，保留 generator 结果也可以，但停止宣称 attention/HF 的独特恢复能力。
- 若 attention 降低旧类 drift，却稳定伤害新类学习或 CUB 局部判别，按 smoothing/stability regularizer 解释；若净收益不成立，及时止损而不是再拼组件。
