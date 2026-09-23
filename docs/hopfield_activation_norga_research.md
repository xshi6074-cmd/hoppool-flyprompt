# Hopfield activation 与 NoRGa score 的对应关系

## 结论

当前 FlyPrompt 的 Hopfield 路径不能把 NoRGa 公式直接作用在生成后的
prompt tensor 上。NoRGa 修改的是 softmax 之前的 attention score，并且只修改
prefix key 对应的 score 子矩阵：

```text
A_prompt_hat = A_prompt + alpha * sigma(tau * A_prompt)
A_hat = concat(A_prompt_hat, A_pretrain)
attention = softmax(A_hat)
```

因此，先前的

```text
prompt = prompt + alpha * sigmoid(tau * prompt)
```

在数学位置和模型作用上都不等价，已经撤回。

## 论文中的准确位置

论文 *Mixture of Experts Meets Prompt-Based Continual Learning*
（arXiv:2405.14124v4）在公式 (12) 中把 prefix expert 的 score
`s_i,N+j` 改为：

```text
s_hat = s + alpha * sigma(tau * s)
```

公式 (13)-(15) 又明确把 attention score 拆为：

```text
A = [A_prompt, A_pretrain]
```

只变换 `A_prompt`，而 `A_pretrain` 保持不变，然后才执行 softmax。
论文还说明 alpha、tau 是可学习标量；实验比较了 tanh、sigmoid 和 GELU，
并在第一个任务训练完成后冻结 alpha、tau。

## 官方实现

官方仓库 `Minhchuyentoancbn/MoE_PromptCL` 的 `attention.py`：

1. 计算 `prompt_attn = q @ key_prefix.T * scale`；
2. 只对 `prompt_attn` 执行残差非线性；
3. 计算普通 token 的 `attn = q @ k.T * scale`；
4. 拼接两部分；
5. 最后执行 softmax。

本仓库的 `models/hide_norga_prefix_vit.py` 已采用相同结构：

```python
prompt_part = attn_logits[..., :self.prefix_len]
base_part = attn_logits[..., self.prefix_len:]
prompt_part = prompt_part + gate_act(prompt_part * tau) * alpha
attn_logits = torch.cat([prompt_part, base_part], dim=-1)
attn_prob = attn_logits.softmax(dim=-1)
```

## 与当前 Hopfield 路径的不等价

当前 `HopfieldPoolingPrompts` 的流程是：

```text
ViT tokens
  -> HopfieldPooling
  -> 生成完整 prompt tokens
  -> prompt tokens 插入 ViT token sequence
  -> 普通 self-attention（prompt 同时参与 Q/K/V）
```

论文使用的是 prefix tuning：

```text
X 产生 Q/K/V
prefix 只提供额外 K/V
只修改 Q 与 prefix K 的 score
```

另外，HopfieldPooling 内部自身也有一次 association：

```text
static pooling states (Q) @ input token keys (K).T
  -> softmax
  -> weighted values
  -> generated prompts
```

因此存在两个候选 score：

1. **ViT prompt-key score**：生成 prompt 插入 backbone 后，原始 token query
   与 prompt key 的 score。它最接近论文的 `A_prompt`，但当前 prompt 同时参与
   Q/K/V，需要明确只修改“保留下来的原始 token query 行 × prompt key 列”。
2. **Hopfield association score**：HopfieldPooling 内部静态 Q 与输入 K 的
   score。修改它会改变 prompt 生成器的检索权重，但这不是论文的 prefix-expert
   score，只能称为受 NoRGa 启发的 Hopfield 变体。

## 推荐方案

若目标是“尽可能忠实地移植论文 NoRGa”，推荐方案 1：

- `activation=False`：完全走当前 block 原始 forward，逐算子保持不变；
- `activation=True`：仅在插入 prompt 的 block 中手工展开 attention；
- 只修改原始 token query 行与 prompt key 列的 pre-softmax score；
- prompt query 行保持不变并在 block 后按现有逻辑移除；
- alpha、tau 按层设置为可学习标量；
- activation function 明确配置，默认值需由实验设计决定；
- 第一个内部 step 结束后冻结 alpha、tau，与论文实验设置一致。

若目标是研究“非线性 Hopfield association”，应单独命名参数与实验，
避免把它报告为 NoRGa 复现。

## 必须验证的不变量

1. `activation=False` 与修改前输出逐元素一致。
2. `activation=True, alpha=0` 与关闭 activation 输出逐元素一致。
3. 只有目标 score 子矩阵发生变化。
4. alpha、tau 在启用时有非空梯度，关闭时不存在或不可训练。
5. 冻结时点与内部 step 边界一致。
6. RPFC 继续使用无 prompt 的 frozen-backbone CLS，不受 activation 影响。
7. 检查 AMP 下 score、mask、softmax 的数值稳定性。

## 来源

- Paper: https://arxiv.org/pdf/2405.14124v4
- Official code: https://github.com/Minhchuyentoancbn/MoE_PromptCL/blob/master/attention.py
- Local reference: `models/hide_norga_prefix_vit.py`
