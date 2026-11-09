# Loop-JEV 核心实现审查包

本仓库不是完整训练工程，也不是实验复现包。它只保留 Loop-JEV 在 Kev
决策模型上的核心实现，目的是让审查者能够沿着“输入编码 -> 前向传播 ->
隐状态回注 -> Pointer Head 打分 -> Loss -> 参数更新 -> Checkpoint 重载”
逐段核验方法是否忠实实现。

源码来自本地最新工作区
`D:\MyProgram\Python\jev_benchmark_engineered\loopjev\kev`，核心文件按字节
原样复制，未在此包内修复。文件哈希和提取范围见
[`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md)。

## 结论先行

当前代码确实实现了可执行的多步 latent recurrence，且现有 CPU 测试覆盖了
K=0、COCONUT、完整 Ponder、Decision-only 和 option+decision 的通过次数、
隐状态读取位置及梯度可达性。核心训练和推理路径也共用同一个
`DecisionModel.forward_batch()`。

但审查后有 2 个需要优先处理的问题：

1. **Full-weight Ponder checkpoint 重载时，冻结的 LM head 没有沿用
   checkpoint 记录的 base revision。** `checkpoint.py:325` 的
   `_full_torch()` 没有把 `meta.base_revision` 传给
   `DecisionModel`；后者在 `model.py:366` 从 Hub 加载 base LM head。base
   分支更新后，同一 backbone/head checkpoint 会得到不同 recurrence。
2. **Partial-depth COCONUT 并不是从第 `start_layer` 层真正的输入状态开始
   loop。** 首轮返回的是模型最终 `norm` 之后的 `last_hidden_state`；
   `loop.py:617-625` 随后把这个最终表示直接送入
   `layers[start_layer:]`。因此 `coconut_start_layer=16` 的实际语义是
   “把最终归一化表示重新送入第 16 层”，不是“保留第 15 层输出并循环后 8
   层”。它可以作为自定义实验，但不能被表述为忠实的 partial-depth
   COCONUT。

这两项分别对应“训练/推理可复现性”和“方法语义正确性”，建议优先于调参
问题处理。

## 0. 核心文件与函数索引

| 文件 | 关键对象 | 作用 |
| --- | --- | --- |
| `kev/model.py` | `SPECIAL` at `17` | 5 个控制 token 的固定映射 |
| `kev/model.py` | `encode()` at `93` | state/question/option 编码，生成 `ids/seg/pos/opt`、`decide_idx`、`opt_idx` |
| `kev/model.py` | `branch_mask_batch()` at `165` | packed block-causal 与 option-isolation mask |
| `kev/model.py` | `rows_of()` at `194` | 将 packed encoding 拆成 state + 每 question causal row |
| `kev/model.py` | `PointerHead` at `211` | query/key projection 与 option logits |
| `kev/model.py` | `DecisionModel.__init__()` at `264` | backbone、LoRA、Ponder LM head、loop flags 的装配 |
| `kev/model.py` | `_loop_forward_batch()` at `495` | COCONUT/Ponder/Decision-only 的实际 row-form forward |
| `kev/model.py` | `forward_batch()` at `656` | 训练主入口；`steps=0` 回退普通 Kev |
| `kev/model.py` | `probs()` at `671` | eval 概率输出 |
| `kev/model.py` | `trainable_parameters()` at `783` | optimizer 参数集合 |
| `kev/loop.py` | `LoopConfig` at `49` | method、K、top-k、official-style flags、pass count |
| `kev/loop.py` | `CoconutOutput` at `188` | 首轮 token states、最终 hidden、latent query |
| `kev/loop.py` | `PonderOutput` at `224` | final hidden、first hidden、decide query、passes |
| `kev/loop.py` | `_deep_coconut_hidden()` at `531` | partial-depth COCONUT 的后层重跑 |
| `kev/loop.py` | `coconut_forward()` at `572` | 完整/partial COCONUT 主体 |
| `kev/loop.py` | `ponder_forward()` at `629` | 普通 Ponder、Decision-only、option+decision 的分派 |
| `kev/loop.py` | `_ponder_decide_recurrence()` at `775` | 只更新 `<decide>` 的实验 recurrence |
| `kev/loop.py` | `_ponder_option_decide_recurrence()` at `836` | 只更新 readout endpoints 的实验 recurrence |
| `kev/train.py` | `question_loss()` at `42` | CE、soft target、Brier、focal、ordinal loss |
| `kev/train.py` | `encode_batch()` at `349` | augmentation 后严格编码 |
| `kev/train.py` | `batch_loss()` at `372` | loop logits、anchor、permutation KL、Multi-K loss |
| `kev/train.py` | `parse_args()` at `409` | 训练参数、loop 组合与合法性检查 |
| `kev/train.py` | `main()` at `622` | 模型构造、optimizer、训练循环、保存 |
| `kev/checkpoint.py` | `Meta` at `49` | loop 配置持久化及旧 checkpoint 默认值 |
| `kev/checkpoint.py` | `Checkpoint.load()` at `269` | LoRA/full checkpoint 重建模型 |
| `kev/checkpoint.py` | `_full_torch()` at `325` | full-weight 模型加载；包含已确认 revision 问题 |
| `kev/checkpoint.py` | `warm_start()` at `379` | `--init_from` 的权重和 Pointer Head 热启动 |
| `kev/full_ft.py` | `MasterAdamW` at `63` | fp32 master 的全量微调 optimizer |
| `kev/api.py` | `to_record()` at `102` | API question 到 internal question/option 的映射 |
| `kev/api.py` | `to_answers()` at `149` | 概率到 Choice/Noul/Score 输出 |
| `kev/data.py` | `materialize()` at `397` | label key 到 option index 的映射 |
| `kev/predictors.py` | `LocalPredictor.__call__()` at `127` | 本地 checkpoint 推理、logits/probabilities |
| `kev/benchmark.py` | `prediction_rows()` at `48` | 预测与 label 对齐并校验分布 |
| `kev/benchmark.py` | `summarize()` at `73` | accuracy/NLL/ECE/heldout 汇总 |
| `kev/metrics.py` | `metrics()` at `64` | benchmark 行级指标计算 |

主要调用关系：

```text
kev.train.main
  -> DecisionModel.__init__                 # loop flags, backbone, LoRA, pointer head
  -> training_requests / record_variants
  -> encode_batch                            # data.materialize + DecisionModel.encode
  -> batch_loss
       -> DecisionModel.forward_batch
            -> _loop_forward_batch
                 -> coconut_forward | ponder_forward
                 -> PointerHead
       -> question_loss
  -> backward -> clip/AdamW/MasterAdamW -> checkpoint

Checkpoint.load
  -> read_meta
  -> DecisionModel.__init__
       -> _full_torch or _adapted_torch
  -> load pointer head

LocalPredictor.__call__
  -> data.materialize
  -> DecisionModel.encode
  -> DecisionModel.forward
  -> softmax(logits)
  -> benchmark.prediction_rows / metrics
```

## 1. 核心方法

### 1.1 Kev 基线

Kev 将一条请求编码成：

```text
[<state> state tokens]
[<q> instruction
     <opt> option 0 </opt>
     <opt> option 1 </opt>
     ...
     <decide>]
```

每个 question 是独立 branch。默认 packed 前向使用 block-causal mask：
state 可见自身；question branch 可见 state 与自身，不可见其他 question。
Pointer Head 以 `<decide>` hidden state 为 query，以每个
`</opt>` hidden state 为 key，产生该题各选项的 logits。

关键实现：

- token 编码与 option endpoint：
  [`kev/model.py:93`](kev/model.py#L93)
- block-causal mask：
  [`kev/model.py:165`](kev/model.py#L165)
- Pointer Head：
  [`kev/model.py:211`](kev/model.py#L211)
- Kev 前向入口：
  [`kev/model.py:656`](kev/model.py#L656)

### 1.2 Loop 变体

| Arm | `K` 的含义 | Backbone passes | Pointer query | Pointer option keys |
| --- | --- | ---: | --- | --- |
| Kev | 无 loop | 1 | 首轮 `<decide>` | 首轮 option endpoints |
| COCONUT | latent recurrence 次数 | `K + 1` | 最后一轮 appended latent | 首轮 option endpoints |
| Full Ponder | expected-embedding recurrence 次数 | `K + 1` | 最后一轮 `<decide>` | 最后一轮 option endpoints |
| Decision-only Ponder | 只更新 `<decide>` 的 recurrence 次数 | `K + 2` | 最后决定轮的 `<decide>` | 首轮 option endpoints |
| option+decision Ponder | 只更新 option endpoints 与 `<decide>` | `K + 1` | 最后一轮 `<decide>` | 最后一轮 option endpoints |

`K=0` 的所有 arm 都会回退到普通 one-pass Kev 前向：
[`kev/model.py:656`](kev/model.py#L656)、
[`kev/loop.py:174`](kev/loop.py#L174)。

## 2. Forward、Hidden State 与回注

### 2.1 COCONUT

入口：
[`kev/loop.py:572`](kev/loop.py#L572)。

标准 `coconut_start_layer=0` 路径为：

1. 用 `input_ids` 做一次完整前向，得到 `H0`。
2. 取 `H0[:, -1, :]` 作为新的 latent token embedding，追加到原始 embedding
   序列末尾。
3. 扩展 position ids 与 mask，重跑完整 backbone。
4. 重复 K 次。
5. Pointer query 使用最后 appended latent；option keys 使用首轮 `H0` 的
   `</opt>` 位置。

gradient 没有 detach，首轮 option states 和 latent recurrence 都留在计算图
中。

### 2.2 Partial-depth COCONUT

参数：
[`kev/loop.py:504`](kev/loop.py#L504)。

实现：
[`kev/loop.py:531`](kev/loop.py#L531)。

调用点：
[`kev/loop.py:617`](kev/loop.py#L617)。

当前实现先执行完整首轮，得到经最终 `norm` 的 `last_hidden_state`，再把该
tensor 送入 decoder `layers[start_layer:]`。这与 decoder 的中间状态语义
不一致：第 `start_layer` 层期望的输入应是第 `start_layer - 1` 层输出、
且尚未执行最终 `norm` 的 residual-stream hidden state。

对 Qwen2 系列，`transformers` 的 `Qwen2Model.forward()` 在返回前执行：

```python
hidden_states = self.norm(hidden_states)
return ... last_hidden_state=hidden_states ...
```

因此这是**已确认的方法语义偏差**，不是单纯的命名问题。

### 2.3 Full Ponder

入口：
[`kev/loop.py:629`](kev/loop.py#L629)。

记输入 embedding 为 \(E_0\)，第 \(t\) 次 recurrence 的 hidden state 为
\(H_t\)，LM head 输出为 \(L_t=\mathrm{lm\_head}(H_t)\)，输入 embedding
矩阵为 \(W_E\)。实现为：

\[
p_t(v)=\mathrm{softmax}(L_t)_v
\]

\[
A_t=\sum_v p_t(v)W_E[v]
\]

默认 legacy 配置是 anchored top-k：

\[
\bar E_{t+1}=E_0+\sum_{j=1}^{t}A_j
\]

`ponder_scale_embeds` 与 `ponder_multiply_steps` 打开后，走 explicit
official-style 路径：

\[
E'_0=E_0\sqrt{d},\qquad
\bar E_0=E'_0\cdot K,\qquad
\bar E_{t+1}=\bar E_t+A_t
\]

`ponder_full_vocab=True` 时 \(p_t\) 对完整词表归一化；否则只对 top-k
logits 归一化。需要注意：当前默认值不是官方 PonderLM 的 full-vocabulary
配置，而是该项目保留的 legacy top-100 approximation。

### 2.4 Decision-only Ponder

入口：
[`kev/loop.py:775`](kev/loop.py#L775)。

实际流程：

1. 普通首轮得到 \(H_0\)，option keys 固定为 \(H_0\) 的 endpoint。
2. 每个 recurrence 都从 \(E_0\) 克隆 embedding，只替换 `<decide>` 位置。
3. 对 `<decide>` hidden state 单独做 `lm_head`，不投影整条序列。
4. 更新 `<decide>` expected embedding。
5. 最后一个 full-row pass 的 `<decide>` hidden state 作为 query。

因此它不是官方 PonderLM，而是“固定 option 表征、只搜索 decision query”
的实验 ablation。`K>0` 时成本为 `K+2` 次 backbone forward。

### 2.5 Option+decision Ponder

入口：
[`kev/loop.py:836`](kev/loop.py#L836)。

该变体没有普通首轮。`update_mask` 只选择每个 question 的
`</opt>` endpoint 与 `<decide>`。每轮只将这些位置的 hidden state 投影到
vocab 并计算 expected embedding。最终 pass 同时提供 query 和 option keys。

该路径比 Decision-only 少一次 ordinary pass，`K>0` 时仍为 `K+1`。

## 3. Tokenizer、候选答案与分数对应

### 3.1 分隔符

代码复用 Qwen 已有特殊 token：
[`kev/model.py:17`](kev/model.py#L17)。

| 语义 | Token |
| --- | --- |
| state | `<|fim_prefix|>` |
| question | `<|fim_middle|>` |
| option start | `<|box_start|>` |
| option end | `<|box_end|>` |
| decide | `<|fim_suffix|>` |

用户输入中的同类控制串会先重写，避免伪造 option boundary：
[`kev/model.py:84`](kev/model.py#L84)。

### 3.2 API 到内部 option 顺序

`api.to_record()` 在
[`kev/api.py:102`](kev/api.py#L102) 生成：

- Choice：按 `criteria.items()` 顺序生成 option 文本；
- Noul：固定 `[no, yes]`；
- Score：按 criteria list 顺序生成有序 level。

`data.materialize()` 在
[`kev/data.py:397`](kev/data.py#L397) 将 label key 映射为 option 下标。

这项映射是正确的：Choice 使用 `keys.index(label)`，Noul/Score 使用
整数 label；`encode()` 的 `opt_idx` 与 option 顺序来自同一个
`q["options"]` 列表。

### 3.3 Pointer 打分

实现：
[`kev/model.py:222`](kev/model.py#L222)。

\[
z_{ik}
=
\frac{
\left(W_k h^{\text{opt}}_{ik}\right)^\top
\left(W_q h^{\text{decide}}_i\right)
}{\sqrt{d_p}}
\]

\[
p_{ik}=\frac{\exp(z_{ik})}{\sum_j\exp(z_{ij})}
\]

`temperature` 只在 eval 模式除到 logits 上，不改变 argmax。

## 4. Loss、梯度与训练机制

### 4.1 主 Loss

入口：
[`kev/train.py:42`](kev/train.py#L42)。

默认是 pointer logits 上的 cross entropy：

\[
\mathcal L_{\mathrm{CE}}
=-\log p(y\mid x,\text{question})
\]

可选 auxiliary terms：

- soft-target CE；
- label smoothing；
- Brier probability penalty；
- focal weighting；
- Score 题的 ordinal CDF penalty；
- anchor KL；
- option-order permutation symmetric KL；
- Multi-K auxiliary CE。

### 4.2 训练 step 与梯度路径

主 batch loss：
[`kev/train.py:372`](kev/train.py#L372)。

主训练入口：
[`kev/train.py:622`](kev/train.py#L622)。

训练 loop：
[`kev/train.py:723`](kev/train.py#L723)。

实际优化步骤：

- 生成 logits；
- 对每个 question 计算 CE/辅助损失；
- 按 accumulation group 的记录数归一化并 `backward()`；
- 梯度裁剪到 global norm `1.0`；
- 更新参数并 `zero_grad()`。

现有测试确认：

- Ponder expected embedding 的梯度可达 backbone、LM projection 和 selected
  embedding rows；
- Decision-only 和 option+decision 的 recurrence 梯度未被 detach；
- LM head 在 Ponder LoRA 路径中冻结；
- Pointer Head 始终参与训练。

### 4.3 LoRA 与冻结策略

实现：
[`kev/model.py:387`](kev/model.py#L387)。

| 项目 | 实际配置 |
| --- | --- |
| LoRA rank | CLI 默认 `16` |
| LoRA alpha | `2 * rank` |
| LoRA dropout | `0.05` |
| target modules | q/k/v/o/gate/up/down projections |
| Pointer Head | 从零训练 |
| Ponder LM head | 冻结 |
| 默认 `special_embeddings` | 关闭；打开时只训练 5 个 delimiter embedding |

LoRA 训练时输入 embedding 矩阵默认冻结。Ponder 的 expected embedding 仍有
梯度并参与 recurrence，但只有 backbone adapter 与 Pointer Head 更新；这不
是梯度断裂，但意味着 recurrence 不能改变词表 embedding 几何。

Full-weight 路径：
[`kev/full_ft.py:63`](kev/full_ft.py#L63)。

该路径保留 fp32 master weights/moments，并把更新后的 bf16 working weights
写回模型；多卡时 backbone 由 FSDP2 分片，Pointer Head 在各 rank 复制。

### 4.4 Optimizer 与 scheduler

参数组及 optimizer：
[`kev/train.py:692`](kev/train.py#L692)。

| 项目 | 代码默认/实现 |
| --- | --- |
| LoRA optimizer | AdamW |
| Full-FT optimizer | `MasterAdamW` |
| base LR | `2e-4` |
| Pointer Head LR | `--head_lr`，0 时与 base LR 相同 |
| weight decay | `0.01` |
| scheduler | OneCycleLR，`pct_start=0.1` |
| gradient clip | `1.0` |

这些是代码默认值。正式实验实际传入的参数不在本源码包内，不能根据 README
默认值反推实验配置。

## 5. 推理、Checkpoint 与预测

### 5.1 Forward 与 inference 一致性

训练使用：

[`kev/train.py:378`](kev/train.py#L378)
调用 `DecisionModel.forward_batch()`。

本地推理使用：

[`kev/predictors.py:127`](kev/predictors.py#L127)
调用同一个 `DecisionModel.forward()`。

对正 loop steps，`forward()` 和 `forward_batch()` 都会进入
[`kev/model.py:495`](kev/model.py#L495) 的同一套 row-form recurrence。
这正是本实现最重要的正确性保证之一：训练和 inference 没有分成两套 loop
代码。

### 5.2 Backbone pass 成本

按每个 question 一条 causal row 计算：

| Arm | 相对 Kev 计算量 |
| --- | --- |
| Kev | 1x |
| COCONUT K | `K+1` backbone passes |
| Full Ponder K | `K+1` backbone passes |
| Decision-only K | `K+2` backbone passes |
| option+decision K | `K+1` backbone passes |

Decision-only 多出的 ordinary pass 用于固定 option keys；它的成本高于
full Ponder 和 option+decision。

### 5.3 Checkpoint 重载

元数据：
[`kev/checkpoint.py:49`](kev/checkpoint.py#L49)。

加载入口：
[`kev/checkpoint.py:269`](kev/checkpoint.py#L269)。

Full-weight：
[`kev/checkpoint.py:325`](kev/checkpoint.py#L325)。

LoRA：
[`kev/checkpoint.py:341`](kev/checkpoint.py#L341)。

Full Ponder 的 checkpoint 只保存 backbone 和 Pointer Head，冻结 LM head
需从 base 重新加载。当前实现存在前述 base revision 未传递的问题。

### 5.4 最终预测

本地预测与概率映射：
[`kev/predictors.py:127`](kev/predictors.py#L127)、
[`kev/api.py:149`](kev/api.py#L149)。

benchmark 评分：
[`kev/benchmark.py:48`](kev/benchmark.py#L48)、
[`kev/benchmark.py:73`](kev/benchmark.py#L73)。

预测路径按 question key 对齐 logits，不会直接依赖 option 文本内容。

## 6. 关键超参数

| 名称 | 实际默认/约束 | 位置 |
| --- | --- | --- |
| `loop_method` | `kev` | `train.py:442` |
| `loop_steps` | `0` | `train.py:444` |
| `coconut_start_layer` | `0`，0 表示完整 backbone | `train.py:445` |
| `ponder_top_k` | `100` | `train.py:447` |
| `ponder_full_vocab` | `False` | `train.py:453` |
| `ponder_scale_embeds` | `False` | `train.py:449` |
| `ponder_multiply_steps` | `False` | `train.py:451` |
| `ponder_decide_only` | `False` | `train.py:457` |
| `ponder_option_decide_only` | `False` | `train.py:461` |
| `head_dim` | `256` | `train.py:440` |
| `lora` | `16` | `train.py:417` |
| batch / accum | `1` / `8` | `train.py:418`、`train.py:433` |
| max state | `384` training default | `model.py:20` |

Positive-step loop 有额外限制：

- 只支持 attention-only backbone；
- 不支持 `option_isolation`；
- 不支持 shared prefix；
- 每个 question 独立 causal row，会重复计算 state；
- 当前 MLX backend 拒绝正 loop-step checkpoint。

## 7. 审查发现

### 7.1 已确认问题

#### P1：Full-weight Ponder 的 LM head 未按 checkpoint revision 重载

证据：

- `checkpoint._full_torch()` 在
  [`kev/checkpoint.py:329`](kev/checkpoint.py#L329) 构造
  `DecisionModel`，但没有传 `revision=meta.base_revision`。
- `DecisionModel` 在
  [`kev/model.py:366`](kev/model.py#L366)
  使用 `revision=None` 加载 base `lm_head`。

影响：

- backbone 和 Pointer Head 是固定的，但 recurrence 的 LM projection 可能
  随 Hub `main` 分支变化；
- 同一 checkpoint 在不同时间或不同缓存状态下，可能得到不同 Ponder
  expected embeddings 和最终概率；
- 这是方法可复现性问题，不只是 tokenizer 版本问题。

已用最小 mock 检查确认 `_full_torch()` 传给 `DecisionModel` 的
`revision` 当前为 `None`。该问题在
`review_checks/test_known_findings.py` 中以 strict xfail 固定。

#### P1：Partial-depth COCONUT 从最终 norm 表示重新进入中间层

证据：

- 首轮 `last_hidden_state` 已是最终 norm 之后的输出；
- [`kev/loop.py:617`](kev/loop.py#L617) 将这个 tensor 作为后续 hidden；
- [`kev/loop.py:625`](kev/loop.py#L625) 将其送入 `_deep_coconut_hidden()`；
- [`kev/loop.py:531`](kev/loop.py#L531) 只运行
  `layers[start_layer:]`，最后再次 `norm`。

影响：

- `coconut_start_layer=16` 不等价于“从第 15 层输出开始重复第 16-23 层”；
- 与标准 COCONUT、以及“loop 中间偏后层”的直觉表述不一致；
- 若该实验用于支撑“后层循环有效”，结论归因会失真。

建议审查者要求补齐真正的 layer-input hidden-state 保留或明确把该 arm
重命名为“final-norm representation re-entry”。

### 7.2 潜在风险

#### P2：默认 Ponder 不是官方 PonderLM fidelity

默认是 top-100、anchored recurrence；official-style 需要同时设置
`ponder_scale_embeds`、`ponder_multiply_steps` 和
`ponder_full_vocab`。任何将默认配置直接称为“official PonderLM”的实验
说明都不准确。

#### P2：LoRA 下 expected embedding 的 embedding matrix 不可训练

Ponder 的梯度确实进入 selected embedding rows，但 LoRA 路径默认冻结整张
embedding matrix。可训练的是 backbone adapter 和 Pointer Head；因此
recurrence 不能改变 LM 的词表几何。这是设计选择，但会限制 Ponder 相对
普通 Kev 的可达函数空间。

#### P2：configuration footgun：`ponder_preserve_options` 未强制要求 Ponder

`DecisionModel` 明确校验 `ponder_decide_only` 和
`ponder_option_decide_only` 必须使用 `loop_method="ponder"`，但
`ponder_preserve_options=True` 没有同样的校验。非 Ponder arm 中该 flag
会被静默忽略。建议配置校验统一。

#### P3：`KEV_LOOP_ROW_BATCH` 未验证

[`kev/model.py:509`](kev/model.py#L509) 直接把环境变量转成整数作为 batching
step。`0`、负数或非数字会在训练/推理时产生不直观错误。默认路径不受影响，
但建议显式限制为正整数。

### 7.3 测试空白

- 没有对 partial-depth COCONUT 的 hidden-state 层级做数值契约测试。
- 没有覆盖 full-weight Ponder checkpoint 的 base revision 传递。
- 没有覆盖 `ponder_preserve_options` 在非 Ponder arm 上的配置拒绝。
- 没有验证 `KEV_LOOP_ROW_BATCH` 的非法值。
- 现有 top-k gradient 测试只证明 selected embedding rows 有梯度，不证明
  冻结 embedding 时 expected-embedding 表征能力足够。

## 8. 测试与复核

推荐环境与 upstream 一致：Python 3.12，PyTorch 2.6-2.8，Transformers 5.x。

```bash
python -m pytest tests/test_loop.py -q
python -m pytest tests/test_loop_integration.py -q
python -m pytest tests/test_multik_loss.py -q
python -m pytest review_checks/test_known_findings.py -q
```

本包在 2026-11-09 的 Windows 本地复核结果：

- `tests/test_loop.py`：54 passed；
- `test_loop_integration.py` 的 COCONUT/checkpoint 定向子集：6 passed；
- `test_loop_integration.py` 全量：31 passed，1 个测试因 Windows 缺少
  Unix-only `resource` 模块而在导入 `kev.train` 时失败；
- `review_checks/test_known_findings.py`：预期为 1 xfailed，代表已知
  full-weight Ponder revision 问题。

`resource` 失败是本地操作系统限制；Linux 审查环境应重新运行全量测试。
