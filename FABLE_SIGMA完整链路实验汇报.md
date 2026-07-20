# FABLE + SIGMA 完整链路阶段结果

结果提交：`f2219db`<br>
模型：`FacebookAI/xlm-roberta-base`<br>
输入：4 条序列 × 128 tokens，共 512 个唯一 Token IDs<br>
安全链路：私密 Token IDs + 私密 Embedding 表 → FABLE → 50-bit 算术份额 → SIGMA 掩码输入 → 真实 XLM-R Encoder Block

## 核心结论

已经实现并验证 FABLE 与 SIGMA 的安全协议衔接。FABLE 完成了 512×768 的私密 Embedding 查询，共检查 393,216 个定点值，结果为 0 mismatch。四条序列均进入真实 XLM-R Encoder Layer 0，P0 与 P1 输出全部一致；sequence 0 与零掩码 correctness oracle 逐字节一致。

该结果证明完整安全链路已经跑通，但当前属于 correctness-first 阶段结果，不是优化后的最终性能，也不是完整 12 层 XLM-R 推理。

## 阶段结果

| 阶段 | 工作量/配置 | 耗时 | 通信或存储 | 正确性 | 关键意义 |
|---|---|---:|---:|---|---|
| FABLE 私密查表 | 512 unique IDs；250,002×768 表；24 chunks | 738.922 s | 双方发送合计 191,512,542,340 B | 393,216 values；0 mismatch | 同时保护 Token IDs 与 Embedding 表；当前主要瓶颈 |
| FABLE→SIGMA 转换 | 393,216 个 50-bit ring elements | P0/P1：59.987/18.665 ms | 每方发送 3,145,736 B | masked input 相同；0 mismatch | 不公开重构 Embedding，将份额安全转换为 `x+r` |
| SIGMA 离线预处理 | 1 layer × 4 sequences | 不计入在线时间 | 每方 batch 密钥 6,028,214,272 B | 四条序列独立密钥 | 体现 FSS 不可复用密钥的离线成本 |
| SIGMA 在线推理 | Layer 0；4×128×768；12 heads | P0/P1：5.376/5.393 s | 每方 batch 354,130,232 B | 四条序列 P0=P1；seq0=oracle | 证明 FABLE 输出能进入真实权重 SIGMA Block |
| 顺序组合总计（推导） | Lookup + 转换 + 4 次 SIGMA online | 约 744.375 s | 约 191,872,964,044 B | 各阶段均通过 | 展示当前实现规模，不是优化后的最终性能 |

## 实验配置

| 配置项 | 数值 | 关键意义 |
|---|---|---|
| 模型 | FacebookAI/xlm-roberta-base | 使用真实公开预训练模型参数 |
| Vocabulary size | 250,002 | 提供适合 FABLE 的大规模查表场景 |
| Hidden size | 768 | 与 SIGMA bert-base 形状一致 |
| Attention heads | 12 | XLM-R-base Encoder 配置 |
| FFN hidden size | 3,072 | 4× hidden size |
| 已运行层数 | 1/12 layers | 当前验证 Layer 0，不是完整 Encoder |
| Sequence batch | 4×128 tokens | 对齐 FABLE 的 512-query batch |
| Embedding element | signed int16，scale 12 | 对齐 FABLE 向量输出接口 |
| SIGMA ring | bitwidth 50，scale 12 | 对齐 SIGMA bert-base 算术配置 |
| FABLE 输出 | `x0+x1=x mod 2^50` | 查询结果始终以算术份额保存 |
| SIGMA 输入 | public `x+r`，dealer 保存 `r` | 安全连接 FABLE 与 SIGMA 的核心接口 |
| 模型参数 | dealer mask + evaluator masked weights | 避免把真实权重错误地明文加载给双方 |

## 关键指标意义

### FABLE 查表正确性

完整检查 393,216 个 Embedding 元素，而不是抽样检查。0 mismatch 证明私密查询、向量打包、符号扩展和 GC→算术份额转换正确。

### FABLE 查表时间与通信

当前 768 维输出被拆成 24 个 32 维 chunk，每个 chunk 都重复 FABLE 查询和 setup 工作。因此 738.922 秒和约 191.5 GB 是 correctness-first 实现的开销，不能当作优化后的最终 FABLE 性能。

### Share→Mask 转换

衡量两个协议之间的安全衔接成本。双方只交换 `x_i+r_i`，不会公开重构 Embedding。该阶段的时间和通信远小于 FABLE Lookup。

### SIGMA 离线密钥大小

FSS 预处理密钥不能跨输入复用。每条序列每方需要 1,507,053,568 B，四条序列每方共需要 6,028,214,272 B，体现方案的离线存储和分发成本。

### SIGMA 在线时间与通信

衡量真实 XLM-R Encoder Block 的安全推理开销。但 A100-2 在运行前 utilization=99%，因此本次约 5.4 秒的 batch 时间只用于证明协议跑通，不能进入正式性能平均值。通信量不直接受 GPU 竞争影响，可以作为阶段性通信指标保存。

### Correctness oracle

P0 与 P1 输出一致只能说明双方得到相同结果；sequence 0 进一步与零掩码同协议 oracle 逐字节一致，排除了“双方一致但共同算错”的风险。

## 必须说明的实验边界

- FABLE 当前使用 24 个输出 chunk，尚未复用查询与 setup，时间和通信没有优化。
- SIGMA 运行时 A100-2 utilization 为 99%，`timing_reliable=false`。
- 当前只运行 XLM-R Encoder Layer 0，尚未运行全部 12 层。
- 当前 word embeddings 直接进入 Encoder Block，尚未加入 position embeddings 和 embedding LayerNorm。
- 因此可以汇报“FABLE+SIGMA 安全链路已跑通”，不能表述为“完整 XLM-R 推理已经实现”。
- 当前尚未实现同隐私目标、同工作负载的唯一 baseline，不能宣称已经获得性能加速。

## 推荐汇报表述

> 我们已经实现 FABLE 与 SIGMA 的安全组合。FABLE 在不公开 Token IDs 和 Embedding 表的情况下完成了 512×768 的私密 Embedding 查询，并将输出转换为 SIGMA 所需的 50-bit 掩码表示。393,216 个查表值全部正确，四条 128-token 序列均成功进入真实 XLM-R Encoder Block，双方输出全部一致，且 sequence 0 与 correctness oracle 逐字节一致。当前实现仍采用 24 个输出分片，且只运行一个 Encoder Block，因此该结果用于证明安全链路与数值正确性，后续还需要进行分片复用优化、补全 Embedding 后处理和 12 层 Encoder，再与统一 baseline 做正式性能比较。
