# FABLE + SIGMA：基线查表实现方案

日期：2026-07-19
目标：在已经跑通的 `FABLE → 50-bit 算术份额 → SIGMA 掩码输入 → XLM-R Encoder Layer 0` 链路之外，实现一个功能、隐私目标和下游计算都对齐的“基线查表 + SIGMA”，用于正式对照实验。

## 一、先给结论

建议把正式主基线定为：

> **基于秘密分享矩阵乘法的安全线性扫描（SS-LinearScan） + 现有 FABLE→SIGMA 适配器 + 原样 SIGMA。**

其核心是把每个私密 token id 表示成私密 one-hot 行向量，再与私密 Embedding 表做安全矩阵乘法：

\[
X\in\{0,1\}^{q\times N},\quad E\in\mathbb{Z}_{2^{50}}^{N\times d},\quad Y=XE.
\]

- `X` 由输入方 P1 持有，不能让 P0 看见；
- `E` 由模型方 P0 持有，不能让 P1 看见；
- `Y` 以 50-bit 算术份额输出，不能在查表与 SIGMA 之间明文重构；
- 下游继续调用已经验证过的 share→mask bridge 和同一个 SIGMA Encoder Block。

这是最适合作为“工程主基线”的原因：

1. 它与当前任务完全同功能：输出每个 token 的 768 维向量并保留 token 顺序，而不是只输出一句话的 embedding 求和。
2. 它与 FABLE 的隐私目标一致：查询 id、Embedding 表和中间输出都不公开。
3. 它可以复用 SIGMA/现有 2PC 系统里的秘密分享矩阵乘法、Beaver triple、GPU GEMM 和通信统计设施，不必先复现整套 SP-LUT+ 或 DORAM。
4. 它本身是典型的 `O(qNd)` 全表扫描方案，正好体现 FABLE 希望优化掉的基线瓶颈。

同时保留两个辅助对照，但不要把它们冒充正式安全主基线：

| 名称 | 用途 | 是否满足相同安全目标 |
|---|---|---|
| PlainGather + SIGMA | 正确性 oracle 和非安全性能下界 | 否，P0 会知道 token id 或客户端会拿到明文 embedding |
| 2P-DUORAM+ / SP-LUT+ + SIGMA | 与 FABLE 论文协议对齐的参考组 | 是，但实现成本高，可先做小规模或第二阶段 |

## 二、为什么不能直接用 SIGMA 自带 LUT 当基线

SIGMA 第 4.1.4 节的 LUT 原语是 `LUT_{n,l,T}`，论文明确把 `T` 定义为**公开表**。SIGMA 中 GeLU、exp、inverse、rsqrt 使用的也都是规模较小的函数表，例如 GeLU 只用 256 项表。

当前要比较的是：

- 逻辑表：`250,002 × 768` 的 XLM-R word embedding；
- 查询数：`512`；
- 查询 id 私密；
- Embedding 表也私密；
- 输出必须继续以秘密状态进入 SIGMA。

如果直接调用 SIGMA 的公开 LUT：

1. 会把私密模型表变成公开函数参数，改变威胁模型；
2. SIGMA 的小函数 LUT 与大规模向量 Embedding 查询不是同一类负载；
3. 用 256 项 GeLU 表做出的结论不能外推到 25 万项 Embedding 表。现有实验已经证明 FABLE 不适合替换这种小表：256 项逻辑表因实现约束扩到 65,536 项后，协议时间和通信都严重膨胀。

因此，正式基线必须直接针对“私密大表 + 私密索引 + 私密向量输出”。

## 三、公平比较必须锁死的功能合同

### 3.1 参与方与威胁模型

- P0：模型方，持有私密 Embedding 表 `E`；
- P1：输入方，持有私密 token ids；
- Dealer：沿用 SIGMA 的预处理方，只生成随机掩码、FSS keys 或 Beaver triples；默认不接触明文 token ids，主基线也应尽量不让其接触明文 Embedding 表；
- 对 P0/P1 采用静态、半诚实、至多腐化一方的安全模型；
- 不允许 P0/P1 合谋；
- 不把 TEE、恶意安全、侧信道和拒绝服务临时混入本轮 lookup 对照。

注意：FABLE lookup 本身不依赖 trusted dealer，但整个 FABLE+SIGMA 链路已经依赖 SIGMA dealer。主基线使用 dealer 生成随机 Beaver triples，不会扩大当前端到端系统的既有信任假设。为了表述严谨，论文中应称其为“dealer-assisted SS-LinearScan baseline”。

### 3.2 固定工作负载

正式 A/B 对照必须使用完全相同的：

- 模型：`FacebookAI/xlm-roberta-base`；
- revision：`e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`；
- Embedding 表：`250,002 × 768`；
- 表元素：signed int16，定点 scale=12；
- SIGMA ring：`Z_(2^50)`；
- 输入：同一组 `4 × 128 = 512` token ids；
- 输出：`512 × 768 = 393,216` 个 50-bit ring elements；
- 下游：同一套 XLM-R Encoder Layer 0、相同权重、相同形状和相同 SIGMA 参数；
- 网络、GPU、线程数、重复次数和计时规则。

不要一边比较 lookup，一边更换 token 分布、模型、维度、bitwidth 或隐私目标。

### 3.3 输出接口

两个 lookup 后端都实现同一个接口：

```text
secure_lookup(
    private_ids,          # q 个 token id
    private_table,        # N × d
    logical_N,
    output_dim=d,
    ring_bits=50,
    fixed_scale=12
) -> (y_share_p0, y_share_p1)

要求：
y_share_p0 + y_share_p1 = Embedding[ids] mod 2^50
```

然后二者都经过同一个：

```text
(y_share_p0, y_share_p1)
    -> share_to_sigma_mask(...)
    -> public masked value y+r
    -> SIGMA Encoder Layer 0
```

## 四、主基线 SS-LinearScan 的具体协议

### 4.1 明文功能

对第 `i` 个 token id `t_i`，定义：

\[
X[i,j]=\mathbf{1}[t_i=j].
\]

则：

\[
Y[i,:]=\sum_{j=0}^{N-1} X[i,j]E[j,:]=E[t_i,:].
\]

这个表达式保留每个 token 的位置和对应向量，能够直接进入 Transformer。不能把 512 个 token 压成 word-count 向量，因为那只适合 bag-of-words 或 embedding 求和，会丢失 token 顺序，无法作为 XLM-R Encoder 的等价输入。

### 4.2 输入秘密分享

P1 已经知道 token ids，因此可以在本地按块生成 one-hot，不需要通过昂贵的安全相等比较来构造选择向量。

对每个 row block `[s, s+B)`：

1. P1 生成 `X_b ∈ {0,1}^{q×B}`；
2. P1 随机采样 `X_b,0`；
3. 令 `X_b,1 = X_b - X_b,0 mod 2^50`；
4. P1 把 `X_b,0` 发给 P0，自己保留 `X_b,1`。

单个份额统计上随机，不泄露 token id。半诚实模型下不需要额外证明每行恰好只有一个 1；如果以后扩展到恶意客户端，再增加 one-hot 合法性验证。

Embedding 表同理由 P0 做算术秘密分享：

1. 对 int16 定点值做二补码符号扩展，映射到 `Z_(2^50)`；
2. P0 随机采样 `E_0`；
3. 令 `E_1 = E - E_0 mod 2^50`；
4. P0 保留 `E_0`，把随机份额 `E_1` 发给 P1。

`E` 是静态模型参数，表份额可以跨请求保留；但后续乘法使用的 Beaver triples 和输入掩码绝对不能跨运行复用。

### 4.3 用 Beaver 矩阵乘法得到输出份额

Dealer 为每个 block 生成随机矩阵：

\[
A\in Z_{2^{50}}^{q\times B},\quad
B_m\in Z_{2^{50}}^{B\times d},\quad
C=A B_m,
\]

并把 `A、B_m、C` 分成两方份额。

在线阶段两方计算并打开：

\[
D=X_b-A,\quad F=E_b-B_m.
\]

随后本地计算结果份额：

\[
[Z_b]=[C]+D[B_m]+[A]F+DF.
\]

其中 `DF` 只由约定的一方加入，避免重复。最后累加所有 block：

\[
[Y]=\sum_b[Z_b].
\]

由于 one-hot 的有效值是 0/1，这里是选择而不是普通定点乘法。首选实现是：

- one-hot scale = 0；
- Embedding scale = 12；
- 输出 scale = 12；
- lookup matmul 后**不做定点截断**。

如果现有 SIGMA 线性层内核硬编码两侧都是 scale=12，可以退而求其次：

- 把 one-hot 的 1 编码为 `2^12`；
- 乘法后对 `q×d` 输出做一次 faithful truncation by 12；
- 禁止使用 CrypTen 风格的 local truncation，因为 SIGMA 论文已经指出它不满足标准 2PC 安全。

两种实现必须在 debug 模式下逐元素与明文 gather 对齐后，才能进入性能实验。

### 4.4 流式分块，避免显存爆炸

绝对不要一次物化完整 `512 × 250,002` one-hot 矩阵和整套 triple。建议默认 `B=4096` 行：

```text
y_share = 0
for s in range(0, N, 4096):
    e = min(s + 4096, N)
    X_block_share = share_one_hot_block(ids, s, e)
    E_block_share = load_table_share_block(s, e)
    triple = load_fresh_triple(run_id, s, e)
    Z_block_share = secure_matmul_no_trunc(
        X_block_share,
        E_block_share,
        triple
    )
    y_share += Z_block_share
return y_share
```

实现注意点：

- 最后一块补零到 block size，但日志必须同时记录 `logical_N=250002` 和 `physical_N`；
- block 内使用现有 CUDA GEMM；
- CPU→GPU 传输与网络通信使用双缓冲，把 block `k+1` 的加载和 block `k` 的计算重叠；
- 表份额可静态存储，triple 按 `run_id/block_id` 一次性消费；
- 每消费一个 triple 就落盘标记，崩溃重跑必须使用新 triple，不能“从中间继续”并复用旧随机性；
- 通信计数器要分别统计 lookup preprocessing、lookup online、bridge 和 SIGMA online。

### 4.5 当前工作负载的量级

在 `N=250,002、q=512、d=768` 下：

- one-hot 元素数：`128,001,024`；
- Embedding 表元素数：`192,001,536`；
- 输出元素数：`393,216`；
- 安全矩阵乘法约需：`98,304,786,432` 次 ring MAC，约 983 亿次；
- one-hot 按 50-bit 紧凑打包约 `0.800 GB`，用 uint64 存储约 `0.954 GiB`；
- 表按 50-bit 紧凑打包约 `1.200 GB`，用 uint64 存储约 `1.431 GiB`；
- 仅打开 Beaver 的 `D` 与 `F`，两方合计理论发送量约为 `4.000 GB`（50-bit 紧凑打包）或 `5.120 GB`（uint64 直接发送），尚未包含协议头、同步和其他掩码数据。

这些数字说明：该基线可能计算很慢，但它是一个语义正确、隐私一致、可解释的基线。基线的职责不是一定要快，而是提供可信参照。

## 五、怎样接入现有 SIGMA 链路

### 5.1 只替换 lookup，不改下游

当前已验证链路是：

```text
private ids + private table
    -> FABLE
    -> 512×768 arithmetic shares over Z_(2^50)
    -> share→mask
    -> SIGMA masked input
    -> XLM-R Encoder Layer 0
```

基线链路应为：

```text
same private ids + same private table
    -> SS-LinearScan
    -> same 512×768 arithmetic shares contract
    -> same share→mask implementation
    -> separately generated fresh SIGMA preprocessing material
    -> same XLM-R Encoder Layer 0
```

不要为基线重新写一套 share→mask，也不要让基线在进入 SIGMA 前明文重构 Embedding。

### 5.2 SIGMA keys 是否可以复用

正式安全实验中，FSS keys、输入/输出 masks 和 Beaver triples都按一次性预处理材料处理，不在 FABLE 组和 baseline 组之间复用。

为了控制变量：

- 两组使用相同参数、相同模型权重、相同 token ids；
- 分别生成独立但同分布的预处理材料；
- 分阶段报告耗时，不用“复用同一批 keys”人为压低某一组；
- 正确性比较使用解码后的 oracle 输出或 hash，而不是要求两组 masked tensor 的字节完全相同。

## 六、辅助基线如何放置

### 6.1 B0：PlainGather + SIGMA

实现：直接执行 `E[ids]`，随后人为秘密分享输出并进入同一个 bridge 和 SIGMA。

用途：

- 明文正确性 oracle；
- lookup 非安全速度下界；
- 检查下游 SIGMA 与模型语义。

报告中必须标成 `insecure lower bound / correctness oracle`，不能与 FABLE 做安全协议 speedup 结论。

### 6.2 B2：2P-DUORAM+ + SIGMA

这是 FABLE 论文第 3.2 节给出的 strawman，更贴近论文的 confidential LUT 接口：

1. 查询索引以 XOR share 表示，P0 的 share 是随机旋转量 `R`；
2. P0 按 `R` 旋转表；
3. P0 用新鲜输出 mask 隐藏表项；
4. P1 用另一份索引 share 发起 CSPIR；
5. P1 得到 masked item，P0 保留 mask，两者形成输出份额；
6. 输出再转换为 50-bit 算术份额并进入现有 bridge + SIGMA。

它的优点是安全接口直接；缺点是每个查询都要做 PIR，服务器工作随 `qN` 增长，没有 FABLE 的 batch PIR 摊销。若 Spiral/CSPIR 代码尚未准备好，可以把它放到第二阶段，不要阻塞主基线。

### 6.3 B3：SP-LUT+ + SIGMA

SP-LUT+ 是 FABLE 论文正式使用的 MPC LUT baseline，保护表内容且客户端较轻。但它的通信随 `N × 输出位数` 线性增长。

当前每行是 768 个 int16，即 12,288 bit。只按原始表 payload 粗略计算：

- 每个查询扫描全部表约 `384 MB`；
- 512 个查询约 `196.6 GB`；
- 如果直接把每个输出元素扩成 50-bit，再做相同计算会更大。

这还没有包含 OT、协议元数据和多轮开销。因此建议：

- 先在 `N=2^12、2^14、2^16` 和较小 `d` 上实测；
- 验证线性趋势；
- 对完整规模只做有明确公式和实测校准的外推；
- 不把“超时/跑不动”替换成臆测的完整规模实测值。

## 七、正式实验矩阵

### 7.1 方法组

1. `PlainGather + SIGMA`：非安全下界；
2. `SS-LinearScan + SIGMA`：主安全基线；
3. `FABLE-current + SIGMA`：当前 correctness-first、24 chunks 实现；
4. `FABLE-optimized + SIGMA`：复用 query translation/setup、减少 24 次重复后的正式方案；
5. `2P-DUORAM+` 或 `SP-LUT+`：第二阶段论文对齐参考。

### 7.2 参数组

至少覆盖：

| 变量 | 建议取值 |
|---|---|
| `N` | `2^16、2^18、250002` |
| `q` | `128、256、512、1024、4096` |
| `d` | `32、128、768` |
| token 分布 | 100% unique；真实 XLM-R 自然重复；人为控制 25%/50% 重复率 |
| row block | `1024、4096、8192` |
| 网络 | 当前真实 GPU 网络；若可控再增加 FABLE 论文式 LAN/WAN |

为什么必须扫 `q`：FABLE 论文报告中，FABLE 超过 SP-LUT+ 的批量阈值约在 `2^8` 之后，而超过 DORAM 协议约在 `2^10` 之后。当前 `q=512=2^9` 正处于两者之间，不能预设 FABLE 必然击败所有安全 lookup baseline。

为什么必须区分 unique 与重复输入：FABLE 内部有 deduplication/expansion，重复率会影响实际 batch 结构；SS-LinearScan 的主要计算量基本仍由 `qNd` 决定。只用 512 unique ids 会漏掉真实文本的重复特征。

### 7.3 指标

每组至少记录：

#### 正确性

- lookup 输出元素数与 mismatch 数；
- share→mask mismatch；
- 双方最终输出是否一致；
- 至少一个 sequence 与明文/零掩码 oracle 是否一致；
- 输出 hash、模型 revision、输入 hash、表 hash。

#### Lookup 离线成本

- 表份额生成时间与存储；
- Beaver triple / PIR setup / FABLE server setup 时间；
- 预处理材料生成、传输和磁盘占用；
- 预处理材料是否可跨请求安全复用。

#### Lookup 在线成本

- wall time；
- P0/P1 分别计算时间；
- 两方发送字节总量；
- rounds / 同步次数；
- GPU 峰值显存、CPU 峰值内存；
- queries/s、bytes/query；
- 每个输出 value 的字节数。

#### 组合链路

- lookup；
- share→mask；
- SIGMA dealer generation/transfer；
- SIGMA online；
- 顺序组合总时间与总通信；
- lookup 在端到端中的占比。

## 八、建议的实现顺序

### M0：冻结数据与接口

- 固定当前 512 ids、XLM-R revision、Embedding table hash；
- 固定 50-bit ring、scale=12；
- 把现有 FABLE 输出 contract 写成单元测试；
- 保存一份 PlainGather oracle。

完成标准：相同 ids 与 table 生成固定的 `512×768` oracle hash。

### M1：实现小规模 SS-LinearScan

- 先用 `N=4096、q=8、d=32`；
- CPU 版或现有 secret matmul 内核；
- 验证负数二补码、scale 和 ring wraparound；
- 输出 50-bit 算术份额。

完成标准：全部元素零 mismatch，并且任何一方单独查看份额不能恢复输入或表。

### M2：流式 GPU 版

- 加入 row block；
- 默认 `B=4096`；
- 加入 triple 一次性消费机制；
- 记录每 block 的 compute、communication、I/O；
- 做 `B=1024/4096/8192` 调优。

完成标准：`N=250002、q=512、d=768` 不 OOM，能够完成一次全链路 correctness run。

### M3：接入现有 bridge + SIGMA

- 不修改 bridge；
- 不明文重构中间 embedding；
- 跑 XLM-R Layer 0；
- 复用现有四项检查：lookup 全量、bridge 全量、P0/P1 一致、oracle 一致。

完成标准：与当前 FABLE+SIGMA 的正确性标准完全相同。

### M4：正式性能实验

- GPU utilization 在启动前和运行中都低于预设阈值，建议 `≤30%`；
- warm-up 后独立运行至少 5 次；
- 报告 mean、std、min/max；
- 预处理与在线阶段严格分开；
- 使用同一网络、同一两张 GPU、同一进程绑定；
- FABLE 先完成 24 chunks 的 query/setup 复用优化，再给最终 speedup。

## 九、最容易犯的错误

1. **把公开 LUT 当私密 Embedding 基线。** 这会直接改变研究问题。
2. **用 word-count × embedding 代替逐 token embedding。** 前者丢失顺序，不能进入 Transformer。
3. **为了方便让 P0 看见 token id。** 这会把安全 lookup 退化成普通 gather。
4. **为了方便让 P1 拿到明文 embedding。** 这会泄露模型表项，且破坏后续秘密计算接口。
5. **查表后明文重构再重新分享。** 即使最终 SIGMA 是安全的，中间泄露已经发生。
6. **one-hot 使用 scale=12，却忘记 faithful truncation。** 输出会额外放大 `2^12`。
7. **one-hot 使用 scale=0，内核却仍自动截断 12 位。** 输出会被错误缩小。
8. **复用 Beaver triples、FSS keys 或 masks。** 这不是性能优化，而是破坏安全性。
9. **只报 online，不报 dealer/triple/storage。** SIGMA 和 SS-LinearScan 的离线成本都很大，必须显式列出。
10. **拿当前 FABLE 191.5 GB/738.9 s 当最终值。** 当前实现重复 24 个输出 chunk 的查询/setup，是 correctness-first 结果。
11. **只测 512 unique ids。** 真实文本重复率和 FABLE dedup 行为都没有被覆盖。
12. **把 `q=512` 下的结果外推到更大 batch。** FABLE 的主要优势来自 batch amortization，必须实际扫 `q`。

## 十、最终论文/汇报中的推荐表述

可以这样写：

> 为保证对照组与 FABLE 具有相同的功能和隐私目标，我们实现了基于秘密分享矩阵乘法的安全线性扫描基线。输入方将私密 token ids 表示为秘密分享 one-hot 矩阵，模型方将私密 Embedding 表表示为算术份额，双方利用一次性 Beaver triples 计算 `X·E` 并得到 50-bit 输出份额。该输出通过与 FABLE 相同的 share-to-mask 适配器进入未修改的 SIGMA Encoder Block。该基线保持查询、表和中间 embedding 的隐私，但需要对整个表执行 `O(qNd)` 的安全扫描，因此可作为衡量 FABLE 批量机密查表收益的直接参照。

现阶段不能写：

> FABLE+SIGMA 已经比 baseline 更快。

原因是正式 baseline 尚未按相同口径实测，当前 FABLE 仍是 24 chunks 的 correctness-first 实现，SIGMA 当次运行还受到 GPU 高负载干扰。

## 十一、依据

- Zotero：FABLE，条目 `ELPT99NH`，重点依据第 3.1、3.2、6.3、6.4 节；
- Zotero：SIGMA，条目 `3YKCP9K8`，重点依据第 2.2、2.4、4.1.4、5、附录 L；
- 本地阶段结果：`FABLE_GELU替换前后对比.xlsx`；
- 本地阶段结果：`FABLE_SIGMA完整链路实验汇报.xlsx`。
