# Byte-level BPE 算法完整逻辑

## 一、算法概述

### 作用

BPE（Byte-Pair Encoding，字节对编码）是大语言模型（LLM）的**核心分词算法**，负责将人类可读的文本转换为模型可处理的整数序列（token IDs），以及反向转换。它是连接"语言"与"模型"的桥梁——模型不直接处理文字，只处理数字，而 BPE 就是这个翻译官。

BPE 的核心优势在于：
- **无损编码**：byte-level 设计保证任何文本都能被编码，几乎不会出现 `<unk>`（未知 token）
- **自适应压缩**：通过从语料中学习高频子词，实现文本的高效压缩（一个 token 可代表多个字符）
- **多语言通用**：同一份 tokenizer 可处理英文、中文、代码、符号等多种语言/格式

### 线上大模型的典型规模与语料

| 维度 | 典型值 | 说明 |
|------|--------|------|
| **词表大小** | 32K ~ 256K | LLaMA 系列 32K，LLaMA 3 为 128K，Gemma 为 256K |
| **合并次数 (num_merges)** | ≈ 词表大小 - 256 | 初始 256 个单字节 token |
| **Tokenizer 训练语料** | 数百 GB ~ 数 TB 原始文本 | 以原始字符/字节为单位，如 Common Crawl 等公开数据集 |
| **模型训练语料** | 数千亿 ~ 数万亿 tokens | 经过 tokenizer 编码后的 token 数，GPT-3 约 300B tokens，GPT-4 估 1T+ tokens |
| **预训练数据来源** | 网页、书籍、代码、论文、对话 | 以公开爬虫数据为主，辅以高质量人工数据 |
| **Embedding 维度** | 768 ~ 4096 | 词表越大，embedding 参数量越大（= vocab_size × embed_dim） |
| **推理速度** | 单 token 生成 10~50ms | 取决于模型规模和硬件，tokenizer 本身只占极少量耗时 |

> **注意**：Tokenizer 训练和模型训练的语料单位不同。Tokenizer 训练使用原始文本（字符/字节），
> 模型训练使用经过 tokenizer 编码后的 token 序列。1 个 token 通常对应 1~1.5 个英文单词或 1~2 个中文字符。

---

## 二、字节↔Unicode 双向映射（bytes_to_unicode）

**目标**：将 256 个字节值（0~255）一一映射到 256 个可打印 Unicode 字符，供 BPE 算法在 `String` 层面操作。

### 映射规则

1. `0x21`~`0x7E`（94 个可打印 ASCII）：直接映射为自身码点，UTF-8 字节数 **1→1**
2. `0xA1`~`0xAC`（12 个 Latin-1 补充，跳过 `0xAD`）：直接映射为自身，UTF-8 字节数 **1→2**
3. `0xAE`~`0xFF`（82 个 Latin-1 补充）：直接映射为自身，UTF-8 字节数 **1→2**
4. 剩余 68 个不可打印字节：映射到 `U+0100` 起始的连续码点，UTF-8 字节数 **1→2**

### 关键特性

- 映射是严格的**一对一关系**（一个字节值 → 一个 Rust `char`），不存在一对多或多对一
- 映射后改变了 UTF-8 字节长度，但**不影响算法正确性**——BPE 操作在 `char`/`String` 层面进行，解码时通过反向映射 `u2b` 可精确还原原始字节值
- 返回 `(b2u, u2b)`：`b2u: HashMap<u8, char>` 字节→字符，`u2b: HashMap<char, u8>` 字符→字节

---

## 三、预分词（pretokenize）

**目标**：将原始文本按词/标点边界**贪婪切割**为 pre-token 列表，防止 BPE 跨词合并。

### 正则模式（按优先级贪婪匹配）

```
空白符 | 英文缩写后缀('s|'t|'re|'ve|'m|'ll|'d) | 词字符序列 | 数字序列 | 标点符号序列
```

每项都是贪婪匹配（正则用 `+`），如 `"hh1122ddd"` → `["hh", "1122", "ddd"]`。

### 处理逻辑

1. 用正则 `find_iter` 获取所有匹配项
2. 遍历每个匹配：
   - 若匹配是**空白字符**：累加到 `pending_spaces`（空白符**永远不作为独立 token**）
   - 若匹配是**非空 token**：
     - 若 `pending_spaces > 0` 或当前非首个 token：添加前缀 `Ġ`（U+0120，GPT-2 空格占位符）
     - 若是**文本首 token** 且无前导空格：不加前缀
     - 追加当前匹配内容到 token，重置 `pending_spaces = 0`
3. 返回 pre-token 字符串列表

### 关键设计

- 空格信息通过 `Ġ` 前缀编码进 token 本身，使 BPE 能学到 `Ġthe`、`Ġcat` 等带前导空格的高频 token
- 纯空格输入会被完全丢弃（`pending_spaces` 累积后无后续非空 token，返回空列表），此场景在生产中由应用层前置校验拦截（前端 `trim()` 检测为空→拒绝提交）

---

## 四、BPE 训练（train_bpe）

**输入**：训练语料 `texts`、最大合并次数 `num_merges`、字节→字符映射 `b2u`
**输出**：`vocab`（token→ID 映射表）、`merges`（有序合并规则列表）

### Step 1：初始化词频统计

构建 `word_freqs: HashMap<Vec<String>, usize>`，key 是"词"的序列表示，value 是出现频率。

遍历每条文本 → `pretokenize` → 每个 pre-token：
1. 转为 UTF-8 原始字节序列 `Vec<u8>`
2. 每个字节通过 `b2u` 映射为单个 Unicode `char` 再转为 `String`
3. 末尾追加字面字符串 `"</w>"`（词结束标记，防止跨词合并）
4. 以该 `Vec<String>` 为 key，累加频率

同时构建 `vocab_set: HashSet<String>`，收集所有初始 token。

### Step 2：迭代合并（最多 num_merges 次）

每次迭代：

#### (a) 统计 pair 频率
- 遍历 `word_freqs` 中每个 `(word, freq)`
- 对 word 中每对相邻 token `(word[j], word[j+1])`，以 `freq` 为权重累加到 `pair_freqs`

#### (b) 选最高频 pair
- 取 `pair_freqs` 中频率最大的 pair
- 新 token = `pair.0 + pair.1`（字符串拼接）

#### (c) 执行合并
- 遍历 `word_freqs` 每个 `(word, freq)`，从左到右贪心扫描：
  - 若 `word[j] == pair.0 && word[j+1] == pair.1`：push 新 token，j 跳 2
  - 否则：push `word[j]`，j 跳 1
- 用合并后的 `new_word_freqs` 替换 `word_freqs`
- 将 `(best_pair, new_token)` 追加到 `merges`（**记录了本次合并的两个相邻 token 和合并结果**）
- 将 `new_token` 加入 `vocab_set`

#### (d) 提前终止
`pair_freqs` 为空时 break

### Step 3：构建最终 Vocab

按以下顺序分配 token ID：

1. **Special tokens**（固定 ID 0~3）：`<unk>=0, <pad>=1, <bos>=2, <eos>=3`
2. **Merged tokens**（按 `merges` 顺序）：遍历 `merges` 中的 `new_token`，未分配过的依次分配 `vocab.len()` 作为 ID
3. **剩余 token**（按字典序）：`vocab_set` 排序后，未分配过的依次分配

---

## 五、BPE 编码 / 推理（encode）

**输入**：待编码文本 `text`、合并规则 `merges`、词表 `vocab`、字节→字符映射 `b2u`
**输出**：token ID 列表

### 逻辑

1. 遍历 `pretokenize(text)` 的每个 pre-token：
   - 转为 UTF-8 字节序列 → `b2u` 映射为 Unicode 字符串列表 → 末尾追加 `"</w>"`
   - **按 `merges` 顺序依次应用每条规则**：
     - 从左到右贪心扫描 tokens
     - 若相邻两个 token 匹配规则的 `pair`：替换为 `merged`，跳 2
     - 否则：保留当前 token，跳 1
     - 每条规则应用完后，tokens 更新为 `new_tokens`，再应用下一条规则
   - 将最终 tokens 中每个 token 通过 `vocab` 映射为 ID（不在 vocab 中则用 `<unk>` 的 ID）
2. 返回所有 token ID 的拼接列表

### 兜底保证

由于 byte-level BPE 的初始 256 个单字节 token 覆盖了所有字节值，任何文本都能被无损编码，**`<unk>` 几乎不会被触发**。

---

## 六、BPE 解码（decode）

**输入**：token ID 列表 `ids`、词表 `vocab`、字符→字节映射 `u2b`
**输出**：原始文本字符串

### 逻辑

1. 构建 `id2tok: HashMap<usize, String>`（vocab 的反向映射，ID→token 字符串）
2. 遍历每个 ID：
   - 若对应 token 是 `<unk>/<pad>/<bos>/<eos>`：跳过（不携带文本信息）
   - 否则：遍历 token 中每个字符
     - `Ġ`(U+0120) → 空格字节 `0x20`
     - 其他字符 → 通过 `u2b` 映射还原为原始字节值
3. 将所有字节拼接后用 `String::from_utf8_lossy` 解码为 UTF-8 字符串
4. 移除所有 `</w>` 字面串 + 将残留 `Ġ` 替换为空格

---

## 七、核心流程总结

```
训练阶段:
  原始语料 → pretokenize → 字节序列+b2u映射+</w> → 统计词频
    → 迭代合并最高频pair(num_merges次) → 记录merges规则 → 构建vocab

推理阶段(编码):
  输入文本 → pretokenize → 字节序列+b2u映射+</w> → 按merges规则贪心合并
    → vocab查表 → 输出token ID列表

推理阶段(解码):
  token ID列表 → id2tok反向查表 → u2b还原字节 → UTF-8解码 → 输出文本
```

---

## 八、关键设计决策汇总

| 决策 | 原因 |
|------|------|
| Byte-level 而非 Char-level | 保证任何文本都能被无损编码，`<unk>` 几乎不会触发 |
| 空格用 `Ġ` 前缀编码 | 使 BPE 能学到 `Ġthe`、`Ġcat` 等带前导空格的高频 token |
| `</w>` 词尾标记 | 防止跨词合并，确保合并只在词内部进行 |
| 映射改变 UTF-8 字节长度 | 不影响算法，映射是一对一的，解码可精确还原 |
| 纯空格输入被丢弃 | 由应用层前置校验拦截，不影响 tokenizer 设计 |
| 合并规则按顺序应用 | 高频合并优先，实现贪心策略 |
| Special tokens 固定 ID 0~3 | 兼容主流 LLM 训练框架 |
