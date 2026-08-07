/// Byte-level BPE Tokenizer (GPT-2 风格) — Rust 实现
///
/// 本程序实现了一个完整的 Byte-level BPE (Byte Pair Encoding) 分词器，
/// 采用与 GPT-2/3/4 相同的分词策略。
///
/// BPE 的核心思想:
///   - 初始时，每个字节(或字符)是一个独立的 token
///   - 反复统计语料中最高频的相邻 token 对(pair)，将其合并为一个新 token
///   - 每次合并使 vocab 增加一个 token，直到达到目标词表大小
///
/// 覆盖三个阶段:
///   1. 训练 (train_bpe):
///      从原始文本学习 merge rules + vocab
///      - 先做 pre-tokenize（按词/标点边界粗切）
///      - 将每个 pre-token 编码为字节序列
///      - 迭代合并最高频的相邻字节对
///
///   2. 编码 (encode):
///      用学到的 merge rules 对新文本做贪心分词 → token ids
///      - 按 merge 的先后顺序依次尝试合并（贪心策略）
///
///   3. 查表 (embedding_lookup):
///      模拟推理时的 token id → embedding vector 查找
///      - 真实场景中这是 nn.Embedding 的可学习权重矩阵
///
/// 运行方式:
///   cargo run --release                                    # 使用内置小语料
///   cargo run --release -- <语料文件路径> <num_merges>      # 使用外部语料
///
/// 示例:
///   cargo run --release -- alice.txt 5000                  # Alice in Wonderland, 5000 次合并

use regex::Regex;
use std::collections::HashMap;
use std::env;
use std::fs;

// ============================================================
// GPT-2 风格的 bytes ↔ unicode 映射
// ============================================================
//
// 为什么需要这个映射？
// BPE 操作在字节级别进行，但某些字节值(如 0x00-0x1F 控制字符、0x7F DEL、0x80-0x9F C1控制字符)
// 是不可打印的，直接作为 token 名称会导致打印/序列化问题。
//
// 解决方案: 将 256 个字节一一映射到 256 个**可打印**的 Unicode 字符。
// - 可打印 ASCII (! ~)、Latin-1 补充(¡ ¬)、Latin Extended(® ÿ) 直接映射为自身
// - 不可打印字节映射到 U+0100 及以上的 Unicode 字符
//
// 这样就保证了每个字节都有唯一的可打印字符表示，便于调试和存储。
//
// 注意: 这不同于将文本编码为 UTF-8。B2U/U2B 映射只是给每个字节值(0-255)
// 分配一个"显示名称"，方便在 HashMap 的 key 中使用。

/// 构建 bytes ↔ unicode 双向映射表。
/// 返回 (b2u, u2b):
///   - b2u: HashMap<u8, char> — 字节值(0-255) → 可打印 Unicode 字符
///   - u2b: HashMap<char, u8> — 可打印 Unicode 字符 → 字节值(0-255)
fn bytes_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    let mut cs: Vec<u32> = Vec::new();

    // 第一批: 可打印 ASCII 字符 (0x21 '!' ~ 0x7E '~')，共 94 个
    // 这些字节直接映射为自身的 Unicode 码点
    for b in b'!'..=b'~' {
        bs.push(b);
        cs.push(b as u32);
    }

    // 第二批: Latin-1 Supplement 中的可打印部分 (0xA1 '¡' ~ 0xAC '¬')，共 12 个
    // 跳过 0xAD (soft hyphen，不可见)
    for b in 0xA1..=0xAC {
        bs.push(b);
        cs.push(b as u32);
    }

    // 第三批: Latin-1 Supplement 剩余可打印部分 (0xAE '®' ~ 0xFF 'ÿ')，共 82 个
    // 0xAE 起跳是因为跳过了 0xAD
    for b in 0xAE..=0xFF {
        bs.push(b);
        cs.push(b as u32);
    }

    // 以上三批共覆盖 94 + 12 + 82 = 188 个字节
    // 剩余 256 - 188 = 68 个不可打印字节，映射到 U+0100 (256) 及以上
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n); // 映射到 U+0100, U+0101, U+0102, ...
            n += 1;
        }
    }

    // 构建双向 HashMap
    let b2u: HashMap<u8, char> = bs
        .iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (b, char::from_u32(c).unwrap()))
        .collect();
    let u2b: HashMap<char, u8> = b2u.iter().map(|(&b, &c)| (c, b)).collect();
    (b2u, u2b)
}

// ============================================================
// Pre-tokenize (GPT-2 风格)
// ============================================================
//
// Pre-tokenize 是 BPE 的第一步: 将原始文本按词/标点边界粗切为"准 token"。
//
// 为什么要 pre-tokenize？
// 如果直接在整段文本上做 BPE，空格可能被合并进 token（如 "th e" 合并成 "the"），
// 导致模型学到无意义的跨词 token。Pre-tokenize 确保 BPE 只在合理的边界内合并。
//
// GPT-2 的 pre-tokenize 策略:
// 1. 用正则将文本切分为: 词、数字、标点、缩写后缀、空白
// 2. 空白不作为独立 token，而是作为 Ġ (U+0120) 前缀附加到下一个词头
//    例如 "Hello world" → ["Hello", "Ġworld"]
//    这样空格信息被编码进了 token 本身，BPE 可以学到 "Ġthe" 这种带前导空格的 token

/// GPT-2 风格 pre-tokenize: 将文本切分为带空格前缀的 pre-token 列表。
///
/// 正则各部分含义:
///   's|'t|'re|'ve|'m|'ll|'d  — 英文缩写后缀（如 don't → "don" + "'t"）
///   [\w\u{...}]+              — 词字符（ASCII + 拉丁扩展 + 中文 + 日文 + 韩文 + 西里尔 + 阿拉伯 + 泰文 + 梵文等）
///   \d+                       — 数字序列（如 "2024"）
///   [^\s\w\u{...}\d]+         — 以上都不匹配的字符（标点符号等）
///   \s+                       — 空白字符（空格、换行等，用于识别词边界）
fn pretokenize(text: &str) -> Vec<String> {
    // 定义所有需要支持的 Unicode 字符范围，用于正则中的"词字符"匹配。
    // 注意: 这个范围会同时出现在"匹配词"和"排除词(匹配标点)"两个位置。
    //
    // 覆盖的语言/文字:
    //   \w                    — ASCII 字母、数字、下划线 (a-z, A-Z, 0-9, _)
    //   \u{00C0}-\u{024F}     — Latin Extended (é, ñ, ü, ç 等欧洲语言带口音字母)
    //   \u{0400}-\u{04FF}     — Cyrillic (俄语、乌克兰语等: А, Б, В, а, б, в)
    //   \u{0600}-\u{06FF}     — Arabic (阿拉伯文、波斯文、乌尔都文: ا, ب, ت)
    //   \u{0900}-\u{097F}     — Devanagari (印地语、梵文、马拉地语: अ, आ, क)
    //   \u{0E00}-\u{0E7F}     — Thai (泰文: ก, ข, ค)
    //   \u{3040}-\u{309F}     — Hiragana (日文平假名: あ, い, う)
    //   \u{30A0}-\u{30FF}     — Katakana (日文片假名: ア, カ, サ)
    //   \u{4E00}-\u{9FFF}     — CJK Unified Ideographs (中文汉字: 你, 好, 世)
    //   \u{AC00}-\u{D7AF}     — Hangul Syllables (韩文音节: 한, 글)
    let word_chars = r"\w\u{00C0}-\u{024F}\u{0400}-\u{04FF}\u{0600}-\u{06FF}\u{0900}-\u{097F}\u{0E00}-\u{0E7F}\u{3040}-\u{309F}\u{30A0}-\u{30FF}\u{4E00}-\u{9FFF}\u{AC00}-\u{D7AF}";

    // 构建完整正则:
    //   \s+                    — 先匹配空白（用于识别词边界）
    //   's|'t|...              — 英文缩写后缀
    //   [{word_chars}]+        — 词字符序列（一个或多个连续的词字符）
    //   \d+                    — 数字序列
    //   [^\s{word_chars}\d]+   — 标点符号等其他字符
    let pattern = format!(
        r"(\s+|'s|'t|'re|'ve|'m|'ll|'d|[{wc}]+|\d+|[^\s{wc}\d]+)",
        wc = word_chars
    );
    let re = Regex::new(&pattern).unwrap();

    // 用 find_iter 获取所有匹配项（而非 split，因为 split 返回的是不匹配的部分）
    let matches: Vec<String> = re.find_iter(text).map(|m| m.as_str().to_string()).collect();

    let mut tokens = Vec::new();
    let mut pending_spaces = 0usize; // 累积的空白字符数

    for part in &matches {
        if part.trim().is_empty() && !part.is_empty() {
            // 当前匹配是空白字符，累积空格数，等待下一个非空 token
            pending_spaces += part.len();
        } else {
            // 当前匹配是非空 token
            // 如果前面有空白，或者这不是第一个 token，则添加 Ġ 前缀表示"词边界"
            let prefix = if pending_spaces > 0 || !tokens.is_empty() {
                '\u{0120}' // Ġ (Latin Capital Letter G With Dot) — GPT-2 约定的空格占位符
            } else {
                '\0' // 文本开头的第一个 token 不加前缀
            };
            let mut token = String::new();
            if prefix != '\0' {
                token.push(prefix);
            }
            token.push_str(part);
            tokens.push(token);
            pending_spaces = 0; // 重置空白计数
        }
    }
    tokens
}

/// 将 pre-token 编码为字节序列。
/// 例如: "Hello" → [72, 101, 108, 108, 111]
///       "Ġworld" → [196, 160, 119, 111, 114, 108, 100]  (Ġ = U+0120, UTF-8 为 0xC4 0xA0)
fn pretoken_to_bytes(pretoken: &str) -> Vec<u8> {
    pretoken.as_bytes().to_vec()
}

/// 将 BPE token (内部 Unicode 字符序列) 转换为人类可读的字节字符串。
///
/// 转换过程:
/// 1. 遍历 token 中的每个字符
/// 2. Ġ (U+0120) → 空格 (0x20)
/// 3. 其他字符通过 U2B 映射还原为原始字节值
/// 4. 将字节序列解码为 UTF-8 字符串
/// 5. 剥离 </w> 结尾标记（合并后会变成字面字符 '<' '/' 'w' '>'，需要移除）
/// 6. 将残留的 Ġ (U+0120) 替换为空格（因为 UTF-8 解码后可能还原出 Ġ）
fn token_to_readable(tok: &str, u2b: &HashMap<char, u8>) -> String {
    let mut bytes = Vec::new();
    for c in tok.chars() {
        if c == '\u{0120}' {
            // Ġ (U+0120) 是 GPT-2 的空格占位符，直接转为空格字节
            bytes.push(b' ');
        } else if let Some(&b) = u2b.get(&c) {
            // 通过反向映射还原为原始字节值
            bytes.push(b);
        }
    }
    let text = String::from_utf8_lossy(&bytes).to_string();
    // </w> 在合并过程中变成了 4 个独立的字面字符 '<' '/' 'w' '>'
    // 它们不是特殊标记，需要显式移除
    text.replace("</w>", "").replace('\u{0120}', " ")
}

// ============================================================
// 阶段 1: 训练 Byte-level BPE
// ============================================================
//
// 训练算法流程:
// 1. 对语料做 pre-tokenize，将每个 pre-token 编码为字节序列 + </w> 结尾标记
// 2. 统计所有"词"(字节序列)的出现频率
// 3. 迭代 num_merges 次:
//    a. 遍历所有"词"，统计其中相邻 token 对(pair)的全局频率
//    b. 找到全局最高频的 pair
//    c. 遍历所有"词"，将其中的 best_pair 替换为合并后的新 token
//    d. 记录这次合并规则 (pair → new_token)
// 4. 构建最终 vocab: special tokens + merged tokens(按合并顺序) + 初始单字节 tokens

/// 训练 Byte-level BPE 分词器。
///
/// 参数:
///   texts      — 训练语料，每行一个句子/段落
///   num_merges — 最大合并次数，也约等于最终 vocab 大小 (vocab_size ≈ num_merges + 256 + 4)
///   b2u        — 字节到 Unicode 的映射表
///
/// 返回:
///   vocab  — HashMap<String, usize>，token 字符串 → token ID
///            例如: {"<unk>": 0, "<pad>": 1, ..., "the": 333, "ing": 456, ...}
///   merges — Vec<((String, String), String)>，有序的合并规则列表
///            例如: [(("h", "e"), "he"), (("t", "he</w>"), "the</w>"), ...]
///            列表中靠前的规则优先级更高（编码时先执行）
fn train_bpe(
    texts: &[String],
    num_merges: usize,
    b2u: &HashMap<u8, char>,
) -> (HashMap<String, usize>, Vec<((String, String), String)>) {

    // ---- Step 1: 初始化 — 将语料转为字节序列 + 词频统计 ----
    //
    // word_freqs 的 key 是字节序列(Vec<String>)，每个元素是一个 B2U 映射后的 Unicode 字符。
    // 例如 "Hello" → 字节 [72,101,108,108,111] → B2U 映射后 → ['H','e','l','l','o'] + ["</w>"]
    // value 是该字节序列在整个语料中出现的次数。
    //
    // 为什么要按"词"统计而不是按字节统计？
    // 因为 BPE 合并只在同一个词的内部进行（不会跨词合并），
    // 所以需要知道每个词出现了几次，才能正确计算 pair 的全局频率。
    let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();

    for text in texts {
        for pretoken in pretokenize(text) {
            // 将 pre-token 编码为字节序列
            let token_bytes = pretoken_to_bytes(&pretoken);
            // 将每个字节通过 B2U 映射为可打印的 Unicode 字符
            let mut chars: Vec<String> = token_bytes
                .iter()
                .map(|&b| b2u[&b].to_string())
                .collect();
            // 在末尾添加 </w> 标记，表示词的结尾
            // 这确保 BPE 不会将两个词的末尾/开头合并在一起
            // 例如: "cat" + </w> → ['c','a','t','</w>']
            //       "the" + </w> → ['t','h','e','</w>']
            //       这样 't' 和 '</w>' 不会与下一个词的首字母合并
            chars.push("</w>".to_string());
            *word_freqs.entry(chars).or_insert(0) += 1;
        }
    }

    // 打印初始词频（前 20 个最高频的词）
    println!("=== 初始词频 (字节级) ===");
    let mut sorted_words: Vec<_> = word_freqs.iter().collect();
    sorted_words.sort_by(|a, b| b.1.cmp(a.1));
    for (word, freq) in sorted_words.iter().take(20) {
        let display: String = word
            .iter()
            .map(|c| if c == "</w>" { "<eos>" } else { c.as_str() })
            .collect();
        println!("  {:<40} : {}", format!("{:?}", display), freq);
    }
    println!();

    // ---- Step 2: 迭代合并 ----
    //
    // merges: 记录每次合并的规则，按合并先后顺序排列
    //   格式: [((左token, 右token), 合并后的新token), ...]
    //   例如: [(("Ä", "ł"), "Äł"), (("Äł", "t"), "Äłt"), ...]
    //   编码时按列表顺序依次尝试合并，先出现的规则优先级更高
    let mut merges: Vec<((String, String), String)> = Vec::new();

    // vocab_set: 收集所有出现过的 token（包括初始单字节 token 和合并产生的新 token）
    // 用于最终构建 vocab 时确保不遗漏
    let mut vocab_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for word in word_freqs.keys() {
        for c in word {
            vocab_set.insert(c.clone());
        }
    }

    // 执行 num_merges 次合并迭代
    for i in 0..num_merges {
        // (a) 统计所有相邻 token 对(pair)在全语料中的频率
        //
        // 遍历每个"词"的字节序列，提取所有相邻的 (token[i], token[i+1]) 对，
        // 并乘以该词的出现频率，累加到全局 pair_freqs 中。
        //
        // 例如: word = ['h','e','l','l','o','</w>'], freq = 3
        //   pairs: ('h','e')×3, ('e','l')×3, ('l','l')×3, ('l','o')×3, ('o','</w>')×3
        let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
        for (word, &freq) in &word_freqs {
            for j in 0..word.len().saturating_sub(1) {
                let pair = (word[j].clone(), word[j + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }

        // 如果没有可合并的 pair（所有词都只剩 1 个 token），提前终止
        if pair_freqs.is_empty() {
            break;
        }

        // (b) 找到全局最高频的 pair
        // 如果有多个 pair 频率相同，取第一个（HashMap 迭代顺序不确定，但不影响正确性）
        let (best_pair, _) = pair_freqs.into_iter().max_by_key(|&(_, freq)| freq).unwrap();
        // 合并后的新 token = 左 token + 右 token 的字符串拼接
        // 例如: best_pair = ("h", "e") → new_token = "he"
        let new_token = format!("{}{}", best_pair.0, best_pair.1);

        // (c) 执行合并: 遍历所有"词"，将其中的 best_pair 替换为 new_token
        //
        // 合并使用贪心从左到右扫描:
        //   - 如果当前位置和下一个位置匹配 best_pair，则替换为 new_token，前进 2 步
        //   - 否则保留当前 token，前进 1 步
        //
        // 例如: best_pair = ("l", "l"), new_token = "ll"
        //   "hello</w>" = ['h','e','l','l','o','</w>']
        //   → ['h','e','ll','o','</w>']  (两个 'l' 被合并为 'll')
        let mut new_word_freqs: HashMap<Vec<String>, usize> = HashMap::new();
        for (word, &freq) in &word_freqs {
            let mut merged = Vec::new();
            let mut j = 0;
            while j < word.len() {
                if j + 1 < word.len()
                    && word[j] == best_pair.0
                    && word[j + 1] == best_pair.1
                {
                    merged.push(new_token.clone());
                    j += 2; // 跳过了被合并的两个 token
                } else {
                    merged.push(word[j].clone());
                    j += 1;
                }
            }
            *new_word_freqs.entry(merged).or_insert(0) += freq;
        }

        // 用合并后的词频表替换旧的
        word_freqs = new_word_freqs;
        // 记录这次合并规则
        merges.push((best_pair.clone(), new_token.clone()));
        // 将新 token 加入 vocab_set
        vocab_set.insert(new_token.clone());

        println!(
            "Merge {:>3}: {:>6} + {:<6} → {:?}",
            i + 1,
            format!("{:?}", best_pair.0),
            format!("{:?}", best_pair.1),
            new_token
        );
    }

    // ---- Step 3: 构建最终 vocab (token → ID 映射表) ----
    //
    // Vocab 的 ID 分配策略:
    //   1. Special tokens (固定 4 个): <unk>=0, <pad>=1, <bos>=2, <eos>=3
    //   2. Merged tokens: 按 merge 先后顺序分配 ID（高频合并的子词获得更小的 ID）
    //   3. 初始单字节 tokens: 按字典序排列，补在末尾
    //
    // 为什么 merged tokens 排在前面？
    // 高频子词获得更小的 ID 在某些编码方案中更紧凑，
    // 同时也方便调试时观察哪些是训练过程中学到的子词。

    let special = vec!["<unk>", "<pad>", "<bos>", "<eos>"];
    let mut vocab: HashMap<String, usize> = HashMap::new();

    // 先注册 special tokens (ID 0-3)
    // <unk>: 未知词(encode 时遇到 vocab 中没有的 token 时使用)
    // <pad>: 填充(将不等长的序列补齐到相同长度时使用)
    // <bos>: 序列开始标记(Beginning of Sequence)
    // <eos>: 序列结束标记(End of Sequence)
    for (idx, tok) in special.iter().enumerate() {
        vocab.insert(tok.to_string(), idx);
    }

    // 按 merge 顺序添加合并产生的 token（保证高频子词获得较小的 ID）
    for (_, new_tok) in &merges {
        if !vocab.contains_key(new_tok) {
            let id = vocab.len();
            vocab.insert(new_tok.clone(), id);
        }
    }

    // 添加初始单字节 token（未被任何 merge 涉及的字节对应的 Unicode 字符）
    // 按字典序排列确保确定性
    let mut sorted_vocab_set: Vec<String> = vocab_set.into_iter().collect();
    sorted_vocab_set.sort();
    for tok in sorted_vocab_set {
        if !vocab.contains_key(&tok) {
            let id = vocab.len();
            vocab.insert(tok, id);
        }
    }

    (vocab, merges)
}

// ============================================================
// 阶段 2: 编码 — 贪心应用 merge rules (字节级)
// ============================================================
//
// 编码过程:
// 1. 对新文本做 pre-tokenize（和训练时相同的方式）
// 2. 将每个 pre-token 编码为字节序列 + </w>
// 3. 按 merges 列表的顺序，依次尝试每条合并规则:
//    - 从左到右扫描 token 序列
//    - 如果相邻两个 token 匹配规则的 pair，则替换为合并后的 token
//    - 贪心策略: 先出现的规则优先执行
// 4. 将最终的 token 序列通过 vocab 映射为 token IDs

/// Byte-level BPE 编码: 原始文本 → token ID 列表。
///
/// 参数:
///   text   — 待编码的文本
///   merges — 训练阶段学到的合并规则列表（有序）
///   vocab  — token → ID 映射表
///   b2u    — 字节到 Unicode 的映射表
///
/// 返回: Vec<usize> — token ID 列表
///
/// 示例:
///   encode("the cat", ...) → [333, 1154]  (假设 "the" 的 ID 是 333)
fn encode(
    text: &str,
    merges: &[((String, String), String)],
    vocab: &HashMap<String, usize>,
    b2u: &HashMap<u8, char>,
) -> Vec<usize> {
    let mut all_ids = Vec::new();

    for pretoken in pretokenize(text) {
        // 将 pre-token 编码为字节序列，再通过 B2U 映射为 Unicode 字符
        let token_bytes = pretoken_to_bytes(&pretoken);
        let mut tokens: Vec<String> = token_bytes
            .iter()
            .map(|&b| b2u[&b].to_string())
            .collect();
        tokens.push("</w>".to_string());

        // 按 merge 顺序依次尝试合并（贪心策略）
        // 注意: 每条规则应用后，tokens 序列会缩短，然后下一条规则在新的序列上操作
        for (pair, merged) in merges {
            let mut new_tokens = Vec::new();
            let mut j = 0;
            while j < tokens.len() {
                if j + 1 < tokens.len() && tokens[j] == pair.0 && tokens[j + 1] == pair.1 {
                    // 匹配到 pair，替换为合并后的 token，跳过两个位置
                    new_tokens.push(merged.clone());
                    j += 2;
                } else {
                    // 不匹配，保留当前 token
                    new_tokens.push(tokens[j].clone());
                    j += 1;
                }
            }
            tokens = new_tokens;
        }

        // 将最终的 token 序列映射为 token IDs
        // 遇到 vocab 中不存在的 token 时使用 <unk> 的 ID
        let unk_id = vocab["<unk>"];
        for tok in &tokens {
            all_ids.push(*vocab.get(tok).unwrap_or(&unk_id));
        }
    }
    all_ids
}

/// Byte-level BPE 解码: token ID 列表 → 原始文本。
///
/// 解码过程:
/// 1. 将每个 token ID 通过 vocab 反向映射为 token 字符串
/// 2. 跳过 special tokens (<unk>, <pad>, <bos>, <eos>)
/// 3. 将 token 中的每个字符通过 U2B 映射还原为原始字节
///    - Ġ (U+0120) 特殊处理为空格 (0x20)
/// 4. 将字节序列解码为 UTF-8 字符串
/// 5. 剥离 </w> 字面标记和残留的 Ġ 字符
fn decode(ids: &[usize], vocab: &HashMap<String, usize>, u2b: &HashMap<char, u8>) -> String {
    // 构建 ID → token 的反向映射
    let id2tok: HashMap<usize, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
    let special = ["<unk>", "<pad>", "<bos>", "<eos>"];

    let mut result_bytes = Vec::new();
    for &id in ids {
        if let Some(tok) = id2tok.get(&id) {
            // 跳过 special tokens，它们不携带文本信息
            if special.contains(&tok.as_str()) {
                continue;
            }
            // 将 token 中的每个字符还原为原始字节
            for c in tok.chars() {
                if c == '\u{0120}' {
                    // Ġ (U+0120) 是空格占位符，还原为空格字节
                    result_bytes.push(b' ');
                } else if let Some(&b) = u2b.get(&c) {
                    // 通过 U2B 映射还原为原始字节值
                    result_bytes.push(b);
                }
            }
        }
    }
    // 解码为 UTF-8 字符串，并清理残留标记
    let text = String::from_utf8_lossy(&result_bytes).to_string();
    text.replace("</w>", "").replace('\u{0120}', " ")
}

// ============================================================
// 阶段 3: Embedding 查表 — 模拟推理时的 lookup
// ============================================================
//
// 在真实的大模型中，token IDs 会被送入 nn.Embedding 层，
// 该层维护一个形状为 [vocab_size, hidden_dim] 的可学习权重矩阵。
// 查表操作就是用 token ID 作为行索引，从矩阵中取出对应的 embedding 向量。
//
// 这里用伪随机数模拟这个过程，演示查表的 O(1) 复杂度。

/// 模拟 nn.Embedding 的查表操作。
///
/// 参数:
///   ids        — token ID 列表
///   vocab_size — 词表大小（矩阵行数）
///   embed_dim  — embedding 维度（矩阵列数，默认为 8）
///
/// 返回: Vec<Vec<f64>> — 每个 token ID 对应的 embedding 向量
fn embedding_lookup(ids: &[usize], vocab_size: usize, embed_dim: usize) -> Vec<Vec<f64>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // 用 hash 函数生成伪随机的 embedding 矩阵（模拟可学习权重）
    // 真实场景中这是通过梯度下降训练得到的
    let mut table = vec![vec![0.0f64; embed_dim]; vocab_size];
    for (i, row) in table.iter_mut().enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            let mut hasher = DefaultHasher::new();
            (i * 1000 + j + 42).hash(&mut hasher);
            let h = hasher.finish();
            // 将 hash 值归一化到 [-0.02, +0.02] 范围（模拟初始化的小随机权重）
            *val = ((h as f64) / (u64::MAX as f64) - 0.5) * 0.04;
        }
    }

    // ★ 核心操作: 纯索引查找，每个 token O(1) ★
    // 等价于 PyTorch 中的: embedding.weight[token_id]
    ids.iter().map(|&id| table[id].clone()).collect()
}

// ============================================================
// 主流程
// ============================================================

fn main() {
    let args: Vec<String> = env::args().collect();

    // ---- 加载语料 ----
    // 如果命令行提供了语料文件路径，则从文件加载
    // 否则使用内置的小型演示语料
    let corpus = if args.len() > 1 {
        let text = fs::read_to_string(&args[1]).expect("无法读取语料文件");
        // 按行切分为句子/段落
        text.lines()
            .map(|l| l.to_string())
            .filter(|l| !l.is_empty())
            .collect()
    } else {
        // 内置小型英文语料，用于快速演示
        vec![
            "The cat sat on the mat.".to_string(),
            "The dog sat on the log.".to_string(),
            "Cats and dogs are great pets.".to_string(),
            "The rain in Spain falls mainly on the plain.".to_string(),
            "No pain, no gain.".to_string(),
            "To be or not to be, that is the question.".to_string(),
            "All that glitters is not gold.".to_string(),
            "The quick brown fox jumps over the lazy dog.".to_string(),
        ]
    };

    // 合并次数: 命令行第二个参数指定，默认 10
    let num_merges: usize = if args.len() > 2 {
        args[2].parse().unwrap_or(10)
    } else {
        10
    };

    println!("语料大小: {} 行, {} 字符", corpus.len(),
             corpus.iter().map(|s| s.len()).sum::<usize>());
    println!("num_merges: {}", num_merges);

    // 构建字节 ↔ Unicode 双向映射表（整个流程共用）
    let (b2u, u2b) = bytes_to_unicode();

    // ==== 阶段 1: 训练 ====
    println!("{}", "=".repeat(60));
    println!("阶段 1: Byte-level BPE 训练");
    println!("{}", "=".repeat(60));

    let (vocab, merges) = train_bpe(&corpus, num_merges, &b2u);

    println!("\n最终 Vocab 大小: {}", vocab.len());

    // 构建 ID → token 的反向映射，用于打印
    let id2tok: HashMap<usize, String> = vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

    // 打印 vocab 中前 15 个 token（包含 special tokens 和最高频的合并 token）
    println!("Vocab 示例 (前 15 个):");
    for i in 0..std::cmp::min(15, vocab.len()) {
        let tok = &id2tok[&i];
        let readable = token_to_readable(tok, &u2b);
        println!("  {:>3}: {:<20}  readable: {:?}", i, format!("{:?}", tok), readable);
    }

    // 打印训练过程中学到的"有意义的"合并 token
    // 筛选条件: 可读长度 >= 3 个字符（过滤掉单字节和双字节组合）
    println!("\n学到的合并 token 示例:");
    let mut merged_tokens: Vec<(usize, String, String)> = merges
        .iter()
        .filter_map(|(_, new_tok)| {
            let readable = token_to_readable(new_tok, &u2b);
            if readable.len() >= 3 {
                vocab.get(new_tok).map(|&id| (id, new_tok.clone(), readable))
            } else {
                None
            }
        })
        .collect();
    merged_tokens.sort_by_key(|&(id, _, _)| id);
    for (id, _raw, readable) in merged_tokens.iter().take(100) {
        println!("  {:>5}: {:?}", id, readable);
    }

    // ==== 阶段 2: 编码 ====
    println!("\n{}", "=".repeat(60));
    println!("阶段 2: 推理时编码 (字节级)");
    println!("{}", "=".repeat(60));

    let test_text = "the cat is a dog";
    let token_ids = encode(test_text, &merges, &vocab, &b2u);
    println!("输入:       {:?}", test_text);
    println!("Token IDs:  {:?}", token_ids);
    println!(
        "Tokens:     {:?}",
        token_ids
            .iter()
            .map(|&id| id2tok.get(&id).map(|s| s.as_str()).unwrap_or("?"))
            .collect::<Vec<&str>>()
    );

    // 解码验证: token IDs → 原始文本
    let decoded = decode(&token_ids, &vocab, &u2b);
    println!("解码还原:   {:?}", decoded);

    // ==== 阶段 3: Embedding 查表 ====
    println!("\n{}", "=".repeat(60));
    println!("阶段 3: Embedding 查表");
    println!("{}", "=".repeat(60));

    let embed_dim = 8; // 演示用 8 维，真实模型通常 768/1024/4096 维
    let vectors = embedding_lookup(&token_ids, vocab.len(), embed_dim);
    for (_i, (&tid, vec)) in token_ids.iter().zip(vectors.iter()).enumerate() {
        let tok_str = id2tok.get(&tid).map(|s| s.as_str()).unwrap_or("?");
        let readable = token_to_readable(tok_str, &u2b);
        let vec_str: String = vec
            .iter()
            .map(|v| format!("{:+.4}", v))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  id={:>3} ({:>8}) → [{}]", tid, readable, vec_str);
    }
}
