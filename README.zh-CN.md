# ai-learn

[English](README.md) | [简体中文](README.zh-CN.md)

> 一个用 Rust 实现的、关于大模型工作原理的"学习 + 简单演示"仓库。

本仓库是学习大语言模型（LLM）内部原理的个人练习场。每个主题以独立的 Cargo crate
形式存在于共享的 Rust **workspace** 中，包含自包含的实现和可运行的演示。

## 目标

- 通过从零实现来理解现代 LLM 的核心构件
- 每个主题隔离在独立 crate 中（拥有独立的 `main` 和 `Cargo.toml`）
- 通过 Cargo workspace 共享编译产物和锁文件，实现快速、可复现的构建

## 仓库结构

```
ai-learn/
├── Cargo.toml          # workspace 根配置（虚拟工作空间）
├── Cargo.lock          # 共享锁文件
├── target/             # 共享编译产物
└── token_embedding/    # 子项目：byte-level BPE 分词器
    ├── Cargo.toml
    ├── src/main.rs
    ├── BPE算法逻辑.md
    └── corpus_tiny.txt
```

## 环境要求

- Rust 工具链（rustup + cargo），通过 https://rustup.rs 安装

## 子项目

### `token_embedding`

GPT-2 风格的 **byte-level BPE（字节对编码）分词器**，附带模拟的 embedding 查表，
演示 LLM 流水线的前端：原始文本如何变成 token id，再变成 embedding 向量。

**BPE 的作用**：BPE（字节对编码）是 GPT-2/3/4 等大模型使用的子词分词算法。它通过迭代合并
语料中最高频的相邻字节对来构建子词词表——切分粒度比"整词"更细、比"单字节"更有语义，在控制
词表大小的同时覆盖任意文本。这是大模型把原始文本转成 token id 的第一步。

覆盖三个阶段：

1. **训练**（`train_bpe`）—— 从语料学习合并规则和词表
   - pre-tokenize（GPT-2 风格正则切分，带 `Ġ` 空格前缀）
   - 迭代合并最高频的相邻字节对
2. **编码**（`encode` / `decode`）—— 用学到的合并规则处理新文本
   - 按 merge 规则顺序贪心从左到右合并
   - token id 与文本可往返还原
3. **Embedding 查表**（`embedding_lookup`）—— 模拟 `nn.Embedding` 查表
   - token id → embedding 向量，O(1) 索引查找

运行方式：

```bash
# 使用内置小语料，默认 10 次合并
cargo run -p ai_token_embedding

# 使用外部语料，自定义合并次数（大语料建议加 --release）
cargo run -p ai_token_embedding -- corpus_tiny.txt 5000
```

**算法文档**：[BPE算法逻辑.md](token_embedding/BPE算法逻辑.md) —— BPE 算法的详细讲解。

## workspace 常用操作

```bash
cargo build                            # 构建所有成员
cargo build -p ai_token_embedding      # 构建单个成员（按包名）
cargo run -p ai_token_embedding        # 运行单个成员
cargo test --workspace                 # 测试所有成员
```

> 注意：`-p` 后跟的是**包名**（`ai_token_embedding`），不是目录名。

### 新增子项目

```bash
cargo new <name>      # 会自动加入 workspace 的 members 列表
cargo run -p <name>
```
