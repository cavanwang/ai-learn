# ai-learn

[English](README.md) | [简体中文](README.zh-CN.md)

> A hands-on learning & demo repository for understanding how large AI models work — implemented in Rust.

This repo is a personal study playground for the internals of large language models (LLMs).
Each topic lives in its own Cargo crate under a shared Rust **workspace**, with self-contained
implementations and runnable demos.

## Goals

- Understand the building blocks of modern LLMs by implementing them from scratch
- Keep each topic isolated in its own crate (independent `main` + `Cargo.toml`)
- Share build artifacts and lockfile via a Cargo workspace for fast, reproducible builds

## Repository structure

```
ai-learn/
├── Cargo.toml          # workspace root (virtual workspace)
├── Cargo.lock          # shared lockfile
├── target/             # shared build output
└── token_embedding/    # sub-project: byte-level BPE tokenizer
    ├── Cargo.toml
    ├── src/main.rs
    ├── BPE算法逻辑.md
    └── corpus_tiny.txt
```

## Prerequisites

- Rust toolchain (rustup + cargo) — install via https://rustup.rs

## Sub-projects

### `token_embedding`

A GPT-2 style **byte-level BPE (Byte Pair Encoding) tokenizer** with a simulated
embedding lookup, demonstrating the front-end of the LLM pipeline: how raw text
becomes token ids and then embedding vectors.

**What BPE does**: BPE (Byte Pair Encoding) is the subword tokenization
algorithm used by GPT-2/3/4. It iteratively merges the most frequent adjacent
byte pairs in a corpus to build a vocabulary of subword units — finer than
whole words, more meaningful than single bytes — balancing vocabulary size
with full coverage of any text. This is the first step that turns raw text
into the token ids a model consumes.

Three stages:

1. **Training** (`train_bpe`) — learn merge rules + vocabulary from a corpus
   - pre-tokenize (GPT-2 style regex split with `Ġ` space prefix)
   - iteratively merge the most frequent adjacent byte pair
2. **Encoding** (`encode` / `decode`) — apply learned merge rules to new text
   - greedy left-to-right merging in merge-rule order
   - token ids ↔ text round-trip
3. **Embedding lookup** (`embedding_lookup`) — simulate `nn.Embedding` lookup
   - token id → embedding vector via O(1) table index

Run it:

```bash
# built-in tiny corpus, default 10 merges
cargo run -p ai_token_embedding

# external corpus with a custom number of merges (add --release for large corpora)
cargo run -p ai_token_embedding -- corpus_tiny.txt 5000
```

**Algorithm docs**: [BPE算法逻辑.md](token_embedding/BPE算法逻辑.md) — detailed walkthrough of the BPE algorithm.

## Working with the workspace

```bash
cargo build                            # build all members
cargo build -p ai_token_embedding      # build a single member (by package name)
cargo run -p ai_token_embedding        # run a single member
cargo test --workspace                 # test all members
```

> Note: `-p` takes the **package name** (`ai_token_embedding`), not the directory name.

### Add a new sub-project

```bash
cargo new <name>      # auto-joins the workspace members list
cargo run -p <name>
```
