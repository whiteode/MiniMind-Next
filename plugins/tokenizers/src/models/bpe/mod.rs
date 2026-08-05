//! [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
// ↑ 模块级文档注释，附带了提出 BPE 算法应用于 NLP 的经典论文链接。

use std::{iter, mem}; // 引入标准库中的迭代器工具和内存操作工具

// ----------------- 1. 模块声明 -----------------
// 将 BPE 的不同功能拆分到不同的文件中，保持代码整洁
mod model;         // 核心模型逻辑（例如合并操作、加载词表）
mod serialization; // 序列化与反序列化逻辑（保存和加载 tokenizer.json）
pub mod trainer;   // BPE 的训练器逻辑（对外公开，所以加了 pub）
mod word;          // 单词级别的处理逻辑

// ----------------- 2. 类型别名 -----------------
/// Pair 代表 BPE 合并操作中的一对符号。
/// 在 BPE 中，我们总是寻找相邻频率最高的两个 Token 进行合并，
/// 用 `u32` 代表 Token 的 ID，所以一对 Token 就是一个 (u32, u32) 元组。
type Pair = (u32, u32);

// ----------------- 3. 错误处理定义 -----------------
/// 使用 `thiserror` 宏来优雅地定义 BPE 模型在使用或构建时可能遇到的错误。
/// #[derive(thiserror::Error, Debug)] 会自动为这个枚举实现标准库的 Error 特征。
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// 主要在读取文件（如 vocab.json 或 merges.txt）时遇到的 IO 错误
    #[error("IoError: {0}")]
    Io(#[from] std::io::Error), // #[from] 允许从 std::io::Error 自动转换过来
    
    /// 解析 JSON 文件时 forwarded（转发）的 Serde 错误
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
    
    /// 当 vocab.json（词汇表）文件格式不正确时触发
    #[error("Bad vocabulary json file")]
    BadVocabulary,
    
    /// 当 merges.txt（合并规则）文件格式不正确时触发。
    /// 包含的 usize 参数用于精确指出是哪一行报错了，方便 debug。
    #[error("Merges text file invalid at line {0}")]
    BadMerges(usize),
    
    /// 如果在 merges 文件中发现了一个 Token，但它却不在词汇表中
    #[error("Token `{0}` out of vocabulary")]
    MergeTokenOutOfVocabulary(String),
    
    /// 如果用户提供的 UNK（未知词）Token 不在词汇表中
    #[error("Unk token `{0}` not found in the vocabulary")]
    UnkTokenOutOfVocabulary(String),
    
    /// BPE Dropout 参数不合法（Dropout 是 BPE 的一种变体，用于提升模型鲁棒性）
    #[error("Dropout should be between 0 and 1, inclusive")]
    InvalidDropout,
}

// ----------------- 4. 巧妙的迭代器扩展 -----------------
/// 这是一个包内可见 (pub(crate)) 的 Trait。
/// 它的作用是为 Rust 中所有的 Iterator（迭代器）添加一个名为 `with_first_and_last` 的新方法。
pub(crate) trait WithFirstLastIterator: Iterator + Sized {
    fn with_first_and_last(self) -> FirstLastIterator<Self>;
}

/// 泛型实现：只要某个类型 I 实现了 Iterator 特征，就自动为它实现 WithFirstLastIterator。
impl<I> WithFirstLastIterator for I
where
    I: Iterator,
{
    fn with_first_and_last(self) -> FirstLastIterator<Self> {
        FirstLastIterator {
            first: true,            // 初始状态下，下一个元素必然是第一个
            iter: self.peekable(),  // 将原生迭代器转换为可偷看（Peekable）的迭代器
        }
    }
}

/// 这个结构体就是具体的迭代器实现。
/// 它能在遍历元素时，额外告诉你当前元素【是不是第一个】以及【是不是最后一个】。
pub(crate) struct FirstLastIterator<I>
where
    I: Iterator,
{
    first: bool,             // 记录当前是否是第一个元素
    iter: iter::Peekable<I>, // Peekable 允许我们在不消耗下一个元素的情况下“看”它一眼
}

impl<I> Iterator for FirstLastIterator<I>
where
    I: Iterator,
{
    /// 迭代器产出的元素类型变成了个三元组：
    /// (是否为首元素: bool, 是否为尾元素: bool, 原本的元素: I::Item)
    type Item = (bool, bool, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        // mem::replace 是一种非常地道的 Rust 写法。
        // 它会把 false 写入 self.first，并把 self.first 原本的值（旧值）取出来赋给变量 first。
        // 这意味着：只有第一次调用时 first 为 true，之后永远为 false。
        let first = mem::replace(&mut self.first, false);
        
        // 尝试从底层迭代器获取下一个元素
        self.iter
            .next()
            // 如果拿到了元素 `e`，我们使用 `peek()` 去看下一位有没有东西。
            // 如果下一位是 `None`（空），说明当前这个 `e` 就是最后一个元素（is_last = true）。
            .map(|e| (first, self.iter.peek().is_none(), e))
    }
}

// ----------------- 5. 模块导出 -----------------
// Re-export（重导出）：把子模块里定义的核心内容提升到当前层级。
// 这样外部调用时，不需要写 `bpe::model::BPE`，直接写 `bpe::BPE` 即可。
pub use model::*;
pub use trainer::*;
use word::*;