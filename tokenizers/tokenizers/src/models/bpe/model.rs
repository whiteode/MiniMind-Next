// 导入BPE相关的核心类型：词汇表迭代器、训练器、错误类型、字符对、单词
use super::{super::OrderedVocabIter, trainer::BpeTrainer, Error, Pair, Word};
// 导入分词器的模型trait、结果类型和Token类型
use crate::tokenizer::{Model, Result, Token};
// 导入缓存相关的常量：默认缓存容量和最大长度限制
use crate::utils::cache::{DEFAULT_CACHE_CAPACITY, MAX_LENGTH};
// 导入结果分流器，用于处理迭代器中的Result类型
use crate::utils::iter::ResultShunt;
// 导入高性能哈希表实现
use ahash::AHashMap;
// 导入JSON值类型，用于解析词汇表文件
use serde_json::Value;
// 导入写时复制智能指针，用于优化字符串处理
use std::borrow::Cow;
// 导入RefCell，用于内部可变性（线程局部缓存）
use std::cell::RefCell;
// 导入原子类型和内存顺序，用于线程安全的计数器
use std::sync::atomic::{AtomicU64, Ordering};

// 导入标准库的HashMap
use std::collections::HashMap;
// 导入不安全的UTF-8转换函数（用于已知有效的UTF-8数据）
use std::str::from_utf8_unchecked;
// 导入文件系统和IO相关类型
use std::{
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

// 词汇表类型：从token字符串映射到token ID
// pub type 是 Rust 里的“公开类型别名（type alias）”声明。

// type：给一个已有类型起别名（不创建新类型，只是换个名字）
// pub：把这个别名导出到当前模块外部可见
pub type Vocab = AHashMap<String, u32>;
// 反向词汇表类型：从token ID映射回token字符串
type VocabR = AHashMap<u32, String>;
// 合并映射类型：从字符对映射到(合并优先级rank, 合并后的新token ID)
pub type MergeMap = AHashMap<Pair, (u32, u32)>;
// Pair 是一个类型别名，定义在 mod.rs:17 为 (u32, u32)。也就是一个由两个 u32 组成的元组。


/// 进程级单调递增计数器，用于为每个`BpeCache`分配唯一的代数ID
/// 这样可以确保每个BPE实例的线程局部缓存永远不会冲突
static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(0);

// AtomicU64 是 Rust 标准库中 std::sync::atomic::AtomicU64，表示一个可以在多线程间原子访问和修改的 64 位无符号整数（u64）。


/// 每个BPE实例的缓存描述符
///
/// BPE不再使用共享的`RwLock<AHashMap>`缓存：编码热路径只读写下面的
/// 线程局部`BPE_LOCAL_CACHE`，通过`(BpeCache::id, sequence)`作为键。
/// 这个结构体只携带每个实例的代数ID和容量，这样现有的`clear_cache()`
/// 和`resize_cache()` API保持其语义：`clear()`会增加ID，
/// 一次性使该BPE在所有线程中的条目失效。
#[derive(Debug)]
pub(crate) struct BpeCache { // pub(crate) 是 Rust 的可见性修饰符，表示“仅在当前 crate（包）内可见”
    id: AtomicU64,           // 缓存代数ID，用于区分不同的缓存实例
    pub capacity: usize,     // 缓存容量限制，usize 是 Rust 的无符号整数类型，其位数与平台指针宽度一致
}

// 匹配之前的`Cache`实现：我们从不按值比较缓存
// 所有BpeCache实例都被认为是相等的（因为它们只是描述符）
impl PartialEq for BpeCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
//PartialEq 是一个 trait，表示“可以用 == 和 != 判断两个值是否相等”。

impl BpeCache { //impl BpeCache 是 Rust 中的 impl 块，用于为 BpeCache 结构体定义方法。
    /// 创建新的BpeCache，分配一个全局唯一的代数ID
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            // 原子地获取并递增全局计数器，为这个缓存分配唯一ID
            id: AtomicU64::new(NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed)),// fetch_add 返回之前的值，所以每个实例都会得到一个独特的ID,Ordering::Relaxed 表示这个操作不需要与其他原子操作同步（因为我们只需要唯一ID，不关心顺序）
            capacity,
        }
    }

    /// 返回一个新的`BpeCache`，容量相同但ID是新的
    /// 用于`impl Clone for BPE`，确保克隆的BPE有独立的缓存
    pub(crate) fn fresh(&self) -> Self {
        Self::new(self.capacity)
    }

    /// 获取当前的代数ID。每次调用`clear()`时会递增
    pub(crate) fn id(&self) -> u64 {
        self.id.load(Ordering::Relaxed)
    }

    /// 通过推进代数ID来使所有线程的线程局部缓存条目失效
    /// 下次查找时会重新计算
    pub(crate) fn clear(&self) {
        self.id.store(
            NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    /// 调整缓存容量
    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity;
    }
    // capacity 限制的是：

    // 每个 BPE 实例
    // 在每个线程中
    // 可以缓存的不同文本序列的数量
    // 默认值：10,000 条
}

thread_local! {
    /// 每个线程的BPE分词缓存。这是热路径上唯一的BPE缓存：
    /// 没有共享的全局映射，所以查找和插入完全不需要原子同步。
    /// 外层映射以`BpeCache::id`为键，这样共享同一个rayon工作线程的
    /// 多个`BPE`实例永远不会看到彼此的条目。
    static BPE_LOCAL_CACHE: RefCell<AHashMap<u64, AHashMap<String, Word>>> =
        RefCell::new(AHashMap::new());
}
// 在热路径上，用空间换时间是值得的：我们为每个线程维护一个独立的缓存，避免了锁的开销，同时通过BpeCache的ID机制确保不同BPE实例之间的缓存不会冲突。
// thread_local! 是 Rust 的线程局部存储宏，用于创建每个线程独立拥有的变量。
//
// Word 是 BPE 分词器中的核心数据结构，用于表示一个单词的符号序列。

// Word 数据结构定义

// pub(super) struct Word {
//     symbols: Vec<Symbol>,  // 符号序列
// }
// 内部 Symbol 结构
// Word 内部使用 Symbol 来表示每个 token：


// struct Symbol {
//     c: u32,          // 符号ID（token ID）
//     prev: isize,     // 前一个符号的索引（-1表示没有）
//     next: isize,     // 后一个符号的索引（-1表示没有）
//     len: usize,      // 符号的字节长度
// }

// 合并列表类型：存储合并规则的向量，每个元素是一对字符串
pub type Merges = Vec<(String, String)>;

// BPE配置结构体，用于构建BPE模型
struct Config {
    files: Option<(String, String)>,              // 可选的词汇表和合并文件路径
    vocab: Vocab,                                  // 词汇表：token到ID的映射
    merges: Merges,                                // 合并规则列表
    cache_capacity: usize,                         // 缓存容量
    dropout: Option<f32>,                          // dropout概率（用于训练时的随机性）
                                                    // 训练时：dropout = 0.1（典型值）
                                                    // 推理时：dropout = None（确定性分词）
    unk_token: Option<String>,                     // 未知token的表示
    continuing_subword_prefix: Option<String>,     // 子词前缀（如"##"）
    end_of_word_suffix: Option<String>,            // 词尾后缀
    // 使用 BERT 风格模型→ 设置 continuing_subword_prefix = "##"
    // 使用 SentencePiece → 设置 end_of_word_suffix = "</w>"
    // 大多数情况 → 两者选其一，不要同时使用



    fuse_unk: bool,                                // 是否合并连续的未知token
    // 合并连续未知 token 的原因：
    // 效率：减少序列长度 → 更快的推理
    // 语义：将连续未知内容视为整体更合理
    // 资源：节省计算和内存
    // 默认建议：大多数情况下使用 fuse_unk = true，除非你有特殊需求需要保留每个未知字符的独立性。
    byte_fallback: bool,                           // 是否使用字节回退（如<0x00>）

    // 字节回退是一种无损编码策略：

    // 特性	说明
    // 原理	将未知字符拆解成UTF-8 字节
    // 表示	每个字节用 <0xXX> token 表示
    // 优势	无信息损失，可完美还原
    // 代价	序列长度增加
    // 典型应用	GPT-2, GPT-3 等通用模型
    // 选择建议：

    // 通用模型 → 使用字节回退
    // 特定领域 → 使用 UNK token
    ignore_merges: bool,                           // 是否忽略合并规则（直接输出词汇表中的词）
    // 使用建议：

    // 通用场景：ignore_merges = false（默认）
    // 特定词汇表 + 性能要求：ignore_merges = true
    // 需要保持词完整性：ignore_merges = true

}


/// `BpeBuilder`可用于创建具有自定义配置的`BPE`模型
pub struct BpeBuilder {
    config: Config,  // 内部配置对象
}

impl Default for BpeBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,                                    // 默认不从文件加载
                vocab: AHashMap::new(),                         // 空词汇表
                merges: vec![],                                 // 空合并规则
                cache_capacity: DEFAULT_CACHE_CAPACITY,         // 使用默认缓存容量
                dropout: None,                                  // 不使用dropout
                unk_token: None,                                // 不设置未知token
                continuing_subword_prefix: None,                // 不使用子词前缀
                end_of_word_suffix: None,                       // 不使用词尾后缀
                fuse_unk: false,                                // 不合并未知token
                byte_fallback: false,                           // 不使用字节回退
                ignore_merges: false,                           // 不忽略合并规则
            },
        }
    }
}

impl BpeBuilder {
    /// 构造一个新的`BpeBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置输入文件路径（词汇表文件和合并文件）
    // 特性	说明
    // 作用	强制使用返回值
    // 警告时机	返回值被忽略时
    // 典型用途	Builder 模式、Result 类型
    // 目的	防止意外丢弃重要返回值
    // #[must_use] 是 Rust 的安全机制，帮助你避免常见的编程错误。
    #[must_use]
    pub fn files(mut self, vocab: String, merges: String) -> Self {
        self.config.files = Some((vocab, merges));
        self
    }

    /// 设置词汇表（token -> ID）和合并规则映射
    #[must_use]
    pub fn vocab_and_merges<V: Into<AHashMap<String, u32>>>(
        mut self,
        vocab: V,
        merges: Merges,
    ) -> Self {
        self.config.vocab = vocab.into();
        self.config.merges = merges;
        self
    }

    // Merges 是一个类型别名，定义如下：
    // pub type Merges = Vec<(String, String)>;
    // 含义
    // Merges = BPE 合并规则列表
    // 每个元素是一个元组 (String, String)，表示哪两个 token 应该合并。


    /// 设置缓存容量。设为0可禁用缓存
    #[must_use]
    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.config.cache_capacity = capacity;
        self
    }

    //缓存是性能优化，不是必需功能。

    /// 使用dropout（参考论文：https://arxiv.org/abs/1910.13267）
    /// dropout可以在训练时增加随机性，提高模型泛化能力
    #[must_use]
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = Some(dropout);
        self
    }

    /// 设置词汇表的`UNK`（未知）token
    /// 当遇到词汇表中不存在的字符时使用
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = Some(unk_token);
        self
    }

    /// 设置`continuing_subword_prefix`选项
    /// 用于标记非首个子词（如BERT中的"##"）
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// 设置`end_of_word_suffix`选项
    /// 用于标记词尾的子词（如某些模型中的"</w>"）
    #[must_use]
    pub fn end_of_word_suffix(mut self, prefix: String) -> Self {
        self.config.end_of_word_suffix = Some(prefix);
        self
    }

    /// 设置`fuse_unk`选项
    /// 如果为true，连续的未知字符会被合并为一个UNK token
    #[must_use]
    pub fn fuse_unk(mut self, fuse_unk: bool) -> Self {
        self.config.fuse_unk = fuse_unk;
        self
    }

    /// 设置`byte_fallback`选项
    /// 如果为true，未知字符会被转换为字节表示（如"<0x61>"）而不是UNK
    #[must_use]
    pub fn byte_fallback(mut self, byte_fallback: bool) -> Self {
        self.config.byte_fallback = byte_fallback;
        self
    }
    /// 设置`ignore_merges`选项
    /// 如果为true，直接输出词汇表中的完整词，不进行BPE合并
    #[must_use]
    pub fn ignore_merges(mut self, ignore_merges: bool) -> Self {
        self.config.ignore_merges = ignore_merges;
        self
    }

    /// 返回使用`BpeBuilder`配置的`BPE`模型
    pub fn build(mut self) -> Result<BPE> {
        // 验证dropout参数：必须在0.0到1.0之间
        if let Some(p) = self.config.dropout {
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::InvalidDropout.into());
            }
        }

        // 如果指定了文件路径，从文件读取词汇表和合并规则
        if let Some((vocab, merges)) = self.config.files {
            let (v, m) = BPE::read_file(&vocab, &merges)?;
            self.config.vocab = v;
            self.config.merges = m;
        }

        // 构建反向词汇表（ID -> token），同时记录最长token的长度
        let mut max_len = 0;
        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| {
                if max_len < key.len() {
                    max_len = key.len();
                }
                (*val, key.to_owned())
            })
            .collect();
        // 根据容量创建缓存，容量为0则不使用缓存
        let cache = match self.config.cache_capacity {
            0 => None,
            capacity => Some(BpeCache::new(capacity)),
        };

        let vocab = self.config.vocab;
        // 计算子词前缀的长度（如果有的话）
        let prefix_len = if let Some(prefix) = &self.config.continuing_subword_prefix {
            prefix.len()
        } else {
            0
        };
        // 创建缓冲区用于合并token时的字符串拼接
        let mut buffer: Vec<u8> = vec![0; max_len];
        // 构建合并映射：将合并规则转换为(token_pair) -> (rank, new_token_id)的映射
        let merge_map: MergeMap = self
            .config
            .merges
            .into_iter()
            .enumerate()
            .map(|(i, (a, b))| -> Result<(Pair, (u32, u32))> {
                // 获取第一个token的ID
                let a_id = vocab
                    .get(&a)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(a.to_owned()))?;
                // 获取第二个token的ID
                let b_id = vocab
                    .get(&b)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(b.to_owned()))?;
                // 将第一个token复制到缓冲区
                buffer[0..a.len()].copy_from_slice(a.as_bytes());
                // 计算第二个token去除前缀后的长度
                let b_len = b.len() - prefix_len;
                let merge_len = a.len() + b_len;
                // 将第二个token（去除前缀）追加到缓冲区
                buffer[a.len()..merge_len].copy_from_slice(&b.as_bytes()[prefix_len..]);
                // 安全性：缓冲区包含两个有效UTF-8字符串的拼接，所以它本身也是有效的UTF-8
                let new_token = unsafe { from_utf8_unchecked(&buffer[..merge_len]) };
                // 获取合并后新token的ID
                let new_id = vocab
                    .get(new_token)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(new_token.to_owned()))?;
                // 返回：(token对) -> (合并优先级rank, 新token的ID)
                Ok(((*a_id, *b_id), (i as u32, *new_id)))
            })
            .collect::<Result<MergeMap>>()?;

            // 这段代码的作用是预处理合并规则：

            // 转换	从	到
            // 输入	Vec<(String, String)>	字符串对
            // 输出	AHashMap<Pair, (u32, u32)>	ID对 → (优先级, 新ID)
            // 目的	运行时快速查找	避免字符串比较
            // 这样在分词时可以直接用数字 ID 操作，比字符串快得多。

        // 将合并规则插入映射：pair -> (rank, new_id)

        // 构造并返回BPE模型实例
        Ok(BPE {
            vocab,                                  // 词汇表
            vocab_r,                                // 反向词汇表
            merges: merge_map,                      // 合并映射
            cache,                                  // 缓存（可选）
            dropout: self.config.dropout,           // dropout概率
            unk_token: self.config.unk_token,       // 未知token
            continuing_subword_prefix: self.config.continuing_subword_prefix,  // 子词前缀
            end_of_word_suffix: self.config.end_of_word_suffix,                // 词尾后缀
            fuse_unk: self.config.fuse_unk,         // 是否合并未知token
            byte_fallback: self.config.byte_fallback,  // 是否使用字节回退
            ignore_merges: self.config.ignore_merges,  // 是否忽略合并
        })
    }
}

/// [字节对编码（Byte Pair Encoding）](https://www.aclweb.org/anthology/P16-1162/)模型
/// BPE是一种子词分词算法，通过迭代合并最频繁出现的字符对来构建词汇表
#[derive(PartialEq)]//#[derive(PartialEq)] 是 Rust 的派生宏，自动为类型实现相等比较功能。
pub struct BPE {
    /// 词汇表：为每个token分配一个数字ID
    pub(crate) vocab: Vocab,
    /// 反向词汇表：用于重建句子（从ID映射回token字符串）
    pub(crate) vocab_r: VocabR,
    /// 包含字符对到其(优先级rank, 新token ID)的映射
    /// rank越小表示该合并规则越优先执行
    pub(crate) merges: MergeMap,
    /// 用于优化编码步骤的缓存
    cache: Option<BpeCache>,
    /// 合并的dropout概率。默认为0.0表示无dropout。
    /// 设为1.0时，分词将不执行任何合并，结果只是字符级别的token
    pub dropout: Option<f32>,
    /// 遇到未知字符时使用的未知token
    pub unk_token: Option<String>,
    /// 可选的前缀，用于标记只存在于另一个子词后面的子词
    /// 例如BERT中的"##"前缀
    pub continuing_subword_prefix: Option<String>,
    /// 可选的后缀，用于标记词尾的子词
    /// 例如某些模型中的"</w>"后缀
    pub end_of_word_suffix: Option<String>,
    /// 是否合并多个连续的未知token
    pub fuse_unk: bool,
    /// 字节回退模式（来自sentence pieces）：
    /// 不使用UNK，而是为未知token中的每个字节使用`"<0x00>"`形式
    pub byte_fallback: bool,
    /// 是否直接输出词汇表中的完整词（如果存在）
    /// 设为true时会跳过BPE合并过程
    pub ignore_merges: bool,
}

impl std::fmt::Debug for BPE {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        // 自定义Debug输出，避免打印整个词汇表和合并表（太大）
        // 只打印配置参数和词汇表/合并表的大小
        fmt.debug_struct("BPE")
            .field("dropout", &self.dropout)
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("end_of_word_suffix", &self.end_of_word_suffix)
            .field("fuse_unk", &self.fuse_unk)
            .field("byte_fallback", &self.byte_fallback)
            .field("vocab", &self.vocab.len())      // 只显示词汇表大小
            .field("merges", &self.merges.len())    // 只显示合并规则数量
            .field("ignore_merges", &self.ignore_merges)
            .finish()
    }
}

impl Default for BPE {
    fn default() -> Self {
        // 使用builder模式创建默认的BPE实例
        Self::builder().build().unwrap()
    }
}

impl Clone for BPE {
    // `Clone`不能自动派生，因为`BpeCache`没有实现Clone
    // 为了简化克隆操作，新的BPE将从一个新的缓存开始
    fn clone(&self) -> Self {
        // 如果有缓存，创建一个新的缓存（不同的ID）
        let fresh_cache = self.cache.as_ref().map(|cache| cache.fresh());
        Self {
            vocab: self.vocab.clone(),
            vocab_r: self.vocab_r.clone(),
            merges: self.merges.clone(),
            cache: fresh_cache,                     // 使用新的缓存ID
            dropout: self.dropout,
            unk_token: self.unk_token.clone(),
            continuing_subword_prefix: self.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.end_of_word_suffix.clone(),
            fuse_unk: self.fuse_unk,
            byte_fallback: self.byte_fallback,
            ignore_merges: self.ignore_merges,
        }
    }
}

/// 将合并字符串（例如从`merges.txt`文件）转换为BPE结构期望的格式
/// 输入格式："{pair_a} {pair_b}"（两个token用空格分隔）
pub(crate) fn convert_merges_to_hashmap<I: Iterator<Item = String>>(
    iter: I,
    _vocab: &Vocab,
) -> Result<Merges> {
    let mut merges = vec![];

    // 过滤掉版本信息行（以"#version"开头）
    let lines = iter.filter(|l| !l.starts_with("#version"));
    for (rank, line) in lines.enumerate() {
        // 按空格分割每一行
        let parts = line.split(' ').collect::<Vec<_>>();
        if parts.len() != 2 {
            // 每行必须恰好有两个token
            return Err(Error::BadMerges(rank + 1).into());
        }

        // 将合并规则添加到列表中
        merges.push((parts[0].to_string(), parts[1].to_string()));
    }

    Ok(merges)
}

impl BPE {
    /// 初始化一个`BpeBuilder`构建器
    pub fn builder() -> BpeBuilder {
        BpeBuilder::new()
    }

    /// 使用给定的词汇表和合并规则创建新的BPE模型
    pub fn new(vocab: Vocab, merges: Merges) -> Self {
        Self::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap()
    }

    /// 从词汇表和合并规则文件初始化BpeBuilder
    pub fn from_file(vocab: &str, merges: &str) -> BpeBuilder {
        Self::builder().files(vocab.to_owned(), merges.to_owned())
    }

    /// 读取给定的文件以提取词汇表和合并规则
    pub fn read_file(vocab: &str, merges: &str) -> Result<(Vocab, Merges)> {
        // 读取vocab.json文件
        let vocab_file = File::open(vocab)?;
        let mut vocab_file = BufReader::new(vocab_file);

        let mut buffer = String::new();
        vocab_file.read_to_string(&mut buffer)?;
        // 解析JSON格式的词汇表
        let json: Value = serde_json::from_str(&buffer)?;
        let mut vocab = AHashMap::new();
        match json {
            Value::Object(m) => {
                // 遍历JSON对象，提取token和对应的ID
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().ok_or(Error::BadVocabulary)? as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => return Err(Box::new(Error::BadVocabulary)),
        };

        // 读取merges文件（通常是merges.txt）
        let merge_file = File::open(merges)?;
        let merge_file = BufReader::new(merge_file);
        // 使用ResultShunt处理文件行，转换为合并规则
        let merges = ResultShunt::process(merge_file.lines(), |iter| {
            convert_merges_to_hashmap(iter, &vocab)
        })??;

        Ok((vocab, merges))
    }

    /// 重置缓存，清除所有已缓存的分词结果
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear()
        }
    }

    /// 调整缓存容量大小
    pub fn resize_cache(&mut self, capacity: usize) {
        if let Some(ref mut cache) = self.cache {
            cache.resize(capacity);
        }
    }

    /// 获取词汇表的副本（转换为标准HashMap）
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    /// 获取未知token的引用
    pub fn get_unk_token(&self) -> &Option<String> {
        &self.unk_token
    }

    /// 获取子词前缀的引用
    pub fn get_continuing_subword_prefix(&self) -> &Option<String> {
        &self.continuing_subword_prefix
    }

    /// 对单词执行BPE合并操作
    /// 这是BPE算法的核心：将字符序列逐步合并成子词token
    /// 将输入字符串转换为Word对象（符号序列）
    ///
    /// 这个函数是BPE分词的第一阶段：将字符串拆分成初始符号序列
    /// 主要步骤：
    /// 1. 逐字符遍历输入字符串
    /// 2. 为每个字符添加前缀/后缀（如果配置了）
    /// 3. 在词汇表中查找对应的token ID
    /// 4. 处理未知字符（字节回退或UNK token）
    /// 5. 最后应用BPE合并规则
    fn merge_word(&self, w: &str) -> Result<Word> {
        // 步骤1：初始化字符迭代器
        // char_indices()返回(字节索引, 字符)对，我们只需要字节索引
        // peekable()允许我们查看下一个元素而不消费它
        let mut indices = w.char_indices().map(|(idx, _)| idx).peekable();

        // 步骤2：创建Word对象，预分配容量以提高性能
        let mut word = Word::with_capacity(w.len());

        // 步骤3：跟踪连续的未知字符
        // Some((unk_id, unk_len))表示当前正在累积未知字符
        // None表示没有待处理的未知字符
        let mut unk: Option<(u32, usize)> = None;

        // 步骤4：逐字符处理
        while let Some(i) = indices.next() {
            // 获取下一个字符的索引（用于确定当前字符的结束位置）
            let end = indices.peek();

            // 判断当前字符的位置
            let is_first = i == 0;          // 是否是第一个字符（用于决定是否添加前缀）
            let is_last = end.is_none();    // 是否是最后一个字符（用于决定是否添加后缀）

            // 步骤5：提取当前字符对应的子串
            // 使用Cow（Clone on Write）避免不必要的字符串分配
            let mut s = if let Some(e) = end {
                // 不是最后一个字符：从i到下一个字符的起始位置
                Cow::Borrowed(&w[i..*e])
            } else {
                // 是最后一个字符：从i到字符串末尾
                Cow::Borrowed(&w[i..])
            };

            // 记录原始字节长度（用于后续的偏移量计算）
            let byte_len = s.len();

            // 步骤6：添加子词前缀（BERT风格：##）
            // 只有非首字符才添加前缀，用于标记这是词的延续部分
            // 例如："playing" → ["play", "##ing"]
            if !is_first {
                if let Some(ref prefix) = self.continuing_subword_prefix {
                    // 将Cow::Borrowed转换为Cow::Owned（因为需要修改）
                    s = format!("{prefix}{s}").into()
                }
            }

            // 步骤7：添加词尾后缀（SentencePiece风格：</w>）
            // 只有最后一个字符才添加后缀，用于标记词的结束
            // 例如："hello" → ["hel", "lo</w>"]
            if is_last {
                if let Some(ref suffix) = self.end_of_word_suffix {
                    s = format!("{s}{suffix}").into()
                }
            }

            // 步骤8：在词汇表中查找当前子串（可能带前缀/后缀）
            if let Some(id) = self.vocab.get(s.as_ref()) {
                // 情况A：找到了对应的token ID

                // 步骤8.1：先处理之前累积的未知字符（如果有）
                // 因为找到了已知token，所以之前的未知字符序列结束了
                if let Some((unk_id, unk_len)) = unk {
                    word.add(unk_id, unk_len);
                    unk = None;  // 清空未知字符累积器
                }

                // 步骤8.2：添加当前找到的token
                // byte_len是原始字节长度（不包含前缀/后缀）
                word.add(*id, byte_len);
            } else {
                // 情况B：词汇表中没有找到 → 尝试字节回退或使用UNK

                // 步骤9：尝试字节回退模式（GPT-2风格）
                // 将未知字符拆解成UTF-8字节，每个字节用<0xXX>表示
                if self.byte_fallback {
                    // 步骤9.1：尝试将每个字节转换为字节token
                    let tokens: Option<Vec<_>> = s
                        .bytes()  // 获取UTF-8字节序列
                        .map(|b| -> Option<&u32> {
                            // 格式化为"<0xXX>"，例如：0x41 → "<0x41>"
                            let code = format!("<{b:#04X}>");
                            // 在词汇表中查找这个字节token
                            self.vocab.get(&code)
                        })
                        .collect();  // 如果所有字节都找到了，返回Some(Vec)，否则返回None

                    // 步骤9.2：如果所有字节都成功转换，添加这些字节token
                    if let Some(tokens) = tokens {
                        for t in tokens {
                            word.add(*t, 1);  // 每个字节token的长度都是1
                        }
                        continue;  // 跳过后续的UNK处理，继续下一个字符
                    }
                    // 如果字节回退失败（词汇表中没有某些字节token），继续执行UNK处理
                }
                // 步骤10：字节回退失败或未启用 → 使用UNK token
                if let Some(unk_token) = &self.unk_token {
                    // 根据fuse_unk配置决定如何处理连续的未知字符
                    unk = match (unk, self.fuse_unk) {
                        // 情况1：已有累积的未知字符 + 启用了fuse_unk
                        (Some((unk_id, unk_len)), true) => {
                            // 将当前未知字符合并到之前的未知字符中
                            // 只累积长度，不立即添加到word
                            // 例如："😀😁😂" → 累积为一个UNK，长度=12字节
                            Some((unk_id, unk_len + byte_len))
                        }

                        // 情况2：已有累积的未知字符 + 未启用fuse_unk
                        (Some((unk_id, unk_len)), false) => {
                            // 先将之前累积的未知字符添加到word
                            word.add(unk_id, unk_len);
                            // 然后开始新的未知字符累积
                            // 例如："😀😁" → 两个独立的UNK token
                            Some((
                                *self.vocab.get(unk_token).ok_or_else(|| {
                                    Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                                })?,
                                byte_len,
                            ))
                        }

                        // 情况3：第一个未知字符（unk为None）
                        _ => Some((
                            // 从词汇表中获取UNK token的ID
                            *self.vocab.get(unk_token).ok_or_else(|| {
                                Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                            })?,
                            byte_len,
                        )),
                    };
                }
            }
        }
        // 步骤11：处理循环结束后可能剩余的未知字符
        // 如果最后几个字符都是未知的，它们会被累积在unk中
        // 现在需要将它们添加到word中
        if let Some((unk_id, unk_len)) = unk {
            word.add(unk_id, unk_len);
        }

        // 步骤12：执行BPE合并操作
        // 此时word包含初始的符号序列（可能包含UNK、字节token等）
        // merge_all会根据合并规则将相邻的符号合并成更大的token
        // 例如：["h", "e", "l", "l", "o"] → ["he", "llo"] 或 ["hello"]
        // dropout参数用于训练时的随机性（推理时为None）
        word.merge_all(&self.merges, self.dropout);

        // 步骤13：返回最终的Word对象
        // 这个Word对象包含了分词后的符号序列，可以转换为Token列表
        Ok(word)
    }

    /// 将Word对象转换为Token迭代器
    /// 结合字符ID和偏移量信息，生成完整的Token对象
    fn word_to_tokens<'a>(&'a self, word: &'a Word) -> impl Iterator<Item = Token> + 'a {
        word.get_chars_iter()
            .zip(word.get_offsets_iter())
            .map(move |(id, offsets)| Token::new(id, self.vocab_r[&id].clone(), offsets))
    }

    /// 使用缓存进行分词
    /// 这是性能优化的关键：对于相同的输入序列，直接返回缓存的结果
    fn tokenize_with_cache(&self, sequence: &str) -> Result<Vec<Token>> {
        // 如果启用了ignore_merges且序列在词汇表中，直接返回
        if self.ignore_merges {
            if let Some(id) = self.vocab.get(sequence) {
                return Ok(vec![Token::new(
                    *id,
                    sequence.to_string(),
                    (0, sequence.len()),
                )]);
            }
        }
        // 如果缓存被禁用（容量为0），使用无缓存路径
        let Some(cache) = self.cache.as_ref() else {
            let word = self.merge_word(sequence)?;
            return Ok(self.word_to_tokens(&word).collect());
        };
        let cache_id = cache.id();
        // 访问线程局部缓存
        BPE_LOCAL_CACHE.with(|cell| {
            let mut by_bpe = cell.borrow_mut();
            // 获取或创建当前BPE实例的缓存映射
            let local = by_bpe.entry(cache_id).or_default();
            // 缓存命中：直接返回缓存的结果
            if let Some(hit) = local.get(sequence) {
                return Ok(self.word_to_tokens(hit).collect());
            }
            // 缓存未命中：执行合并操作
            let word = self.merge_word(sequence)?;
            let ret: Vec<Token> = self.word_to_tokens(&word).collect();
            // 如果序列长度和缓存大小都在限制内，将结果加入缓存
            if sequence.len() < MAX_LENGTH && local.len() < cache.capacity {
                local.insert(sequence.to_owned(), word);
            }
            Ok(ret)
        })
    }
}

// 为BPE实现Model trait，使其可以作为通用的分词模型使用
impl Model for BPE {
    type Trainer = BpeTrainer;

    /// 获取词汇表（返回标准HashMap）
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    /// 获取词汇表大小
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 对输入序列进行分词
    /// 这是Model trait的核心方法
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        // 空字符串直接返回空结果
        if sequence.is_empty() {
            return Ok(vec![]);
        }

        // 如果没有dropout或dropout为0，使用缓存加速
        if self.dropout.is_none() || self.dropout == Some(0.0) {
            self.tokenize_with_cache(sequence)
        } else {
            // 有dropout时不使用缓存（因为结果是随机的）
            let word = self.merge_word(sequence)?;
            Ok(self.word_to_tokens(&word).collect())
        }
    }

    /// 将token字符串转换为对应的ID
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// 将ID转换为对应的token字符串
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    /// 将模型保存到指定文件夹
    /// 会生成两个文件：vocab.json和merges.txt
    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        // 构造词汇表文件名
        let vocab_file_name = match name {
            Some(name) => format!("{name}-vocab.json"),
            None => "vocab.json".to_string(),
        };

        // 写入vocab.json文件
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        // 使用OrderedVocabIter确保词汇表按ID顺序输出
        let order_vocab_iter = OrderedVocabIter::new(&self.vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter)?;
        vocab_file.write_all(serialized.as_bytes())?;

        // 构造合并规则文件名
        let merges_file_name = match name {
            Some(name) => format!("{name}-merges.txt"),
            None => "merges.txt".to_string(),
        };

        // 写入merges.txt文件
        let merges_path: PathBuf = [folder, Path::new(merges_file_name.as_str())]
            .iter()
            .collect();
        let mut merges_file = File::create(&merges_path)?;
        // 提取并按rank排序合并规则
        let mut merges: Vec<(&Pair, &u32)> = self
            .merges
            .iter()
            .map(|(pair, (rank, _))| (pair, rank))
            .collect();
        merges.sort_unstable_by_key(|k| *k.1);
        // 写入版本信息
        merges_file.write_all(b"#version: 0.2\n")?;
        // 写入每条合并规则（格式：token_a token_b）
        merges_file.write_all(
            &merges
                .into_iter()
                .flat_map(|(pair, _)| {
                    format!("{} {}\n", self.vocab_r[&pair.0], self.vocab_r[&pair.1]).into_bytes()
                })
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path, merges_path])
    }

    /// 获取BPE训练器
    fn get_trainer(&self) -> BpeTrainer {
        BpeTrainer::default()
    }
}

// 测试模块
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cache_is_per_bpe_instance() {
        // 测试：两个具有不同合并规则的BPE实例必须对相同的输入进行不同的分词
        // 即使它们共享同一个线程，BPE的线程局部缓存也不能在实例之间泄漏条目
        let vocab_a: Vocab = [
            ("h", 0u32),
            ("e", 1),
            ("l", 2),
            ("o", 3),
            ("he", 4),
            ("hel", 5),
            ("hell", 6),
            ("hello", 7),
        ]
        .iter()
        .map(|(s, i)| ((*s).into(), *i))
        .collect();
        let merges_a: Merges = vec![
            ("h".into(), "e".into()),
            ("he".into(), "l".into()),
            ("hel".into(), "l".into()),
            ("hell".into(), "o".into()),
        ];
        let bpe_a = BpeBuilder::default()
            .vocab_and_merges(vocab_a, merges_a)
            .build()
            .unwrap();

        let vocab_b: Vocab = [("h", 0u32), ("e", 1), ("l", 2), ("o", 3)]
            .iter()
            .map(|(s, i)| ((*s).into(), *i))
            .collect();
        let bpe_b = BpeBuilder::default()
            .vocab_and_merges(vocab_b, vec![])
            .build()
            .unwrap();

        // 交替使用两个模型，这样任何跨实例的缓存污染都会在第二次查找时显现
        let ids_a: Vec<u32> = bpe_a
            .tokenize("hello")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        let ids_b: Vec<u32> = bpe_b
            .tokenize("hello")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        let ids_a2: Vec<u32> = bpe_a
            .tokenize("hello")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        let ids_b2: Vec<u32> = bpe_b
            .tokenize("hello")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();

        assert_eq!(ids_a, vec![7u32], "bpe_a必须合并为[hello]");
        assert_eq!(ids_b, vec![0u32, 1, 2, 2, 3], "bpe_b没有合并规则");
        assert_eq!(ids_a2, ids_a, "bpe_a第二次调用必须与第一次匹配");
        assert_eq!(ids_b2, ids_b, "bpe_b第二次调用必须与第一次匹配");
    }

    #[test]
    fn test_ordered_vocab_iter() {
        // 测试有序词汇表迭代器
        let vocab_r: VocabR = [
            (0, "a".into()),
            (1, "b".into()),
            (2, "c".into()),
            (3, "ab".into()),
        ]
        .iter()
        .cloned()
        .collect();
        let order_vocab_iter = OrderedVocabIter::new(&vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter).unwrap();
        assert_eq!(serialized, "{\"a\":0,\"b\":1,\"c\":2,\"ab\":3}");
    }

    #[test]
    fn test_unk_not_fused() {
        // 测试未知token不合并的情况
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        // 测试单个未知字符
        let tokens = bpe.tokenize("cc").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(0u32, "<unk>".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 2)),
            ]
        );

        // 测试混合已知和未知字符
        let tokens = bpe.tokenize("accb").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1u32, "a".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 2)),
                Token::new(0u32, "<unk>".into(), (2, 3)),
                Token::new(2u32, "b".into(), (3, 4)),
            ]
        );
    }
    #[test]
    fn test_unk_get_fused() {
        // 测试未知token合并的情况（fuse_unk=true）
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .fuse_unk(true)  // 启用未知token合并
            .build()
            .unwrap();
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        // 连续的未知字符会被合并为一个token
        let tokens = bpe.tokenize("cc").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 2)),]);

        // 测试混合情况：连续的未知字符被合并
        let tokens = bpe.tokenize("accb").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1u32, "a".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 3)),  // "cc"被合并
                Token::new(2u32, "b".into(), (3, 4)),
            ]
        );
    }

    #[test]
    // 测试带dropout和不带dropout的分词
    // dropout设为0时分词是确定性的，所以我们知道结果应该是什么
    //
    // 为了测试这个，我们构建一个简单的模型来分词单词'unrelated'
    fn test_tokenize_with_and_without_dropout() {
        let vocab: Vocab = [
            ("u".into(), 0),
            ("n".into(), 1),
            ("r".into(), 2),
            ("e".into(), 3),
            ("l".into(), 4),
            ("a".into(), 5),
            ("t".into(), 6),
            ("d".into(), 7),
            ("re".into(), 8),
            ("at".into(), 9),
            ("ed".into(), 10),
            ("un".into(), 11),
            ("ated".into(), 12),
            ("rel".into(), 13),
            ("related".into(), 14),
            ("unrelated".into(), 15),
        ]
        .iter()
        .cloned()
        .collect();
        let merges: Merges = vec![
            ("r".to_string(), "e".to_string()),
            ("a".to_string(), "t".to_string()),
            ("e".to_string(), "d".to_string()),
            ("u".to_string(), "n".to_string()),
            ("at".to_string(), "ed".to_string()),
            ("re".to_string(), "l".to_string()),
            ("rel".to_string(), "ated".to_string()),
            ("un".to_string(), "related".to_string()),
        ];
        let mut bpe = BPE::new(vocab, merges);

        // 不使用dropout：应该完全合并
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

        // dropout = 0.0（等价于没有dropout）
        bpe.dropout = Some(0.0);
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

        // 现在设置dropout为1.0。结果应该是不执行任何合并
        bpe.dropout = Some(1.0);
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(0u32, "u".into(), (0, 1)),
                Token::new(1u32, "n".into(), (1, 2)),
                Token::new(2u32, "r".into(), (2, 3)),
                Token::new(3u32, "e".into(), (3, 4)),
                Token::new(4u32, "l".into(), (4, 5)),
                Token::new(5u32, "a".into(), (5, 6)),
                Token::new(6u32, "t".into(), (6, 7)),
                Token::new(3u32, "e".into(), (7, 8)),
                Token::new(7u32, "d".into(), (8, 9)),
            ]
        );

        // 现在尝试0到1之间的dropout
        bpe.dropout = Some(0.5);
        let tokens = bpe.tokenize("unrelated").unwrap();
        // 结果应该在1到9个token之间（随机）
        assert!(!tokens.is_empty() && tokens.len() <= 9);
    }

    #[test]
    // 确保`BPE::from_file`按预期工作
    fn test_bpe_from_file() {
        // 设置词汇表文件
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all(b"{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}")
            .unwrap();

        // 设置合并规则文件
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b").unwrap();

        // 确保我们可以从文件实例化BPE模型
        let builder = BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        );
        let bpe = builder.build().unwrap();

        // 检查合并规则
        assert_eq!(bpe.merges.get(&(0, 1)).unwrap(), &(0u32, 3u32));

        // 检查词汇表
        assert_eq!(bpe.vocab.get("a").unwrap(), &0u32);
        assert_eq!(bpe.vocab.get("b").unwrap(), &1u32);
        assert_eq!(bpe.vocab.get("c").unwrap(), &2u32);
        assert_eq!(bpe.vocab.get("ab").unwrap(), &3u32);
    }

    #[test]
    // 确保BPEBuilder使用dropout = 0.0不会出错
    fn test_bpe_with_dropout_0() {
        let bpe = BPE::builder().dropout(0.0).build().unwrap();
        assert_eq!(bpe.dropout, Some(0.0));
    }

    #[test]
    // 测试带有子词前缀的BPE
    fn test_bpe_with_continuing_subword_prefix() {
        let vocab: Vocab = vec![
            ("a".to_string(), 0),
            ("##b".to_string(), 1),
            ("##c".to_string(), 2),
            ("ab".to_string(), 3),
            ("abc".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let merges = vec![
            ("a".to_string(), "##b".to_string()),
            ("ab".to_string(), "##c".to_string()),
        ];

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .build()
            .unwrap();

        let res = bpe.tokenize("ab");
        assert_eq!(
            res.unwrap(),
            vec![Token {
                id: 3,
                value: "ab".to_string(),
                offsets: (0, 2)
            }]
        );
        let res = bpe.tokenize("abc");
        assert_eq!(
            res.unwrap(),
            vec![Token {
                id: 4,
                value: "abc".to_string(),
                offsets: (0, 3)
            }]
        );
    }

    #[test]
    // 确保在应该返回`MergeTokenOutOfVocabulary`错误时正确返回
    fn test_bpe_from_file_merge_token_oov() {
        // 设置词汇表文件
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all(b"{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}")
            .unwrap();

        // 设置合并规则文件（包含词汇表中不存在的token "d"）
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b\na d").unwrap();

        // 确保BPE::from_file的结果是MergeTokenOutOfVocabulary错误
        match BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        )
        .build()
        {
            Ok(_) => unreachable!(),
            Err(err) => match err.downcast_ref::<Error>() {
                Some(Error::MergeTokenOutOfVocabulary(token)) => {
                    assert_eq!(*token, String::from("d"))
                }
                _ => unreachable!(),
            },
        }
    }

    #[test]
    // 确保当merges.txt文件中有无效行时返回`BadMerges`错误
    fn test_bpe_from_file_bad_merges() {
        // 设置词汇表文件
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all("{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}".as_bytes())
            .unwrap();

        // 设置包含错误行的合并规则文件（"c"这一行只有一个token，应该有两个）
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b\nc").unwrap();

        // 确保BPE::from_file的结果是BadMerges错误
        match BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        )
        .build()
        {
            Ok(_) => unreachable!(),
            Err(err) => match err.downcast_ref::<Error>() {
                Some(Error::BadMerges(line)) => assert_eq!(*line, 2),
                _ => unreachable!(),
            },
        }
    }

    #[test]
    fn test_bpe_byte_fallback() {
        // 测试字节回退功能
        // 0x61 == 'a'的字节表示
        let vocab: Vocab = [("<unk>".into(), 0), ("<0x61>".into(), 1)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .byte_fallback(true)  // 启用字节回退
            .build()
            .unwrap();
        // 'c'没有对应的字节token，使用UNK
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        // 'a'有对应的字节token <0x61>
        let tokens = bpe.tokenize("a").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "<0x61>".into(), (0, 1)),]);
    }

    #[test]
    fn test_bpe_byte_fallback_newline() {
        // 测试换行符的字节回退
        // 0x0A == '\n'的字节表示
        let vocab: Vocab = [("<unk>".into(), 0), ("<0x0A>".into(), 1)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .byte_fallback(true)
            .build()
            .unwrap();
        // 换行符被转换为<0x0A>
        let tokens = bpe.tokenize("\n").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "<0x0A>".into(), (0, 1)),]);
    }

    #[test]
    fn test_ignore_merges() {
        // 测试ignore_merges功能：直接输出词汇表中的完整词，跳过BPE合并
        let vocab: Vocab = [
            (".:.:".into(), 0),
            ("Ġbelirtilen".into(), 1),
            (".".into(), 2),
            (":".into(), 3),
            ("bel".into(), 4),
            ("irtilen".into(), 5),
            ("Ġ".into(), 6),
            (".:".into(), 7),
            ("belirtilen".into(), 8),
            (".:.".into(), 9),
            ("be".into(), 10),
            ("l".into(), 11),
            ("ir".into(), 12),
            ("ti".into(), 13),
            ("en".into(), 14),
            ("irtil".into(), 15),
            ("irti".into(), 16),
            ("i".into(), 17),
            ("r".into(), 18),
            ("t".into(), 19),
            ("b".into(), 20),
            ("e".into(), 21),
            ("n".into(), 22),
        ]
        .iter()
        .cloned()
        .collect();
        let mut bpe = BpeBuilder::default()
            .vocab_and_merges(
                vocab,
                vec![
                    (".".into(), ":".into()),
                    ("b".into(), "e".into()),
                    ("be".into(), "l".into()),
                    ("i".into(), "r".into()),
                    ("t".into(), "i".into()),
                    ("ir".into(), "ti".into()),
                    ("e".into(), "n".into()),
                    ("irti".into(), "l".into()),
                ],
            )
            .ignore_merges(true)  // 启用ignore_merges
            .build()
            .unwrap();
        // 启用ignore_merges时，直接返回词汇表中的完整词
        let tokens = bpe.tokenize(".:.:").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, ".:.:".into(), (0, 4))]);

        let tokens = bpe.tokenize("Ġbelirtilen").unwrap();
        assert_eq!(
            tokens,
            vec![Token::new(1u32, "Ġbelirtilen".into(), (0, 12))]
        );

        // 禁用ignore_merges，执行正常的BPE合并
        bpe.ignore_merges = false;

        let tokens = bpe.tokenize(".:.:").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(7u32, ".:".into(), (0, 2)),
                Token::new(7u32, ".:".into(), (2, 4))
            ]
        );

        let tokens = bpe.tokenize("Ġbelirtilen").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token {
                    id: 6,
                    value: "Ġ".into(),
                    offsets: (0, 2)
                },
                Token {
                    id: 4,
                    value: "bel".into(),
                    offsets: (2, 5)
                },
                Token {
                    id: 15,
                    value: "irtil".into(),
                    offsets: (5, 10)
                },
                Token {
                    id: 14,
                    value: "en".into(),
                    offsets: (10, 12)
                }
            ]
        )
    }
}
