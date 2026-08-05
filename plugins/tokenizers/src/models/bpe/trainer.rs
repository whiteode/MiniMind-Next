#![allow(clippy::map_entry)]

// 导入BPE相关类型和工具
use super::{Pair, WithFirstLastIterator, Word, BPE};
use crate::parallelism::*;
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::progress::{ProgressBar, ProgressFormat, ProgressStyle};
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;  // 八叉堆，用于高效的优先队列
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;

// 合并操作结构体：表示一个待执行的BPE合并
#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,              // 要合并的字符对
    count: u64,              // 该字符对在语料中出现的次数
    pos: AHashSet<usize>,    // 包含该字符对的单词位置集合
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            // 按出现次数降序排列（次数多的优先）
            self.count.cmp(&other.count)
        } else {
            // 次数相同时，按字符对升序排列（保证确定性）
            other.pair.cmp(&self.pair)
        }
    }
}

// BPE训练器配置结构体
struct Config {
    min_frequency: u64,                          // 最小频率：字符对出现次数低于此值不会被合并
    vocab_size: usize,                           // 目标词汇表大小
    show_progress: bool,                         // 是否显示训练进度
    progress_format: ProgressFormat,             // 进度输出格式
    special_tokens: Vec<AddedToken>,             // 特殊token列表
    limit_alphabet: Option<usize>,               // 限制初始字母表大小
    initial_alphabet: AHashSet<char>,            // 初始字母表（必须包含的字符）
    continuing_subword_prefix: Option<String>,   // 子词前缀
    end_of_word_suffix: Option<String>,          // 词尾后缀
    max_token_length: Option<usize>,             // 单个token的最大长度限制
}

/// `BpeTrainerBuilder`可用于创建具有自定义配置的`BpeTrainer`
pub struct BpeTrainerBuilder {
    config: Config,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                min_frequency: 0,                           // 默认不限制最小频率
                vocab_size: 30000,                          // 默认词汇表大小30000
                show_progress: true,                        // 默认显示进度
                progress_format: ProgressFormat::default(), // 默认进度格式
                special_tokens: vec![],                     // 默认无特殊token
                limit_alphabet: None,                       // 默认不限制字母表
                initial_alphabet: AHashSet::new(),          // 默认空初始字母表
                continuing_subword_prefix: None,            // 默认无子词前缀
                end_of_word_suffix: None,                   // 默认无词尾后缀
                max_token_length: None,                     // 默认不限制token长度
            },
        }
    }
}

impl BpeTrainerBuilder {
    /// 构造一个新的`BpeTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置期望的最小频率
    /// 字符对出现次数低于此值不会被合并
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    /// 设置词汇表大小
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// 设置是否显示进度
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    /// 设置进度输出格式
    ///
    /// 控制训练期间如何报告进度信息：
    /// - `Indicatif`（默认）：交互式终端进度条
    /// - `JsonLines`：机器可读的JSON行输出到stderr
    /// - `Silent`：无进度输出
    #[must_use]
    pub fn progress_format(mut self, format: ProgressFormat) -> Self {
        self.config.progress_format = format;
        self
    }

    /// 设置特殊token列表
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// 设置是否限制字母表大小
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    /// 设置初始字母表
    /// 这些字符会被强制包含在词汇表中
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        let mut initial_alphabet = AHashSet::with_capacity(alphabet.len());
        initial_alphabet.extend(alphabet);
        self.config.initial_alphabet = initial_alphabet;
        self
    }

    /// 设置子词前缀（如"##"）
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// 设置词尾后缀（如"</w>"）
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }
    /// 设置单个token的最大长度限制
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }

    /// 构造最终的BpeTrainer
    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            progress_format: self.config.progress_format,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            words: AHashMap::new(),
        }
    }
}

/// 负责训练`BPE`模型
///
/// # 示例
///
/// ```
/// use tokenizers::tokenizer::Trainer;
/// use tokenizers::models::bpe::{BPE, BpeTrainer};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// let mut model = BPE::default();
/// let special_tokens = trainer.train(&mut model).unwrap();
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct BpeTrainer {
    /// 字符对必须达到的最小频率才能产生合并操作
    pub min_frequency: u64,
    /// 目标词汇表大小
    pub vocab_size: usize,
    /// 训练时是否显示进度
    pub show_progress: bool,
    /// 进度输出格式（Indicatif、JsonLines或Silent）
    pub progress_format: ProgressFormat,
    /// 模型应该知道的特殊token列表
    pub special_tokens: Vec<AddedToken>,
    /// 是否限制在计算合并前可以保留的初始token数量
    pub limit_alphabet: Option<usize>,
    /// 我们绝对想要包含的初始字母表
    /// 这允许覆盖一些不一定在训练集中的字符
    pub initial_alphabet: AHashSet<char>,
    /// 可选的前缀，用于标记只存在于另一个子词后面的子词
    pub continuing_subword_prefix: Option<String>,
    /// 可选的后缀，用于标记词尾子词
    pub end_of_word_suffix: Option<String>,
    /// 可选参数，限制任何单个token的最大长度
    pub max_token_length: Option<usize>,

    words: AHashMap<CompactString, u64>,  // 单词计数映射
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl BpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// 返回语料库中唯一单词的数量（在feed之后）
    /// 可用于在开始训练前估计训练时间
    pub fn get_word_count(&self) -> usize {
        self.words.len()
    }

    /// 如果要求显示进度，则设置进度条（仅适用于Indicatif格式）
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress && self.progress_format == ProgressFormat::Indicatif {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// 向stderr输出JSON格式的进度行（用于JsonLines格式）
    fn emit_json_progress(&self, stage: &str, current: usize, total: usize) {
        if self.progress_format == ProgressFormat::JsonLines {
            eprintln!(
                r#"{{"stage":"{}","current":{},"total":{}}}"#,
                stage, current, total
            );
        }
    }

    /// 将进度条设置为完成状态
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize, stage: &str) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
        self.emit_json_progress(stage, final_len, final_len);
    }

    /// 使用新提供的长度和消息更新进度条
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
        // 为此阶段输出初始JSON进度
        self.emit_json_progress(message, 0, len);
    }

    /// 将提供的特殊token添加到初始词汇表
    fn add_special_tokens(
        &self,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        for token in &self.special_tokens {
            // 获取内容的哈希
            if !w2id.contains_key(&CompactString::from(&token.content)) {
                id2w.push(CompactString::from(&token.content));
                w2id.insert(CompactString::from(&token.content), (id2w.len() - 1) as u32);
            }
        }
    }

    /// 计算初始字母表，并在相关时进行限制
    fn compute_alphabet(
        &self,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        // 从看到的单词中计算字母表
        let mut alphabet: AHashMap<char, usize> = AHashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                *alphabet.entry(c).or_default() += *count as usize;
            }
        }

        // 同时包含提供的初始字母表中的任何内容
        for c in &self.initial_alphabet {
            *alphabet.entry(*c).or_default() = usize::MAX;
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // 计算需要从字母表中移除的字符数量
        // 如果`limit_alphabet < initial_alphabet.len()`，
        // 一些初始字符将被移除
        let to_remove = self
            .limit_alphabet
            .map(|limit| alphabet.len().saturating_sub(limit))
            .unwrap_or(0);

        // 移除不需要的字符
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // 保留初始字母表（为确定性排序）
        kept.sort_unstable_by_key(|k| *k.0 as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            /*
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s, (id2w.len() - 1) as u32);
            }
            */
            // u64哈希版本
            if !w2id.contains_key(&CompactString::from(&s)) {
                id2w.push(CompactString::from(&s));
                w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
            }
        });
    }

    /// 对单词进行分词，并在相关时将子词添加到词汇表
    fn tokenize_words(
        &self,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
        p: &Option<ProgressBar>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count);

            // 遍历单词的每个字符，标记是否为首字符和尾字符
            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                if w2id.contains_key(&CompactString::from(&s)) {
                    // 在授权字母表中找到了初始字符

                    // 如果相关，添加`continuing_subword_prefix`
                    if !is_first {
                        if let Some(prefix) = &self.continuing_subword_prefix {
                            s.insert_str(0, prefix);
                        }
                    }
                    // 如果相关，添加`end_of_word_suffix`
                    if is_last {
                        if let Some(suffix) = &self.end_of_word_suffix {
                            s.push_str(suffix);
                        }
                    }

                    // 如果必要，插入新形成的字符串
                    if !w2id.contains_key(&CompactString::from(&s)) {
                        id2w.push(CompactString::from(&s));
                        w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&CompactString::from(&s)], 1); // 这里不关心长度
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }

        (words, counts)
    }

    /// 统计单词中的字符对出现次数
    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
        p: &Option<ProgressBar>,
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        words
            .maybe_par_iter()  // 可能并行迭代（取决于配置）
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                // 使用滑动窗口遍历单词中的相邻字符对
                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // 如果刚看到这个字符对，初始化pair_counts和where_to_update
                    // 然后更新计数
                    *pair_counts.entry(cur_pair).or_default() += counts[i] as i32;
                    where_to_update.entry(cur_pair).or_default().insert(i);
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    // 合并各个线程的结果
                    for (k, v) in pc {
                        *pair_counts.entry(k).or_default() += v;
                    }
                    for (k, v) in wtu {
                        where_to_update.entry(k).or_default().extend(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    /// 执行BPE训练的核心方法
    pub fn do_train(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        model: &mut BPE,
    ) -> Result<Vec<AddedToken>> {
        let mut word_to_id: AHashMap<CompactString, u32> = AHashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<CompactString> = Vec::with_capacity(self.vocab_size);
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress();

        //
        // 1. 将所有特殊token添加到词汇表
        //
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. 计算初始字母表
        //
        self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);

        //
        // 3. 对单词进行分词
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (mut words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &progress);
        self.finalize_progress(&progress, words.len(), "Tokenize words");

        //
        // 4. 统计单词中的字符对
        //
        self.update_progress(&progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);
        // 将它们插入优先队列
        let mut queue = OctonaryHeap::with_capacity(pair_counts.len());
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        });
        self.finalize_progress(&progress, words.len(), "Count pairs");

        //
        // 5. 执行合并操作
        //
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        let mut merges: Vec<(Pair, u32)> = vec![];
        loop {
            // 一旦词汇表足够大就停止
            if word_to_id.len() >= self.vocab_size {
                break;
            }

            let Some(mut top) = queue.pop() else {
                break;
            };

            // 检查计数是否仍然有效（可能已过时）
            if top.count != pair_counts[&top.pair] as u64 {
                top.count = pair_counts[&top.pair] as u64;
                queue.push(top);
                continue;
            }

            // 如果计数太低或低于最小频率，停止
            if top.count < 1 || self.min_frequency > top.count {
                break;
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].as_str();

            // 构建新token
            if let Some(prefix) = &self.continuing_subword_prefix {
                if let Some(rest) = part_b.strip_prefix(prefix) {
                    part_b = rest;
                }
            }

            // 如果新token不存在，则插入
            let new_token = format!("{part_a}{part_b}");
            let new_token_id = word_to_id
                .get(&CompactString::from(&new_token))
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if !word_to_id.contains_key(&CompactString::from(&new_token)) {
                id_to_word.push(CompactString::from(&new_token));
                word_to_id.insert(CompactString::from(&new_token), new_token_id);
            }
            merges.push((top.pair, new_token_id));

            // 在每个单词中合并新的字符对
            // 安全性：这只是一个类型断言，如果`pos`的类型改变，下面的代码可能不再安全
            let pos: &AHashSet<usize> = &top.pos;

            let words_len = words.len();
            struct WordPtr(*mut Word);
            // 安全性：我们实际上不使用这个进行对同一内存的并发访问，
            // 只用于访问同一分配中的不同块
            unsafe impl Sync for WordPtr {}
            let word_start = WordPtr(words.as_mut_ptr());

            let changes = pos
                .maybe_par_iter()
                .flat_map(|&i| {
                    // 我们可以在这里并行合并每个单词，因为每个位置
                    // 只能出现一次（AHashSet）。所以这是安全的。
                    unsafe {
                        assert!(i < words_len);
                        // 这是words[i]，但避免通过&T（会触发UB）
                        let word = word_start.0.add(i);
                        // let word: &mut Word = &mut (*word);
                        (*word)
                            .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                            .into_iter()
                            .map(|c| (c, i))
                            .collect::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();

            // 引入新形成的字符对
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i32;
                *pair_counts.entry(pair).or_default() += count;
                if change > 0 {
                    where_to_update.entry(pair).or_default().insert(iw);
                }
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Merge {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            });

            if let Some(p) = &progress {
                p.inc(1);
            }
            self.emit_json_progress("Compute merges", merges.len(), self.vocab_size);
        }
        self.finalize_progress(&progress, merges.len(), "Compute merges");

        // 将新词汇表和选项转移到模型
        //model.vocab = word_to_id;
        model.vocab = word_to_id
            .into_iter()
            // 我们必须在id_to_word中查找字符串，因为word_to_id中的键是哈希
            .map(|(_key, val)| (id_to_word[val as usize].to_string(), val))
            .collect();
        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        model.merges = merges
            .into_iter()
            .enumerate()
            .map(|(i, (pair, new_token_id))| (pair, (i as u32, new_token_id)))
            .collect();

        model.continuing_subword_prefix = self.continuing_subword_prefix.clone();
        model.end_of_word_suffix = self.end_of_word_suffix.clone();

        Ok(self.special_tokens.clone())
    }
}

// 为BpeTrainer实现Trainer trait
impl Trainer for BpeTrainer {
    type Model = BPE;

    /// 训练BPE模型
    fn train(&self, model: &mut BPE) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model)
    }

    /// 是否应该显示进度
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    /// 向训练器提供训练数据
    /// 接收一个迭代器，处理每个序列并统计单词频率
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<AHashMap<CompactString, u64>> = iterator
            .maybe_par_bridge()  // 可能并行处理
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = AHashMap::new();
                for word in words {
                    *map.entry(CompactString::from(word)).or_default() += 1;
                }
                Ok(map)
            })
            .reduce(
                || Ok(AHashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        *acc.entry(k).or_default() += v;
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }
}

// 测试模块
#[cfg(test)]
mod tests {
    use super::{BpeTrainer, Pair, BPE};
    use ahash::AHashMap;
    use compact_str::CompactString;

    #[test]
    fn test_train() {
        // 测试BPE训练功能
        let word_counts: AHashMap<CompactString, u64> = [
            ("roses".into(), 1),
            ("are".into(), 2),
            ("red".into(), 1),
            ("voilets".into(), 1),
            ("blue".into(), 1),
            ("BERT".into(), 1),
            ("is".into(), 2),
            ("big".into(), 1),
            ("and".into(), 1),
            ("so".into(), 1),
            ("GPT-2".into(), 1),
        ]
        .iter()
        .cloned()
        .collect();
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(2)  // 只合并出现至少2次的字符对
            .build();
        let mut model = BPE::default();
        trainer.do_train(&word_counts, &mut model).unwrap();

        // 词汇表应包含`word_counts`映射中的所有字符
        // 以及三个合并：'re'、'are'和'is'
        let expected_vocab: AHashMap<String, u32> = [
            ("-".into(), 0),
            ("2".into(), 1),
            ("B".into(), 2),
            ("E".into(), 3),
            ("G".into(), 4),
            ("P".into(), 5),
            ("R".into(), 6),
            ("T".into(), 7),
            ("a".into(), 8),
            ("b".into(), 9),
            ("d".into(), 10),
            ("e".into(), 11),
            ("g".into(), 12),
            ("i".into(), 13),
            ("l".into(), 14),
            ("n".into(), 15),
            ("o".into(), 16),
            ("r".into(), 17),
            ("s".into(), 18),
            ("t".into(), 19),
            ("u".into(), 20),
            ("v".into(), 21),
            ("re".into(), 22),
            ("are".into(), 23),
            ("is".into(), 24),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.vocab, expected_vocab);

        // `merges`中的键是符号对，值是(rank, id)元组，
        // 其中'rank'决定了在分词期间应用此合并的顺序，
        // 'id'是合并对应键中的符号对后得到的符号的词汇表ID
        let expected_merges: AHashMap<Pair, (u32, u32)> = [
            ((17, 11), (0, 22)), // 'r' + 'e'  -> 're'
            ((8, 22), (1, 23)),  // 'a' + 're' -> 'are'
            ((13, 18), (2, 24)), // 'i' + 's'  -> 'is'
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.merges, expected_merges);
    }
    #[test]
    fn bpe_test_max_token_length_16() {
        /* bpe_test_max_token_length系列测试用于测试bpetrainer的max_token_length标志
        // 这是更健壮的版本，只测试学习到的token的最大长度
        // （预）分词器设置或词汇表可以在必要时轻松修改
         */

        let max_token_length = 16;
        let long_word_counts: AHashMap<CompactString, u64> = [
            ("singlelongtokenwithoutcasechange", 2),
            ("singleLongTokenWithCamelCaseChange", 2),
            ("Longsingletokenwithpunctu@t!onwithin", 2),
            ("Anotherlongsingletokenwithnumberw1th1n", 2),
            ("짧은한글문자열짧은한", 2),             // 韩语 10字符
            ("긴한글문자열긴한글문자열긴한글문", 2), // 韩语 16字符
            ("短字符串短字符串短字", 2),             // 简体中文 10字符
            ("长字符串长字符串长字符串长字符串", 2), // 简体中文 16字符
            ("短い文字列短い文字列", 2),             // 日语 10字符
            ("長い文字列長い文字列長い文字列長", 2), // 日语 16字符
            ("so", 2),
            ("GPT-2", 2),
        ]
        .iter()
        .map(|(key, value)| (CompactString::from(key.to_string()), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(max_token_length))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let vocab = model.get_vocab();
        // 验证所有学习到的token长度不超过max_token_length
        for token in vocab.keys() {
            assert!(
                token.chars().count() <= max_token_length,
                "token too long : {} , chars().count() = {}",
                token,
                token.chars().count()
            )
        }
    }
    #[test]
    fn bpe_test_max_token_length_direct_assert() {
        /* bpe_test_max_token_length测试的更直接版本
        // 直接将token与已知的预期值进行比较
        // 可能不稳定，取决于特定设置或更改
         */
        let long_word_counts: AHashMap<CompactString, u64> = [
            ("sin", 2),
            ("Sin", 2),
            ("Lon", 2),
            ("Ano", 2),
            ("짧은한", 2),
            ("긴한글", 2),
            ("短字符", 2),
            ("长字符", 2),
            ("短い文", 2),
            ("長い文", 2),
            ("so", 2),
            ("GP", 2),
        ]
        .iter()
        .map(|(key, value)| (CompactString::from(key.to_string()), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(2))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = BPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let trained_vocab: AHashMap<String, u32> = model.get_vocab().into_iter().collect();
        let expected_vocab: AHashMap<String, u32> = [
            ("短", 12),
            ("n", 6),
            ("i", 5),
            ("s", 8),
            ("字符", 23),
            ("長", 14),
            ("긴", 17),
            ("い文", 22),
            ("L", 2),
            ("in", 21),
            ("o", 7),
            ("은한", 29),
            ("S", 4),
            ("P", 3),
            ("so", 27),
            ("符", 13),
            ("文", 11),
            ("字", 10),
            ("짧", 19),
            ("GP", 25),
            ("글", 16),
            ("G", 1),
            ("An", 24),
            ("长", 15),
            ("A", 0),
            ("Lo", 26),
            ("긴한", 28),
            ("い", 9),
            ("한", 20),
            ("은", 18),
        ]
        .iter()
        .cloned()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(trained_vocab, expected_vocab)
    }
}
