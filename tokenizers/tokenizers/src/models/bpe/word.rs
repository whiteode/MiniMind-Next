// 导入BPE相关类型和工具
use super::Pair;
use ahash::AHashMap;
use dary_heap::QuaternaryHeap;  // 四叉堆，用于优先队列
use rand::{rng, Rng};
use std::cmp::Ordering;

// 合并操作结构体：表示一个待执行的合并
#[derive(Debug, Eq)]
struct Merge {
    pos: usize,      // 合并位置（在符号数组中的索引）
    rank: u32,       // 合并优先级（rank越小越优先）
    new_id: u32,     // 合并后的新符号ID
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.pos == other.pos
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // 通过手动实现这个，我们使包含的BinaryHeap成为
        // 首先按rank排序的最小堆，否则按pos排序
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.rank != other.rank {
            // rank小的优先（降序）
            other.rank.cmp(&self.rank)
        } else {
            // rank相同时，pos大的优先（降序）
            other.pos.cmp(&self.pos)
        }
    }
}

// 符号结构体：表示单词中的一个token
#[derive(Debug, Clone, Copy)]
struct Symbol {
    c: u32,          // 符号ID（token ID）
    prev: isize,     // 前一个符号的索引（-1表示没有）
    next: isize,     // 后一个符号的索引（-1表示没有）
    len: usize,      // 符号的字节长度
}
impl Symbol {
    /// 将当前符号与另一个符号合并
    /// 为了更新prev/next，我们认为Self是左边的符号，
    /// other是右边的下一个符号
    pub fn merge_with(&mut self, other: &Self, new_c: u32) {
        self.c = new_c;              // 更新为新的符号ID
        self.len += other.len;       // 累加长度
        self.next = other.next;      // 继承右边符号的next指针
    }
}

// Word结构体：表示一个单词，由符号序列组成
#[derive(Clone, Default)]
pub(super) struct Word {
    symbols: Vec<Symbol>,  // 符号序列（使用双向链表结构）
}
impl std::fmt::Debug for Word {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Word")
            .field(
                "chars",
                &self
                    .symbols
                    .iter()
                    .map(|s| s.c.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )
            .field("symbols", &self.symbols)
            .finish()
    }
}

impl Word {
    /// 创建一个新的空Word
    pub(super) fn new() -> Self {
        Word { symbols: vec![] }
    }

    /// 创建一个具有指定容量的Word
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(capacity),
        }
    }

    /// 向Word添加一个新符号
    pub(super) fn add(&mut self, c: u32, byte_len: usize) {
        let (prev, next) = {
            let len = self.symbols.len() as isize;
            if let Some(last) = self.symbols.last_mut() {
                // 更新前一个符号的`next`指针
                last.next = len;
                (len - 1, -1)
            } else {
                (-1, -1)
            }
        };
        self.symbols.push(Symbol {
            c,
            prev,
            next,
            len: byte_len,
        });
    }

    /// 合并单词中所有出现的指定字符对
    /// 返回受影响的字符对及其计数变化
    pub(super) fn merge(
        &mut self,
        c1: u32,
        c2: u32,
        replacement: u32,
        max_length: usize,
    ) -> Vec<(Pair, i32)> {
        let mut changes: Vec<(Pair, i32)> = vec![];
        let mut i = 0;
        loop {
            if i >= self.symbols.len() {
                break;
            }

            // 找到一个字符对
            if self.symbols[i].c == c1 && i + 1 < self.symbols.len() && self.symbols[i + 1].c == c2
            {
                let first = self.symbols[i];
                let second = self.symbols[i + 1];

                // 就地移除并替换
                let new_s = Symbol {
                    c: replacement,
                    prev: first.prev,
                    next: second.next,
                    len: first.len + second.len,
                };

                // 如果字符对前面还有其他字符
                if i > 0 {
                    changes.push(((self.symbols[i - 1].c, first.c), -1));
                    if self.symbols[i - 1].len + new_s.len < max_length {
                        changes.push(((self.symbols[i - 1].c, replacement), 1));
                    }
                }

                self.symbols.insert(i, new_s); // 在字符对的第一个字符前插入替换符号
                self.symbols.remove(i + 1); // 移除字符对的第一个字符
                self.symbols.remove(i + 1); // 然后移除第二个字符

                // 如果字符对后面还有其他字符
                if i < self.symbols.len() - 1 {
                    changes.push(((second.c, self.symbols[i + 1].c), -1));
                    if self.symbols[i + 1].len + new_s.len < max_length {
                        changes.push(((replacement, self.symbols[i + 1].c), 1));
                    }
                }
            }

            i += 1;
        }

        changes
    }

    /// 对单词应用所有BPE合并规则
    /// merges: 合并映射表，包含(字符对) -> (优先级rank, 新token ID)
    /// dropout: 可选的dropout概率，用于训练时增加随机性
    pub(super) fn merge_all(&mut self, merges: &AHashMap<Pair, (u32, u32)>, dropout: Option<f32>) {
        // 创建优先队列，用于按rank顺序处理合并操作
        let mut queue = QuaternaryHeap::with_capacity(self.symbols.len());
        // 用于存储因dropout而跳过的合并操作
        let mut skip = Vec::with_capacity(queue.len());

        // 初始化队列：遍历所有相邻的符号对
        queue.extend(
            self.symbols
                .windows(2)  // 滑动窗口，每次取两个相邻符号
                .enumerate()
                .filter_map(|(index, window)| {
                    let pair = (window[0].c, window[1].c);
                    // 如果这个字符对在合并表中存在，创建一个Merge对象
                    merges.get(&pair).map(|m| Merge {
                        pos: index,      // 合并位置
                        rank: m.0,       // 合并优先级
                        new_id: m.1,     // 合并后的新符号ID
                    })
                }),
        );

        // 主循环：从优先队列中取出并处理合并操作
        while let Some(top) = queue.pop() {
            // Dropout机制：以一定概率跳过当前合并
            if dropout.map(|d| rng().random::<f32>() < d).unwrap_or(false) {
                // 将跳过的合并暂存到skip列表
                skip.push(top);
            } else {
                // 重新插入之前跳过的合并操作
                queue.extend(skip.drain(..));

                // 检查1：如果符号已被标记为删除（len=0），跳过
                if self.symbols[top.pos].len == 0 {
                    continue;
                }
                // 检查2：如果是最后一个符号（没有next），无法合并，跳过
                // Do nothing if we are the last symbol
                if self.symbols[top.pos].next == -1 {
                    continue;
                }

                // 获取右边符号的位置和内容
                let next_pos = self.symbols[top.pos].next as usize;
                let right = self.symbols[next_pos];

                // 检查3：确保队列中的合并操作仍然有效（未过期）
                // 因为符号可能已经被之前的合并改变了
                // Make sure we are not processing an expired queue entry
                let target_new_pair = (self.symbols[top.pos].c, right.c);
                if merges
                    .get(&target_new_pair)
                    .is_none_or(|(_, new_id)| *new_id != top.new_id)
                {
                    continue;
                }

                // 执行合并：将当前符号与右边符号合并
                // Otherwise, let's merge
                self.symbols[top.pos].merge_with(&right, top.new_id);
                // 标记右边符号为已删除（通过设置len=0）
                // Tag the right part as removed
                self.symbols[next_pos].len = 0;

                // 更新双向链表：如果合并后的符号有next，更新其prev指针
                // Update `prev` on the new `next` to the current pos
                if right.next > -1 && (right.next as usize) < self.symbols.len() {
                    self.symbols[right.next as usize].prev = top.pos as isize;
                }

                // 合并完成后，需要将新形成的字符对加入队列
                // Insert the new pair formed with the previous symbol
                let current = &self.symbols[top.pos];
                if current.prev >= 0 {
                    // 如果当前符号有前驱符号，检查(前驱, 当前)是否可以合并
                    let prev = current.prev as usize;
                    let prev_symbol = self.symbols[prev];
                    let new_pair = (prev_symbol.c, current.c);
                    // 如果新字符对在合并表中存在，加入优先队列
                    if let Some((rank, new_id)) = merges.get(&new_pair) {
                        queue.push(Merge {
                            pos: current.prev as usize,
                            rank: *rank,
                            new_id: *new_id,
                        });
                    }
                }

                // 检查(当前, 后继)是否可以合并
                // Insert the new pair formed with the next symbol
                let next = current.next as usize;
                if next < self.symbols.len() {
                    // 如果当前符号有后继符号，检查(当前, 后继)是否可以合并
                    let next_symbol = self.symbols[next];
                    let new_pair = (current.c, next_symbol.c);
                    // 如果新字符对在合并表中存在，加入优先队列
                    if let Some((rank, new_id)) = merges.get(&new_pair) {
                        queue.push(Merge {
                            pos: top.pos,
                            rank: *rank,
                            new_id: *new_id,
                        });
                    }
                }
            }
        }

        // 过滤掉所有被标记为删除的符号（len=0）
        // Filter out the removed symbols
        self.symbols.retain(|s| s.len != 0);
    }

    /// 获取单词中所有符号的ID列表
    pub(super) fn get_chars(&self) -> Vec<u32> {
        self.symbols.iter().map(|s| s.c).collect()
    }

    /// 获取单词中所有符号ID的迭代器
    pub(super) fn get_chars_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.symbols.iter().map(|s| s.c)
    }

    /// 获取单词中每个符号的字节偏移量迭代器
    /// 返回(起始位置, 结束位置)元组
    pub(super) fn get_offsets_iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut pos = 0;
        self.symbols.iter().map(move |symbol| {
            let new_pos = pos + symbol.len;
            let offset = (pos, new_pos);
            pos = new_pos;
            offset
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        // 测试merge方法：合并单词中的字符对
        // 假设我们有单词'hello'，词汇表为: {'h': 0, 'e': 1, 'l': 2, 'o': 3}
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'

        // 对字符对('l', 'l') ~= (2, 2)执行合并操作
        // 假设'll'在更新后的词汇表中的ID为4
        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let changes = word.merge(2, 2, 4, usize::MAX);

        // 合并后，单词应该变成这样：
        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        // 返回值`changes`用于在训练期间更新字符对计数
        // 这次合并影响了以下字符对的计数：
        // ('e', 'l') ~= (1, 2),
        // ('e', 'll') ~= (1, 4),
        // ('l', 'o') ~= (2, 3), 和
        // ('ll', 'o') ~= (4, 3).
        // changes应该反映这些变化：
        // The return value `changes` will be used to update the pair counts during
        // training. This merge affects the counts for the pairs
        // ('e', 'l') ~= (1, 2),
        // ('e', 'll') ~= (1, 4),
        // ('l', 'o') ~= (2, 3), and
        // ('ll', 'o') ~= (4, 3).
        // So the changes should reflect that:
        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // ('e', 'l')的计数应减少1 / count for ('e', 'l') should be decreased by 1.
                ((1u32, 4u32), 1i32),  // ('e', 'll')的计数应增加1 / count for ('e', 'll') should be increased by 1.
                ((2u32, 3u32), -1i32), // ('l', 'o')的计数应减少1 / count for ('l', 'o') should be decreased by 1.
                ((4u32, 3u32), 1i32),  // ('ll', 'o')的计数应增加1 / count for ('ll', 'o') should be increased by 1.
            ]
        );
    }

    #[test]
    fn test_merge_max_length() {
        // 测试带有max_length限制的merge方法
        // 同样使用单词'hello'，词汇表为: {'h': 0, 'e': 1, 'l': 2, 'o': 3}
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'

        // 对字符对('l', 'l') ~= (2, 2)执行合并，但设置max_length=2
        // 这意味着如果新形成的字符对长度超过2，就不会被添加到changes中
        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let changes = word.merge(2, 2, 4, 2);
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        // 由于max_length=2的限制，某些新字符对不会被添加
        // ('e', 'll')和('ll', 'o')的长度都超过了2，所以不会出现在changes中
        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // ('e', 'l')的计数应减少1 / count for ('e', 'l') should be decreased by 1.
                // ((1u32, 4u32), 1i32),  缺失，因为长度会超过2 / Missing since this would be larger than 2
                ((2u32, 3u32), -1i32), // ('l', 'o')的计数应减少1 / count for ('l', 'o') should be decreased by 1.
                                       // ((4u32, 3u32), 1i32), 缺失，因为长度会超过2 / Missing since this would be larger than 2
            ]
        );
    }
}
