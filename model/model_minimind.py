# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig

class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型的配置类，用于定义模型的超参数、注意力机制配置以及混合专家(MoE)的详细设置。
    继承自 HuggingFace 的 PretrainedConfig，以便完美兼容 Transformers 库。
    """
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,                   # 全局的 Dropout 概率，用于防止过拟合
            # 如果开启 Dropout，在标准的 Transformer 架构中，它具体会发生在以下几个关键的“流水线交接处”：
            # 1. 词嵌入层之后 (After Embedding)
            # 当输入的文本刚刚被转换成词向量（Token Embeddings），还没正式进入第一个 Transformer Block 之前。
            # 效果：相当于随机抹掉句子中某些词的部分初始特征，强迫模型不要过度依赖某几个特定的词。
            # 2. 注意力机制内部 (Inside Attention)
            # 这是 Dropout 最常出没的地方之一。在计算完 Query 和 Key 的相似度（得到 Attention 分数）之后，乘以 Value 之前。
            # 效果：模型本来发现“苹果”和“吃”的关联度很高（比如 0.9），Dropout 会以一定概率直接把这个关联度变成 0。这等于告诉模型：“别老盯着‘苹果’看，也去看看句子里的其他词！”这能有效防止模型产生狭隘的注意力偏见。
            # 3. 残差连接之前 (Before Residual Add)
            # 回想一下我们上一节讲的 Transformer Block。无论是 Attention 模块还是 FFN 模块，它们计算完结果后，都要通过“残差连接 (Add)”并入主干道。Dropout 通常就设在这个并入主干道之前的大门口。
            # 效果：Attention 刚算完一堆复杂的特征，准备交给老板（下一层），结果走到门口，Dropout 随机抽走了报告里的几页纸。这就强迫模型的主干道（原始数据）必须保持强大，不能完全指望新算出来的特征。
            # 4. 前馈神经网络内部 (Inside FFN)
            # 在 FFN 内部的两层线性网络之间（通常在激活函数之后）。
            # 效果：在特征进行深层加工时，随机让一部分中间结果失效，确保没有任何一个单一的神经元变得“不可或缺”。
            # 进阶思考：为什么现代大模型都把 Dropout 设为 0.0？
            # 既然 Dropout 这么好，为什么你看到的这份 LLaMA/ChatGLM 风格的配置里，它却是 0.0 呢？
            # 数据量太大，根本“背不下来”：Dropout 主要是为了防止模型在小数据集上“死记硬背”。但现代大模型是在动辄几万亿 Token 的海量数据上训练的。在如此庞大的数据面前，模型面临的最大问题是“脑容量不够用”（欠拟合），而不是“死记硬背”（过拟合）。
            # 拖慢训练速度：Dropout 会让模型学习的效率变低（因为总有神经元在罢工），在动辄花费上千万美元算力的预训练阶段，关闭 Dropout 可以让模型收敛得更快。
            # 唯一的例外：当你拿到一个训练好的大模型，想要用自己的私有小数据集对它进行微调（Fine-tuning，比如 LoRA 微调）时，由于你的数据量很小，很容易过拟合。这时候，工程师们通常会把 Dropout 重新打开（比如设为 0.1）。

            bos_token_id: int = 1,                  # 序列起始符 (Begin of Sequence) 的 token ID
            eos_token_id: int = 2,                  # 序列结束符 (End of Sequence) 的 token ID
            # 1. bos_token_id = 1 (Begin of Sequence)
            # 它的含义：BOS 代表“序列的起始符”。在这个配置中，数字 1 被专门保留下来，作为这个特定起始信号的代码。
            # 它的作用：当模型准备阅读一段新的文本，或者准备开始回答你的问题时，系统会悄悄在最前面加上这个 1（或者对应的特殊字符）。
            # 通俗比喻：它就像是导演在片场喊的一句“Action！（开机）”。它在提醒模型：“注意了，这是一段全新的、独立的内容，请准备好开始处理。”
            # 2. eos_token_id = 2 (End of Sequence)
            # 它的含义：EOS 代表“序列的结束符”。在这里，数字 2 被指定为结束信号。
            # 它的作用（极其关键）：
            # 在训练时：当把两篇毫无关联的文章拼在一起喂给模型去学习时，工程师会在中间插一个 EOS。这能告诉模型：“上一篇到此结束，接下来的内容是另一篇文章了，别把它们强行联系在一起。”
            # 在生成回答时（推理）：这是决定模型什么时候闭嘴的关键所在！模型生成文本是像接龙一样，一个词一个词往外蹦的。如果不加以控制，它会一直喋喋不休地生成下去，直到达到之前讲过的 max_position_embeddings（比如 32768）的上限，变成满屏胡言乱语的“复读机”。
            # 当模型在某一步，经过计算认为下一个最应该输出的 Token 是 2（即 EOS）时，底层的程序就会立刻介入，打断模型的生成循环，并把已经生成好的话呈现给你。

            # 通俗比喻：它就像是导演喊的“Cut！（停）”，或者是对讲机通话结束时说的一句“Over（完毕）”。
            # 这两个数字是由模型“出厂时自带的分词器字典”决定的，它就像是不同操作系统的底层快捷键。换一个模型（分词器不同），这套数字密码就要跟着换。

            hidden_act: str = 'silu',               # 前馈神经网络(FFN)中使用的激活函数，通常配合 SwiGLU 使用
            # 在 Transformer 架构（目前大多数语言模型的基础）中，模型主要由两层交替组成：注意力层（Attention）和前馈神经网络层（FFN）。
            # 它的作用：如果说“注意力层”是为了让模型理解句子中不同单词之间的关系（比如“苹果”和“吃”的关系），那么“前馈神经网络”就是为了对每个单词的特征进行深加工和非线性转换，让模型能够学习到更复杂的规律。
            # 传统结构：标准的 FFN 通常包含两个线性变换（矩阵乘法），中间夹着一个激活函数。它的运算过程大致是：先把数据映射到一个更高维度的空间（提取更丰富的特征），通过激活函数过滤一下，然后再映射回原来的维度。
            # SiLU 也被称为 Swish 函数（由 Google 提出）。它的数学公式如下：f(x) = x *sigma(x) = x /{1 + e^{-x}}其中 sigma(x) 是 Sigmoid 函数。为什么用它：在早期的深度学习中，大家最常用的是 ReLU 函数（大于0就是原值，小于0就是0）。相比之下，SiLU 有几个显著的优势：平滑性：它的曲线非常平滑，这在数学上意味着它在训练（求导）时更加稳定。非单调性：在 x 略微小于 0 的区域，SiLU 允许出现一点点负值，而不是像 ReLU 那样直接“一刀切”变成 0。这被证明能帮助模型保留更多的微小梯度，从而在深层网络中表现更好。

            # SwiGLU 结构 (Swish-Gated Linear Unit)
            # SiLU（Sigmoid Linear Unit）通常与 SwiGLU 配合使用。SwiGLU 并不是一个单独的激活函数，而是一种改进版的前馈神经网络（FFN）结构，由学者 Noam Shazeer 提出。

            # 传统 FFN vs. SwiGLU：
            # - 传统 FFN 是串行的处理流程：  
            #    x -》{线性层} -》{激活函数} -》{线性层} 
            # - SwiGLU 引入了 “门控（Gating）”机制。它将输入 x  分成两路，分别进行线性变换：  
            #   - 一路通过 SiLU 激活函数，起到“门”的作用（控制信息通过的比例）；  
            #   - 另一路直接进行线性变换；  
            #   然后将两路结果进行逐元素相乘（Hadamard 积），实现门控效果。

            # 数学表达式如下：  
            #
            # {SwiGLU}(x, W_1, W_2) = {SiLU}(x * W_1) Hadamard (x * W_2)
            # Hadamar表示逐元素乘法
            # （注：后续通常还会用一个额外的线性层  W_3  将结果映射回目标维度）

            # 为什么现在都在用它？

            # 当前主流的开源大模型（如 LLaMA、ChatGLM 等）几乎全部将 Transformer 中的传统 FFN 替换为 SwiGLU。大量实验表明，尽管 SwiGLU 增加了参数量和计算开销，但其在模型性能上的提升（例如语言理解、生成能力等）显著优于传统的 ReLU 或 GELU 激活函数组合，因此成为现代大模型设计中的标准选择。
        
            hidden_size: int = 512,                 # 模型隐藏层(特征)的维度 d_model
            
            # 可以把 512 想象成模型内部流水线上的“标准集装箱大小”。无论数据走到哪里，都要装在这个大小的箱子里：
            # - 最初的起点（词嵌入层 Embedding）：当你输入“苹果”时，模型首先查字典，把“苹果”翻译成一个长度为 512 的数字列表。
            # - 在 Attention 模块中：这 512 个特征会被拆分给不同的“头”（之前讲过的 num_attention_heads = 8，那么每个头分到的维度就是 512 ÷ 8 = 64），算完注意力之后，再拼接回 512 维度。
            # - 在 FFN 模块中：数据为了进行深度加工，会先从 512 被“膨胀”到 intermediate_size（比如 1365），加工完后，必须再次被压缩回 512 维度。
            # - 在残差连接（Add）中：因为原始数据和处理后的数据都是 512 维度，它们才能完美地按位对齐相加。

            intermediate_size: int = None,          # FFN 的中间层维度，若为 None 则会在后续代码中自动计算 (通常是 hidden_size 的 8/3 倍)
            # 要理解这个“8/3 倍”的设定，先要理解两个问题：中间层指的是哪一层，以及为什么偏偏是 8/3。
            # ### 1. “中间层” (Intermediate Layer) 指的是哪一层？
            # 在 Transformer 模型中，数据在网络中流动的尺寸（特征向量的长度）主要由 `hidden_size`（隐藏层维度，通常记为 d）决定。
            # 但是在前馈神经网络（FFN）内部，为了让模型能够把特征“展开”去学习更复杂的细节，它会先将数据映射到一个更宽的维度，处理完后再压缩回原来的维度。这个“更宽的维度”，就是所谓的中间层维度 (`intermediate_size`)。
            # 形状变化过程：输入向量 (d) -》 扩大到中间层 (intermediate_size) -》 压缩回输出向量 (d)。
            # 你可以把它想象成一个“两头窄、中间宽”的橄榄球形状。模型在中间最宽的地方（中间层）进行非线性激活操作。



            # 在结合SwiGLU架构后，这里的“中间层”具体对应的是那三个权重矩阵：
            # 1. Gate 矩阵 ($W_1$)：把输入从 $d$ 维度映射到 $intermediate_size$ 维度。
            # 2. Up 矩阵 ($W_2$)：同样把输入从 $d$ 维度映射到 $intermediate_size$ 维度。
            # 3. Down 矩阵 ($W_3$)：把经过激活和相乘后的 $intermediate_size$ 维度的特征，重新降维回 $d$。


            # ### 2. 为什么通常是 `hidden_size` 的 8/3 倍？
            # 这是一个非常巧妙的“为了控制计算成本和参数量”的工程学设计。它的推导过程如下：

            # #### A. 传统 Transformer 的标准（对比基准）
            # 在经典的 Transformer（比如 GPT-2、GPT-3、BERT）中，传统的 FFN 只有两个矩阵，并且默认中间层是 hidden_size 的 4 倍。
            # * 参数量计算：两个矩阵的大小都是 d * 4d。
            # * 传统 FFN 总参数量 = (d * 4d) + (4d * d) = {8d^2}。

            # #### B. SwiGLU 带来的“超载”问题
            # 正如上一个回答中提到的，SwiGLU 使用了 3 个矩阵（Gate, Up, Down），而不是传统的 2 个。
            # 如果我们依然保持传统 Transformer 的 4 倍扩展率：
            # 3 个矩阵的参数量 = (d * 4d) * 3 = {12d^2}。
            # * 这会导致模型的参数量和计算量凭空暴增 50%！这在训练大模型时是极其昂贵的。

            # #### C. “8/3 魔法”的诞生（数学平衡）
            # 为了让使用 SwiGLU 的模型和传统 Transformer 在参数量和计算开销上保持一致（即公平对比），LLaMA 的作者们决定缩小 SwiGLU 的扩展倍率。

            # 我们要让 SwiGLU 的 3 个矩阵的总参数量，等于传统 FFN 的 2 个矩阵的总参数量（8d^2）：
            # 设新的中间层维度为 I。
            # SwiGLU 总参数量 = 3 * (d * I)
            # 让  3 * (d * I) = 8d^2
            # 推导出：I = 8/3 * d

            # 这就是为什么 `intermediate_size` 通常是 `hidden_size` 的 8/3 倍的原因所在！

            # ### 补充一个工程小细节 (对齐与加速)
            # 虽然理论上是 8/3 倍，但在实际的代码（如 LLaMA 的源码）中，如果你去看它计算 `intermediate_size` 的函数，通常还会多做一步：向上取整到 256 的倍数。

            # 例如，如果 d = 4096，那么 4096 * (8/3) == 10922。模型往往不会直接用 10922，而是把它调整到最接近的 256 的倍数（比如 11008）。
            # 这是因为像 NVIDIA GPU 这样的硬件底层设计（Tensor Cores），在处理矩阵维度是 64 或 256 的倍数时，计算效率是最高的。
            
            
            max_position_embeddings: int = 32768,   # 模型支持的最大上下文长度

            # ### 1. 硬件与算力极限（显存墙）  
            # 由于 Transformer 注意力机制的数学特性，计算量和显存占用会随着文本长度呈 平方级增长（O(N^2)）。当上下文长度被强行拉长时，极易导致 GPU 显存溢出（OOM），进而引发系统崩溃。

            # ### 2. 训练成本与数据稀缺  
            # 高质量的超长文本数据（例如几十万字的书籍、财报等）在现实中相对稀少。同时，使用此类超长文本训练大模型所需的算力成本极其高昂，许多公司难以承担，处于“烧不起”的经济压力之下。

            # ### 3. 数学机制的局限（位置编码）  
            # 模型依赖底层数学公式（如 RoPE 旋转位置编码）来捕捉词元之间的先后顺序。当输入序列长度远超模型训练时所见的最大长度时，该编码机制会“失效”，导致模型丧失逻辑一致性，甚至产生胡言乱语的现象。
            # """
            # 失效的根本原因：模型从未"见过"那个旋转角
            # 训练时，假设最大长度是 4096，模型接触过的旋转角范围是：
            # [0θ, 1θ, 2θ, ..., 4096θ]
            # 此时推理输入长度为 8000，位置 5000、6000、7000... 对应的旋转角：
            # 5000θ, 6000θ, 7000θ...
            # 模型在训练中从未见过这些角度值，它的注意力权重完全没有针对这些角度优化过。
            # """

            # ### 4. 算法底层的优化程度  
            # 实现 32K 甚至更长的上下文支持，必须依赖高度先进的底层算法优化技术，例如 FlashAttention 或 稀疏注意力机制，以有效压缩内存占用。若模型架构本身缺乏此类优化，则无法稳定支持长文本处理。


            num_attention_heads: int = 8,           # 多头注意力机制中的 Query 头数量
            
            # 以下是简化整理后的 Transformer 自注意力机制 7 步流程，结构清晰、语言通俗，便于理解：

            # ---

            # ### **第 1 步：输入嵌入（Embedding）**
            # - 一句话有 10 个词，每个词通过词嵌入层转为 512 维向量。
            # - 输入形状：`[10, 512]`（10 个词，每词 512 维）

            # ---

            # ### **第 2 步：生成 Q、K、V 矩阵（线性映射）**
            # - 通过乘以三个权重矩阵 $W^Q, W^K, W^V$（均为 `[512, 512]`），将输入映射为：
            # - Query（查询）、Key（键）、Value（值）
            # - 结果形状：Q、K、V 均为 `[10, 512]`

            # ---

            # ### **第 3 步：拆分多头（Reshape + Transpose）**
            # - 将 512 维拆成 8 个头，每个头 64 维：
            # - Reshape：`[10, 512]` → `[10, 8, 64]`
            # - Transpose：交换维度 → `[8, 10, 64]`
            # - 含义：8 个“小侦探”各自独立工作，每个关注 64 维特征。

            # ---

            # ### **第 4 步：计算注意力分数（Attention Scores）**
            # - 每个头计算：`Q × K^T`
            # - `[8, 10, 64] × [8, 64, 10]` → `[8, 10, 10]`
            # - 得到 8 个 `10×10` 的“关联打分表”，表示 10 个词两两之间的相关性。
            # - 缩放（除以 √64 = 8），再 Softmax 归一化为概率。

            # ---

            # ### **第 5 步：融合 Value 向量**
            # - 用注意力分数加权融合 Value：
            # - `[8, 10, 10] × [8, 10, 64]` → `[8, 10, 64]`
            # - 每个词吸收了全句相关信息，8 个头各自输出结果。

            # ---

            # ### **第 6 步：拼接多头输出（Concat）**
            # - 逆操作第 3 步：
            # - Transpose：`[8, 10, 64]` → `[10, 8, 64]`
            # - Reshape：拼接 8 个头 → `[10, 512]`
            # - 数据回到原始维度，但已融合多视角信息。

            # ---

            # ### **第 7 步：最终线性映射（Output Projection）**
            # - 乘以输出权重矩阵 $W^O$（`[512, 512]`）：
            # - `[10, 512] × [512, 512]` → `[10, 512]`
            # - 输出与输入同形，但已包含上下文感知的语义信息。

            # ---

            # ✅ **总结**：  
            # 数据从 `[10, 512]` 出发，经多头“分身”计算、交互融合，最终变回 `[10, 512]`，完成一次自注意力编码。整个过程高度并行，高效捕捉词间依赖。

            num_hidden_layers: int = 8,             # Transformer block 的层数

            #这个参数指的是 Transformer block（也就是模型的处理层）的数量。
            # 总结一个 Block 的标准流水线：输入 $\rightarrow$ Norm $\rightarrow$ Attention $\rightarrow$ Add (残差) $\rightarrow$ Norm $\rightarrow$ FFN (SwiGLU) $\rightarrow$ Add (残差) $\rightarrow$ 输出给下一层

            #RMSNorm 是一种特殊的归一化方法，和 LayerNorm 不同，它只计算均方根（RMS）而不计算均值。拿到计算好的均方根后，把原来的词向量 $x$ 除以这个均方根，然后再乘上一个可学习的权重参数 $g$.

            num_key_value_heads: int = 2,           # Key 和 Value 的头数量。用于实现 GQA (Grouped-Query Attention)。当其小于 Query 头数时即为 GQA

            # 1. 传统多头注意力（MHA）中，每个头的 $W^Q, W^K, W^V$ 都不一样吗？  

            # 是的，完全不一样！这也是多头注意力能生效的核心原因。  

            # **逻辑上的独立性**：如果 8 个头用的权重矩阵是一模一样的，那算出来的 8 份结果也会完全一样，这就失去“多头”的意义了。为了让 8 个“小侦探”掌握不同的查案技巧（比如一个查语法，一个查实体），模型在初始化时，赋予了这 8 个头完全独立、随机生成的权重参数。在海量数据的训练下，它们会自然而然地朝着不同的方向进化（即映射到不同的特征子空间）。  

            # **代码实现上的小把戏（切分法）**：在实际写代码时，为了让 GPU 算得更快，工程师不会真的去创建 8 个小的 $W^Q$ 矩阵（比如 8 个 [512, 64] 的矩阵）。相反，他们会只创建一个巨大的 $W^Q$ 矩阵（[512, 512]）。当输入 [10, 512] 乘以这个大矩阵得到结果后，再用代码将其硬生生切分成 8 块（每块 64 维）。因为这个大矩阵里的每一个数字都是不一样且独立训练的，所以切出来的 8 个小块，自然就相当于 8 个完全不同的权重参数了。  

            # ---

            # 2. 分组查询注意力（GQA）具体是怎么算的？  

            # 理解了前面的矩阵切分，GQA 的计算过程其实就像变魔术一样巧妙。它的核心思想是：“映射时偷懒，计算时复制。”  

            # 我们依然以你的配置为例：  
            # - `num_attention_heads = 8` （8 个 Q 头）  
            # - `num_key_value_heads = 2` （2 个 K 头，2 个 V 头）  

            # **分组情况**：$8 \div 2 = 4$。也就是每 4 个 Q 头组成一个小组，共享 1 个 K 头和 1 个 V 头。  

            # 接下来，我们跟着张量形状，一步步看它是怎么算的：  

            # **第 1 步：生成不对等的 Q、K、V**  

            # 在传统的 MHA 中，大家都要生成 512 维再切成 8 份。但在 GQA 里，$W^K$ 和 $W^V$ 变小了！  

            # - **计算 Q**：输入 [10, 512] 乘以大矩阵 $W^Q$ [512, 512]，切分并换位后，得到 8 个头。形状是 [8, 10, 64]。  
            # - **计算 K 和 V（偷懒开始）**：输入 [10, 512] 乘以一个变小了的矩阵 $W^K$ 和 $W^V$（形状只有 [512, 128]，因为 $2 \times 64 = 128$）。切分并换位后，K 和 V 只有 2 个头！形状是 [2, 10, 64]。  

            # **第 2 步：广播复制 (Broadcast / Repeat) —— GQA 的核心魔法**  

            # 现在问题来了：Q 是 8 份（[8, 10, 64]），K 却只有 2 份（[2, 10, 64]）。在数学上，维度不同是没法直接做矩阵乘法的！怎么办？  

            # 答案是：**物理上复制它们（广播机制）！**  

            # 程序会把 K 和 V 的这 2 个头，按组复制扩充成 8 个头，来跟 Q 对齐：  
            # - 把 K 的第 1 个头，原封不动地复制 4 份，发给 Q 的第 1、2、3、4 头。  
            # - 把 K 的第 2 个头，原封不动地复制 4 份，发给 Q 的第 5、6、7、8 头。  
            # - V 也是同样的复制操作。  

            # 经过“复制”后，原本由于节省显存而只有两份的 K 和 V，在参与计算的瞬间，被强行拉伸（Broadcast）成了和 Q 一样的形状：[8, 10, 64]。  

            # **第 3 步：像传统 MHA 一样计算**  

            # 既然现在 Q、K、V 的形状都变成了完美的 [8, 10, 64]，接下来的事情就和我们上一节讲的传统计算一模一样了：  
            # - Q × K转置 算出注意力分数 [8, 10, 10]。  
            # - 乘以被复制扩充后的 V，得到结果 [8, 10, 64]。  

            # **第 4 步：拼接与输出**  

            # 最后，把算出来的 8 个头的结果重新拼装回 512 维（[10, 512]），再乘上最后的输出权重 $W^O$，结束战斗。  

            # ---

            # **总结 GQA 的大智慧**  

            # 你可能会觉得奇怪：“既然你第 2 步还是要复制成 8 份参与计算，那你省了什么呢？”  

            # 这就是大模型工程的精妙之处：**GQA 省的从来都不是 GPU 瞬间计算的算力，它省的是保存在显存里的“长期记忆”（KV Cache）的物理空间！**  

            # 在推理生成文本时，模型只需要把那 2 个 K 头和 2 个 V 头的数据存在显存条里（空间直接砍掉 75%）。只有在每次需要计算的那个几毫秒内，它才会在 GPU 的高速缓存（SRAM）里瞬间把这 2 份数据复制成 8 份去跟 Q 碰撞。算完立刻销毁复制品，显存里依然只保留那 2 份“原件”。  

            # 通过这种“用一点点复制时间，换取海量存储空间”的策略，GQA 成功让大模型在普通显卡上跑起了超长上下文的对话.


            vocab_size: int = 6400,                 # 词表大小
            rms_norm_eps: float = 1e-05,            # RMSNorm 为了防止除零错误引入的微小常数 epsilon
            rope_theta: int = 1000000.0,            # 旋转位置编码 (RoPE) 的基数 theta (常见有 10000, 500000, 1000000 等)
            #“请把位置编码的旋转速度调到最慢档！因为我这个模型的设计目标，是用来处理几万甚至十几万字的超长文档的，必须要保证远距离的词也能被精准区分先后顺序。”rope_theta为何设置为 1000000.0？因为 RoPE 的旋转速度与 theta 成反比，theta 越大，旋转越慢，模型就能更好地处理长文本中的远距离依赖关系。
            #RoPE 选择在每次计算 Attention（Q 和 K 匹配）的前一瞬间注入
            #rope_theta 算出来的那些 $\cos$ 和 $\sin$ 值，其实是在模型加载的时候或者开始生成文本的第一步就一次性算好并缓存（Cache）起来的。在后续不断生成新词时，模型只需要查表拿来用就行
            #第一步：拆分与换位 (Rotate Half)  
            # 先把原始的 $Q$ 向量（假设为了简单只看 4 个维度 $[q_1, q_2, q_3, q_4]$）一分为二，把后半部分取负号，然后放到前面来。这就生造出了一个辅助向量，我们叫它 $Q_{\text{rotate}}$：  
            # $$Q_{\text{rotate}} = [-q_3, -q_4, q_1, q_2]$$

            # 第二步：查表拿到算好的 $\cos$ 和 $\sin$  
            # 拿着词的位置和 rope_theta，查表得到对应的 $\cos$ 向量和 $\sin$ 向量。

            # 第三步：按位相乘再相加  
            # 这是最核心的一行代码！直接把原始 $Q$ 乘以 $\cos$，加上 $Q_{\text{rotate}}$ 乘以 $\sin$：  
            # $$Q_{\text{new}} = (Q \odot \cos) + (Q_{\text{rotate}} \odot \sin)$$
            inference_rope_scaling: bool = False,   # 是否在推理时启用 RoPE 缩放（用于上下文长度外推）
            # 1. 痛点：模型遇到了没见过的“远方”假设你下载了一个开源大模型，它的官方说明写着：“本模型在 4000 个 Token 的长度下训练完毕”。这意味着在训练期间，模型最多只见过时钟指针转到第 4000 步的位置。它对 0~4000 步的角度变化非常熟悉。灾难发生：如果你在跟它聊天时，强行塞给它一篇 8000 字的文章让它总结。模型的反应：当读到第 4001 个字时，RoPE 算出了一个模型在娘胎（训练集）里从来没见过的旋转角度！模型瞬间就懵了，各个注意力头（Attention Heads）开始瞎匹配，最终输出一堆胡言乱语。这就好比你有一把 40 厘米的尺子，现在非要用它去量 80 厘米的东西，尺子直接不够长了。

            # 2. 解法：RoPE 缩放 (RoPE Scaling) 的空间折叠魔法

            # 以前，为了让模型能处理 8000 字，工程师只能花几百万美元，拿 8000 字的数据把模型重新训练一遍。后来，天才的研究人员（比如 Meta 的团队）发现了一个极其取巧的数学方法：既然模型只认识 0~4000 的角度，那我们能不能把 8000 的长度“压缩”进 4000 的空间里？

            # 这就是 RoPE Scaling（缩放）的原理：

            # - 线性缩放 (Linear Scaling)：直接除以一个缩放因子 $s$。比如你想扩展到原来的 2 倍长，那就让 $s=2$。当处理第 8000 个词时，原本它应该转动角度 $8000 \times \theta$。现在加上缩放后，我们让它转动 $(8000 / 2) \times \theta = 4000 \times \theta$。奇迹出现了：第 8000 个词，套上了第 4000 个词的角度外衣！模型一看：“哎呀，这个角度我熟啊（假装它在第 4000 的位置）”，于是模型就能继续正常运算了。

            # 通俗比喻：这就好比把一把 40 厘米的尺子，刻度线保持不变，但强行把皮尺拉长了一倍。虽然每个刻度之间的物理距离变密了（分辨率下降了），但尺子现在确实能量 80 厘米了！在数学上，这其实是一种插值（Interpolation）操作，用来达到外推（Extrapolation）的目的。

            # 3. 为什么配置里默认是 False（关闭状态）？

            # 既然缩放这么好，能直接让上下文翻倍，为什么默认不打开呢？因为天下没有免费的午餐。

            # - 分辨率受损：当你把 8000 个词硬挤进 4000 个词的空间里时，词与词之间的“角度差”变小了。就像尺子的刻度变密了，模型在寻找相邻词汇时的精确度会稍微下降（容易“看走眼”）。

            # - 需要微调配合：虽然单纯依靠数学缩放（Zero-shot）也能强行跑起来，但表现会有所下降。通常的做法是开启缩放后，再用少量长文本稍微训练（微调）几十步，让模型适应这种“变密了的刻度”，效果才会完美。

            # 总结这行代码的意义：

            # `inference_rope_scaling: bool = False`

            # 意味着：“当前模型保持原汁原味的原始长度，不使用空间压缩魔法。”

            # 如果你哪天想在自己电脑上强行让一个 8K 的模型去读 16K 的小说，你就可以把这个开关改成 True，并配置相应的缩放倍数（通常需要配合 rope_scaling_factor 参数使用），它就能顶着稍微下降一点点的精准度，帮你把长篇小说啃完。
            flash_attn: bool = True,                # 是否启用 FlashAttention-2 加速注意力计算，极大地降低显存占用和提升速度
            
            
            


            
            ####################################################
            # 以下是混合专家模型 (MoE) 的专属配置
            # 当 use_moe 为 False 时，以下配置不会生效
            ####################################################
            use_moe: bool = False,                  # 是否启用 MoE (Mixture of Experts) 架构
            # 含义：是否把 FFN 层替换成 MoE 架构。

            # 背景：如果为 False，模型就是传统的 Dense（稠密）模型（像 LLaMA 1/2/3），每一个词都要经过 FFN 里的每一个神经元，计算量极大。如果设为 True，模型就变成了 Sparse（稀疏）模型，开启专家分流模式。
            num_experts_per_tok: int = 2,           # 每个 token 在路由时选择激活的专家数量 (Top-K 路由的 K 值)

            # 含义：这是 MoE 能够**“省算力”**的核心！虽然我们有 4 个专家，但每个词（Token）进来时，最多只允许看 2 个专家。

            # 通俗理解：当输入词是“def”（Python 定义函数的关键词）时，路由系统（门控）一看，这明显是代码！于是只把它派给专家 B（逻辑）和专家 C（代码）。专家 A 和 D 直接处于“休眠”状态，完全不消耗算力！

            # 收益：通过只激活 Top-2 的专家，模型的总参数量可能很大（变聪明了），但在处理每一个词时，实际参与计算的参数量却很小（跑得极快）。

            n_routed_experts: int = 4,              # 参与路由的总独立专家数量
            # 这里设为 4，相当于模型把 FFN 拆成了 4 个小网络。你可以想象它们在训练中会自动进化出不同的特长：专家 A 擅长语法，专家 B 擅长逻辑，专家 C 擅长写代码，专家 D 擅长情感分析。

            n_shared_experts: int = 1,              # 共享专家的数量 (类似 DeepSeek-MoE 架构，所有 token 都会经过共享专家以保持通用知识)

            # 痛点解决：如果所有词都只去专科看病（常规 MoE），久而久之，那些专科医生可能会忘记“基础常识”（比如标点符号怎么处理、最基础的语法是什么）。

            # 运作方式：只要这个参数大于 0，就意味着**不管是什么词，都必须强制先经过这 1 个“共享专家”**提取通用知识，然后再去被分配给那 2 个“路由专家”提取专业知识。这极大地提升了模型的下限。
            scoring_func: str = 'softmax',          # 门控网络 (Gate) 计算路由权重的评分函数，默认为 softmax
            # 含义：词是怎么知道自己该去哪个专家的？模型里有一个极小的神经网络叫 Gate（门控网络/路由器），就像医院的分诊台护士。

            # 运作方式：词向量走到分诊台，护士会给它去 4 个专家的“契合度”打分。打完分后，使用 softmax 函数把分数转化为概率百分比（比如专家 A: 10%, 专家 B: 60%, 专家 C: 25%, 专家 D: 5%），然后挑出概率最高的两个（B 和 C）。

            aux_loss_alpha: float = 0.01,           # 辅助损失 (Auxiliary Loss) 的权重因子，用于防止 MoE 路由坍塌（即所有 token 都只去少数几个专家）

            seq_aux: bool = True,                   # 是否在序列(Sequence)级别上计算辅助损失，否则在整个 Batch 级别上计算
            # 痛点：神经网络是极其“偷懒”的。如果早期训练时，专家 A 碰巧表现好了一点点，门控网络就会倾向于把所有的词都塞给专家 A。越塞给 A，A 就越强；其他专家接不到词，得不到训练，彻底“饿死”。最后 4 个专家退化成了 1 个专家在干活，这就是路由坍塌。

            # aux_loss_alpha: float = 0.01：为了防止这种马太效应，工程师引入了 辅助损失 (Auxiliary Loss，即负载均衡损失)。它相当于给门控网络定了一个强制 KPI：“你必须保证各个专家的接客量大致平均！”如果门控网络偏心，就会在训练时被扣分（惩罚）。0.01 就是这个惩罚的严厉程度。

            # seq_aux: bool = True：表示这个“平均接客量”的考核，是在这一整句话（Sequence）的范围内计算的，确保这句话里的词能均匀地分给各个专家。

            norm_topk_prob: bool = True,            # 是否将选出的 Top-K 专家的概率分数进行归一化处理

            # 含义：针对选出来的 Top-2 专家的概率进行重新计算。为什么要做：接着上面的例子，选出的 B (60%) 和 C (25%)，加起来只有 85% ($0.6 + 0.25 = 0.85$)。如果不处理直接用，会让信号衰减。运作方式：归一化就是把它们按比例放大，让它们加起来等于 1（100%）。比如 B 变成 $0.6 \div 0.85 \approx 0.70$，C 变成 $0.25 \div 0.85 \approx 0.30$。然后把专家 B 和 C 计算出的结果，分别乘以 0.7 和 0.3 的权重，最后相加，作为这个词走出 MoE 层的最终结果。

            **kwargs
    ):
        
        """
        初始状态输入数据 X：形状为 [10, 512]。（10 个词，每个词 512 维特征）

        配置清单：1 个共享专家，4 个路由专家，每个词挑 2 个路由专家。

        第 1 步：全员进入“共享专家” (Shared Expert)  
        因为配置了 n_shared_experts = 1，所以这 10 个词不需要任何筛选，直接全部被送进这个共享专家（你可以把它当成一个普通的 SwiGLU FFN 层）。  
        计算：这 10 个词在共享专家内部经历了维度膨胀（比如到 1365 维）和收缩（回到 512 维）。  
        输出：得到一个基础特征张量 H_shared，形状依然是 [10, 512]。（此时，这 10 个词都获得了保底的“通用知识”）

        第 2 步：路由器打分 (Gate Scoring)  
        在进入共享专家的同时，原始数据 X 也要去分诊台（Gate）算一下该去哪几个“专科”。  
        Gate 其实是一个极小的线性层（权重矩阵 W_gate 的形状是 [512, 4]，因为有 4 个路由专家）。  
        计算：[10, 512] × [512, 4]  
        输出：得到一个形状为 [10, 4] 的打分矩阵。（这意味着，这 10 个词中的每一个，都获得了去 4 个专家那里的原始匹配分数）

        第 3 步：挑选 Top-K 与归一化 (Top-K Routing & Normalization)  
        拿到了 [10, 4] 的打分表后，路由器开始执行筛选逻辑：  
        Softmax 转化：把这 4 个分数变成加起来等于 100% 的概率。  
        挑选 Top-2：由于 num_experts_per_tok = 2，程序会为这 10 个词，各自挑出概率最高的 2 个专家。  
        此时，程序提取出了两个张量：一个是专家编号（形状 [10, 2]），一个是对应的概率（形状 [10, 2]）。  
        归一化 (norm_topk_prob = True)：假设第 1 个词选了专家 A（概率 0.5）和专家 C（概率 0.3）。归一化就是把它们按比例放大，使其加起来等于 1。放大后 A 变成 0.625，C 变成 0.375。更新后的概率张量形状依然是 [10, 2]。

        第 4 步：专家分流计算 (Expert Execution)  
        这是 MoE 最神奇、最省算力的一步！程序不会让 10 个词把 4 个专家都跑一遍。  
        它会根据刚才的专家编号进行“大挪移”（物理上的数据分发，通常用到 scatter 操作）：  
        假设只有 3 个词被分配到了专家 A，那么专家 A 的输入形状就是 [3, 512]。  
        假设有 5 个词被分配到了专家 B，那么专家 B 的输入形状就是 [5, 512]。  
        （注意：因为每个词选了 2 个专家，所以所有专家处理的词的总量加起来刚好是 20 个）  
        每个专家各自处理完自己领到的词后（也是一个变宽再变窄的 FFN 过程），程序会根据词原本的顺序，把打散的数据重新拼装回来。  
        加权求和：拼装时，不是直接拿来用，而是要乘以第 3 步算出来的归一化概率！  
        对于第 1 个词：  
        Output_1 = 0.625 × 专家A的结果 + 0.375 × 专家C的结果  
        输出：经过复杂的拼装和加权求和，4 个路由专家联合交出了一份最终答卷，我们叫它 H_routed，它的形状完美复原为 [10, 512]。

        第 5 步：终极大融合  
        现在，我们手里有两份 [10, 512] 的数据：  
        第 1 步里全科医生给出的通用知识 H_shared  
        第 4 步里专科医生联合给出的专业知识 H_routed  
        计算：直接将它们按位对齐相加。  
        X_final = H_shared + H_routed  
        最终输出：形状依然是 [10, 512]！
        
        """
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        
        # RoPE 缩放配置，用于支持超长上下文窗口 (例如 YaRN 算法)
        # 外推长度 = factor * original_max_position_embeddings = 16 * 2048 = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,                                  # 扩展因子
            "original_max_position_embeddings": 2048,      # 模型预训练时的原始最大长度
            "attention_factor": 1.0,                       # 注意力温度缩放因子
            "type": "yarn"                                 # 使用 YaRN (Yet another RoPE extensioN) 外推方法
        } if self.inference_rope_scaling else None
        
        self.flash_attn = flash_attn
        
        ####################################################
        # MoE 具体配置赋值
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  
        self.n_routed_experts = n_routed_experts  
        self.n_shared_experts = n_shared_experts  
        self.scoring_func = scoring_func  
        self.aux_loss_alpha = aux_loss_alpha  
        self.seq_aux = seq_aux  
        self.norm_topk_prob = norm_topk_prob  


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(torch.nn.Module):
    """
    均方根归一化 (Root Mean Square Normalization)。
    相较于 LayerNorm，RMSNorm 省略了均值计算，只计算均方根，在保持性能的同时提高了计算效率。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 weight (缩放向量)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算公式: x / sqrt(mean(x^2) + eps)
        # 使用 rsqrt (平方根倒数) 会比 1 / sqrt 运行得更快：1.现代处理器（特别是 GPU）为了加速 3D 图形渲染和科学计算，专门为“平方根倒数”设计了独立的硬件指令。2.它在芯片内部将其打包成了一个高吞吐量的单一操作，消耗的 CPU/GPU 时钟周期（Clock Cycles）显著少于先开方再做除法。3.在计算机底层的算术逻辑单元（ALU）中，运算速度的排名通常是：加法/减法 ≈ 乘法 > 乘加 (FMA) >>> 开方 >>> 除法。
        # 乘加 (FMA)作用是在一个时钟周期内，一步到位地完成乘法和加法的组合运算。它的基本数学公式是：$d = a \times b + c$
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 保证在 float32 精度下进行归一化计算，防止溢出，之后再转回原始数据类型 (如 fp16/bf16)
        # 1. 为什么 _norm 算完之后，还要乘以 self.weight？这是为了“在保持训练稳定性的同时，把数据的表达能力还给神经网络”。归一化（Norm）的副作用： 当我们执行 _norm(x) 时，我们强行把这一层输出的数据“按压”到了一个固定的尺度（均方根为 1）。这虽然让数据分布变得老实了，防止了梯度消失或爆炸，让模型很好训练。但是，这也破坏了数据原本携带的特征信息！ 如果下一层网络真的需要数值很大或者很小的数据来激活某个特定的特征，归一化操作就把它给扼杀了。weight 的救场（仿射变换）： 为了弥补这个副作用，设计者引入了一个可学习的缩放参数 self.weight。
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    预计算旋转位置编码 (RoPE) 的正弦和余弦频率矩阵。
    支持 YaRN 等长度外推算法的频率缩放。
    
    :param dim: 每个 Attention Head 的维度
    :param end: 预计算的最大序列长度
    :param rope_base: RoPE 的基数 (Theta)
    :param rope_scaling: 缩放配置字典
    """
    # 以dim = 64 为例，torch.arange(0, dim, 2) 会生成 [0, 2, 4, ..., 62]，共 32 个元素（dim//2）。每个元素 i 对应一个频率。
    # 频率计算公式: 1 / (base ^ (2i / dim))
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 问题rope_base由什么决定？它是 RoPE 旋转的基数，直接影响旋转速度。常见的值有 10000、500000、1000000 等。值越大，旋转越慢，模型就能更好地处理长文本中的远距离依赖关系。选择合适的 rope_base 是根据模型预训练时的最大上下文长度来定的，比如如果模型在 2048 长度下训练完毕，通常会选择 1000000 来确保在更长文本时旋转足够慢。

    attn_factor = 1.0
    
    # 如果启用了上下文长度外推 (RoPE Scaling)
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), 
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), 
            rope_scaling.get("beta_slow", 1.0), 
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN 算法核心逻辑: f'(i) = f(i)((1-γ) + γ/s)
            # 对高频部分不缩放，对低频部分进行线性缩放，中间部分进行过渡
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 计算斜坡函数 (Ramp function) 以实现平滑过渡
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 构造长度为 end 的位置索引 t
    t = torch.arange(end, device=freqs.device)
    # 将位置 t 乘上预计算的频率，得到 (end, dim//2) 的矩阵
    freqs = torch.outer(t, freqs).float()
    # 将频率拼接以匹配 dim 维度，生成实部 (cos) 和虚部 (sin)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    将预计算的 RoPE 正余弦值应用到 Query 和 Key 张量上。
    """
    def rotate_half(x):
        # 旋转向量的一半： [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用复数乘法的实数等价形式: x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    用于 GQA (Grouped-Query Attention)。
    当 KV 头数少于 Query 头数时，按组 (n_rep) 复制 Key 和 Value 张量，以匹配 Query 头数。
    相比于 torch.repeat_interleave，使用 expand + reshape 在内存和速度上更高效。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力机制模块。支持 标准多头注意力 (MHA)、分组查询注意力 (GQA) 和 KV Cache 机制。
    集成了 Flash Attention 2 的原生 PyTorch 实现以加速计算。
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # Query 数量是 KV 数量的几倍 (即组内重复的次数)
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # Q, K, V 投影矩阵 (不带偏置)
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出映射矩阵
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 检测环境是否支持 PyTorch 原生的 Flash Attention (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 预先切片好的 RoPE (cos, sin)
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        
        bsz, seq_len, _ = x.shape
        # 计算 Q, K, V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆分出多头维度 -> (batch, seq_len, n_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # 应用旋转位置编码
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV Cache 拼接逻辑 (用于推理阶段的自回归生成加速)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 更新用于下次生成的 KV Cache
        past_kv = (xk, xv) if use_cache else None

        # 调整维度以适配注意力计算格式，同时对 KV 进行复制扩展 (GQA 逻辑)
        # shape 变为: (batch, n_heads, seq_len, head_dim)
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 注意力计算核心逻辑
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 路径 1: 训练阶段使用高效的 Flash Attention 计算，自动构建因果掩码 (is_causal=True)
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 路径 2: 推理阶段 (带 cache) 或遇到特定 attention_mask 时使用手动计算的 Attention
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 添加下三角的因果掩码 (Causal Mask) 以防止看到未来的 token
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            # 如果传入了自定义的 attention_mask (通常用于 padding 屏蔽)
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 将多头输出拼接回一起: (batch, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 最后经过一个线性映射层与 Dropout
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    基于 SwiGLU 结构的 Transformer 前馈神经网络 (FFN)。
    LLaMA、Mistral 等主流模型均使用此结构代替传统带有 ReLU/GELU 的两层 MLP。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 如果未指定，自动计算隐藏层中间维度（设为 hidden_size 的 8/3 倍，并对齐至 64 的整数倍）
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act] # 例如 Swish/SiLU

    def forward(self, x):
        # 计算公式: DownProj( ActFn( GateProj(x) ) * UpProj(x) )
        # 也就是左侧通过门控网络计算激活值，右侧直接进行线性映射，两边对应元素相乘
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    混合专家模型 (MoE) 的门控/路由网络 (Router/Gate)。
    负责计算各个专家被激活的概率权重，并处理负载均衡。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok      # 激活数
        self.n_routed_experts = config.n_routed_experts # 总专家数

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha           # 辅助损失系数
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        # 路由权重矩阵
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        # 计算每个 token 在各个专家上的 logits
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选出分数最高的 top_k 个专家及其权重索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化 Top-K 的分数，确保它们相加为 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失 (Auxiliary Loss) 防止路由崩溃 (Expert Collapse, 比如所有的 token 都堆在某一个专家上)
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 序列级辅助损失计算 (更精细化控制一条序列内部的负载均衡)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 批次级辅助损失 (传统 MoE 做法)
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
            
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    稀疏混合专家前馈网络 (Sparse Mixture-of-Experts FFN)。
    包含了多个路由专家和可选的共享专家，用于提升模型容量。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 初始化 N 个独立/路由专家
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        
        # 共享专家 (类似 DeepSeek V2 设计，共享专家能够捕获公共通用的表示模式，不会参与路由，每个 token 必过)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制获取 Top-K 专家的索引、权重以及辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 训练时的逻辑：通过 repeat_interleave 将 token 扩展到对应的专家数量，方便批处理计算
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                # 对分给专家 i 的 token 进行计算
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: 
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: 
                    # 防止由于空专家而导致的梯度图断裂 (DDP同步中出现无梯度的参数问题)
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            
            # 将多专家的输出乘上对应门控概率权重并相加还原
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理时的逻辑：使用了更高效的索引分发逻辑 (免去了 repeat_interleave 显存开销)
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
            
        # 叠加共享专家的计算结果
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
                
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        MoE 的高效推理函数，通过索引收集分发，降低显存占用并提升运算速度。
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        # 统计每个专家负责的 token 总数并求累加和，用于界定切片范围
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 例如: 当 tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且 token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味着 token_idxs[:6] -> [3, 7, 19, 21, 24, 25] 这 6 个位置属于专家 0 处理的 token 
        # (每个 token 有可能被多个专家处理，这取决于 num_experts_per_tok)
        # 接下来 9 个位置 token_idxs[6:15] -> [4, 5, 6, 10, 11, 12...] 属于专家 1 处理的 token...依此类推
        
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue # 没有 token 分配给当前专家
                
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 提取属于当前专家的 token
            expert_tokens = x[exp_token_idx]
            # 经过 FFN 层计算
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 乘以对应路由权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add 将算好的结果累加回对应 token 缓存的原始位置上
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    构成 Transformer 主体的单个 Decoder 层。
    包含了 Pre-Norm (RMSNorm)、注意力机制 (Attention)、残差连接 和 FFN/MoE。
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.self_attn = Attention(config)
        self.layer_id = layer_id
        
        # 针对 Attention 和 MLP 的两个前置归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 根据配置决定使用标准的 FFN 还是 MoE 模块
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 第一个残差块：Attention
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        
        # 第二个残差块：MLP / MoE
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 模型主干 (不包含语言建模头 Causal LM Head)。
    处理词嵌入、构建多层 Transformer 堆叠结构，以及处理位置编码张量。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 构建 L 层 Decoder Blocks
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最后一层输出的 LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算整个模型生命周期所需的位置编码 (直到最大长度 end)，存储在 buffer 中不进行梯度更新
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        
        batch_size, seq_length = input_ids.shape
        # 兼容一些过去的 kv cache 格式
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 从 past_key_values 确定当前的偏移位置 (如果存在 cache，就无需从 0 开始获取位置编码)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Token 嵌入映射
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 根据当前的上下文长度切片提取相应的 RoPE 预计算矩阵
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        # 逐层前向传播
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 进行最后的归一化
        hidden_states = self.norm(hidden_states)

        # 累加所有 MoE 层的辅助损失，如果是传统 FFN 层则为 0
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    用于自回归生成任务 (Causal Language Modeling) 的上层包装类。
    继承自 HuggingFace 的 GenerationMixin，因此可以直接使用 .generate() 等生成工具。
    包含了主干网络以及用于输出预测词表分布的 LM Head。
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # 实例化主干网络
        self.model = MiniMindModel(self.config)
        # 定义语言建模头，将特征维度映射到词表大小
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # 权重绑定 (Weight Tying) 技巧：
        # 将输入 Embedding 层和输出 Linear 层的权重共享，能够极大地减少模型参数量并稳定训练
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0, # 用于在训练/推理时节省内存，只保留序列后部的 logits
                **args):
        
        # 主干网络特征提取
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 为节约显存和计算资源，截取需要计算 logits 的序列部分
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 输出分类对数几率分布 (Logits)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        # 若存在标签，则计算交叉熵损失 (Cross Entropy Loss)
        if labels is not None:
            # 偏移 (Shift) 对齐逻辑：第 i 个位置的 token 输出必须用来预测第 i+1 个位置的 label
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平矩阵，并在计算中忽略 index 为 -100 的 padding token
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        # 封装为兼容 HuggingFace API 的输出格式
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output