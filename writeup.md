# Writeup

## Problem (unicode1)

(a) chr(0) 会返回 Unicode NULL 字符, 也就是 U+0000 (\x00).

(b) 它的 __repr__() 是转义的 "'\\x00'"

(c) 直接出现在字符串中，会显示出\x00，不过print打印出来是不可见字符。

## Problem (unicode2)

(a) UTF-8 是使用最广泛的编码形式，在效率上也最优

(b) 这段程序错误地认为一个unicode字符对应一个字节，但是实际上一个字符可以对应多个字节。例如输入中文汉字时就会出错。

(c) 例如b"\x80\x80"，两个字节都是接续字节没有起始字节

## BPE Training on TinyStories
(a) 最长的token有：
b' accomplishment'
b' disappointment'
b' responsibility'
b' uncomfortable'
b' compassionate'
b' understanding'
b' neighbourhood'
b' Unfortunately'
b' determination'
b' encouragement'
挺合理的。

(b) pretokenize花了大部分时间（605.26s）。后面的merge只花了20s。
    内存占用scalene测多进程测出来的似乎不准。

## BPE Training on OWT

(a) 训练得到的最长的token是这些：
Longest tokens in the vocabulary:
b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
b'----------------------------------------------------------------'
b'\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94'
b'--------------------------------'
b'________________________________'
b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
b'================================'
b'................................'
b'********************************'
b'\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94'
b' disproportionately'
b' telecommunications'
b' environmentalists'
b' responsibilities'
b' unconstitutional'
b' cryptocurrencies'
b' disproportionate'
b' misunderstanding'
b' counterterrorism'
b' characterization'
b'----------------'
b'________________'
b' representatives'
b' recommendations'
b' characteristics'
b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'
b'================'
b' straightforward'
b' Representatives'
b'................'

b'\xe2\x80\x94'是破折线，还有一些乱码

充分体现了网络上爬下来的数据非常杂乱



(b) 对比TinyStories_tokenizer和OpenWebText_tokenizer，

## Transformer LM resource accounting

(a)
全部的可训练参数有：
embedding的 vacab_size \* d_model
transformerblock的num_layers个以下
    attn包括q,k,v,output_proj：d_model \* d_model*3
    ln1: d_model
    ffn包括w1,w2,w3: d_model \* d_ff \* 3
    ln2: d_model
ln_final的：d_model
lm_head的：d_model*vocab_size

总共约2B参数，要8GB内存

(b)
单个block的矩阵乘操作有：
QKV投影：3\*2\*context_length\*d_model^2
15.7B
计算注意力分数：2\*context_length^2\*d_model
3.3B
计算分数乘V：2\*context_length^2\*d_model
3.3B
计算输出的投影：2\*context_length\*d_model…2
5.2B
FFN: 6\*context_length\*d_model\*d_ff
62.9B

一个block约90B，48次就是4348B

最后的mlp：2\*context_length\*d_model\*vacab_size
164.6B

总共是4.51*10^12

(c)
计算量最大的是ffn

(d)
GPT-2 small

12 层，d_model = 768，12 heads

单个 block 的矩阵乘操作有：

QKV 投影
≈ 3.62B FLOPs

计算注意力分数（QKᵀ）
≈ 1.61B FLOPs

注意力分数乘 V（AV）
≈ 1.61B FLOPs

注意力输出投影
≈ 1.21B FLOPs

FFN（三个矩阵乘）
≈ 14.50B FLOPs

单个 block 合计
≈ 22.55B FLOPs

12 个 block 合计

≈ 270.6B FLOPs

最后的 LM head

≈ 79.1B FLOPs

总 FLOPs

≈ 3.50 × 10¹¹ FLOPs

FLOPs 占比

FFN：49.8%

QKV + 注意力输出投影：16.6%

注意力（QKᵀ + AV）：11.1%

LM head：22.6%

GPT-2 medium

24 层，d_model = 1024，16 heads

单个 block 的矩阵乘操作有：

QKV 投影
≈ 6.44B FLOPs

计算注意力分数（QKᵀ）
≈ 2.15B FLOPs

注意力分数乘 V（AV）
≈ 2.15B FLOPs

注意力输出投影
≈ 2.15B FLOPs

FFN（三个矩阵乘）
≈ 25.77B FLOPs

单个 block 合计
≈ 38.65B FLOPs

24 个 block 合计

≈ 927.7B FLOPs

最后的 LM head

≈ 105.4B FLOPs

总 FLOPs

≈ 1.03 × 10¹² FLOPs

FLOPs 占比

FFN：59.9%

QKV + 注意力输出投影：24.9%

注意力（QKᵀ + AV）：10.0%

LM head：10.2%

GPT-2 large

36 层，d_model = 1280，20 heads

单个 block 的矩阵乘操作有：

QKV 投影
≈ 10.07B FLOPs

计算注意力分数（QKᵀ）
≈ 2.68B FLOPs

注意力分数乘 V（AV）
≈ 2.68B FLOPs

注意力输出投影
≈ 3.36B FLOPs

FFN（三个矩阵乘）
≈ 40.27B FLOPs

单个 block 合计
≈ 59.06B FLOPs

36 个 block 合计

≈ 2126.0B FLOPs

最后的 LM head

≈ 131.8B FLOPs

总 FLOPs

≈ 2.26 × 10¹² FLOPs

FLOPs 占比

FFN：64.2%

QKV + 注意力输出投影：21.4%

注意力（QKᵀ + AV）：8.6%

LM head：5.8%

模型规模变化带来的比例变化总结（简述）

随着模型规模增大，FFN 和 QKV / 输出投影这类与 d_model 平方相关的计算在总 FLOPs 中占比持续上升；而注意力打分与加权、以及 LM head 这类只随 d_model 线性增长的部分，占比则明显下降。
因此，小模型更接近 attention-bound，而大模型在计算上越来越 MLP-bound。

(e)
把 GPT-2 XL 的 context length 从 1,024 提到 16,384（16 倍）后，由于注意力相关矩阵乘的计算量随序列长度按平方增长，一次前向传播的总矩阵乘 FLOPs 从约 4.51×10¹² 增加到约 1.50×10¹⁴（约 33.1 倍）。同时 FLOPs 构成从“FFN/投影占主导”转为“注意力占主导”：QKᵀ+AV 合计占比由约 7.1% 升至约 55.2%，FFN 占比由约 66.9% 降至约 32.3%（QKV、输出投影和 LM head 占比均明显下降）。

## Tuning the learning rate

在lr=1e1和1e2的时候，能下降。1e2的时候下降得更快
在lr=1e3的时候，就爆炸了。

## Resource accounting for training with AdamW

(a)
对于每个参数，要存储的内容有：参数本身，梯度，一阶矩，二阶矩，共四个
参数量：

- Transformer block:
  - RMSNorm: 2×d_model
    - Multi-head: self-attention sublayer
    - QKV投影: 3×d_model×d_model
    - 输出投影: 1×d_model×d_model
  - FFN
    - W1: d_model×(4×d_model)
    - W2: (4×d_model)×d_model
- 最终的RMSNorm: d_model
- output embedding: vocab_size×d_model

激活值:

- Transformer block:
  - RMSNorm: 2×batch_size×context_length×d_model
  - Multi-head: self-attention sublayer
    - QKV投影: 3×batch_size×context_length×d_model
    - QK乘积: batch_size×num_heads×context_length×context_length
    - softmax: batch_size×num_heads×context_length×context_length
    - weighted sum of values: batch_size×context_length×d_model
    - 输出投影: batch_size×context_length×d_model
  - FFN
    - W1: batch_size×context_length×(4×d_model)
    - SiLU: batch_size×context_length×(4×d_model)
    - W2: batch_size×context_length×d_model
- 最终的RMSNorm: batch_size×context_length×d_model
- output embedding: batch_size×context_length×vocab_size
- cross-entropy on logits: batch_size×context_length×vocab_size

显存占用是4*参数量+激活量

N_parameters=num_layers×(12×d_model×d_model+2×d_model)+d_model+vocab_size×d_model
N_activations=num_layers×(16×batch_size×context_length×d_model+2×batch_size×num_heads×context_length×context_length)+(batch_size×context_length×d_model)+2×(batch_size×context_length×vocab_size)

(b)
待补充

(c)
待补充

(d)
待补充