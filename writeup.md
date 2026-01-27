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



(b) 对比TinyStories_tokenizer和OpenWebText_tokenizer：

```bash
Average token length (OpenWebText tokenizer): 6.31
Average token length (TinyStories tokenizer): 5.79

Compression ratio on English text (TinyStories tokenizer): 4.73
Compression ratio on English text (OpenWebText tokenizer): 4.82
Compression ratio on Chinese text (TinyStories tokenizer): 1.00
Compression ratio on Chinese text (OpenWebText tokenizer): 1.19
```

## Experiments with tokenizers

(a)

Compression ratio on TinyStoriesV2 valid set (TinyStories tokenizer): 4.09
Compression ratio on OpenWebText valid set (OpenWebText tokenizer): 4.53

更大的词表带来了更大的压缩率

(b)

Compression ratio on OpenWebText valid set (TinyStories tokenizer): 3.35

由于词表不匹配以及vocab_size更小，压缩率由4.53下降到3.35。

(c)

我的tokenizer的throughtput约为7MB/s

要tokenize完 The Pile，需要1.4天

(d)

uint16的范围有65536，而token_id的范围是32000，足够存下。能比uint32节省一半的存储空间。

## Transformer LM resource accounting

(a)
全部的可训练参数有：
embedding的 $vocab\_size * d\_model$
transformerblock的num_layers个: 

​	attention包括q,k,v,output_proj: $4\times d\_model^2$

​	SwiGLU：$3\times d\_model \times d\_ff$

Output head的 $d\_model \times vocab\_size$

总共约2B参数，要8GB内存

(b)

| 矩阵运算           | 公式                                         | 计算量（FLOPS） |
| ------------------ | -------------------------------------------- | --------------- |
| 求Q矩阵            | $2\times T \times d_{model}\times d_{model}$ | 5.24B           |
| 求K矩阵            | $2\times T \times d_{model}\times d_{model}$ | 5.24B           |
| 求V矩阵            | $2\times T \times d_{model}\times d_{model}$ | 5.24B           |
| 计算注意力分数     | $2\times h \times T \times d_k \times T$     | 3.36B           |
| 计算注意力加权输出 | $2\times h \times T \times d_k \times T$     | 3.36B           |
| 输出投影           | $2\times T \times d_{model}\times d_{model}$ | 5.24B           |
| SwiGLU W1          | $2 \times T \times d_{model} \times d_{ff}$  | 20.97B          |
| SwiGLU W2          | $2 \times T \times d_{model} \times d_{ff}$  | 20.97B          |
| SwiGLU W3          | $2 \times T \times d_{model} \times d_{ff}$  | 20.97B          |
| 求logits           | $2 \times T \times d_{model} \times V$       | 164.68B         |

表格中除了最后求logits的，其余的都要进行num_layer次

共4.51T FLOPs

(c)
计算量最大的是ffn

(d)

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
M = 31.7G + B * 14.45G

(c)
对每个参数，AdamW要进行16FLOPs。

(d)
Forward 要4.51T FLOPs，backward要forward的两倍

对应一个token的FLOPs 是 13.212B

总共需要 13.212B * 400,000 * 1024 * 1024 = 419.43B tokens

训练共需要 5.54e21 FLOPs

时间是 5.54e21/(19.5e12 / 2) = 5.65e8秒 18年左右