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
