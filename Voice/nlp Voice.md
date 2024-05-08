# NLP算法工程师 -【AI/算法类】

## NLP 运用场景

```
# NLP 运用场景：使用深度学习研究语言（文本/语音）的识别、理解、生成等问题。

# 1. 语音识别 ASR (Automatic Speech Recognition) 语音 -> 文本
# 2. 语音合成 TTS (Text to Speech) 文本 -> 语音
# 3. 语音分离和转换 语音 -> 语音
# 4. 语音标注 Recognition 语音 -> 类别
# 5. 文本分类 Classification 文本 -> 类别
# 6. 文本生成 文本 -> 文本
```

![截屏2023-12-02 上午8.13.59](./assets/%E6%88%AA%E5%B1%8F2023-12-02%20%E4%B8%8A%E5%8D%888.13.59.png)

## Input & Output

```
# Speech Recognition: the ability of a machine or program to identify words and phrases in spoken language and convert them to a machine-readable format.

# Token

# Phoneme: a unit of sound in speech. 发音的基本单位
# 需要语言学知识去划分单词

# Grapheme: the smallest meaningful contrastive unit in a writing system. 语言文字的最小单位 
# e.g. 26个字母 + 空格 + 标点符号

# Lexicon: the vocabulary of a person, language, or branch of knowledge. 词典 word to phoneme: 词转音素
# 中文: Lexicon free language.

# word: 单词。英文单词的划分比较简单，中文单词的划分比较复杂。英文用空格划分。

# Morpheme: the smallest meaningful unit in a language. 最小的有意义的单元。e.g. un + break + able
# 可以表示意思的最小单位。e.g. un + break + able
# < word, > grapheme.

# Byte: UTF-8, 使 language independent. 使得可以语言无关。e.g. 0x41 -> A


# Acoustic Features: 声学特征
# Window: 词窗口
# frame, 25ms, 400 samples, 16kHz 帧向量
# 移动窗口，每次移动10ms，每次移动160个sample，有重叠的部分。

# Waveform* -- DFT 离散傅立叶变换 --> Spectrum 频谱图* --> Filter Bank Output* 滤波器组-- Log + DCT 离散余弦变换 --> MFCC* Mel 频率倒谱系数

# 数据集
# 英文语料库 Corpus 
# e.g. TIMIT* (4hr), WSJ* (80hr), LibriSpeech* (960hr), TED-LIUM, Switchboard* (300hr), Fisher* (2000hr), VoxForge, Common Voice, LibriSpeech, TED-LIUM, Switchboard, Fisher, VoxForge, Common Voice
# MNIST 49min, ISLVRC ImageNet 4096hr
```

## Model Seq2Seq

### Summary

![image.png](./assets/1645108192091-b7e0e8ed-ba63-4128-9e7d-e01e21704a45.png)

### LAS

```
# Model: Sequence to Sequence Model, HMM.

# LAS: Listen, Attend and Spell
# Listen: Encoder. 
# Acoustic Features as input -> Encoder -> high level representation as output. 去除噪音，提取特征。
# Use 1-D CNN, RNN, CNN + RNN, Self-Attention.
# Down Sampling: 节省计算量，提取特征。1. 多个输出的向量合并成一个向量, Pyramidal RNN. 2. Pooling Over Time, 只选择最重要的特征。
# Attend: z0 + h1(Output of Encoder) -> Attention(e.g. dot production att 相似度， Additive att) -> alpha(0,1) -> softmax -> new alpha(0,i) (weight)-> weighted sum of hi -> c0 as input of Decoder.
# z0 -> Attention -> c0
# alpha(i) = att(z0, hi) with softmax
# c0 = sum of alpha(i) * hi
# Spell: Decoder, RNN
# z0 + Attention -> c0 -- dec --> z1 -> 'c' (z1 是 c0 decode的 hidden state。get spell token c from z1 (Distribution 概率最大)) argmax
# z1 + Attention -> c1 (Context Vector) + 'c'(上一个输出的token) -- dec -> z2 -> 'a' (概率最大) argmax
# ----> '<EOS>' (end of sentence)  
# typical seq2seq model with Attention.

# 每次选择最大概率的一条路径，
# 前一个token的选择，会影响后一个token的概率分布。一直选择最大概率的token，不一定是最优的。因为后面的token的概率分布会受到前面token的影响。
# 因此：Beam Search，每次选择概率大的多个路径作为候选集合。
# Beam Search: 选择概率最大的前k个token，作为下一个token的候选集合。然后再从候选集合中选择概率最大的前k个token，作为下一个token的候选集合。以此类推。
# k: beam size. k越大，计算量越大，效果越好。k越小，计算量越小，效果越差。

# 训练
# e.g. "cat"
# 训练目标: 最大化概率, max p(first token = 'c') * p(second token = 'a') * p(third token = 't')
# Teacher Foring: 训练时，使用正确的token作为输入，而不是使用上一个输出的token作为输入。
# 训练时，使用正确的token作为输入，而不是使用上一个输出的token作为输入。因为上一个输出的token可能是错误的，会影响后面token的概率分布。
# 防止错误不断传递和累积。

# Token, Ground truth, Prediction
# Ground truth Token -- One-hot Encoding --> One-hot Vector
# Prediction:                 RNN Output --> Softmax Vector
# Cross Entropy Loss --> Loss, minimize loss.

# Attention
# 1. Attention 的结果影响下一个token的概率输出。zt -- Attention --> ct, previous output token -> Decoder -> zt+1
# 2. Attention 的结果影响当前token的概率输出。  zt -- Attention --> ct + zt -- Decoder --> output token (current for t, previous for t+1) -> zt+1
# 3. Attention 的结果影响当前token的概率输出，同时也影响下一个token的概率输出。

# LAS 适应很复杂的语言环境，多种表达，triple a， aaa

# 限制：
# 需要听完整个句子，才能输出结果。
# 倒装可能会出现问题。e.g. "I am a student" -> "student a am I"
```

### CTC & RNA

```
# CTC: Connectionist Temporal Classification
# Decoder: Linear Classifier with Softmax -> Distribution over all possible tokens. 不涉及 Attention, 输出结果各自独立。
# Encoder: RNN... 可能是一个很深的LSTM
# 可实现Online的语音识别，不需要等待整个句子。

# 引入一个特殊的token: blank token，可能是前一个重复的token，也可能为空。下面记为'_'。
# e.g. "ca_av" -> "caav", "caat" -> "cat"
# ca的多种alignment方式: _ca_, ccaa, caaa, c_a_, _c_a, 

# RNA：Recurrent Neural Aligner
# 添加依赖：将 CTC Decoder 中线性分类器换成 RNN/LSTM，使得输出结果依赖于前一个输出结果。（+ 前一个的输出结果，+ 前一个的隐藏状态)
```

![image.png](./assets/1645091308455-658132b6-8baf-4683-a690-577ad29ffd2c.png)

### RNN-T

```
# RNN-T
# CTC模型无法处理一帧中含有多个token的问题。
# 提出Encoder的输出的每一个h再经过一个RNN网，不断输出，直到最终输出$\phi$表示该帧信息已经全部输出完毕。
# ht -> t1 -> t2 -> t3 -> ... -> $\phi$, 
# ht+1 -> t1 ... -> $\phi$,
```

![image.png](./assets/1645093441595-7ad302fe-613d-4bd0-98b4-a2b25762ca08.png)

会存在alignment（对齐）的问题，例如的对于label 好棒， RNN-T存在多重排列，例如
$$
$\underline{好\phi_{1}\phi_{2}棒\phi_{3}\phi_{4}} \ ,\underline{\phi_{1}\phi_{2}\phi_{3}好棒\phi_{4}}$
$$
引入额外的RNN，建立在真正的token上，而无视blank token,$\phi$. 便于穷举所有的alignment。

![img](./assets/1645095037688-ddbe49e3-7bd2-4168-9005-51f2f999aa93.png)![img](./assets/1645095157404-ed907fc3-7b03-40fd-b816-5df073116a75.png)

### Neural Transducer

累积一个窗口的Encoder输出，h1, h2, h3,...

使用Attention + RNN 预测该窗口预测的tokens，直至产生$\phi$。

![image.png](./assets/1645107121376-d47b723e-195f-49c4-bde5-e7bd8775ad69.png)

### MoChA

Dynamically shift the window. 由一个Module决定是否将window设在此处hi -> yes / no.

if yes, 使用 Attention + RNN decode 出一个token 即可。仅执行一次Attention操作。

if no, 向右移动一格。

![image.png](./assets/1645108091939-d12d5863-446f-4124-9ce2-edce39da08a2.png)

## Model HMM

统计学视角。Hidden Markov Model

![img](./assets/01ecf3997fc1aadc2828da8281d475b3.svg)

Phoneme -> Tri-phone -> State 更小的细分。

![image-20231202034414367](./assets/image-20231202034414367.png)

### All the alignments

![image.png](./assets/1645410371467-cd74c90e-f27b-48a7-94eb-3e75be296493.png)

#### HMM

可以选择向右下方走，即获得下一个token。或者向右走，即duplicate当前token。

![img](./assets/1645411276581-9540ff43-f85b-44a0-ae5f-d91bf90fbf58.png)

#### CTC

1. 当前token为![img](./assets/1ed346930917426bc46d41e22cc525ec.svg)时，只能选择向右复制![img](https://cdn.nlark.com/yuque/__latex/1ed346930917426bc46d41e22cc525ec.svg)或者右下一个，拿到下一个非空token。
2. 当前token不为![img](https://cdn.nlark.com/yuque/__latex/1ed346930917426bc46d41e22cc525ec.svg)时，可以选择向右复制该token或者右下获得![img](https://cdn.nlark.com/yuque/__latex/1ed346930917426bc46d41e22cc525ec.svg)，或者右下跳步获得下一个非空token。
3. 例外：如果连续两个token相同，则不能右下跳步获得下一个非空token。

![img](./assets/1645412292187-d9991bd3-c0f8-4615-8aea-618f33c97436.png)

![img](./assets/1645412454841-9b6ee67a-4d41-40d7-a371-7f1a6da1f821.png)

#### RNN-T

![img](./assets/1645412885160-451b7270-b7e3-4d54-ab58-acdfebddd305.png)注意，最后一定要产生![img](./assets/1ed346930917426bc46d41e22cc525ec-20231202042713459.svg)。

![img](./assets/1645413406327-33049ed2-d16c-4d58-8ba2-4493d36a247a.png)

#### 概率计算 RNN-T

![image.png](./assets/1645428288771-73525ad9-fae5-4f43-a4c6-b4ce29156450.png)

无视$\phi$，因此无论走哪条路线，在同一位置的softmax概率都是相同的。$p_{4,2} = p(l^2,h^4)$ 

##### 算法

定义![img](./assets/f5f2dd3ae6310f4b9f7b0070e8075c39.svg)为，已经读了i个h，输出了j个token的所有概率的总和。

![img](./assets/1645495646487-0461b20b-d8c4-471f-9e3e-488e47cbc02e.png)

![img](./assets/107d85aa8f28c6d0d08bfa792e83e849.svg)，最后一个![img](./assets/7b7f9dbfea05c83784f8b85149852f08.svg)的值即![img](./assets/7082016653a57c45dae6f7d57ceea8ba.svg)。Dynamatic Programming

#### Summary

![image.png](./assets/1645414044787-4838367d-3217-4bc8-9251-5a50f37d7eeb.png)

![image.png](./assets/1645497423496-02dbd84b-aa49-4161-bbb8-8c2511ab6eec.png)

梯度下降算法：使用 back propagation 更新模型参数，以最小化损失函数。

## Language model

LM（language model）用来估测一个token序列（word，character，phoneme）可能的几率。

- Token sequence：![img](./assets/499a2bdfb3e3e90881b65177da787ca3.svg)
- 估测：![img](./assets/b09d60379e92c242e9537d37142a5c24.svg)

为什么需要LM呢？

1. 对于HMM来说，需要![img](./assets/944ea13a98aa027ea5cb3002964f24b9.svg)，需要P(Y)。
2. 对于LAS来说，需要![img](./assets/41dc755d25d30c826ab233ed1d298c02.svg)，但是P(Y|X)是需要成对的资料，较难收集。而P(Y)只需收集token序列，比较容易。加上P(Y)获得![img](./assets/8a3a3ae11c7c41bf287d439b65b15c05.svg)，对训练模型很有帮助，我们总是想sequence Y的概率越大越常见越好。

![img](./assets/1645499586574-0ff04776-2e97-4a92-8f4a-cdcd7915cfa6.png)

### N-gram

假设第n个token只与前n-1个token有关。将总概率转换成条件概率相乘。

例如，2-gram：

![img](./assets/1645517144884-1557fbb3-4f37-489f-9450-97817541be06.png)

问题是即使n-gram，训练资料也不够大，很多n-gram的组合未曾出现在训练资料中，使用language model smoothing 添加小的概率。

### Continuous LM

来源于Recommendation System，使用matrix factorization解决n-gram组合未出现问题。

每一个词汇有其对应向量，该向量通过最小化L学出，最终使用向量之间点乘代表出现次数。

![img](./assets/1645518445868-be647639-e28c-4c4d-b1a0-3874ab3c5e33.png)

![img](./assets/1645527424124-bcbcce30-3870-4aa8-8539-dc7d7549a970.png)

有同样属性的token（dog，cat）往往具有相似的词向量。

后来就逐渐进入了deep learning的时代....上面使用matrix factorization的方案可以转化为下图使用DNN的方案，使用L2作为损失函数。

![img](./assets/1645527470276-5c545ca7-d13d-46de-a4ba-575204bfe348.png)

### NN-based LM

替代n-gram，使用NN训练下一个词汇出现的概率。

![img](./assets/1645527865408-55a6a443-fa12-48c8-8fd3-fd5835710627.png)

## RNN-based LM

可能会使用很长的history来决定下一个token的出现几率。

![img](./assets/1645528504598-883d1e1d-df78-4cc0-a03b-33db61fe6af9.png)

##  LM and LAS

![image.png](./assets/1645844677039-83227852-5c47-4c85-9967-f3cbc56f1d65.png)

####  Shallow Fusion

![image.png](./assets/1645529760685-bb2eaf38-458f-4530-9e90-00b6e5fe783d.png)

已经训练好的LAS的Decoder的output拿到token概率，和LM的softmax得到的概率log相加，最后得到概率最大的作为output。（Beam Search 同样适用）

####  Deep Fusion

![image.png](./assets/1645529141937-b90b29a0-0105-46b6-9b53-af21e47688c4.png)

已经训练好的LAS的hidden layer和LM output拿出来，再经过一个Network后得到softmax概率最大的。对于不同的domain，可能需要更换LM。接入Softmax之前的结果。

####  Cold Fusion

![image.png](./assets/1645529443444-6ab020be-d9f2-4386-a867-a2af99f19af0.png)

LAS 未经过训练，LM已被训练。加快LAS的训练速度，但与LM牢牢绑定，不可轻易更换。

## Voice Conversion 声音转换

#### 应用场景

> Deep Fake: Fool humans / speaker verification system 
>
> Personalized TTS.
>
> Style转换：情感转换 / Normal 2 Lombard / 悄悄话 2 Normal
>
> 改善清晰度 / 口音转换

![image-20231202072815931](./assets/image-20231202072815931.png)

> 通常假设，T' = T，此时 seq2seq 不一定需要。
>
> 可使用:
>
> Rule-based: Griffin-Lim algorithm 
>
> Deep Learning: WaveNet
>
> Vocoder: feature, vector -> Voice

#### 类型

![image-20231202073054420](./assets/image-20231202073054420.png)

Unparallel data: 1. Feature disentangle 2. 

![image-20231202074321640](./assets/image-20231202074321640.png)

![image-20231202074444409](./assets/image-20231202074444409.png)

**Instance Normalize**: 使 Content Encoder 移除语者信息，只保留内容信息。1-D CNN提取特征，对同一Row Normalize化。-mean / std.

![image-20231202074800968](./assets/image-20231202074800968.png)

![image-20231202075034079](./assets/image-20231202075034079.png)

![image-20231202075526479](./assets/image-20231202075526479.png)

...

