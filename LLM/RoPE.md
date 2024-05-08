https://www.zhihu.com/tardis/zm/art/647109286?source_id=1003

十分钟读懂旋转编码（RoPE）

![绝密伏击](./assets/v2-005eaa34a1a2f96d53f0ba4d27db3bbb_l-20240301015745147.jpg)

绝密伏击

《推荐系统技术原理与实践》作者，欢迎知友京东购买。

545 赞同

22 评论

953 收藏

旋转位置编码（Rotary Position Embedding，RoPE）是论文[Roformer: Enhanced Transformer With Rotray Position Embedding](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.09864.pdf) 提出的一种能够将相对位置信息依赖集成到 self-attention 中并提升 transformer 架构性能的位置编码方式。而目前很火的 LLaMA、GLM 模型也是采用该位置编码方式。

和相对位置编码相比，RoPE 具有更好的**外推性**，目前是大模型相对位置编码中应用最广的方式之一。

**备注：什么是大模型外推性？**

外推性是指大模型在训练时和预测时的输入长度不一致，导致模型的泛化能力下降的问题。例如，如果一个模型在训练时只使用了512个 token 的文本，那么在预测时如果输入超过512个 token，模型可能无法正确处理。这就限制了大模型在处理长文本或多轮对话等任务时的效果。

**备注：下一篇文章将详细介绍如何基于RoPE提升大模型的外推能力（2k推8k，甚至更长）**

[绝密伏击：再论大模型位置编码及其外推性（万字长文）](https://zhuanlan.zhihu.com/p/675243992)

## 1. 旋转编码 RoPE

### 1.1 基本概念

在介绍 RoPE 之前，先给出一些符号定义，以及基本背景。

首先定义一个长度为 ![N](./assets/equation-20240301015746077) 的输入序列为：

![\mathbb{S}_{N}=\{ w_i \}_{i=1}^{N} \\\tag1](./assets/equation-20240301015745467)

其中 ![w_i](./assets/equation-20240301015745460) 表示输入序列中第 ![i](./assets/equation-20240301015745122) 个 token，而输入序列 ![\mathbb{S}_N](./assets/equation-20240301015745464) 对应的 embedding 表示为：

![\mathbb{E}_{N}=\{ \bm{x}_i \}_{i=1}^N\\\tag2](./assets/equation-20240301015745521)

其中 ![\bm{x}_i](./assets/equation-20240301015745452) 表示第 ![i](./assets/equation-20240301015745122) 个 token ![w_i](./assets/equation-20240301015745460) 对应的 ![d](./assets/equation-20240301015745723) 维词嵌入向量。

接着在做 self-attention 之前，会用词嵌入向量计算 ![\bm{q,k,v}](https://www.zhihu.com/equation?tex=%5Cbm%7Bq%2Ck%2Cv%7D&consumer=ZHI_MENG) 向量同时加入位置信息，函数公式表达如下：

![\bm{q}_m=f_q(\bm{x}_m,m) \\ \bm{k}_n=f_k(\bm{x}_n,n) \\ \bm{v}_n=f_v(\bm{x}_n,n) \\\tag3](./assets/equation-20240301015745887)

其中 ![\bm{q}_m](./assets/equation-20240301015745721) 表示第 ![m](./assets/equation-20240301015745768) 个 token 对应的词向量 ![\bm{x}_m](./assets/equation-20240301015745983) 集成位置信息 ![m](https://www.zhihu.com/equation?tex=m&consumer=ZHI_MENG) 之后的 query 向量。而 ![\bm{k}_n](./assets/equation-20240301015746012) 和 ![\bm{v}_n](./assets/equation-20240301015745992) 则表示第 ![n](./assets/equation-20240301015746236) 个 token 对应的词向量 ![\bm{x}_n](./assets/equation-20240301015746154) 集成位置信息 ![n](https://www.zhihu.com/equation?tex=n&consumer=ZHI_MENG) 之后的 key 和 value 向量。

而基于 transformer 的位置编码方法都是着重于构造一个合适的 ![f\left( \bm{q},\bm{k},\bm{v} \right)](./assets/equation-20240301015746228) 函数形式。

而计算第 ![m](https://www.zhihu.com/equation?tex=m&consumer=ZHI_MENG) 个词嵌入向量 ![\bm{x}_m](./assets/equation-20240301015745983) 对应的 self-attention 输出结果，就是 ![\bm{q}_m](./assets/equation-20240301015745721) 和其他 ![\bm{k}_n](./assets/equation-20240301015746012) 都计算一个 attention score ，然后再将 attention score 乘以对应的 ![\bm{v}_n](./assets/equation-20240301015745992) 再求和得到输出向量 ![\bm{o}_m](./assets/equation-20240301015746232) ：

![a_{m,n}=\frac{\text{exp}(\frac{\bm{q}_m^{\textbf{T}}\bm{k}_n}{\sqrt{d}})}{\sum_{j=1}^N\text{exp}(\frac{\bm{q}_m^{\textbf{T}}\bm{k}_j}{\sqrt{d}})} \\ \bm{o}_m=\sum_{n=1}^Na_{m,n}\bm{v}_n \\\tag4](./assets/equation-20240301015746308)

### 1.2 绝对位置编码

对于位置编码，常规的做法是在计算 query, key 和 value 向量之前，会计算一个位置编码向量 ![\bm{p}_i](./assets/equation-20240301015746413) 加到词嵌入 ![\bm{x}_i](./assets/equation-20240301015745452) 上，位置编码向量 ![\bm{p}_i](https://www.zhihu.com/equation?tex=%5Cbm%7Bp%7D_i&consumer=ZHI_MENG) 同样也是 ![d](./assets/equation-20240301015745723) 维向量，然后再乘以对应的变换矩阵 ![\bm{W}](./assets/equation-20240301015746415)：

![f_{t:t\in\{q,k,v\}}(\bm{x}_i,i):=\bm{W}_{t:t\in\{q,k,v\}}(\bm{x}_i+\bm{p}_i) \\\tag5](./assets/equation-20240301015746466)

而经典的位置编码向量 ![\bm{p}_i](https://www.zhihu.com/equation?tex=%5Cbm%7Bp%7D_i&consumer=ZHI_MENG) 的计算方式是使用 Sinusoidal 函数：

![\bm{p}_{i,2t}=\text{sin}\left( k/10000^{2t/d} \right)\\ \bm{p}_{i,2t+1}=\text{cos}\left( k/10000^{2t/d} \right)\\\tag6](./assets/equation-20240301015746456)

其中 ![\bm{p}_{i,2t}](./assets/equation-20240301015746511) 表示位置 ![d](./assets/equation-20240301015745723) 维度向量 ![\bm{p}_i](https://www.zhihu.com/equation?tex=%5Cbm%7Bp%7D_i&consumer=ZHI_MENG) 中的第 ![2t](./assets/equation-20240301015746542) 位置分量也就是偶数索引位置的计算公式，而![\bm{p}_{i,2t+1}](./assets/equation-20240301015746775)就对应第 ![2t+1](./assets/equation-20240301015746736) 位置分量也就是奇数索引位置的计算公式。

### 1.3 2维旋转位置编码

论文中提出为了能利用上 token 之间的相对位置信息，假定 query 向量 ![\bm{q}_m](./assets/equation-20240301015745721) 和 key 向量 ![\bm{k}_n](./assets/equation-20240301015746012) 之间的内积操作可以被一个函数 ![g](./assets/equation-20240301015746754) 表示，该函数 ![g](https://www.zhihu.com/equation?tex=g&consumer=ZHI_MENG) 的输入是词嵌入向量 ![\bm{x}_m](./assets/equation-20240301015745983) ， ![\bm{x}_n](./assets/equation-20240301015746154) 和它们之间的相对位置 ![m-n](./assets/equation-20240301015746692) ：

![\left<\bm{f}_q(\bm{x}_m,m),f_k(\bm{x}_n,n)\right>=g(\bm{x}_m,\bm{x}_n,m-n) \\\tag7](./assets/equation-20240301015746839)

接下来的目标就是找到一个等价的位置编码方式，从而使得上述关系成立。

假定现在词嵌入向量的维度是两维 ![d=2](./assets/equation-20240301015746936) ，这样就可以利用上2维度平面上的向量的几何性质，然后论文中提出了一个满足上述关系的 ![f](./assets/equation-20240301015746937) 和 ![g](https://www.zhihu.com/equation?tex=g&consumer=ZHI_MENG) 的形式如下：

![f_q(\bm{x}_m,m)=\left(\bm{W}_q\bm{x}_m\right)e^{im\theta} \\ f_k(\bm{x}_n,n)=(\bm{W}_k\bm{x}_n)e^{in\theta} \\ g(\bm{x}_m,\bm{x}_n,m-n)=\text{Re}\left[(\bm{W}_q\bm{x}_m)(\bm{W}_k\bm{x}_n)^{*}e^{i(m-n)\theta}\right] \\\tag8](./assets/equation-20240301015746957)

这里面 Re 表示复数的实部。

进一步地， ![f_q](./assets/equation-20240301015747134) 可以表示成下面的式子：

![\begin{align} f_q\left( \bm{x}_m,m \right)  &= \begin{pmatrix}  \cos m\theta & -\sin m\theta) \\  \sin m \theta &  \cos m \theta \end{pmatrix}   \begin{pmatrix}  W^{(1,1)}_{q} & W^{(1,2)}_{q}  \\  W^{(2,1)}_{q}  &  W^{(2,2)}_{q} \end{pmatrix} \begin{pmatrix}  x_m^{(1)}  \\  x_m^{(2)}    \end{pmatrix} \\ &= \begin{pmatrix}  \cos m\theta & -\sin m\theta) \\  \sin m \theta &  \cos m \theta \end{pmatrix}\begin{pmatrix}  q_m^{(1)}  \\  q_m^{(2)}    \end{pmatrix}  \end{align}\tag9](./assets/equation-20240301015746995)

看到这里会发现，这不就是 query 向量乘以了一个旋转矩阵吗？这就是为什么叫做旋转位置编码的原因。

同理， ![f_k](./assets/equation-20240301015747143) 可以表示成下面的式子：

![\begin{align} f_k\left( \bm{x}_m,m \right)  &= \begin{pmatrix}  \cos m\theta & -\sin m\theta) \\  \sin m \theta &  \cos m \theta \end{pmatrix}   \begin{pmatrix}  W^{(1,1)}_{k} & W^{(1,2)}_{k}  \\  W^{(2,1)}_{k}  &  W^{(2,2)}_{k} \end{pmatrix} \begin{pmatrix}  x_m^{(1)}  \\  x_m^{(2)}    \end{pmatrix} \\ &= \begin{pmatrix}  \cos m\theta & -\sin m\theta) \\  \sin m \theta &  \cos m \theta \end{pmatrix}\begin{pmatrix}  k_m^{(1)}  \\  k_m^{(2)}    \end{pmatrix}  \end{align}\tag{10}](./assets/equation-20240301015747218)

最终 ![g(\bm{x}_m,\bm{x}_n,m-n)](./assets/equation-20240301015747186) 可以表示如下：

![g(\bm{x}_m,\bm{x}_n,m-n)  =\begin{pmatrix}  \bm{q}_m^{(1)} &  \bm{q}_m^{(2)}  \\ \end{pmatrix}   \begin{pmatrix}  \cos((m-n)\theta) & -\sin((m-n)\theta) \\  \sin((m-n)\theta) &  \cos((m-n)\theta) \end{pmatrix}   \begin{pmatrix}  k_n^{(1)}  \\  k_n^{(2)}    \end{pmatrix} \\\tag{11}](./assets/equation-20240301015747262)

关于上面公式（8）~（11）的具体推导，可以参见文章最后的**附录**，或者参考文章：[一文看懂 LLaMA 中的旋转式位置编码（Rotary Position Embedding）](https://zhuanlan.zhihu.com/p/642884818)。

### 1.4 扩展到多维

将2维推广到任意维度，可以表示如下：

![f_{\left\{ q,k \right\}}\left( \bm{x}_m,m \right)=\bm{R}^d_{\Theta,m}\bm{W}_{\left\{ q,k \right\}}\bm{x}_m\\\tag{12}](./assets/equation-20240301015747274)

内积满足线性叠加性，因此任意偶数维的RoPE，我们都可以表示为二维情形的拼接，即

![\bm{R}^d_{\Theta,m}=\begin{equation}\scriptsize{\underbrace{\begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\ \end{pmatrix}}_{\boldsymbol{W}_m}}\end{equation}\\\tag{13}](./assets/equation-20240301015747489)

![\Theta=\left\{ \theta_i=10000^{-2(i-1)/d}, i \in [1,2,...,d/2] \right\} \\](./assets/equation?tex=/Theta%253D/left/%7B+/theta_i%253D10000%5E%7B-2(i-1)%252Fd%7D%252C+i+/in+%5B1%252C2%252C..-20240301015747352.%252Cd%252F2%5D+/right/%7D+/&consumer=ZHI_MENG)

将 RoPE 应用到前面公式（4）的 Self-Attention 计算，可以得到**包含相对位置信息的Self-Attetion**：

![\bm{q}^{\textbf{T}}_m\bm{k}_n=\left( \bm{R}^d_{\Theta,m}\bm{W}_q\bm{x}_m \right)^{\textbf{T}}\left( \bm{R}^d_{\Theta,n}\bm{W}_k\bm{x}_n \right)=\bm{x}_m^{\textbf{T}}\bm{W}_q\bm{R}^d_{\Theta,n-m}\bm{W}_k\bm{x}_n\tag{14}](./assets/equation-20240301015747551) 其中， ![\bm{R}^d_{\Theta,n-m}=\left( \bm{R}^d_{\Theta,m} \right)^{\textbf{T}}\bm{R}^d_{\Theta,n}](./assets/equation-20240301015747461) 。

值得指出的是，由于![\bm{R}^d_{\Theta}](./assets/equation-20240301015747462)是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

### 1.5 RoPE 的高效计算

由于![\bm{R}^d_{\Theta,m}](https://www.zhihu.com/equation?tex=%5Cbm%7BR%7D%5Ed_%7B%5CTheta%2Cm%7D&consumer=ZHI_MENG)的稀疏性，所以直接用矩阵乘法来实现会很浪费算力，**推荐通过下述方式来实现 RoPE**：

![\bm{R}^d_{\Theta,m}\bm{x}=\begin{equation}\begin{pmatrix}x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_{d-2} \\ x_{d-1}  \end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1} \end{pmatrix} + \begin{pmatrix}-x_1 \\ x_0 \\ -x_3 \\ x_2 \\ \vdots \\ -x_{d-1} \\ x_{d-2}  \end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} \end{pmatrix}\end{equation}\\\tag{15}](./assets/equation-20240301015747696)

其中![\otimes](./assets/equation-20240301015747582)是逐位对应相乘，即计算框架中的![*](./assets/equation-20240301015747679)运算。从这个实现也可以看到，RoPE 可以视为是乘性位置编码的变体。

总结来说，RoPE 的 self-attention 操作的流程是：对于 token 序列中的每个词嵌入向量，首先计算其对应的 query 和 key 向量，然后对每个 token 位置都计算对应的旋转位置编码，接着对每个 token 位置的 query 和 key 向量的元素按照 **两两一组** 应用旋转变换，最后再计算 query 和 key 之间的内积得到 self-attention 的计算结果。

论文中有个很直观的图片展示了旋转变换的过程：

![img](./assets/v2-6db340ee9d34709e9c25921f5a0c2a0e_b-20240301015744514.jpg)

### 1.6 远程衰减

可以看到，RoPE 形式上和前面公式（6） Sinusoidal 位置编码有点相似，只不过 Sinusoidal 位置编码是加性的，而 RoPE 可以视为乘性的。在 ![\theta_i](./assets/equation-20240301015747832) 的选择上，RoPE 同样沿用了 Sinusoidal 位置编码的方案，即![\theta_i = 10000^{-2i/d}](./assets/equation-20240301015747787)，它可以带来一定的远程衰减性。

具体证明如下：将 ![\boldsymbol{q},\boldsymbol{k}](./assets/equation-20240301015747828) 两两分组后，它们加上 RoPE 后的内积可以用复数乘法表示为：

![\begin{equation}  \left( \bm{R}^d_{\Theta,m}\bm{W}_q\bm{x}_m \right)^{\textbf{T}}\left( \bm{R}^d_{\Theta,n}\bm{W}_k\bm{x}_n \right)= \text{Re}\left[\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i}\right]\end{equation}\\\tag{16}](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D++%5Cleft%28+%5Cbm%7BR%7D%5Ed_%7B%5CTheta%2Cm%7D%5Cbm%7BW%7D_q%5Cbm%7Bx%7D_m+%5Cright%29%5E%7B%5Ctextbf%7BT%7D%7D%5Cleft%28+%5Cbm%7BR%7D%5Ed_%7B%5CTheta%2Cn%7D%5Cbm%7BW%7D_k%5Cbm%7Bx%7D_n+%5Cright%29%3D+%5Ctext%7BRe%7D%5Cleft%5B%5Csum_%7Bi%3D0%7D%5E%7Bd%2F2-1%7D%5Cboldsymbol%7Bq%7D_%7B%5B2i%3A2i%2B1%5D%7D%5Cboldsymbol%7Bk%7D_%7B%5B2i%3A2i%2B1%5D%7D%5E%2A+e%5E%7B%5Ctext%7Bi%7D%28m-n%29%5Ctheta_i%7D%5Cright%5D%5Cend%7Bequation%7D%5C%5C%5Ctag%7B16%7D&consumer=ZHI_MENG)

记 ![h_i = \boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^*, S_j = \sum\limits_{i=0}^{j-1} e^{\text{i}(m-n)\theta_i}](./assets/equation-20240301015747990)，并约定![h_{d/2}=0,S_0=0](./assets/equation-20240301015747991)，那么由**[Abel变换（分部求和法）](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/wiki/%E5%88%86%E9%83%A8%E6%B1%82%E5%92%8C%E6%B3%95)**可以得到：

![\begin{equation}\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i} = \sum_{i=0}^{d/2-1} h_i (S_{i +1} - S_i)  = \sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)\end{equation}\\\tag{17}](./assets/equation-20240301015748112)

所以

![\begin{equation}\begin{aligned} \left|\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i}\right| =&\, \left|\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)\right| \\ \leq&\, \sum_{i=0}^{d/2-1} |S_{i+1}| |h_{i+1} - h_i| \\ \leq&\, \left(\max_i |h_{i+1} - h_i|\right)\sum_{i=0}^{d/2-1} |S_{i+1}|  \end{aligned}\end{equation}\\\tag{18}](./assets/equation-20240301015747990-9218667.)

因此我们可以考察 ![\frac{1}{d/2}\sum\limits_{i=1}^{d/2} |S_i|](./assets/equation-20240301015748042) 随着相对距离的变化情况来作为衰减性的体现：

![img](./assets/v2-6376b397b8ea3e8f05d74d433e98a3a4_b-20240301015744512.jpg)

从图中我们可以看到**随着相对距离的变大，内积结果有衰减趋势**的出现。因此，选择 ![\theta_i = 10000^{-2i/d}](./assets/equation-20240301015747787)，确实能带来一定的远程衰减性。论文中还试过以 ![\theta_i = 10000^{-2i/d}](https://www.zhihu.com/equation?tex=%5Ctheta_i+%3D+10000%5E%7B-2i%2Fd%7D&consumer=ZHI_MENG) 为初始化，将 ![\theta_i](./assets/equation-20240301015747832) 视为可训练参数，然后训练一段时间后发现 ![\theta_i](https://www.zhihu.com/equation?tex=%5Ctheta_i&consumer=ZHI_MENG) 并没有显著更新，因此干脆就直接固定![\theta_i = 10000^{-2i/d}](https://www.zhihu.com/equation?tex=%5Ctheta_i+%3D+10000%5E%7B-2i%2Fd%7D&consumer=ZHI_MENG)了。

## 2. RoPE实验

我们看一下 RoPE 在预训练阶段的实验效果：

| Stage | Max seq length | Batch size | Training steps | Loss | Accuracy |
| ----- | -------------- | ---------- | -------------- | ---- | -------- |
| 1     | 512            | 256        | 200k           | 1.73 | 65.0%    |
| 2     | 1536           | 256        | 12.5k          | 1.61 | 66.8%    |
| 3     | 256            | 256        | 120k           | 1.75 | 64.6%    |
| 4     | 128            | 512        | 80k            | 1.83 | 63.4%    |
| 5     | 1536           | 256        | 10k            | 1.58 | 67.4%    |
| 6     | 512            | 512        | 30k            | 1.66 | 66.2%    |

从上面可以看出，增大序列长度，预训练的准确率反而有所提升，这体现了 **RoPE 具有良好的外推能力**。

下面是在下游任务上的实验结果：

| Model         | Validation | Test   |
| ------------- | ---------- | ------ |
| BERT-512      | 64.13%     | 67.77% |
| WoBERT-512    | 64.07%     | 68.10% |
| RoFormer-512  | 64.13%     | 68.29% |
| RoFormer-1024 | 66.07%     | 69.79% |

其中 RoFormer 是一个绝对位置编码替换为 RoPE 的**[WoBERT](https://link.zhihu.com/?target=https%3A//github.com/ZhuiyiTechnology/WoBERT)**模型，后面的参数（512）是微调时截断的maxlen，可以看到 RoPE 确实能较好地处理长文本语义。

## 3. RoPE代码实现

Meta 的 LLAMA 和 清华的 ChatGLM 都使用了 RoPE 编码，下面看一下具体实现。

### 3.1 在LLAMA中的实现

```python3
# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wq = Linear(...)
        self.wk = Linear(...)
        self.wv = Linear(...)
        
        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, dim)
        xk = xk.view(batch_size, seq_len, dim)
        xv = xv.view(batch_size, seq_len, dim)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # scores.shape = (bs, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)  # (batch_size, seq_len, dim)
  # ......
```

这里举一个例子，假设batch_size=10, seq_len=3, d=8，则调用函数precompute_freqs_cis(d, seq_len)后，生成结果为：

```python3
In [239]: freqs_cis
Out[239]: 
tensor([[ 1.0000+0.0000j,  1.0000+0.0000j,  1.0000+0.0000j,  1.0000+0.0000j],
        [ 0.5403+0.8415j,  0.9950+0.0998j,  0.9999+0.0100j,  1.0000+0.0010j],
        [-0.4161+0.9093j,  0.9801+0.1987j,  0.9998+0.0200j,  1.0000+0.0020j]])
```

以结果中的第二行为例（对应的 m = 1），也就是：

![\begin{align} cos\left( 1*\theta_0 \right)&=cos\left( 1 \right)=0.5403,&sin\left( 1*\theta_0 \right)=sin\left( 1 \right)=0.8415\\ cos\left( 1*\theta_1\right)&=cos\left( 0.1 \right)=0.9950,&sin\left( 1*\theta_1 \right)=sin\left( 0.1 \right)=0.0998\\ cos\left( 1*\theta_2 \right)&=cos\left( 0.01 \right)=0.9999,&sin\left( 1*\theta_2 \right)=sin\left( 0.01 \right)=0.0100\\ cos\left( 1*\theta_3 \right)&=cos\left( 0.001 \right)=1.0000,&sin\left( 1*\theta_3 \right)=sin\left( 0.001 \right)=0.0010 \end{align}\tag{19}](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+cos%5Cleft%28+1%2A%5Ctheta_0+%5Cright%29%26%3Dcos%5Cleft%28+1+%5Cright%29%3D0.5403%2C%26sin%5Cleft%28+1%2A%5Ctheta_0+%5Cright%29%3Dsin%5Cleft%28+1+%5Cright%29%3D0.8415%5C%5C+cos%5Cleft%28+1%2A%5Ctheta_1%5Cright%29%26%3Dcos%5Cleft%28+0.1+%5Cright%29%3D0.9950%2C%26sin%5Cleft%28+1%2A%5Ctheta_1+%5Cright%29%3Dsin%5Cleft%28+0.1+%5Cright%29%3D0.0998%5C%5C+cos%5Cleft%28+1%2A%5Ctheta_2+%5Cright%29%26%3Dcos%5Cleft%28+0.01+%5Cright%29%3D0.9999%2C%26sin%5Cleft%28+1%2A%5Ctheta_2+%5Cright%29%3Dsin%5Cleft%28+0.01+%5Cright%29%3D0.0100%5C%5C+cos%5Cleft%28+1%2A%5Ctheta_3+%5Cright%29%26%3Dcos%5Cleft%28+0.001+%5Cright%29%3D1.0000%2C%26sin%5Cleft%28+1%2A%5Ctheta_3+%5Cright%29%3Dsin%5Cleft%28+0.001+%5Cright%29%3D0.0010+%5Cend%7Balign%7D%5Ctag%7B19%7D&consumer=ZHI_MENG)

最终按照公式（12）可以得到编码之后的 ![\bm{q},\bm{k}](./assets/equation-20240301015748247) 。

**注意：**在代码中是直接用freqs_cis[0] * xq_[0]的结果表示第一个 token 对应的旋转编码（和公式12计算方式有所区别）。其中将原始的 query 向量 ![\bm{q}](./assets/equation-20240301015748648) 转换为了复数形式。

```python3
In [351]: q_ = q.float().reshape(*q.shape[:-1], -1, 2)

In [352]: q_[0]
Out[352]: 
tensor([[[ 1.0247,  0.4782],
         [ 1.5593,  0.2119],
         [ 0.4175,  0.5309],
         [ 0.4858,  0.1850]],

        [[-1.7456,  0.6849],
         [ 0.3844,  1.1492],
         [ 0.1700,  0.2106],
         [ 0.5433,  0.2261]],

        [[-1.1206,  0.6969],
         [ 0.8371, -0.7765],
         [-0.3076,  0.1704],
         [-0.5999, -1.7029]]])

In [353]: xq = torch.view_as_complex(q_)

In [354]: xq[0]
Out[354]: 
tensor([[ 1.0247+0.4782j,  1.5593+0.2119j,  0.4175+0.5309j,  0.4858+0.1850j],
        [-1.7456+0.6849j,  0.3844+1.1492j,  0.1700+0.2106j,  0.5433+0.2261j],
        [-1.1206+0.6969j,  0.8371-0.7765j, -0.3076+0.1704j, -0.5999-1.7029j]])
```

**这里为什么可以这样计算？**

主要是利用了复数的乘法性质。

我们首先来复习一下复数乘法的性质：

![(a+ib) \cdot (c+id) = ac + ibc + iad + i^2bd=(ac-bd)+i(bc+ad) \\](./assets/equation-20240301015748357)

因此要计算：

![\begin{align} f_q\left( \bm{x}_m,m \right)  &= \begin{pmatrix}  \cos m\theta & -\sin m\theta \\  \sin m \theta &  \cos m \theta \end{pmatrix}\begin{pmatrix}  q_m^{(1)}  \\  q_m^{(2)}    \end{pmatrix} \\ &= \left( \cos m\theta *q_m^{(1)}-\sin m\theta *q_m^{(2)} ,\sin m\theta *q_m^{(1)}-\cos m\theta *q_m^{(2)} \right) \end{align}](./assets/equation-20240301015749034)

可以转化为计算：

![\left( \cos m\theta+i \sin m\theta \right)\cdot \left( q_m^{(1)}+i q_m^{(2)} \right)\\](./assets/equation-20240301015748332)

所以可以将公式（12）转化为两个复数的乘法运算。

### 3.2 在ChatGLM中的实现

和 LLAMA 的实现方式相差不大。代码如下：

```python3
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
         # 计算 \theta_i
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # 对应m * \theta
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # 将 m * \theta 拼接两次，对应复数的实部和虚部
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]  # 计算得到cos(m*\theta)
            sin_cached = emb.sin()[:, None, :]  # 计算得到cos(m*\theta)
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  
```

## 4. RoPE的外推性

我们都知道 RoPE 具有很好的外推性，前面的实验结果也证明了这一点。这里解释下具体原因。

RoPE 可以通过旋转矩阵来实现位置编码的外推，即可以通过旋转矩阵来生成超过预期训练长度的位置编码。这样可以提高模型的泛化能力和鲁棒性。

我们回顾一下 RoPE 的工作原理：假设我们有一个 ![d](./assets/equation-20240301015745723) 维的绝对位置编码 ![P_i](./assets/equation-20240301015748414) ，其中 ![i](./assets/equation-20240301015745122) 是位置索引。我们可以将 ![P_i](https://www.zhihu.com/equation?tex=P_i&consumer=ZHI_MENG) 看成一个 ![d](https://www.zhihu.com/equation?tex=d&consumer=ZHI_MENG) 维空间中的一个点。我们可以定义一个 ![d](https://www.zhihu.com/equation?tex=d&consumer=ZHI_MENG) 维空间中的一个旋转矩阵 ![\bm{R}](./assets/equation-20240301015749142) ，它可以将任意一个点沿着某个轴旋转一定的角度。我们可以用 ![\bm{R}](https://www.zhihu.com/equation?tex=%5Cbm%7BR%7D&consumer=ZHI_MENG) 来变换 ![P_i](https://www.zhihu.com/equation?tex=P_i&consumer=ZHI_MENG) ，得到一个新的点 ![Q_i=\bm{R}*P_i](./assets/equation-20240301015749128) 。我们可以发现， ![Q_i](./assets/equation-20240301015749558) 和 ![P_i](https://www.zhihu.com/equation?tex=P_i&consumer=ZHI_MENG) 的距离是相等的，即 ![\left|\left| Q_i-P_i \right|\right| = 0](./assets/equation-20240301015748884) 。这意味着 ![Q_i](https://www.zhihu.com/equation?tex=Q_i&consumer=ZHI_MENG) 和 ![P_i](https://www.zhihu.com/equation?tex=P_i&consumer=ZHI_MENG) 的相对关系没有改变。但是， ![Q_i](https://www.zhihu.com/equation?tex=Q_i&consumer=ZHI_MENG) 和 ![P_i](https://www.zhihu.com/equation?tex=P_i&consumer=ZHI_MENG) 的距离可能发生改变，即 ![\left|\left| Q_i-P_j \right|\right| \ne \left|\left| P_i-P_j \right|\right|](./assets/equation-20240301015749064)。这意味着 ![Q_i](https://www.zhihu.com/equation?tex=Q_i&consumer=ZHI_MENG) 和 ![P_j](./assets/equation-20240301015749327) 的相对关系有所改变。因此，我们可以用 ![\bm{R}](https://www.zhihu.com/equation?tex=%5Cbm%7BR%7D&consumer=ZHI_MENG) 来调整不同位置之间的相对关系。

如果我们想要生成超过预训练长度的位置编码，我们只需要用 ![\bm{R}](https://www.zhihu.com/equation?tex=%5Cbm%7BR%7D&consumer=ZHI_MENG) 来重复变换最后一个预训练位置编码 ![P_n](./assets/equation-20240301015749364) ，得到新的位置编码 ![Q_{n+1} = \bm{R} * P_n ，Q_{n+2} = \bm{R} * Q_{n+1} ， Q_{n+3} = \bm{R} * Q_{n+2} ](./assets/equation-20240301015749449) ，依此类推。这样就可以得到任意长度的位置编码序列 ![Q_1, Q_2, …, Q_m ](./assets/equation-20240301015749422) ，其中 ![m](https://www.zhihu.com/equation?tex=m&consumer=ZHI_MENG) 可以大于 ![n](https://www.zhihu.com/equation?tex=n&consumer=ZHI_MENG) 。由于 ![\bm{R}](https://www.zhihu.com/equation?tex=%5Cbm%7BR%7D&consumer=ZHI_MENG) 是一个正交矩阵，它保证了 ![ Q_i ](./assets/equation-20240301015749467) 和 ![Q_j ](./assets/equation-20240301015749650) 的距离不会无限增大或缩小，而是在一个有限范围内波动。这样就可以避免数值溢出或下溢的问题。同时，由于 ![\bm{R}](https://www.zhihu.com/equation?tex=%5Cbm%7BR%7D&consumer=ZHI_MENG) 是一个可逆矩阵，它保证了 ![ Q_i ](https://www.zhihu.com/equation?tex=+Q_i+&consumer=ZHI_MENG)和 ![Q_j ](https://www.zhihu.com/equation?tex=Q_j+&consumer=ZHI_MENG) 的距离可以通过 ![\bm{R}](https://www.zhihu.com/equation?tex=%5Cbm%7BR%7D&consumer=ZHI_MENG) 的逆矩阵 ![\bm{R}^{-1}](./assets/equation-20240301015749705) 还原到 ![ P_i](./assets/equation-20240301015749732) 和 ![P_j ](./assets/equation-20240301015749667) 的距离，即 ![||\bm{R}^{-1} * Q_i - \bm{R}^{-1} * Q_j|| = ||P_i - P_j||](./assets/equation-20240301015749926) 。这样就可以保证位置编码的可逆性和可解释性。

总结而言：

**旋转编码 RoPE 可以有效地保持位置信息的相对关系**，即相邻位置的编码之间有一定的相似性，而远离位置的编码之间有一定的差异性。这样可以增强模型对位置信息的感知和利用。这一点是其他绝对位置编码方式（如正弦位置编码、学习的位置编码等）所不具备的，因为它们只能表示绝对位置，而不能表示相对位置。

**旋转编码 RoPE 可以通过旋转矩阵来实现位置编码的外推**，即可以通过旋转矩阵来生成超过预训练长度的位置编码。这样可以提高模型的泛化能力和鲁棒性。这一点是其他固定位置编码方式（如正弦位置编码、固定相对位置编码等）所不具备的，因为它们只能表示预训练长度内的位置，而不能表示超过预训练长度的位置。

**旋转编码 RoPE 可以与线性注意力机制兼容**，即不需要额外的计算或参数来实现相对位置编码。这样可以降低模型的计算复杂度和内存消耗。这一点是其他混合位置编码方式（如Transformer-XL、XLNet等）所不具备的，因为它们需要额外的计算或参数来实现相对位置编码。

## 总结

最近一直听到旋转编码这个词，但是一直没有仔细看具体原理。今天花时间仔细看了一遍，确实理论写的比较完备，而且实验效果也不错。目前很多的大模型，都选择了使用了这种编码方式（LLAMA、GLM等）。

## 附录

这里补充一下前面公式1.3.2节中，公式（8）~（11）是怎么推导出来的。

回到之前的公式（8），编码之后的 ![\bm{q},\bm{v}](./assets/equation-20240301015749786) 以及内积 ![\left< \bm{q},\bm{v} \right>](./assets/equation-20240301015749945) 的形式如下：

![f_q(\bm{x}_m,m)=(\bm{W}_q\bm{x}_m)e^{im\theta} \\ f_k(\bm{x}_n,n)=(\bm{W}_kx_n)e^{in\theta} \\ g(\bm{x}_m,x_n,m-n)=Re[(\bm{W}_\bm{q}x_m)(\bm{W}_k\bm{x}_n)^{*}e^{i(m-n)\theta}] \\](./assets/equation-20240301015749944)

上面的公式为什么满足： ![\left<\bm{f}_q(\bm{x}_m,m),f_k(\bm{x}_n,n)\right>=g(\bm{x}_m,\bm{x}_n,m-n) ](./assets/equation-20240301015750083) 。

首先我们得先了解一下基本的复数相关知识。

首先看到上述 ![f](./assets/equation-20240301015746937) 和 ![g](https://www.zhihu.com/equation?tex=g&consumer=ZHI_MENG) 公式中有个指数函数： ![$$e^{ix} $$](https://www.zhihu.com/equation?tex=%24%24e%5E%7Bix%7D+%24%24&consumer=ZHI_MENG)

这个其实是欧拉公式，其中 ![x](./assets/equation-20240301015750083-9218670.) 表示任意实数， ![e](./assets/equation-20240301015750172) 是自然对数的底数， ![i](./assets/equation-20240301015745122) 是复数中的虚数单位，则根据欧拉公式有：

![e^{ix} = \cos x + i\sin x \\](./assets/equation-20240301015750159)

则是上述指数函数可以表示为实部为 ![\cos x](./assets/equation-20240301015750262) ，虚部为 ![\sin x](./assets/equation-20240301015750328) 的一个复数，欧拉公式建立了指数函数、三角函数和复数之间的桥梁。

则上述 ![f](./assets/equation-20240301015746937) 和 ![g](https://www.zhihu.com/equation?tex=g&consumer=ZHI_MENG) 公式的

![ e^{im\theta}=\cos (m\theta) + i\sin (m\theta) \\ e^{in\theta}=\cos (n\theta) + i\sin (n\theta) \\ e^{i(m-n)\theta}=\cos ((m-n)\theta) + i\sin ((m-n)\theta) \\](./assets/equation-20240301015750396)

然后我们看回公式：

![f_q(\bm{x}_m,m)=(\bm{W}_q\bm{x}_m)e^{im\theta} \\](./assets/equation-20240301015750384)

其中 ![\bm{W}_q](./assets/equation-20240301015750420) 是个二维矩阵， ![\bm{x}_m](./assets/equation-20240301015745983) 是个二维向量，相乘的结果也是一个二维向量，这里用 ![\bm{q}_m](./assets/equation-20240301015745721) 表示：

![ q_m= \begin{pmatrix}  q_m^{(1)}  \\  q_m^{(2)}    \end{pmatrix} = \bm{W}_q\bm{x}_m =\begin{pmatrix}  {W}_q^{(11)} & W_q^{(12)} \\  W_q^{(21)} & W_q^{(22)}    \end{pmatrix} \begin{pmatrix}  x_m^{(1)}  \\  x_m^{(2)}    \end{pmatrix} \\](./assets/equation-20240301015750407)

然后首先将 ![\bm{q}_m](./assets/equation-20240301015745721) 表示成复数形式：

![\bm{q}_m = [q_m^{(1)}, q_m^{(2)}] = [q_m^{(1)} + iq_m^{(2)}] \\](./assets/equation-20240301015750609)

接着

![\bm{f}_q(\bm{x}_m,m)=(\bm{W}_q\bm{x}_m)e^{im\theta}=\bm{q}_me^{im\theta} \\](./assets/equation-20240301015750736)

其实就是两个复数相乘：

![\bm{q}_me^{im\theta}=(q_m^{(1)} + iq_m^{(2)}) * (\cos (m\theta) + i\sin (m\theta)) \\](./assets/equation-20240301015750609-9218670.)

然后就有：

![\bm{q}_me^{im\theta}=(q_m^{(1)} + iq_m^{(2)}) * (\cos (m\theta) + i\sin (m\theta)) \\ =(q_m^{(1)}cos (m\theta) -  q_m^{(2)} \sin (m\theta) ) + i(q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta)) \\](./assets/equation-20240301015750627)

将结果重新表达成实数向量形式就是：

![ \bm{q}_me^{im\theta}=[q_m^{(1)} \cos (m\theta) -  q_m^{(2)} \sin (m\theta), q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta)] \\](./assets/equation-20240301015750648)

**这里不难发现就是 query 向量乘以了一个旋转矩阵**。

![f_q(\bm{x}_m,m)=(\bm{W}_q\bm{x}_m)e^{im\theta}=\bm{q}_me^{im\theta}\\ =[q_m^{(1)} \cos (m\theta) -  q_m^{(2)} \sin (m\theta), q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta)] \\ = \begin{pmatrix}  \cos (m\theta) & -\sin (m\theta) \\  \sin (m\theta) & \cos (m\theta)    \end{pmatrix} \begin{pmatrix}  q_m^{(1)}  \\  q_m^{(2)}    \end{pmatrix} \\](./assets/equation-20240301015750820)

**这就是为什么叫做旋转式位置编码的原因。**

同理可得 key 向量 ![\bm{k}_n](./assets/equation-20240301015746012) ：

![f_k(\bm{x}_n,n)=(\bm{W}_k\bm{x}_n)e^{in\theta}=\bm{k}_ne^{in\theta}\\ =[k_n^{(1)} \cos (n\theta) -  k_n^{(2)} \sin (n\theta), k_n^{(2)}\cos (n\theta) + k_n^{(1)}\sin (n\theta)] \\ = \begin{pmatrix}  \cos (n\theta) & -\sin (n\theta) \\  \sin (n\theta) & \cos (n\theta)    \end{pmatrix} \begin{pmatrix}  k_n^{(1)}  \\  k_n^{(2)}    \end{pmatrix} \\](./assets/equation-20240301015750977)

最后还有个函数 ![g](https://www.zhihu.com/equation?tex=g&consumer=ZHI_MENG) ：

![g(\bm{x}_m,\bm{x}_n,m-n)=Re[(\bm{W}_q\bm{x}_m)(\bm{W}_k\bm{x}_n)^{*}e^{i(m-n)\theta}] \\](./assets/equation-20240301015750886)

其中 ![Re\left( x \right)](./assets/equation-20240301015751030) 表示一个复数 ![x](./assets/equation-20240301015750083-9218670.) 的实部部分，而 ![ (\bm{W}_k\bm{x}_n)^{*} ](./assets/equation-20240301015751025) 则表示复数 ![ \bm{W}_k\bm{x}_n ](./assets/equation-20240301015751134) 的共轭。

复习一下共轭复数的定义：![ z=a+ib\\ z^*=a-ib  \\](./assets/equation-20240301015751045)

所以可得：

![\bm{W}_q\bm{x}_m = \bm{q}_m = q_m^{(1)} + iq_m^{(2)} \\ \bm{W}_k\bm{x}_n=\bm{k}_n= k_n^{(1)} + ik_n^{(2)} \\ (\bm{W}_k\bm{x}_n)^*=\bm{k}_n^*= k_n^{(1)} - ik_n^{(2)} \\ e^{i(m-n)\theta}=\cos((m-n)\theta) + i \sin((m-n)\theta) \\](./assets/equation-20240301015751292)

继续可得：

![\begin{align} g(\bm{x}_m,\bm{x}_n,m-n) &=Re[(\bm{W}_q\bm{x}_m)(\bm{W}_k\bm{x}_n)^{*}e^{i(m n)\theta}] \\ & = Re[(q_m^{(1)} + iq_m^{(2)})(k_n^{(1)} - ik_n^{(2)})(\cos((m-n)\theta) + i \sin((m-n)\theta))] \\  &= Re[((q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)}) + i(q_m^{(2)}k_n^{(1)} - q_m^{(1)}k_n^{(2)}))(\cos((m-n)\theta) + i \sin((m-n)\theta))] \\  &= (q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)})\cos((m-n)\theta) - (q_m^{(2)}k_n^{(1)} - q_m^{(1)}k_n^{(2)})\sin((m-n)\theta)  \end{align}](./assets/equation-20240301015751229)

接下来我们就要证明函数 ![g](https://www.zhihu.com/equation?tex=g&consumer=ZHI_MENG) 的计算公式是成立的。

首先回顾一下 attention 操作， 位置 ![m](https://www.zhihu.com/equation?tex=m&consumer=ZHI_MENG) 的 query 和位置 ![n](https://www.zhihu.com/equation?tex=n&consumer=ZHI_MENG) 的 key 会做一个内积操作：

![\begin{align} f_q(x_m,m)&=[q_m^{(1)} \cos (m\theta) -  q_m^{(2)} \sin (m\theta), q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta)] \\  f_k(x_n,n)& =[k_n^{(1)} \cos (n\theta) -  k_n^{(2)} \sin (n\theta), k_n^{(2)}\cos (n\theta) + k_n^{(1)}\sin (n\theta)] \\  <f_q(x_m,m),f_k(x_n,n)> &=  (q_m^{(1)} \cos (m\theta) -  q_m^{(2)} \sin (m\theta))(k_n^{(1)} \cos (n\theta) -  k_n^{(2)} \sin (n\theta)) \\ &+ (q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta))(k_n^{(2)}\cos (n\theta) + k_n^{(1)}\sin (n\theta))\\ & =q_m^{(1)} \cos (m\theta) k_n^{(1)} \cos (n\theta) - q_m^{(1)} \cos (m\theta)k_n^{(2)} \sin (n\theta)\\ & - q_m^{(2)} \sin (m\theta)k_n^{(1)} \cos (n\theta) + q_m^{(2)} \sin (m\theta)k_n^{(2)} \sin (n\theta) \\ & + q_m^{(2)}\cos (m\theta)k_n^{(2)}\cos (n\theta) + q_m^{(2)}\cos (m\theta)k_n^{(1)}\sin (n\theta) \ + q_m^{(1)}\sin (m\theta)k_n^{(2)}\cos (n\theta) + q_m^{(1)}\sin (m\theta)k_n^{(1)}\sin (n\theta)  \end{align}](./assets/equation-20240301015751258)

接着进行推导，我们整理一下：

![\begin{align} <f_q(\bm{x}_m,m),f_k(\bm{x}_n,n)>  &=  {q}_m^{(1)}{k}_n^{(1)}(\cos(m\theta)\cos(n\theta) + \sin(m\theta)\sin(n\theta) ) \\  &+ {q}_m^{(1)}{k}_n^{(2)}(-\cos(m\theta)\sin(n\theta) + \sin(m\theta)\cos(n\theta) ) \\ & + {q}_m^{(2)}{k}_n^{(1)}(-\sin(m\theta)\cos(n\theta) + \cos(m\theta)\sin(n\theta) ) \\  &+ {q}_m^{(2)}{k}_n^{(2)}(\sin(m\theta)\sin(n\theta) + \cos(m\theta)\cos(n\theta) ) \\ & = q_m^{(1)}k_n^{(1)}\cos((m-n)\theta) \\  &+ q_m^{(1)}k_n^{(2)}\sin((m-n)\theta) \\  &- q_m^{(2)}k_n^{(1)}\sin((m-n)\theta) \\ & + q_m^{(2)}k_n^{(2)}\cos((m-n)\theta) \\  &= (q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)})\cos((m-n)\theta) + (q_m^{(1)}k_n^{(2)}- q_m^{(2)}k_n^{(1)})\sin((m-n)\theta) \\ & = (q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)})\cos((m-n)\theta) - (q_m^{(2)}k_n^{(1)} - q_m^{(1)}k_n^{(2)})\sin((m-n)\theta) \\ &=g(x_m,x_n,m-n)  \end{align}](./assets/equation-20240301015751262)

这就证明上述关系是成立的，位置 ![m](https://www.zhihu.com/equation?tex=m&consumer=ZHI_MENG) 的 query 和位置 ![n](https://www.zhihu.com/equation?tex=n&consumer=ZHI_MENG) 的 key 的内积就是函数 ![g](https://www.zhihu.com/equation?tex=g&consumer=ZHI_MENG) 。

把上面的式子用矩阵向量乘的形式来表达就是：

![<f_q(\bm{x}_m,m),f_k(\bm{x}_n,n)> \\ =\begin{pmatrix} \begin{pmatrix}  \cos (m\theta) & -\sin (m\theta) \\  \sin (m\theta) & \cos (m\theta)    \end{pmatrix} \begin{pmatrix}  q_m^{(1)}  \\  q_m^{(2)}    \end{pmatrix} \end{pmatrix}^T  \begin{pmatrix}  \begin{pmatrix}  \cos (n\theta) & -\sin (n\theta) \\  \sin (n\theta) & \cos (n\theta)    \end{pmatrix} \begin{pmatrix}  k_n^{(1)}  \\  k_n^{(2)}    \end{pmatrix} \end{pmatrix}  \\ = \begin{pmatrix}  q_m^{(1)} &  q_m^{(2)}  \\ \end{pmatrix}  \begin{pmatrix}  \cos (m\theta) & \sin (m\theta) \\  -\sin (m\theta) & \cos (m\theta)    \end{pmatrix}    \begin{pmatrix}  \cos (n\theta) & -\sin (n\theta) \\  \sin (n\theta) & \cos (n\theta)    \end{pmatrix} \begin{pmatrix}  k_n^{(1)}  \\  k_n^{(2)}    \end{pmatrix} \\ = \begin{pmatrix}  q_m^{(1)} &  q_m^{(2)}  \\ \end{pmatrix}   \begin{pmatrix}  \cos(m\theta)\cos(n\theta) + \sin(m\theta)\sin(n\theta) & -\cos(m\theta)\sin(n\theta) + \sin(m\theta)\cos(n\theta) \\  -\sin(m\theta)\cos(n\theta) + \cos(m\theta)\sin(n\theta) & \sin(m\theta)\sin(n\theta) + \cos(m\theta)\cos(n\theta) \end{pmatrix}   \begin{pmatrix}  k_n^{(1)}  \\  k_n^{(2)}    \end{pmatrix} \\ =\begin{pmatrix}  q_m^{(1)} &  q_m^{(2)}  \\ \end{pmatrix}   \begin{pmatrix}  \cos((m-n)\theta) & -\sin((m-n)\theta) \\  \sin((m-n)\theta) &  \cos((m-n)\theta) \end{pmatrix}   \begin{pmatrix}  k_n^{(1)}  \\  k_n^{(2)}    \end{pmatrix} \\](./assets/equation-20240301015751386)

## 参考

[ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.09864.pdf)

[梁德澎：一文看懂 LLaMA 中的旋转式位置编码（Rotary Position Embedding）](https://zhuanlan.zhihu.com/p/642884818)

[马梦之：一步一步，推导旋转位置编码 (Rotary Position Embedding, RoPE)](https://zhuanlan.zhihu.com/p/644585013)

[苏剑林：Transformer升级之路：2、博采众长的旋转式位置编码](https://zhuanlan.zhihu.com/p/359502624)