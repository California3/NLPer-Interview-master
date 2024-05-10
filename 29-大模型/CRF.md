# 条件随机场 CRF

![img](./assets/v2-df921363ad92958df76328a400dc7647_1440w.webp)

![image-20240311161445293](./assets/image-20240311161445293.png)

其中 Z(x) 是归一化因子， K 是特征函数的数量， λk 是与特征函数 fk 相关联的权重， fk 是定义在输入序列 *x* 和输出序列的某个特定位置 *i* 上的函数。

![截屏2024-03-28 下午5.07.14](./assets/%E6%88%AA%E5%B1%8F2024-03-28%20%E4%B8%8B%E5%8D%885.07.14.png)

```python
log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, self.outputs_seq, inputs_seq_len) # B * (S+2) * V, B * (S+2), B

		#  B * (S+2), 
preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, inputs_seq_len) 
				
```


  	

[![img](./assets/ex_14_5.png)](https://watermelon-1253263790.cos.ap-shanghai.myqcloud.com/ex_14_5.png)