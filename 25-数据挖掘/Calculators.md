## Colab 

https://colab.research.google.com/drive/1QGLKMEKpPqify6qHKoRNAmYnLdljRMIT?usp=sharing

``` python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import copy
import math
```

``` python
# Information Gain Calculators
def Info(probsOrcounts):
  probs = probsOrcounts
  probs /= np.sum(probsOrcounts)
  sum =  0
  for p in probs:
    if p == 0:
      continue
    sum += p * math.log2(p)
  return -sum

def Info_AInD(A_in_D):
  sum =  0
  D_sum = np.sum(A_in_D)
  for dj in A_in_D:
    Dj_sum = np.sum(dj)
    sum += Info(dj) * Dj_sum / D_sum
  return sum

def Gain_AInD(A_in_D):
  sum_p_n = np.sum(A_in_D, axis = 0)

  sum =  0
  D_sum = np.sum(A_in_D)
  for dj in A_in_D:
    Dj_sum = np.sum(dj)
    sum += Info(dj) * Dj_sum / D_sum
  return Info(sum_p_n.tolist()) - sum

def SplitInfo_AInD(A_in_D):
  sum =  0
  D_sum = np.sum(A_in_D)
  for dj in A_in_D:
    Dj_sum = np.sum(dj)
    cal = Dj_sum / D_sum
    sum += cal * math.log2(cal)
  return - sum

    # p, n
D =  [9, 5]

         # p, n
A_in_D = [[2,2],
          [4,2],
          [3,1]]

InfoD = Info(D)
Info_A_in_D = Info_AInD(A_in_D)
Gain_A = InfoD - Info_A_in_D

print("Expected Information (D): " ,round(InfoD, 3))
print("Expected Information (A in D): " , round(Info_A_in_D, 3))

print("Gain (A):", round(Gain_AInD(A_in_D), 3))
print("SplitInfo (A):", round(SplitInfo_AInD(A_in_D), 3))
print("Gain_ratio (A):", round(Gain_AInD(A_in_D) / SplitInfo_AInD(A_in_D), 3))
```

    Expected Information (D):  0.94
    Expected Information (A in D):  0.911
    Gain (A): 0.029
    SplitInfo (A): 1.557
    Gain_ratio (A): 0.019

``` python
# gini calculator 
      # p, n
s =  [[7,3],
      [2,2]] 

def cal_gini_of_D(probsOrcounts):
  sum = 1
  probs = probsOrcounts
  probs /= np.sum(probsOrcounts)
  for p in probs:
    sum -= p**2
  return sum

def cal_gini_a_of_D(A_in_D):
  sum =  0
  D_sum = np.sum(A_in_D)
  for dj in A_in_D:
    Dj_sum = np.sum(dj)
    sum += cal_gini_of_D(dj) * Dj_sum / D_sum
  return sum
print("Gini between A and B:", round(cal_gini_a_of_D(s), 3))
```

    Gini between A and B: 0.443

``` python
L = [23, 35, 40]

def gaussian_prob(x, mu, sig):
  return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

# Min
print("Min:", np.min(L))
# Max
print("Max:", np.max(L))
# Mean
print("Mean:", np.mean(L))
# Std
print("Std:",np.std(L))
# Sample std
print("Sample Std(sigma):", np.std(L, ddof=1))
# Var
print("Var:",np.var(L))
# Sample Var = Sigma^2
print("Sample Var(Sigma^2):", np.var(L, ddof=1))
# Median
print("Median:",np.median(L))
# Mad #MAD = median(|xi – xm|)
print("MAD:",np.median(np.abs(L - np.median(L))))
# Mode
vals,counts = np.unique(L, return_counts=True)
print("Mode:",vals[np.argmax(counts)])
# Norm 2
print("Norm 2:",np.linalg.norm(L,ord=2))
# Norm 1
print("Norm 1:",np.linalg.norm(L,ord=1))
# Sorted
print("Sorted:",np.sort(L))
# feature - mean / std
print("Normalize L with std:", (L - np.mean(L)) / np.std(L))
print("Normalize L with min & max, From Zero to One:", (L - np.min(L)) / (np.max(L) - np.min(L)))
```

    Min: 23
    Max: 40
    Mean: 32.666666666666664
    Std: 7.1336448530109
    Sample Std(sigma): 8.736894948054106
    Var: 50.88888888888889
    Sample Var(Sigma^2): 76.33333333333334
    Median: 35.0
    MAD: 5.0
    Mode: 23
    Norm 2: 57.91372894228103
    Norm 1: 98.0
    Sorted: [23 35 40]
    Normalize L with std: [-1.35508101  0.32708852  1.02799249]
    Normalize L with min & max, From Zero to One: [0.         0.70588235 1.        ]

``` python
x = 30 # Input

miu = np.mean(L) # Mean
sigma_square = np.var(L, ddof=1) # Sample Var(Sigma^2)
sigma = np.std(L, ddof=1) # np.sqrt(sigma_square) # Sample Std 8.736894948054106

print("Gaussian_Prob:",gaussian_prob(x, miu, sigma))
```

    Gaussian_Prob: 0.04358367082808079

``` python
def similiar(Vec1,Vec2):
    Vec1 = np.array(Vec1)
    Vec2 = np.array(Vec2)
    return np.sum(Vec1 * Vec2) / (np.linalg.norm(Vec1,ord=2) * np.linalg.norm(Vec2,ord=2))
```

``` python
a = [1, 0, 0.5, 0.7, 0.1]
b = [0, 1, 0.5, 0.7, 0.2]

q = 0
r = 0
s = 0
t = 0

for i in range(len(a)):
    a_e = a[i]
    b_e = b[i]
    if a_e == 1 and b_e == 1:
        q = q + 1
    elif a_e == 1 and b_e == 0:
        r = r + 1
    elif a_e == 0 and b_e == 1:
        s = s + 1
    elif a_e == 0 and b_e == 0:
        t = t + 1

print('q,r',q,r)
print('s,t',s,t)

d = (r + s) / (q + r + s + t)
print("Symmetric binary dissimiliarty: ",d)
d = (r + s) / (q + r + s)
print("ASymmetric binary dissimiliarty: ",d)
JaccardSimiliry = 1 - d
print("JaccardSimiliry: ",JaccardSimiliry)
# Cosin Similiar
print("Cosin Similiar: ",similiar(a,b))
# Manhattan Distance
print("Manhattan Distance: ",np.linalg.norm(np.array(a) - np.array(b),ord=1))
# Euclidean Distance
print("Euclidean Distance: ",np.linalg.norm(np.array(a) - np.array(b),ord=2))
```

    q,r 0 1
    s,t 1 0
    Symmetric binary dissimiliarty:  1.0
    ASymmetric binary dissimiliarty:  1.0
    JaccardSimiliry:  0.0
    Cosin Similiar:  0.4306104517492559
    Manhattan Distance:  2.1
    Euclidean Distance:  1.4177446878757824

``` python
w0 = 5.5
w_list  = [0.15, 0.0000, 0.18, 2.0]
feature = [30, 70000, 80, 0]

print(np.sum(np.array(w_list) * np.array(feature)) + w0)
```

    24.4
