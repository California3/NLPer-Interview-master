# Classiﬁcation & Prediction: SVM, Lazy Learners, & Variants - 7.2

## 1. Introduction

Most of this material is derived from the text, Han, Kamber and Pei, Chapter 9, or the corresponding powerpoint slides made available by the publisher. Where a source other than the text or its slides was used for the material, attribution is given. Unless otherwise stated, images are copyright of the publisher, Elsevier.



This topic concludes our work on the data mining problems of classification and prediction with a look at the very popular [Support Vector Machine](https://wattlecourses.anu.edu.au/mod/resource/view.php?id=2472101) kernel method as well as lazy learning algorithms including k-nearest neighbour, and finishing up with a brief summary of some variants of the classification and prediction data mining problem.



## 2. Support Vector Machine (Text: 9.3)

**Support vector machines (SVMs)**



- One of the most successful *classification* methods for ***both linear and nonlinear data***
- **It uses a nonlinear mapping to transform the original training data into a higher dimension**
- With the new dimension, it searches for the linear optimal separating **hyperplane** (i.e., “decision boundary”)
- With an appropriate nonlinear mapping to a sufficiently high dimension, data from two classes can always be separated by a hyperplane
- SVM finds this hyperplane using **support vectors** (“essential” training tuples) and **margins** (defined by the support vectors)



**History and Applications**



- Vapnik and colleagues (1992): [Support Vector Machine](https://wattlecourses.anu.edu.au/mod/resource/view.php?id=2472101)
  - groundwork from Vapnik & Chervonenkis’ statistical learning theory in 1960s
- Features: training can be slow but accuracy is high owing to their ability to model complex nonlinear decision boundaries (margin maximisation)
- Used for: classification and numeric prediction(regression)
- Applications:
  - handwritten digit recognition, object recognition, speaker identification, benchmarking time-series prediction tests



**Algorithms**

- Training of SVM can be distinguished by

  - Linearly separable case

  - Non-linearly separable case

    

### 2.1. Linearly Separable Case

**SVM with Linearly Separable Training Data**

- Let's consider buys_computer example with two input attributes [![A_1](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846959.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_1) and [![A_2](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846111.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_2). If the training tuples can be plotted as follows (x-axis and y-axis represent [![A_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4be60c01260fad068dd84cb934d15c36.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_1) and [![A_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e7fb081e7d6a49314607f263a85eef3c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_2), respectively), then the dataset is linearly separable:

![lssvm](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270851239.png)

- Because **a straight line (hyperplane) can be drawn to separate all the tuples** of class +1 from all the tuples of class -1.
- There are **infinite lines** (hyperplanes) separating the two classes
  - e.g., all of the dotted lines separate the training tuples exactly the same in the above example.
- We want to find the best one (the one that minimises classification error on unseen data)



**Maximum marginal hyperplane**

SVM searches for the hyperplane with **the largest margin**, i.e., *maximum marginal hyperplane* (**MMH**)

- **Margin**: Draw a perpendicular line from the hyperplane to a tuple. The distance between the hyperplane and the tuple is the margin of that hyperplane.

![mmh](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270852055.png)

In this example, the hyperplane on the right figure has a larger margin than the one on the left.



**Support Vectors:**

![support_vectors](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270853825.png)

- **Support vectors:** the training tuples that determine the largest margin hyperplane. In the above example, red-circled tuples are the support vectors of the hyperplane.



**Formal definition of hyperplanes and support vectors:**

**Two dimensional training tuple case:**

- In two dimensional space ([![A_1-A_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9627e8db8b2416eb82bab33564f1ab51.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_1-A_2) plane), a hyperplane corresponds to a line, and every hyperplane can be written as:
  - [![A_2 = a\times A_1 + b](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4ec75a62c7166a777ffb831978c28ab0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_2 %3D a\times A_1 %2B b)
- For a more general representation, if we replace [![A_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4be60c01260fad068dd84cb934d15c36.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_1) and [![A_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e7fb081e7d6a49314607f263a85eef3c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_2) by [![x_1](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846764.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x_1) and [![x_2](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846145.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x_2), then the above hyperplane can be rewritten as:
  - [![0 = w_1\times x_1 + w_2 \times x_2 + w_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/cdcdee132fbd22274900f21edfe0da75.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=0 %3D w_1\times x_1 %2B w_2 \times x_2 %2B w_0),
  - where [![w_1 = a, w_2 = -1, w_0 = b](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e6a5b07b0305cbfca1551037e7a534bb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_1 %3D a%2C w_2 %3D -1%2C w_0 %3D b).
  - We can represent any hyperplane(line) in two dimensional space with [![w_1, w_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3b0bf20890dcaf25bcacdbc3e5779c51.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_1%2C w_2), and [![w_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ac1052c8c41fa0e8d67714e0723a068b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0).
- In the linearly separable case, every training tuple satisfies the following condition:
  - H1 (positive class)
    - If [![w_1 \times x_1 + w_2 \times x_2 + w_0 \geq +1](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846456.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_1 \times x_1 %2B w_2 \times x_2 %2B w_0 \geq %2B1)
  - H2 (negative class):
    - If [![w_1 \times x_1 + w_2 \times x_2 + w_0 \leq -1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4d38ea77c56be2cbe19346a20b92ff0d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_1 \times x_1 %2B w_2 \times x_2 %2B w_0 \leq -1)
- Support vector: Therefore, every training tuple that satisfies [![w_1 \times x_1 + w_2 \times x_2 + w_0 = \pm 1](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846630.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_1 \times x_1 %2B w_2 \times x_2 %2B w_0 %3D \pm 1) is a support vector.

**N-dimensional training tuple case:**

- Let [![X = (x_1, x_2, x_3, ..., x_n)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/221f124c3477f622d2201a1dbe9f2691.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X %3D (x_1%2C x_2%2C x_3%2C ...%2C x_n)) be a training tuple with class label [![y_i ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/eea02ee7e0b4b6f8ea52e4d2632fc73c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y_i ) then a separating hyperplane can be written as[![W X^\top + w_0 = 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c863c733cbe77fa5ebaaaafaa496b4bf.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=W X^\top %2B w_0 %3D 0)where [![W={w_1, w_2, ..., w_n}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846987.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=W%3D{w_1%2C w_2%2C ...%2C w_n}) is a weight vector and [![w_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ac1052c8c41fa0e8d67714e0723a068b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0) a scalar (bias)[![W X^\top =W \cdot X = \sum_{i=1}^{n} w_i \times x_i](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846136.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=W X^\top %3DW \cdot X %3D \sum_{i%3D1}^{n} w_i \times x_i)
- The hyperplane defining the sides of the margin:
  - H1: [![w_0 + W X^\top\geq 1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/95d177a3107788f5d01790dbbd18f5fc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0 %2B W X^\top\geq 1) for [![y_i = +1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/33bd807c42c7a6e72bfabed3df5817ee.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y_i %3D %2B1), and
  - H2: [![w_0 + W X^\top\leq -1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/50e2228fb81b5a491b1d68a6cb3c609e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0 %2B W X^\top\leq -1) for [![y_i = -1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9e9d8c2fda87ed188a6638898ec1f238.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y_i %3D -1)
- These two equations can be combined into one equation:
  - [![y_i (w_0 + W X^\top) \geq 1 \quad \forall i](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846497.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y_i (w_0 %2B W X^\top) \geq 1 \quad \forall i)
  - This equation can be solved as a ***constrained (convex) quadratic optimisation problem*** that maximises the margins to estimate the weights [![W](https://wattlecourses.anu.edu.au/filter/tex/pix.php/61e9c06ea9a85a5088a499df6458d276.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=W) from the training set, and is the SVM version of *training the model.*



**Classify test tuple using trained model:**

During the testing phase, the trained model classifies a new tuple [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846627.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) using the rules:

- Using hyperplane
  - H1 (positive class)
    - If [![w_0 + W X^\top \geq 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/b0a54b57f7883dbaec367711f25ccc52.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0 %2B W X^\top \geq 0)
    - Then [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846627.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) will be classified as a positive class
  - H2 (negative class):
    - If [![w_0 + W X^\top < 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/914d45d25ac9e5bbc2ca3b7cc69a2e5f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0 %2B W X^\top < 0)
    - Then [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846627.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) will be classified as a negative class
- **Alternatively**, we can use the support vectors [![X_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a97118fb9e8d7e006a466bfc0771f888.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X_i), each with class label [![y_i](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846015.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y_i), to classify test tuples. For test tuple [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846627.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X), 
  - [![d(X) = \sum_{i=1}^{\ell} y_i \alpha_i X_i X^\top +b_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1423ae2cac62d83827237ef63aa6dbe8.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=d(X) %3D \sum_{i%3D1}^{\ell} y_i \alpha_i X_i X^\top %2Bb_0)
  - where [![\ell](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ee5e5c003694e7cd5ae404923c665edb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\ell) is the number of support vectors, and [![\alpha](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7b7f9dbfea05c83784f8b85149852f08.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\alpha) and [![b_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/2e426000b92cfbc7286b0e2cc2a37482.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=b_0) are automatically determined by the optimisation/training algorithm.
  - If the sign of [![d(X)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846150.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=d(X)) is positive then [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270846627.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) is classified as H1, otherwise H2.
  - Note that we need to **keep only** **the support vectors** for testing
    - This fact will be used in the non-linearly separable case



**Why Is SVM effective on high-dimensional data?**

- The **complexity** of a trained classifier is characterised by the number of support vectors rather than the dimensionality of the data
- The **support vectors** are the essential or critical training examples —they lie closest to the decision boundary (MMH)
- If all other training examples are removed and the training is repeated, the same separating hyperplane would be found from the support vectors alone
- The number of support vectors found can be used to compute an (upper) bound on the expected error rate of the SVM classifier, which is independent of the data dimensionality
- Thus, an SVM with a small number of support vectors can have good generalisation, even when the dimensionality of the data is high



### 2. Support Vector Machine (Text: 9.3)

#### 2.2. Linearly Inseparable Case

So far we've discussed the case where there is a straight line that separates two classes in the training dataset. Now, we will discuss the case when the data are not linearly separable.



Example of linearly inseparable data

![dataset_nonsep](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270856618.png)

\> A two-class dataset that is not linearly separable. The outer ring (cyan) is class '0', while the inner ring (red) is class '1'

- Linearly inseparable training data. Unlike the linearly separable data, it is not possible to draw a **straight (linear) line** to separate the classes.
- Basic SVM would not be able to find a feasible solution here.
- But there is a way to extend the linear approach in this case!



**Kernel Trick**



The idea is to obtain linear separation by **mapping the data to a higher dimensional space**. Let's see the example first:

![data_2d_to_3d](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270856363.png)

\> (Left) A dataset in [![\mathbb{R}^2](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270855367.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\mathbb{R}^2), not linearly separable. (Right) The same dataset transformed by the transformation: ![ffe8444d63922077a7e16fb2af302d39](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270857457.png)

![data_2d_to_3d_hyperplane](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270858017.png)

\> Hyperplane (green plane) that **linearly separates two classes** in the higher dimensional space.

In the above example, we can train a linear SVM classifier that **successfully finds a good decision boundary in [![\mathbb{R}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270855500.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\mathbb{R}).**

**However, we are given the dataset in [![\mathbb{R}^2](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270855367.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\mathbb{R}^2). The challenge is to \**find a transformation*\*[![\phi:\mathbb{R}^2 \rightarrow \mathbb{R}^3](https://wattlecourses.anu.edu.au/filter/tex/pix.php/29b2bdd88cbbcf95973ac502a0e2141c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\phi%3A\mathbb{R}^2 \rightarrow \mathbb{R}^3), such that the transformed dataset is linearly separable in [![\mathbb{R}^3](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a6bcd1eddcf2923b077bd5e08d5731c6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\mathbb{R}^3) .** 

**,![a69fb0dda90642931dd8b16e54fa994d](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270858713.png) which after applied to every point in the original tuples yields the linearly separable dataset.**

***\*ACTION: Watch this video visualisation of polynomial kernel.\**** 



Assuming we have such a transformation [![\phi](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1ed346930917426bc46d41e22cc525ec.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\phi) , the new classification pipeline is as follows.

- First transform the training set [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270859066.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) to [![X'](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c164e630313c7e71508c5c046f83c6f5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X') with [![\phi](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1ed346930917426bc46d41e22cc525ec.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\phi) . 
- Train a linear SVM on [![X'](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c164e630313c7e71508c5c046f83c6f5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X') to get a new SVM. 
- At test time, a new example [![\bar{x}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6fbdf291cda891b99cf211417ad1df18.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\bar{x}) will first be transformed to [![\bar{x}' = \phi(\bar{x}')](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9af68d28996118aac1d30c9e08fa9d5f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\bar{x}' %3D \phi(\bar{x}')) during the testing time.

This is exactly the same as the train/test procedure for regular linear SVMs, but with an added data transformation via [![\phi](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1ed346930917426bc46d41e22cc525ec.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\phi).

We have improved the **expressiveness** of the Linear SVM classifier by working in a higher-dimensional space.



**Kernels**

- The decision rule used in the linearly separable case, for test vector [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270859250.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) was

  - [![d(X) = \sum_{i=1}^{\ell} y_i \alpha_i X_i X^\top +b_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1423ae2cac62d83827237ef63aa6dbe8.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=d(X) %3D \sum_{i%3D1}^{\ell} y_i \alpha_i X_i X^\top %2Bb_0)

- Now it is converted, in the higher-dimensional space to

  - [![d(X) = \sum_{i=1}^{\ell} y_i \alpha_i \phi(X_i) \cdot \phi(X)^\top +b_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bb239d222ca89bee12a7f661a8190ac.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=d(X) %3D \sum_{i%3D1}^{\ell} y_i \alpha_i \phi(X_i) \cdot \phi(X)^\top %2Bb_0)
  - But all these dot-products over potentially very high-dimensional vectors are expensive.

- We define a kernel function K(.,.) on the data in the original lower-dimensional space:

  - Let [![X_i, X_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d61983dc0c9a47f15cf87e30a3b88377.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X_i%2C X_j) be vectors. Then [![K(X_i, X_j) = \phi(X_i) \cdot \phi(X_j)^\top](https://wattlecourses.anu.edu.au/filter/tex/pix.php/b782499ede3f56c769337af62621c822.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=K(X_i%2C X_j) %3D \phi(X_i) \cdot \phi(X_j)^\top)

- Now, every [![\phi(X_i) \cdot \phi(X_j)^\top](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1f5a9f6719f4162d88eb3debc97e0c36.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\phi(X_i) \cdot \phi(X_j)^\top) can be replaced by [![K(X_i, X_j)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/66f39de07fd52c7f448b0921c9fb393d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=K(X_i%2C X_j)) in the training algorithm and decision rule. We do not need to make calculations over the higher-dimensional transformed vectors. We do not even need a formulation of [![\phi](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1ed346930917426bc46d41e22cc525ec.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\phi), as a kernel function is enough for determining the SVM. But we do need the kernel function to have certain properties, so that not just any function will do.

- Here are some widely used kernel functions. They do result in different classifiers, although they are commonly of similar accuracy and the only way to find the best one is to try it.

- - [Polynomial kernel](https://en.wikipedia.org/wiki/Polynomial_kernel) of degree [![h](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270859834.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=h): [![K(X_i, X_j) = (X_i \cdot X_j^\top + 1)^h](https://wattlecourses.anu.edu.au/filter/tex/pix.php/57e3456c7165311644163d54c0a2f12c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=K(X_i%2C X_j) %3D (X_i \cdot X_j^\top %2B 1)^h)
  - Gaussian radial basis function kernel: [![K(X_i, X_j) = e^{-dist(X_i, X_j)^2 / 2\sigma^2}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6d1fa8d9275b1ecec8b06d94ca819e83.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=K(X_i%2C X_j) %3D e^{-dist(X_i%2C X_j)^2 %2F 2\sigma^2})
  - Sigmoid Kernel: [![K(X_i, X_j) = tanh(\kappa X_i \cdot X_j^\top - \delta)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/b274a4cc05e641d00cef419b15f65d36.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=K(X_i%2C X_j) %3D tanh(\kappa X_i \cdot X_j^\top - \delta))
  - Linear Kernel: [![K(X_i, X_j) = X_i \cdot X_j^\top](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270859333.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=K(X_i%2C X_j) %3D X_i \cdot X_j^\top)

**ACTION: Watch this video if you'd like to know more details about learning SVM**

**[ Support Vector Machines by Patrick Winston(MIT)](https://wattlecourses.anu.edu.au/mod/url/view.php?id=2472100)**

Some of the examples used in this note are from http://www.eric-kim.net/



#### 2.3. Neural Network vs. SVM

**Comparison between neural network and SVM**

Decision hyperplanes found by non-linear SVM are similar to those found by neural network classifiers. e.g. SVM with Gaussian radial basis function is the same as a radial basis function neural network. e.g. A sigmoid kernel SVM is the same as a two-layer neural network (i.e with no hidden layers).

| Neural Network                                               | SVM                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Nondeterministic algorithm which finds local minima                        Generalises well but doesn't have strong mathematical foundation.            Can easily be learned in incremental fashion.                                                                       To learn complex function, use multi-layer neural network, but adding more layers adds more training time                                                                             Gets more complex with higher dimensions | Deterministic algorithm that finds the global minimum                                                                        Nice generalisation property.                                          Hard to learn - learned in batch mode using quadratic programming techniques                                          Using kernels can learn very complex functions       Well suited to high-dimensional data |



### 3. Practical Exercises: Support Vector Machines in Rattle

**Objectives**

The objectives of this lab are to experiment with the support vector machines (SVM) package available in **R** and**Rattle**, in order to better understand the issues involved with this data mining technique; to compare the SVM classification results with the results from decision trees; and to gain more experience with the different evaluation methods for supervised classification available in the **Rattle** tool.

------

**Preliminaries**

Read through the following section in the Rattle online documentation:



- **[Support Vector Machine](http://datamining.togaware.com/survivor/Support_Vector.html)**
- [**Risk Chart**](https://www.togaware.com/datamining/survivor/Risk_Charts.html)

------

For this lab, we will mainly use the **audit.csv** data set which you have used in the previous labs. If you want to use another data set to conduct more experiments at the end of the lab please do so.

The [support vector machine](https://wattlecourses.anu.edu.au/mod/resource/view.php?id=2472101) classifier in **Rattle** is based on the **R** package **[kernlab](http://cran.r-project.org/web/packages/kernlab/index.html)** (Kernel Methods Lab), and specifically on the **ksvm** class from this package. You can get help on this class by typing the following two commands into the **R** console (the terminal window where you started **R** and **Rattle**), assuming you have started **R**: 

- library(kernlab)
- help(ksvm)

To re-familiarise yourself with the evaluation of (classification) models, you might want to read the corresponding chapter [Evaluation and Deployment](http://datamining.togaware.com/survivor/Evaluation.html) in the **Rattle Data Miner** documentation (before coming to the lab).

------

**Tasks**

1. Start **Rattle** as described in the first lab sheet. Here is a quick repeat of the steps involved:

   a) Open a console/terminal window.
   b) Start R by typing **R** (capitalised!) followed by 'Enter'.
   c) Type: library(rattle) followed by 'Enter'.
   d) Type: rattle() followed by 'Enter'.

2. The following steps (up to step 7) are the same as the first steps for the lab on decision tree:

   Load the **CSV** data set **audit** (make sure you have **CSV File** selected in the **Data** tab, and the **Header** box is ticked).

   Click **Execute** to load the data into **Rattle**.

3. Now make sure the variable (attribute) **TARGET_Adjusted** is selected as Target variable, and that you partition the data (e.g. leave the 70/15/15 percentage split in the Partition box - which must be ticked). This means that we will use 70% of all records in the **audit** data set for training, 15% for validation (tuning) and 15% for testing.

4. Also make sure that the variable **ID** is set to role Ident(ifier).

5. You can select or set to **Ignore** other variables if you feel they are not suitable for classification (after having built a classification model you might later want to come back to the **Data** tab and change your variable selection).

6. Next you might want to explore the data set in order to again become familiar with it. Specifically, you should examine the values of the *target* variable **TARGET_Adjusted**. 

7. You might also want to have a look at the actual data (which you can do in the Data tab by clicking on the View button).

   

8. Now go to the Model tab and make sure the SVM type radio button is selected. As you can see, there is one main parameter you can modify, the **Kernel** function (the mathematical function that is at the core of the SVM), and further parameters can be entered into the **Options** input box. Please read the **Rattle** documentation on [support vector machines](http://datamining.togaware.com/survivor/Support_Vector.html) for more information.

9. To build a SVM, click on Execute and inspect what is printed into the main **Rattle** output area. How many support vectors are required (out of how many training records)? Go to the Evaluate tab and examine the error matrix results you get with this SVM (make sure the Validation button is activated and not the Training one). Write them down so you can compare them with the results from other SVMs you will construct later on in this lab. Next check the error and accuracy you get on the Testing data - why do they differ between Validation and Testing?

10. Also, do you remember the accuracy you achieved with the best decision trees on this **audit.csv** data set in the lab on decision tree?

11. Now experiment with the **Kernel** function, and for each SVM you build examine the resulting error matrix. Which one gives you the best results? Also check the **Training error** printed on the **Model** page. Is there a correlation between training and validation error?

12.  Next select the Tree classifier (as previously done in the lab on decision tree) and re-create the best decision tree classifier you got in the lab on decision tree. Once you have done this, go to the **Evaluate** tab and you will see that you can now also tick the Tree model box.

    Make sure both the SVM and Tree boxes are ticked, select Error Matrix and click on Execute. This should give you two error matrices each (two for the decision tree and two for the SVM). Which one is the better classifier?

13. Next select ROC and once you've executed you should see a graph window being shown which contains two curves - one for the decision tree (rpart) and one for the SVM (ksvm) classifier. Compare these graphs - again, which is the better classifier, and how do they differ?

14. Let's look at the [risk charts](http://www.togaware.com/datamining/survivor/Risk_Charts.html) implemented in **Rattle** (please read the documentation provided at the previous link). Go back to the **Data** tab, and select the **RISK_Adjustment** variable (attribute) as **Risk** variable (make sure you click on **Execute** before you go back to the **Model** tab).

    -  Again build your 2-class classifiers (decision tree and SVM), and then go to the Evaluation tab and select Risk (make sure both Tree and SVM are ticked). Once you click on Execute you will see two risk charts popping up (one per classifier). Analyse them to see which of the two classifiers is better. What is the purpose of the **Risk_Adjustment** variable? Can you think of other examples where a risk variable might be more useful than a simple class label?

    - In a risk chart, all tuples are sorted by their classification score(probability) and aligned along the caseload-axis from the highest score (0%) to the lowest score (100%).

    - The green and blue line represents the recall and precision, respectively, when x% of sorted tuples(caseload) are classified as a positive class.

    - The red line shows the proportion of risk taken into account, when x% of sorted tuples(caseload) are considered.

      

15.  If you have time, you might want to use different data sets, e.g from the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml/), and explore how you can build SVMs and decision trees on them.

16. Make sure you log out from your computer before you leave the lab room!



## Solution to Lab: Support Vector Machines in Rattle

9. The number of support vectors is 583.

The training data size is 2000*0.70 = 1400.

The error rates on validation dataset are:

Overall error: 17.1%, Averaged class error: 29.1%

Training and validation errors are different because they use different datasets (validation dataset, and train dataset) to measure the errors.



10. It depends on your parameter setting of the decision tree. For example, in my case, I obtained the following result:

Overall error: 15.3%, Averaged class error: 24.2%



11. Here are some examples that I obtained with different kernels:

With the polynomial kernel:

Overall error: 18.1%, Averaged class error: 29.8%

With the linear kernel:

Overall error: 18.1%, Averaged class error: 29.8%

With the laplacian kernel:

Overall error: 16.7%, Averaged class error: 30.45%

There are some positive correlation between training and validation errors (when the training error decreases, the validation error also decreases).

"However, the positive correlation is only valid when there is no overfitting." Fortunately, I couldn't find any overfitting in above examples.



12. Comparison between the decision tree and SVM(radial basis kernel).

The decision tree is better in my case in terms of the error rate. The result may vary with respect to a different configuration of parameters.



13. The AUC-ROC of the decision tree is 0.83, and the AUC-ROC of the SVM(radial basis kernel) is 0.86.

In terms of the AUC-ROC, the SVM performs better than the decision tree.

NB: a comparison between models depends on a choice of evaluation metric.



14. What is the purpose of the Risk_Adjustment variable? Can you think of other examples where a risk variable might be more useful than a simple class label? 

Sometimes, the importance of tuples in your data set is not the same. For example, let's assume that you are working at the lending section of a bank, and you want to build a classification model that classifies whether a loan application is safe or not. Based on the classification result, you will decide to lend or not. In this case, the risk_adjustment variable can be used to indicate the amount of money requested by a borrower, which represents an importance (or a risk) of that tuple. An application with a huge amount of money will have high risk_adjustment value. Therefore, classifying high risk applications will be much more important than just classifying a given application to safe or not.



### 4. Lazy Learners (Text: 9.5)

**Lazy vs. Eager Learning**

- Lazy vs. eager learning
  - **Lazy learning** (e.g., instance-based learning): Simply stores training data (or only minor processing) and **waits until it is given a test tuple**
  - **Eager learning** (the discussed methods so far): Given a set of training tuples, **constructs a classification model** before receiving new (e.g., test) data to classify
- Lazy: less time in training but more time in predicting
- Accuracy
  - Lazy method effectively uses a richer hypothesis space since it uses many local linear functions to form an implicit global approximation to the target function
  - Eager: must commit to a single hypothesis that covers the entire instance space

**Lazy Learner: Instance based method**

- Instance-based learning: 
  - Store training examples and delay the processing (“lazy evaluation”) until a new instance must be classified
- Typical approaches
  - k-nearest neighbor approach (KNN)
    - Instances represented as points in a Euclidean space.
  - Locally weighted regression
    - Constructs local approximation
  - Case-based reasoning
    - Uses symbolic representations and knowledge-based inference



#### 4.1. kNN-Classifier (Text: 9.5.1)

**K-Nearest Neighbourhood (k-NN)**

- kNN classifiers are based on learning by analogy

- A training tuple described by *n* attributes represents a point in the n-dimensional space

  - All training tuples are stored in an n-dimensional space

- Given a tuple of unknown class or unknown target value, KNN classifier searches k nearest neighbours of the unknown tuple.

  - The nearest neighbours are defined in terms of Euclidean distance or other metrics, [![dist(X_1, X_2)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270903622.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=dist(X_1%2C X_2))

    ![knn](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270903340.png)

- Two ways of classifying the unknown tuple in kNN

  - Discrete method (discrete-valued method)

    - - k-NN returns the most common value among the k training examples nearest to [![X_q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c5dd3dafba597ca70d29041140bb6146.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X_q)(test tuple)
      - Decision function:
        - [![D(X_q) = \sum_{i=1}^{k} y_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/04889ddd4c4463bc9e2920f4f40c6cfb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D(X_q) %3D \sum_{i%3D1}^{k} y_i)
        - where [![y_i \in \{+1, -1\}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/70084c365c474b4b731f84e96cd3a991.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y_i \in \{%2B1%2C -1\}) is the class of [![i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/865c0c0b4ab0e063e5caa3387c1a8741.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=i)th nearest neighbour
        - If [![D(X_q) > 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/32c91da6fa61df87a453add525a13b04.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D(X_q) > 0) then [![X_q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c5dd3dafba597ca70d29041140bb6146.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X_q) is positive class otherwise negative class

    - - Voronoi diagram: the decision surface induced by 1-NN for a typical set of training examples

        ![vonoroi](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270904169.png)
        Example of Voronoi diagram with Euclidean distance. The lines represent a decision surface induced by 1-NN. from Wikipedia

- - Continuous method (real-valued prediction):
    - Returns the mean values of the k nearest neighbours
    - Distance-weighted nearest neighbour algorithm
      - Weight the contribution of each of the k neighbours according to their distance to the query [![X_q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c5dd3dafba597ca70d29041140bb6146.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X_q)
      - Give greater weight to closer neighbours
        [![ w_i = \frac{1}{dist(X_q, X_i)^2} ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/66a04ce40613e242406e56b1fbe11352.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp= w_i %3D \frac{1}{dist(X_q%2C X_i)^2} )
    - Decision function:
      - [![D(X) = \sum_{i=1}^{k}\frac{w_i}{\sum_{j=1}^{k}w_j} \cdot y_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4d1dddd9e636f09cc4fec6f7d2473c4e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D(X) %3D \sum_{i%3D1}^{k}\frac{w_i}{\sum_{j%3D1}^{k}w_j} \cdot y_i)
      - where [![y_i \in \{+1, -1\}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/70084c365c474b4b731f84e96cd3a991.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y_i \in \{%2B1%2C -1\}) is the target value of [![i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/865c0c0b4ab0e063e5caa3387c1a8741.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=i)th nearest neighbour



**Characteristics of KNN**

- **Robust** to noisy data by averaging k-nearest neighbours

- **Extremely slow** when classifying test tuples

  - With [![|D|](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a2d2184975ac156e22c2994baecc5a3a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=|D|) training tuples, [![|D|](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a2d2184975ac156e22c2994baecc5a3a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=|D|) comparisons are required to find k-nearest neighbourhood

  - For example, SVM only requires [![\ell](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270903701.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\ell) comparisons where [![\ell](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ee5e5c003694e7cd5ae404923c665edb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\ell) is the number of support vectors

  - Partial distance method:

    - Compute a distance on a subset of n attributes. 

    - - If the distance exceeds a threshold, further distance computation will be halted
      - Otherwise keep computing the distance on the remaining attributes

#### 4.2. Exercise: kNN Classifier

## Example: KNN Classifier

We are manufacturers of paper products and we wish to understand whether our new types of paper tissue are good or not. 

We have data from both a survey of customer opinion and objective lab testing with two attributes: acid durability and strength.

Here are four training samples:



| X1 = Acid Durability (seconds) | X2 = Strength(kg/square meter) | Y = Classification |
| ------------------------------ | ------------------------------ | ------------------ |
| 7                              | 7                              | Bad                |
| 7                              | 4                              | Bad                |
| 3                              | 4                              | Good               |
| 1                              | 4                              | Good               |



Now the product developers produce a new paper tissue that passes the laboratory test with X1 = 3 and X2 = 7. 

What will be the assigned class (Good or Bad) if we use a discrete 3-NN classifier to classify the new paper tissue (k=3, distance metric: Euclidean distance)?



## Solution to Exercise: KNN-Classifier

\1. Calculate the distance between the query-instance and all the training samples.

\2. Sort the distance and determine nearest neighbours based on the K-th minimum distance 

\3. Gather the category of the nearest neighbours. Notice in the second row last column that the category of nearest neighbour is not included because the rank of this data is more than 3 (=K). 

| X1 = Acid Durability (seconds) | X2 = Strength(kg/square meter) | Squared Distance to query instance (3, 7) | Rank minimum distance | Is it included in 3-Nearest neighbors? | Y = Category of nearest Neighbor |
| ------------------------------ | ------------------------------ | ----------------------------------------- | --------------------- | -------------------------------------- | -------------------------------- |
| 7                              | 7                              | 16                                        | 3                     | Yes                                    | Bad                              |
| 7                              | 4                              | 25                                        | 4                     | No                                     | -                                |
| 3                              | 4                              | 9                                         | 1                     | Yes                                    | Good                             |
| 1                              | 4                              | 13                                        | 2                     | Yes                                    | Good                             |



Use simple majority of the category of nearest neighbours as the prediction value of the query instance. We have 2 good and 1 bad, since 2>1 then we conclude that a new paper tissue that passes laboratory test with X1 = 3 and X2 = 7 is included in **Good** category.





The example from http://people.revoledu.com/kardi/tutorial/KNN/KNN_Numerical-example.html



#### 4.3. Case-Based Reasoning (CBR) (Text: 9.5.2)

- CBR(Case-Based Reasoning): Uses a database of problem solutions to solve new problems
- Store symbolic description (tuples or cases)—not points in a Euclidean space
- Applications: Customer-service (product-related diagnosis), legal ruling
- Methodology
  - Instances represented by rich symbolic descriptions
  - If there is an identical training case, given a test case, the solution of the training case will be returned
  - If not, search for similar cases, multiple retrieved cases may be combined
  - Tight coupling between case retrieval, knowledge-based reasoning, and problem solving
- Challenges
  - Find a good similarity metric 
  - Indexing based on syntactic similarity measure, and when failure, backtracking, and adapting to additional cases



### 5. Variants of Classification (Text 9.7)

**Multiclass-Classification**

- Classification involving more than two classes (i.e., > 2 Classes) 
- Method 1. **One-vs-all** (OVA): Learn a classifier for *one class at a time* Given [![m](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6f8f57715090da2632453988d9a1501b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m) classes, train [![m](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6f8f57715090da2632453988d9a1501b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m) classifiers: one for each classClassifier [![j](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270906028.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j):treats tuples in class [![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j) as positive & all others as negativeTo classify a tuple [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X), the set of classifiers vote as an ensemble. If classifier [![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j) predicts the positive class, then class [![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j) gets one vote. If classifier [![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j) predicts the negative class then all non-[![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j) classes get one vote. 
- Method 2. **All-vs-all** (AVA): Learn a classifier for *each pair of classes*Given m classes, construct [![m(m-1)/2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/fa029bfe3b7f8cc3b8e2262eff51cf1f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m(m-1)%2F2) binary classifiersA classifier is trained using tuples of the two classes, ignoring other tuplesTo classify a tuple [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X), each classifier votes. [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) is assigned to the class with maximal vote
- ComparisonAll-vs.-all tends to be superior to one-vs.-allProblem: Binary classifier is sensitive to errors, and errors affect vote count



**Semi-supervised Classification**



- Semi-supervised: Uses labeled and unlabeled data to build a classifier
- **Self-training**: 
  - Build a classifier using the labeled data
  - Use it to label the unlabeled data, and those with the most confident label prediction are added to the set of labeled data
  - Repeat the above process
  - Advantage: easy to understand; disadvantage: may **reinforce errors**
- **Co-training**: Use two or more classifiers to teach each other
  - Use two disjoint and independent selections of attributes of each tuple to train two good classifiers, say [![f_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c354bdd39692a0ba3f80f7c733f4e0eb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f_1) and [![f_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7de62936dedfe1edafd9147c61f6f8ef.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f_2) 
  - Then [![f_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c354bdd39692a0ba3f80f7c733f4e0eb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f_1) and [![f_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7de62936dedfe1edafd9147c61f6f8ef.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f_2) are used to predict the class label for unlabeled data tuples [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X)
  - Teach each other: The tuples in [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) having the most confident prediction from [![f_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c354bdd39692a0ba3f80f7c733f4e0eb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f_1) are added to the set of labeled training data for [![f_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7de62936dedfe1edafd9147c61f6f8ef.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f_2), & vice versa
  - Retrain two classifiers using the extended training sets, using the same disjoint attribute selections



**Active-Learning**

- Class labels are expensive to obtain

- Active learner: **query human (oracle) for labels**

- Pool-based approach: Uses a pool of unlabeled data

  - [![L](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270906550.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=L): a small subset of [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) is labeled, [![U](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4c614360da93c0a041b22e537de151eb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=U): a pool of unlabeled data in [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D)
  - Use a query function to carefully select one or more tuples from [![U](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4c614360da93c0a041b22e537de151eb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=U) and request labels from an oracle (a human annotator)
  - The newly labeled samples are added to [![L](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270906550.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=L), and learn a model
  - Goal: Achieve high accuracy using as few labeled data as possible

- Evaluated using learning curves: Accuracy as a function of the number of instances queried (# of tuples to be queried should be small)

- Research issue: How to choose the data tuples to be queried?

- - Uncertainty sampling: choose the least certain ones
  - Reduce version space, the subset of hypotheses consistent with the training data
  - Reduce expected entropy over [![U](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4c614360da93c0a041b22e537de151eb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=U): Find the greatest reduction in the total number of unlabelled data



**Transfer-Learning**

- Transfer learning: Build classifiers for one or more similar source tasks and apply to a target task
- vs Traditional learning: Build a new classifier for each new task



**Want to know more? 
**

**[COMP8420](http://programsandcourses.anu.edu.au/2018/course/COMP8420)/[COMP4660](http://programsandcourses.anu.edu.au/course/COMP4660)** **Neural Networks, Deep Learning and Bio-inspired Computing** covers, in much more depth, neural, deep learning, fuzzy, evolutionary and hybrid methods.

**[COMP8600](http://programsandcourses.anu.edu.au/2018/course/COMP8600)/[COMP4670](http://programsandcourses.anu.edu.au/course/COMP4670) Statistical Machine Learning** covers, in much more mathematical depth, Bayesian learning, regression, neural networks, and support vector machines.