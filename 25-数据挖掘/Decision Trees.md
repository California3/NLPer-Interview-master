## Introduction and Decision Trees - 6.1

### 1. Introduction

Most of this material is derived from the text, Han, Kamber and Pei, Chapter 8, or the corresponding powerpoint slides made available by the publisher. Where a source other than the text or its slides was used for the material, attribution is given. Unless otherwise stated, images are copyright of the publisher, Elsevier.

In this module of work we aim to introduce the data mining problems of *classification and prediction*, and to understand the basic classification technique of *decision trees*, so that you can recognise a problem to which they may apply; can apply them to the problem; and can evaluate the quality of the results. 

### 2. Classification (Text: 8.1)

Classification builds models that describe interesting **classes** of data. The models are called **classifiers** because, once built, the model may be used to classify **unseen** data. Sometimes the model itself is more important than its use in ongoing classification because it provides a **compact summary** of the data, that is explanatory for humans.

Most commonly classification is **binary,** that is, objects are determined to belong to a class or not. For example, taxpayers are classified as fraudulent, or not. However, the generalisation to classfying data into more than two classes is important.

Classification is often classified as a machine-learning problem due to its origins in AI research, although data mining research has developed the scalability to handle large disk-stored data sets.

Nowadays it is widely used in application to problems in science, marketing, fraud detection, performance prediction, medical diagnosis, and fault diagnosis. 

**Classification**

- Used to predict **categorical class labels** (discrete or nominal) from unlabelled data.
- Constructs (or **learns**) a **classifier** (or **model**) from **training** data that includes, for each **example** in the data, **data values** as well as a pre-determined **class label.**
- Uses the model to **predict** the class label for new,unseen, unlabelled data.

**Classification vs Prediction**

Although classifiers predict the values of unknown class labels, classification is usually distinguished from the problem of **numerical prediction** (commonly called simply *prediction*) that **builds models of continuous-valued functions** and so predicts unknown or missing **numeric** values. We will also study some popular prediction techniques. 

**Supervised Learning vs Unsupervised learning**

Again, we see the AI influence in the language here, where **supervised** learning refers to classification as we have defined it -- where the training data (observations, measurements, etc.) is accompanied by labels indicating the class of the observations, and new data is classified based on the training set. In this AI-oriented view of classification we often talk about **batch** vs **incremental** learning. The former is usually an unstated assumption for data mining. In the latter case the labelled data becomes available to the learning algorithm in a sequence and a working classifier developed initially from a small amount of data must be continually updated to account for new data. 

On the other hand, for **unsupervised** learning there are no class labels in the training data and the learning algorthm must find some interesting classes, or classifications with which to classify new data. This is commonly called **clustering.** We will study some popular clustering techniques later. 

So classification can also be defined as **supervised learning of categorical variables.**



#### 2.1. Two steps: Construct and Evaluate

**Step 1: Training phase or learning step: Build a model from the labelled training set.**

Each **tuple/sample/record/object/example/instance/feature vector** of the training dataset is assumed to belong to a predefined class, as determined by the class label attribute. Ideally, the tuples are a random sample from the full population of data.

- The set of tuples used for model construction is the *training set*:

[![T = \{X: X = (x_1,x_2,..,x_n)\}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e999046f2b9170f8b6fb0b2073552e04.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=T %3D \{X%3A X %3D (x_1%2Cx_2%2C..%2Cx_n)\}) *and each* [![x_i](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260551878.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x_i) *is an attribute value and* [![X\in C_j ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/03df1bb85db8209196c537942bfeea05.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X\in C_j ) *for some*[![j=1,..,k, k\geq 2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/254de7c2a082eb8fe843d637509bef72.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j%3D1%2C..%2Ck%2C k\geq 2) *and* [![ j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/231fec80038de71293915cb76e01f49e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp= j) *is the class label for* [![ X\} ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8a54c909bf3304eb65bd972703e47e11.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp= X\} ).

- Commonly, each [![X \in T](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260551492.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X \in T) is assumed to belong to exactly one class [![C_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1ae38954f6cba2eafda4e9c34df8d944.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_j)
- In the very common special case of exactly 2 classes, i.e. binary learning, the training classes are called the **positive** **examples** [![C_+](https://wattlecourses.anu.edu.au/filter/tex/pix.php/401c451d881f4a504c2ee2404685bf72.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_%2B) or *P* and **negative examples** [![C_- ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/60e50badc40a957ac92049f0b0a1f74f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_- ) or *N*.
- The model is represented as classification rules, decision trees, mathematical formulae, or a "black box". The model can be viewed as a function [![f(X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/41c2f2136110516f7d332adc5041b0fe.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f(X)) that can predict the class label for some unlabelled tuple [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X).
- For classification models, the built model may be called a **classifier**. 

**Step 2: Use the model to classify unseen objects**

- Need to estimate the **accuracy** of the model
  - The known labels of a set of independent **test samples** is compared with the classified results for those same samples from the model
  - **Accuracy** is the proportion of test set samples that are correctly classified by the model

- If the accuracy and all other evaluation measures are acceptable, apply the model to classify data objects whose class labels are not known in the world.

Example:

The data classification process:

(a) Learning: Training data is analysed by a classification algorithm. Here, the class label attribute is **loan_decision**, and the learned model or classifier is represented in the form of classification rules.

(b) Classification: Test data are used to estimate the accuracy of the classification rules. If the accuracy is considered acceptable, the rules can be applied to the classification of new, unlabelled, data tuples.

<img src="https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260554535.png" alt="782324872_1794046507." style="zoom: 150%;" />

#### 2.2. Evaluation

*Learning algorithms* (or l*earners*) that build models built for classification and prediction are generally evaluated in the following ways. These ways are applied to the case of *inventing* new algorithms and wanting to assess them against others, or for *selecting* an algorithm for its suitability for a particular learning problem. Once an algorithm(s) has been selected and a model(s) built, these overarching principles may be revisited to choose which, if any, to put into practice.

- **Accuracy** often on *benchmark* data sets so they can be compared with other learning algorithms
  - Classifier accuracy: Predicting class label
  - Predictor accuracy: Guessing value of predicted attributes

- **Speed and complexity**
  - Time to construct the model (training time)
  - Time to use the model (classification/prediction time)
  - Worst case or average case theoretical complexity

- **Scalability**
  - Efficiency in handling disk-based databases
  - Potential for speed up by parallel computation

- **Robustness**
  - Handling noise and outliers

- **Interpretability**
  - Understanding and insight provided by the model

- **Other measures**
  - goodness of rules
  - decision tree size
  - compactness or simplicity of model



### 3. Decision Tree Induction (Text: 8.2)

A decision tree classifies labelled data by applying a sequence of logical tests on attributes that partition the data into finer and finer sets. The model that is learnt is a tree of logical tests.

The decision tree is a **flowchart-like structure**, where

- each **internal node** as well as the topmost **root node** represents a test on an attribute; commonly the tests can have only two outcomes, in which case the tree is **binary**
- each **branch** directed out and down from an internal node represents an outcome of the test
- each **leaf node** (or terminal node) represents represents <u>a decision and holds a class label</u>
- a **path** from the root to a leaf traces out the classification for a tuple

​	<!-- leaf -> decision, node -> test -->

Decision tree induction is **very popular for classification** because:

- relatively fast learning speed 

- convertible to simple and easy to understand classification rules

- can work with SQL queries to access databases while tree-building

- comparable classification accuracy with other methods

  

Exercise

Here is some data for a binary classification problem, with label *buys_computer.*

**ACTION:** **Consider what sequence of decisions you would propose to identify "who buys a computer"? Is your tree binary, or not?
**![78807923_247356216.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260612333.png)



And here is a decision tree model to classify the data.

**ACTION: Consider, how does it differ from yours? Can you always design a tree that correctly classifies every object in the training dataset? 
**![78807923_1843931984.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260613337.png)

### 4. Basic, greedy, decision tree algorithm

A typical basic algorithm follows. It is **greedy** (it makes decisions optimising the next step context and never backtracks to reconsider).

It is a **recursive**, top-down divide-and-conquer approach to build a tree. 

Attributes may be **nominal, ordinal,** or **continuous.**

- At the **start,** all the training examples are at the root
- At a node, **test attributes are selected** on the basis of a heuristic or statistical measure (e.g., information gain)
- Examples at the node are **partitioned** to sub-nodes based on selected attributes
- **Recurse** over subnodes 
- Paritioning **stops** when
  - All samples for a given node belong to the same class; or
  - There are no remaining attributes for further partitioning – majority voting is employed for classifying the leaf; or
  - There are no samples left

The slightly more generic algorithm sketch below permits n-ary (*multiway*) trees and can discretise continuous attributes dynamically, according to local context in the tree.

<img src="https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260632804.png" alt="712367564_1574077900." style="zoom:150%;" />



**ACTION: Go back to the previous page and build a tree stepping through the algorithm as shown here.** 

**Is your tree any different this time? What attribute_selection_method did you use?**

### 5. Attribute Selection Methods (Text 8.2.2)

Attribute selection methods are also called **splitting rules.**

- Techniques to choose a **splitting criterion** comprised of a **splitting attribute** and a **split point** or **splitting subset**
- Aim to have partitions at each branch as **pure** as possible -- i.e. all examples at each sub- node belong in the same class.

Example

This figure shows three possibilities for partitioning tuples based on the splitting criterion, each with examples. Let A be the **splitting attribute**. (a) If A is **nominal-valued**, then one branch is grown for **each known value** of A. (b) If A is **continuous-valued or ordinal**, then two branches are grown, corresponding to A [![\leq](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260701051.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\leq)**split_point** and A > split_point. (c) If A is **nominal and a binary** tree must be produced, then the test is of the form A [![\in](https://wattlecourses.anu.edu.au/filter/tex/pix.php/986c22f151c46acac223b858e3fcf6fd.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\in)SA, where SA is the **splitting subset** for A.

![771758716_1940992285.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260702797.png)



**Heuristics,** (or **attribute selection measures**) are used to choose the best splitting criterion. <!--启发式-->

- Information Gain, Gain ratio and Gini index are most popular.

- Information gain:

​			biased towards multivalued attributes

- Gain ratio:

​			tends to prefer unbalanced splits in which one partition is much smaller than the others

- Gini index:

​			biased towards multivalued attributes

​			has difficulty when number of classes is large

​			tends to favour tests that result in equal-sized partitions and purity in both partitions



#### 5.1. Information Gain

**Information Gain**

- This was a very early method that sprang from AI research in ID3, and was refined further to become *Gain Ratio* in C4.5.
- It selects the attribute to split on with the highest information gain

- Let [![p_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/eca91c83a74a2373ca5f796700e99fd3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_i) be the probability that an arbitrary tuple in [![D](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260737633.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) belongs to class [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i), of [![m](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6f8f57715090da2632453988d9a1501b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m) classes, where [![C_{i,D}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/b6d6ed29e92f7232e4638baad87229f2.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_{i%2CD}) is the set of tuples in [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D)labelled with class [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i)estimated by [![p_i=\frac{|C_{i, D}|}{|D|}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/490b205cba4c3be1bd7c8d481d9f171c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_i%3D\frac{|C_{i%2C D}|}{|D|})
- Expected information (**entropy**) needed to classify a tuple in [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D)is defined by 

[![\it{Info}(D) = - \sum_{i=1}^{m} p_i\it{log}_2(p_i)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/49d66392800726196289141703b70040.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\it{Info}(D) %3D - \sum_{i%3D1}^{m} p_i\it{log}_2(p_i))

- After using attribute [![A](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260737638.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A) to split [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) into [![v](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9e3669d19b675bd57058fd4664205d2a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=v) partitions, corresponding to each attribute value for [![A](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7fc56270e7a70fa81a5935b72eacbe29.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A) , each one of these partitions being [![D_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/5b58cc0cefa1115cdeb54f391b25591d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D_j), the information that is still needed to separate the classes is:

[![\it{Info}_A(D) = \sum_{j=1}^{v}\frac{|D_j|}{|D|}\times\it{Info}(D_j)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260737007.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\it{Info}_A(D) %3D \sum_{j%3D1}^{v}\frac{|D_j|}{|D|}\times\it{Info}(D_j))

- Therefore, information gained by branching on attribute [![A](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7fc56270e7a70fa81a5935b72eacbe29.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A) is given by 

  [![\it{Gain}(A) = \it{Info}(D) - \it{Info_A}(D)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/59eae82ef9285cc3028b72044b862c0b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\it{Gain}(A) %3D \it{Info}(D) - \it{Info_A}(D))

Example (continued from previous)

![78807923_247356216.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260739121.png)

Consider 2 classes: Class *P* is buys_computer = “yes”. Class *N* is buys_computer = “no”

For some partition on [![D](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260740420.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) with [![p_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/eca91c83a74a2373ca5f796700e99fd3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_i) examples of *P* and [![n_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/584a81dbf5bf6aa737ba43567ad6307b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=n_i) examples of *N*, let [![\it{Info}(D)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/15237321e8988467a40fde1129529fa0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\it{Info}(D)) be written as [![I(p_i, n_i)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/fbe9a420c103f6824ea3ca4091f57959.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=I(p_i%2C n_i)). 

Using the definition [![\it{Info}(D) = - \sum_{i=1}^{m} p_i\it{log}_2(p_i) ](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260740680.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\it{Info}(D) %3D - \sum_{i%3D1}^{m} p_i\it{log}_2(p_i) ) from above,

we have ![621264696_1957654102.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260740213.png)

Now consider the first partition on *age.* We have the following

![621264696_1474042360.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260741716.png)

![image-20220426上午74227527](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260742577.png)

![303872180_429071810.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260743462.png)



#### 5.2. Information Gain for continuous-valued attributes

Let attribute *A* be a continuous-valued attribute

To determine the *best split point* for A

- Sort the values of *A* in increasing order

- Typically, the **midpoint between each pair of adjacent values** is considered as a possible split point  <!--每个相邻点的中间值皆有可能。-->
  - *(ai+ai+1)/2* is the midpoint between the values of *a*i and *ai+1*

- Of these, the point with the minimum expected information requirement for *A,* [![\it{Info}_A(D)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260850362.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\it{Info}_A(D)) is selected as the split-point for *A* <!--选取 min (Info（A in D）)-->

Then Split:

*D1* is the set of tuples in *D* satisfying *A ≤ split-point*, and *D2* is the set of tuples in *D* satisfying *A > split-point*

This method can also be used for ordinal attributes with many values (where treating them simply as nominals may cause too much branching).



#### 5.3. Gain Ratio

- Used in C4.5 (a successor of ID3) to overcome bias towards attributes with many values 
- Normalises information gain

![456413725_1930007997.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260906735.png)

splitInfoA represents the *potential i*nformation generated by splitting *D* into *v p*artitions, corresponding to the *v* outcomes of a test on *A*. 

Now we define

​	*GainRatio(A) = Gain(A)/SplitInfo(A)* 

Example (continued from previous):

![456413725_956118889.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260906248.png)

​	*gain_ratio(income) = 0.029/1.557 = 0.019*

The attribute with the **maximum** gain ratio is **selected** as the splitting attribute.



#### 5.4. Gini index

**Gini index** is used in CART and IBM Intelligent miner decision tree learners

- All attributes are assumed **nominal**

If a data set [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) contains examples from [![n](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7b8b965ad4bca0e41ab51de7b31363a1.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=n) classes, gini index, [![gini(D)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260920378.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=gini(D)), measures the **impurity** of [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) and is defined as <!--杂质-->

[![ gini(D) = 1 - \sum\limits_{j=1}^{n} p_j^{\ 2}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/cb891e8edbb3dc3ea5eb0a8138f6b7a5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp= gini(D) %3D 1 - \sum\limits_{j%3D1}^{n} p_j^{\ 2}) where [![p_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8b6f59f2af8f45b773cb64ac76c9b095.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_j) is the relative frequency of class[![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j)in [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) 
` `

If a data set [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) is split on attribute [![A](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260920923.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A) into two subsets [![D_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/323b515dec6e9a6563cad1790f7590bc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D_1) and [![D_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a701eb15aaebdd365911d0df1da9c8f7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D_2), the gini index [![gini_A(D)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a1d42cc53d55b73507a8df8674ef1977.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=gini_A(D)) is defined as the size-weighted sum of the impurity of each partition:

[![gini_A(D) = \frac{D_1}{D} gini(D_1) + \frac{D_2}{D} gini(D_2)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e018b567f90b028075358864131058b4.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=gini_A(D) %3D \frac{D_1}{D} gini(D_1) %2B \frac{D_2}{D} gini(D_2)) 

 

**To split a node in the tree:**

- Enumerate all the possible ways of splitting all the possible attributes
- The attribute *split* that provides the **<u>smallest</u>** *ginisplit(D)* (i.e the greatest purity) is chosen to split the node

Example (continued from previous)

*D* has 9 tuples in class buys_computer = “yes” and 5 in “no”

Then [![gini(D) = 1 - \left(\frac{9}{14} \right)^2 -\left(\frac{5}{14} \right)^2 = 0.459](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ad1f01bf2c9958ab13c3d9dbe6739390.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=gini(D) %3D 1 - \left(\frac{9}{14} \right)^2 -\left(\frac{5}{14} \right)^2 %3D 0.459) 

Now consider the attribute *income.* Partition *D* into 10 objects in *D1*with income in *{low, medium}* and 4 objects in *D2* with income in*{high}* 

 

We have

[![gini_{income \in \{low, medium\}}(D) ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f1baeda923d566fde7addc86091568d3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=gini_{income \in \{low%2C medium\}}(D) )

[![ = \left( \frac{10}{14} \right) gini(D_1) + \left(\frac{4}{14} \right) gini(D_2) ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4b1f7048a752481ea438f289cb30b070.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp= %3D \left( \frac{10}{14} \right) gini(D_1) %2B \left(\frac{4}{14} \right) gini(D_2) ) 

[![ = \frac{10}{14}\left(1 - \left({\frac{7}{10}}\right)^2 - \left({\frac{3}{10}}\right)^2\right) + \frac{4}{14}\left(1 - \left({\frac{2}{4}}\right)^2 - \left({\frac{2}{4}}\right)^2\right)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204260920458.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp= %3D \frac{10}{14}\left(1 - \left({\frac{7}{10}}\right)^2 - \left({\frac{3}{10}}\right)^2\right) 

[![ = 0.443 ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/56d593b55b3cea3e7b7bccb9914335ef.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp= %3D 0.443 )

[![ = gini_{income \in \{high\}}(D) ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3b01e69f7f78469ad3c92bd72753c5a4.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp= %3D gini_{income \in \{high\}}(D) ) 



Similarly, *giniincome{low,high}* is 0.458; and giniincome{medium,high} is 0.450. 

Thus, we split on the {low,medium} (and the other partition is {high}) since it has the lowest Gini index

When attributes are **continuous or ordinal**, the method for selecting the **midpoint between each pair of adjacent values** ([described earlier](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472055&chapterid=439335)) may be used. 



#### 5.5. Other Attribute Selection Methods

Of course, researchers have experimented with many other attribute selection methods that you might come across. Here are some of the most well-known that you might like to look into further.

- **CHAID**: a popular decision tree algorithm, measure based on χ2 test for independence
- **C-SEP**: performs better than info. gain and gini index in certain cases
- **G-statistic**: has a close approximation to χ2 distribution
- **MDL (Minimal Description Length) principle** (i.e., the simplest solution is preferred):
  - The best tree as the one that requires the fewest number of bits to both (1) encode the tree, and (2) encode the exceptions to the tree
- Multivariate splits (partition based on multiple variable combinations), e.g. **CART**: finds multivariate splits based on a linear combination of attributes.



#### 5.6 Coding of methods

```python
# Information Gain Calculators
# import numpy as np
# import math

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

# gini calculator 
      # p, n
s =  [[7,3],
      [2,2]] 

print("Gini between s[0] and s[1]:", round(cal_gini_a_of_D(s), 3))
```

```shell
Expected Information (D):  0.94
Expected Information (A in D):  0.911
Gain (A): 0.029
SplitInfo (A): 1.557
Gain_ratio (A): 0.019
Gini between s[0] and s[1]: 0.443
```



### 6. Overfitting and tree pruning (Text 8.2.3)

If you have consistently labelled data, and you allow attributes to be re-used when growing the tree, and you stop growing when every training tuple is classified, it is always possible to have a tree that describes the training set with 100% accuracy. Such a tree typically has very poor ***generalisation*** and is said to ***overfit*** the training data. 

**Overfitting**: An induced tree may **overfit** the training data

- Too many branches, some may reflect anomalies due to noise or outliers
- <u>Poor accuracy for unseen samples</u> is observed because anomalies are modelled and objects *like* the training data are not modelled. 100% accuracy on the training data can be a *bad thing* for accuracy on unseen data.

There are two typical approaches to avoid overfitting:

- **Prepruning**: Stop tree construction early ̵ do not split a node if this would result in the goodness measure falling below a threshold. But it is difficult to choose an appropriate <u>threshold</u>. 

- **Postpruning**: <u>Remove branches from a “fully grown” tree.</u> This produces a sequence of progressively pruned trees. Then <u>use a set of data different from the training data to decide which is the “best pruned tree”</u>

- - 

  - **ACTION:** Overfitting is an important but rather qualitative, loosely-defined concept. If you need more explanation of overfitting, refer to https://en.wikipedia.org/wiki/Overfitting



### 7. Enhancements to the Basic Algorithm (not in text)

These enhancements may be embedded within the decision tree induction algorithm

- Handle missing attribute values
  - Assign the most common value of the attribute
  - Assign probability to each of the possible values
- Attribute construction
  - Create new attributes based on existing ones that are sparsely represented
  - This reduces fragmentation, repetition, and replication
- Continuous target variable
  - In this case the tree is called **a *regression tree***, the leaf node classes are represented by their **<u>mean</u>** values, and the tree <u>performs prediction</u> (using that mean value) rather than classification. 
- Probabilistic classifier
  - Instead of majority voting to assign a class label to a leaf node, <u>the *proportion* of training data objects of each class in the leaf node can be interpreted to as the *probability* of the class,</u> and this probability can be assigned to the classification for unknown objects falling in that node at use-time.

**ACTION: There is a nice non-technical overview of decision trees, that you might like to read if you are needing more:**

https://algobeans.com/2016/07/27/decision-trees-tutorial/



### 8. Extracting Rules from Decision Trees (Text 8.4.2)

Decision trees can become large and difficult to interpret. We look at how to build a rule-based classifier by extracting IF-THEN rules from a decision tree. In comparison with a decision tree, the IF-THEN rules may be easier for humans to understand, particularly if the decision tree is very large.

- Rules are easier to understand than large trees
- One rule is created for each path from the root to a leaf
- Each attribute-value pair along a path forms a conjunction: the leaf holds the class prediction

Example:

 ![585476863_2128868346.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261146969.png)

From the decision tree above, we can extract IF-THEN rules by tracing them from the root node to each leaf node:

![979650026_535670513.](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261147688.png)

**Properties of extracted rules**

- Mutually exclusive

  - We cannot have rule conflicts here because no two rules will be triggered for the same tuple.
  - We have one rule per leaf, and any tuple can map to only one leaf.

- Exhaustive:

- - There is one rule for each possible attribute–value combination, so that this set of rules does not require a default rule.
  - The order if rules is irrelevant.

\* Rattle can generate rules from a trained decision tree.



### 9. Lab

<div class="ratio ratio-16x9" style="margin-bottom:15px;">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/8zc4AiKi4ZQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### 10. Ending