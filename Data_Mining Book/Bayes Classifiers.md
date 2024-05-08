## Classification & Prediction: Bayes Classifiers - 6.3

### 1. Introduction

Most of this material is derived from the text, Han, Kamber and Pei, Chapter 8 and 9, or the corresponding powerpoint slides made available by the publisher. Where a source other than the text or its slides was used for the material, attribution is given. Unless otherwise stated, images are copyright of the publisher, Elsevier.

Here, we will discuss the probabilistic classifiers derived from Bayes' theorem, including Bayes classifier, naive Bayes classifier and Bayesian belief networks. 



### 2. Probabilistic Classifier (Text:8.3.1)

 **What is Bayesian classifier?**

- A statistical (probabilistic) classifier: Predicts the probability of a given tuple belonging to a particular class
- Foundation: Based on Bayes' theorem. Bayes was a mid-18th century monk (apparently).
- Performance: Comparable accuracy performance to decision tree and neural network classifiers
- Computational performance is much enhanced by assuming *class-conditional independence*, in which case the method is called *Naive Bayes.*
- Incremental: Each training example can incrementally contribute to the classification probabilities, so this allows adapting over time to gradual or incremental changes in (labelled) training data.
- It is not really possible to humanly interpret the results (i.e. it is known as a "black box" method), although it's relationship to its training data is straightforward to understand. 



#### 2.1. Basic Probabilities (not in text)

**Basic Probability Theory**

Before discussing probabilistic classifiers, we recap basic probability theory first.

- **Event** [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324483.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X): A subset of outcomes of an experiment (a subset of event space).
  - Let's assume that we roll a dice with six faces. If we observe number 3 from a single roll, then 3 is the event, [![X=3](https://wattlecourses.anu.edu.au/filter/tex/pix.php/972e61c3ab8b8c3e006f7f2332f12c8e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X%3D3)
  - A set of observations can also be an event, signifying any of the observations in the set. For example, an event from a dice roll [![A=\{1,3,5\}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/165db378e3efa27c89e8550c2541fabf.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A%3D\{1%2C3%2C5\}) can signify the outcome that either 1, 3, or 5 is rolled.
- **Event space (sample space)**: the set of all possible outcomes
  - e.g. {1,2,3,4,5,6} with a six-faced dice
- **Probability** of event [![P(X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0c3d72395d7576ab13b9e9389f865960.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X)): probability of observing an event
  - e.g. probability of observing 5 from a single dice roll, [![P(X=5) = 1/6](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324645.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%3D5) %3D 1%2F6)
- **Joint probability** [![P(X, Y)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/eb58c4cfd17b18317dbf1ff80dd5945c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%2C Y)): probability of observing multiple distinguishable events.
  - e.g. roll a dice and flip a coin, simultaneously. What would be the probability of observing 3 from the dice and HEAD from the coin [![P(X=3, Y=HEAD)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324966.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%3D3%2C Y%3DHEAD))? 

Example 

For an experiment, we roll a dice and flip a coin simultaneously, and record the first six trials as follows:

| Trial # | Dice | Coin |
| :------ | ---- | ---- |
| 1       | 1    | H    |
| 2       | 2    | T    |
| 3       | 1    | T    |
| 4       | 3    | H    |
| 5       | 4    | H    |
| 6       | 1    | T    |



Q: Given the above experiments, what is the probability of observing 3 from the dice? 

A: [![P(Dice=3)=1/6](https://wattlecourses.anu.edu.au/filter/tex/pix.php/de09b264deff36587f7744ae30e67894.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Dice%3D3)%3D1%2F6)

Q: Given above experiments, what is the probability of observing Dice={1,2} from the dice? 

A: [![P(Dice=1 or 2)=4/6=2/3](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8869a1d0f6bef61d85d6034f7cd6c202.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Dice%3D1 or 2)%3D4%2F6%3D2%2F3)

Q: Given above experiments, what is the probability of observing 1 and TAIL from a single execution?

A: [![P(Dice=1, Coin=TAIL) = 2/6](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324133.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Dice%3D1%2C Coin%3DTAIL) %3D 2%2F6)



**Conditional probability**

A conditional probability measures the probability of event [![X](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324483.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) given that another event [![Y](https://wattlecourses.anu.edu.au/filter/tex/pix.php/57cec4137b614c87cb4e24a3d003a3e0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Y) has occurred. If [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) and [![Y](https://wattlecourses.anu.edu.au/filter/tex/pix.php/57cec4137b614c87cb4e24a3d003a3e0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Y) are events with [![P(Y) > 0](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324340.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Y) > 0), the conditional probability of [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) given [![Y](https://wattlecourses.anu.edu.au/filter/tex/pix.php/57cec4137b614c87cb4e24a3d003a3e0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Y) is [![P(X|Y) = \frac{P(X, Y)}{P(Y)}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324524.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X|Y) %3D \frac{P(X%2C Y)}{P(Y)}). 

**Example**: Drug test

Let's assume that we have 4000 patients who have taken a drug test. The following table summarises the result of the drug test. We categorise the result based on gender and test result.

|         | Women | Men  |
| ------- | ----- | ---- |
| Success | 200   | 1800 |
| Failure | 1800  | 200  |



Let

[![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) represent gender

[![Y](https://wattlecourses.anu.edu.au/filter/tex/pix.php/57cec4137b614c87cb4e24a3d003a3e0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Y) represent a result of a drug test

Then what is the probability of a patient being a woman when the patient fails on a drug test, i.e., [![P(X=woman|Y=fail)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324711.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%3Dwoman|Y%3Dfail))?

[![P(X=woman) = \frac{2000}{4000} = \frac{1}{2}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261324898.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%3Dwoman) %3D \frac{2000}{4000} %3D \frac{1}{2})

[![P(Y=fail) = \frac{2000}{4000} = \frac{1}{2}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ca56ce76e78f36337e40b1f1eec0ef9c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Y%3Dfail) %3D \frac{2000}{4000} %3D \frac{1}{2})

[![P(X=woman, Y=fail) = \frac{1800}{4000} = \frac{9}{20}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d827926604fe7d61e01537764f43de16.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%3Dwoman%2C Y%3Dfail) %3D \frac{1800}{4000} %3D \frac{9}{20})

From these probabilities, we can compute the conditional probability

[![P(X=woman|Y=fail) = \frac{P(X=woman, Y=fail)}{P(Y=fail)} = \frac{9/20}{1/2} = \frac{18}{20} = 0.9](https://wattlecourses.anu.edu.au/filter/tex/pix.php/5e4a42bd2beb52c3e61d5eecd3aa9aa4.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%3Dwoman|Y%3Dfail) %3D \frac{P(X%3Dwoman%2C Y%3Dfail)}{P(Y%3Dfail)} %3D \frac{9%2F20}{1%2F2} %3D \frac{18}{20} %3D 0.9)



#### 2.2. Bayes' Theorem (Text 8.3.1)

**Terminology**

**A running example**: Let's assume that you are a owner of a computer shop. You may want to identify which customers buy a computer for targeting your advertising. So you decide to record a customer's *age* and *credit rating* whether the customer buys a computer or not for future predictions.

- **Evidence** [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X): A Bayesian term for observed data tuple, described by measurements made on a set of [![n](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332049.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=n) attributes.

  - E.g., record of customer's information such as *age* and *credit rating*.
  - [![X = (x_1, x_2, ..., x_n)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/158cda808fe8b98b447dee199940d2fd.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X %3D (x_1%2C x_2%2C ...%2C x_n))
  - Sometimes the probability [![P(X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0c3d72395d7576ab13b9e9389f865960.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X)) is also called *evidence*.

- **Hypothesis** [![H](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c1d9f50f86825a1a2302ec2449c17196.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=H): A target of the classification. Hypothesis such that [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) belongs to a specified class [![C](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0d61f8370cad1d412f80b84d143e1257.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C).

  - E.g., [![C_1](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332580.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_1) = buy computer, [![C_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f0350e5818b058dbcfd95f155e417f6a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_2) = not buy computer

- **Prior** probability, [![P(H)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6ba6e0e2b139069e480184bb3a47a0e9.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(H)): the *a priori* *probability* of [![H](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c1d9f50f86825a1a2302ec2449c17196.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=H) 

  - E.g., [![P(C_1)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6b0470e6f7f368461731183b7e7e24b8.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_1)) = the probability that any given customer will buy a computer regardless of *age,* or *credit rating*.

- **Likelihood**, [![P(X|H)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ffdc780e1e96d2c450c15963bf53a2cc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X|H)): the probability of observing the sample [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) given that the hypothesis holds.

  - E.g., Given that a customer, [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X), will buy a computer, the probability that the customer is *35 years old* and has *fair credit rating*.

  

- **Posterior** probability, [![P(H|X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4708d8e5720b31c1089c2da2216bc265.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(H|X)) : the *a posteriori probability,* that is the probability that the hypothesis holds given the observed data [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X). 

  - E.g., Given that a customer, [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) is *35 years old* and has *fair credit rating,* the probability that [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) will buy a computer. 





The prediction of a class for some new tuple [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) for which the class is unknown, is determined by the class which has the **highest** **posterior probability**. 





**Bayes' Theorem**

- In many cases, it is easy to estimate the posterior probabilty through estimating the prior and likelihood of given problem from historical data (i.e a *training* set).

  - E.g., to estimate the prior [![P(C_1)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6b0470e6f7f368461731183b7e7e24b8.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_1)), we can count the number of customers who bought a computer and divide it by the total number of customers.
  - E.g., to estimate the likelihood [![P(X=(35, fair)|C_1)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0f50fdc6ff677723b7c70e37265fa3c8.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%3D(35%2C fair)|C_1)), we can measure the proportion of customers whose age is *35* and have *fair* credit rating among the customers who bought a computer.
  - E.g., to estimate the evidence [![P(X=(35, fair)) ](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332076.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X%3D(35%2C fair)) ) we can measure the proportion of customers whose age is *35* and have *fair* credit rating amongst *all* the customers, irrespective of computer-buying. 
  - The posterior probability can then be computed from the prior and likelihood through Bayes' theorem.

  

- Bayes' theorem provides a way to relate likelihood, prior, and posterior probabilities in the following way, when [![P(X) > 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/241e300952ec4279d266707977082a4b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X) > 0)
  ![bayes_theorem](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261342220.png)

- Informally, this equation can be interpreted as

Posterior = likelihood x prior / evidence

- Bayes' theorem is used to predict [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) belongs to [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i) iff the posterior [![P(C_i|X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3cd2306f3c7783a619159b3a15d910ff.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_i|X)) is the highest among all other [![P(C_k|X)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332552.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_k|X)) for all the *k* classes. We can also state the *probability* that [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) belongs to [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i) is [![P(C_i|X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3cd2306f3c7783a619159b3a15d910ff.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_i|X)). Because we can give this probability, we call Bayes classification a *probablistic classifier.* 

- For determining the classification of some [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X), we are looking to find the [![C_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/72abb72bf267ec1843c375f2d32f4e09.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_k) that maximises [![P(C_k|X)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332552.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_k|X)) yet [![P(X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0c3d72395d7576ab13b9e9389f865960.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X)) is the same for every [![C_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/72abb72bf267ec1843c375f2d32f4e09.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_k), so [![P(X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0c3d72395d7576ab13b9e9389f865960.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X)) can be ignored in all the calculations as long as we don't need to know the probability.

**ACTION**: Bayes' Theorem can be derived straightforwardly from [conditional probability](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472057&chapterid=439365). The derivation is given [here](https://brilliant.org/wiki/bayes-theorem/) if you want to know. 





Example: with training data

Let's assume that you are a owner of a computer shop. You may want to identify which customers buy a computer for a targeted advertisement. So the owner decided to record a customers's age and credit rating no matter the customer buys a computer or not. The following table shows a set of customer records in the computer shop. What is the probability of a customer who is *youth* and has *fair credit* rating buying a computer?

| **age**     | **credit** | **buys_computer** |
| ----------- | ---------- | ----------------- |
| youth       | fair       | no                |
| youth       | fair       | yes               |
| middle_aged | excellent  | yes               |
| middle_aged | fair       | no                |
| youth       | fair       | no                |
| middle_aged | excellent  | no                |
| middle_aged | fair       | yes               |



- Prior: probability of a customer buying a computer regardless of their information.
  - [![P(buys\ computer=yes) = 3/7](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9985ba28c50eb015de1b0f13ebf3b59b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(buys\ computer%3Dyes) %3D 3%2F7)
  - [![P(buys\ computer=no) =4/7](https://wattlecourses.anu.edu.au/filter/tex/pix.php/37e1cd65fca2d60300a4cb835142d8c4.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(buys\ computer%3Dno) %3D4%2F7)
- Likelihood
  - [![P(age=youth, credit=fair |buys\ computer=yes) = 1/3](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332845.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(age%3Dyouth%2C credit%3Dfair |buys\ computer%3Dyes) %3D 1%2F3)
  - [![P(age=youth, credit=fair |buys\ computer=no) = 2/4](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332046.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(age%3Dyouth%2C credit%3Dfair |buys\ computer%3Dno) %3D 2%2F4)
- Evidence
  - [![P(age=youth, credit=fair) = 3/7](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e76de52f59ce4ceb8e4e67c1b492121a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(age%3Dyouth%2C credit%3Dfair) %3D 3%2F7)
- Posterior
  - [![P(buys\ computer=yes|age=youth, credit=fair)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/964f548324467df9a13b9a7f9a59740b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(buys\ computer%3Dyes|age%3Dyouth%2C credit%3Dfair)) 
    [![\quad = \frac{3/7 \times 1/3}{3/7} = 0.33](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3e3938566ddb8fb39e4225959881ae38.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\quad %3D \frac{3%2F7 \times 1%2F3}{3%2F7} %3D 0.33)
  - [![P(buys\ computer=no|age=youth, credit=fair)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332362.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(buys\ computer%3Dno|age%3Dyouth%2C credit%3Dfair)) 
    [![\quad = \frac{4/7 \times 2/4}{3/7} = 0.66](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6e86dc3d7245687ee95c701f0263a267.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\quad %3D \frac{4%2F7 \times 2%2F4}{3%2F7} %3D 0.66)
- Therefore, the customer would not buy a computer
  - When computing a posterior, the evidence term is the same for all hypothesis classes. Since our goal is to find the highest class, the evidence term is often ignored in practice.



Example: with estimated probabilities



You might be interested in finding out a probability of patients having liver cancer if they are an alcoholic. In this scenario, we discover by using Bayes' Theorem that “being an alcoholic” is a useful diagnostic examination for liver cancer.

- **Prior**: [![C_1](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332580.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_1) means the event “Patient has liver cancer.” Past data tells you that 1% of patients entering your clinic have liver disease. [![C_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f0350e5818b058dbcfd95f155e417f6a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_2) means the event "Patient does not have liver disease".[![P(C_1) = 0.01](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332545.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_1) %3D 0.01), [![P(C_2) = 0.99](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a3d239484446ef92439cd2868c0bd1f3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_2) %3D 0.99)

- **Evidence**: [![A](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7fc56270e7a70fa81a5935b72eacbe29.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A) could mean the examination that “Patient is an alcoholic.” Five percent of the clinic’s patients are alcoholics.

  - [![P(A) = 0.05](https://wattlecourses.anu.edu.au/filter/tex/pix.php/bb1dda7b450f7ff57cb62c7be8315b80.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(A) %3D 0.05)

- **Likelihood**: You may also know from the medical literature that among those patients diagnosed with liver cancer, 70% are alcoholics.

  - [![P(A|C_1) = 0.70](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332867.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(A|C_1) %3D 0.70); the probability that a patient is alcoholic, given that they have liver cancer, is 70%.

- Bayes’ theorem tells you: If the patient is an alcoholic, their chances of having liver cancer is 0.14 (14%). This is much more than the 1% prior probability suggested by past data.

- - [![P(C_1|A) = (0.7 * 0.01)/0.05 = 0.14](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261332008.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_1|A) %3D (0.7 * 0.01)%2F0.05 %3D 0.14)

**ACTION**: This 6.5 minute video explains the application of Bayes' Theorem by example if you want more. https://www.khanacademy.org/partner-content/wi-phi/wiphi-critical-thinking/wiphi-fundamentals/v/bayes-theorem 



#### 2.3. Limitation (Text 8.3.2)

In the following example, we would like to classify whether a certain customer would buy a computer or not. We have a customer purchase history as follows:

| **age**     | **credit** | **buys_computer** |
| ----------- | ---------- | ----------------- |
| youth       | fair       | no                |
| youth       | fair       | yes               |
| middle_aged | excellent  | yes               |
| middle_aged | fair       | no                |
| youth       | excellent  | no                |
| middle_aged | excellent  | no                |
| middle_aged | fair       | yes               |



What is the probability of *(youth, excellent)* customer buying a computer?

- If we compute the likelihood [![P(X|H)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261354052.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X|H)):As we can see, we observe 0 likelihood for buying a computer with attribute (*age=youth, credit=excellent*).

[![P((age=youth, credit=excellent)|buys\ computer=yes) = 0/3 = 0](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261354340.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P((age%3Dyouth%2C credit%3Dexcellent)|buys\ computer%3Dyes)

- Therefore, posterior probability of tuples with (age=youth, credit=excellent) will be 0:

[![P((age=youth, credit=excellent)|buys\ computer=yes) \\ \times P(buys\ computer=yes)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6fc1391e3349414968486ed8c6bbb3ea.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P((age%3Dyouth%2C credit%3Dexcellent)|buys\ computer%3Dyes) 
[![= 0 \times 3/7](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261354547.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=%3D 0 \times 3%2F7)
[![= 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/b3df24f360f2df5a72c996bb1c142c5e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=%3D 0)

- This does not mean that every buyer with (*age=youth, credit=excellent*) would not buy a computer.
  - The data contains some information about customers who are youth *or* have excellent credit.
  - But the classifier ignores it because there are no who are youth *and* have excellent credit.
- It is usual to interpret this to mean that the number of observations is too small to obtain a reliable posterior probability.
- This tendency toward having zero probability will increase as we incorporate more and more attributes.
  - Because we need **at least one observation** for every possible combination of attributes and target classes.
- In the next section, we will see that this problem is mitigated somewhat with **naive Bayes** that assumes class conditional independence, but we will still need the **Laplacian correction** when there is some attribute value which has not been seen in some class in the training data.



### 3. Naive Bayes (Text:8.3.2)



**Naive Bayes Classification method
**

- Let [![D](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261400435.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) be a training set of tuples and their associated class labels, and each tuple is represented by an n-Dimensional attribute vector [![X = (x_1, x_2, ..., x_n)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/158cda808fe8b98b447dee199940d2fd.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X %3D (x_1%2C x_2%2C ...%2C x_n))
- Suppose there are [![m](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6f8f57715090da2632453988d9a1501b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m) classes [![C_1, C_2, ..., C_m.](https://wattlecourses.anu.edu.au/filter/tex/pix.php/13041721c290b332dcf8509d69a42aab.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_1%2C C_2%2C ...%2C C_m.)
- Classification aims to derive the maximum posteriori, i.e., the maximal [![P(C_i|X)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3cd2306f3c7783a619159b3a15d910ff.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_i|X)) using Bayes’ theorem [![P(C_i|X) = \frac{P(X|C_i)P(C_i)}{P(X)}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261400571.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_i|X) %3D \frac{P(X|C_i)P(C_i)}{P(X)})
  - Since P(X) is constant for all classes, we only need to maximise 
    [![P(C_i|X) \propto P(X|C_i)P(C_i)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261400951.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_i|X) \propto P(X|C_i)P(C_i))

For Naive Bayes, we simplify Bayes' theorem to reduce the computation cost of each likelihood in the training phase. Instead of a computing and recording a likelihood for each tuple for each class in our training set, we summarise by computing a likelihood for each attribute value for each class, that is, the class distribution for each attribute value. Statistically, we are making an assumption that, within each class, each attribute is independent of all the others.

**Class conditional independence**: We *assume* the object's attribute values are conditionally independent of each other given a class label, so we can write

![class_conditional_probability](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261403988.png)

- In other words, we factorise each attribute in the likelihood function, by *<u>assuming that there are no dependence relationships amongst the attributes.</u>*
- This greatly reduces the computation cost as it only counts the class distribution
- If [![A_k](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261400096.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_k) is categorical, [![P(x_k|C_i)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261400303.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(x_k|C_i)) is the number of tuples in [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i) having value [![x_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83b08453f4197d78025b7af0f4b71186.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x_k) for [![A_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/36b7a5a0150dd4e04b4078a7f3aeeac7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_k) divided by [![|C_{i, D}|](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261400486.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=|C_{i%2C D}|) (number of tuples of [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i) in [![D](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261400435.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D))
- <u>Blithely assuming class conditional independence of attributes is **naive,** hence the name of the method. It is not checked, and is commonly even known to be untrue, however, it seems to work, mostly.</u> 



Example

Let's compute the likelihood of the previous example using the assumption of class conditional independence

| **age**     | **credit** | **buys_computer** |
| ----------- | ---------- | ----------------- |
| youth       | fair       | no                |
| youth       | fair       | yes               |
| middle_aged | excellent  | yes               |
| middle_aged | fair       | no                |
| youth       | excellent  | no                |
| middle_aged | excellent  | no                |
| middle_aged | fair       | yes               |

 

- With the conditional independence assumption, the likelihood of tuple (youth, excellent) is
  [![P((age=youth, credit=excellent) | buys\ computer=yes)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261400742.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P((age%3Dyouth%2C credit%3Dexcellent) | buys\ computer%3Dyes))
  [![= P(age=youth | buys\ computer=yes) \times P(credit=excellent | buys\ computer=yes)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3ae25558ba376cfdcb2a8ac08ff7bdbf.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=%3D P(age%3Dyouth | buys\ computer%3Dyes) \times P(credit%3Dexcellent | buys\ computer%3Dyes))
  [![= 1/3 \times 1/3](https://wattlecourses.anu.edu.au/filter/tex/pix.php/37b33c962f280a24890dfdf7701743e7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=%3D 1%2F3 \times 1%2F3)
  [![= 1/9](https://wattlecourses.anu.edu.au/filter/tex/pix.php/97f66c8eeeaa2e5a5a9e0c2e460c9455.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=%3D 1%2F9)

- We can also see here that we have mitigated the [limitation observed earlier ](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472057&chapterid=439358)caused by the lack of observations for (youth, excellent) actually buying a computer.





Example 2

- Here we have some more complex customer history with four different attributes.

| **age**     | **income** | **student** | **credit** | **buys_computer** |
| ----------- | ---------- | ----------- | ---------- | ----------------- |
| youth       | high       | no          | fair       | no                |
| youth       | high       | no          | excellent  | no                |
| middle_aged | high       | no          | fair       | yes               |
| senior      | medium     | no          | fair       | yes               |
| senior      | low        | yes         | fair       | yes               |
| senior      | low        | yes         | excellent  | no                |
| middle_aged | low        | yes         | excellent  | yes               |
| youth       | medium     | no          | fair       | no                |
| youth       | low        | yes         | fair       | yes               |
| senior      | medium     | yes         | fair       | yes               |
| youth       | medium     | yes         | excellent  | yes               |
| middle_aged | medium     | no          | excellent  | yes               |
| middle_aged | high       | yes         | fair       | yes               |
| senior      | medium     | no          | excellent  | no                |

Compute prior probability on hypothesis: [![P(C_i)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/901d2ceda80fb3d275929602dbe05624.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(C_i))

- [![P(buys\ computer = yes) = 9/14 = 0.643](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d0a54454f8acdf999eb864258b2f4378.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(buys\ computer %3D yes) %3D 9%2F14 %3D 0.643)
- [![P(buys\ computer = no) = 5/14= 0.357](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261411225.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(buys\ computer %3D no) %3D 5%2F14%3D 0.357)

<img src="https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261413618.png" alt="image-20220426下午21308558" style="zoom:150%;" />

#### 3.1. Laplacian Correction

**Zero-probability problem**

Naïve Bayesian prediction requires each class conditional probability to be non-zero, as otherwise the predicted probability will be zero. 

Example

Let's assume that we extract following two tables for *student* and *credit* attributes from a customer history, where each entry represents a number of customers:



| Buy computer \ Student | Yes  | No   |
| ---------------------- | ---- | ---- |
| Yes                    | 0    | 5    |
| No                     | 3    | 7    |



| Buy computer \ credit | Fair | Excellent |
| --------------------- | ---- | --------- |
| Yes                   | 4    | 1         |
| No                    | 6    | 4         |



Using naive Bayes, let's classify the probability of a *student* with *fair credit* buying a computer. First, we need to compute the likelihood:



[![P(Student = Yes, Credit=Fair|Buy = Yes) \\ = P(Student = Yes|Buy=Yes) \times P(Credit=Fair|Buy=Yes) \\ =0/5 \times 4/5 \\ = 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/148b77d0cf421fc38aca4bbc6eb82ed6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Student %3D Yes%2C Credit%3DFair|Buy %3D Yes) \\ %3D P(Student %3D Yes|Buy%3DYes) \times P(Credit%3DFair|Buy%3DYes) \\ %3D0%2F5 \times 4%2F5 \\ %3D 0)



[![P(Student = Yes, Credit=Fair|Buy = No) \\ = P(Student = Yes|Buy=No) \times P(Credit=Fair|Buy=No) \\ =3/10 \times 6/10 \\ = 0.18](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261418714.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Student %3D Yes%2C Credit%3DFair|Buy %3D No) \\ %3D P(Student %3D Yes|Buy%3DNo) \times P(Credit%3DFair|Buy%3DNo) \\ %3D3%2F10 \times 6%2F10 \\ %3D 0.18)



Therefore, the classifier will classify that the student will not buy a computer irrespective of the prior. This is because no student has bought a computer ever before. In other words, the likelihood of student buying a computer: [![P(Student=Yes|Buy=Yes) = 0/5 = 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6bc8c3b4f157362aedd418b8c52d5dba.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Student%3DYes|Buy%3DYes) %3D 0%2F5 %3D 0), indicates **irrespective of the other attributes**, the classifier will always classify a student tuple as *not* buy a computer. During the classification of an unlabelled tuple, all the other attributes have no effect if the *student* attribute is *Yes*. This is not wrong, but inconvenient, as in some cases, the other attributes may have a different opinion to contribute to the classification of the tuple.



**Laplace correction**

To avoid the zero probability in the likelihood, we can simply add a small constant to the summary table as follows:

| Buy computer \ Student | Yes                                                          | No                                                           |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Yes                    | 0+[![\alpha](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7b7f9dbfea05c83784f8b85149852f08.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\alpha) | 5+[![\alpha](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7b7f9dbfea05c83784f8b85149852f08.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\alpha) |
| No                     | 3                                                            | 7                                                            |



If we let [![\alpha = 1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/5019ef8cb51410d8ca44c6be8b89f1cd.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\alpha %3D 1), which is the usual value, then the likelihoods of naive Bayes are:

[![P(Student = Yes, Credit=Fair|Buy = Yes) \\ = P(Student = Yes|Buy=Yes) \times P(Credit=Fair|Buy=Yes) \\ = 1/7 \times 4/5 = 0.11](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8c9d96a18dd7832782ded9c5f518a8c6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Student %3D Yes%2C Credit%3DFair|Buy %3D Yes) \\ %3D P(Student %3D Yes|Buy%3DYes) \times P(Credit%3DFair|Buy%3DYes) \\ %3D 1%2F7 \times 4%2F5 %3D 0.11)

Using the Laplacian correction with [![\alpha](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7b7f9dbfea05c83784f8b85149852f08.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\alpha) of 1, we pretend that we have 1 more tuple for each possible value for Student (i.e., Yes and No, here) but we only pretend this while computing the likelihood factors for the **attribute and class combination which has a zero count in the data**for at least one of its values. 

Likelihood for alternative(non-zero count) values of the affected attribute are also affected, but this will come into play when we are predicting a *different* customer at a different time: e.g.

[![P(Student  = No, Credit=Fair|Buy = Yes) \\ = P(Student = No|Buy=Yes) \times  P(Credit=Fair|Buy=Yes) \\ = 6/7 \times 4/5 = 0.69 ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9c9e5621eb98c921944c70156f13f766.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Student %3D No%2C Credit%3DFair|Buy %3D Yes) \\ %3D P(Student %3D No|Buy%3DYes) \times  P(Credit%3DFair|Buy%3DYes) \\ %3D 6%2F7 \times 4%2F5 %3D 0.69 )

The likelihood for the other class (for the same student with fair credit) is unchanged as before:

[![P(Student = Yes, Credit=Fair|Buy = No) \\ = P(Student = Yes|Buy=No) \times P(Credit=Fair|Buy=No) \\ =3/10 \times 6/10 = 0.18](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261418248.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(Student %3D Yes%2C Credit%3DFair|Buy %3D No) \\ %3D P(Student %3D Yes|Buy%3DNo) \times P(Credit%3DFair|Buy%3DNo) \\ %3D3%2F10 \times 6%2F10 %3D 0.18)

The “corrected” probability estimates are close to their “uncorrected” counterparts, yet the zero probability value is avoided.



#### 3.2. Numerical attributes

So far, we've only considered the case when every attribute is a categorical or binary variable.However, numerical variables are common.

In this section, we will show how to use a naive-Bayes classifier with a continuous (numerical) attribute. This approach can also be used for ordinal variables, although depending on the application, and where the range of possible values is small, it may be more useful to treat ordinals as categorical even though the information of the order will not be used for prediction.

It is common to assume that a continuous attribute follows a *Gaussian* distribution (also called *normal*, or *bell curve*). 

- Two parameters define a Gaussian distribution mean: [![\mu](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261425715.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\mu) and standard deviation [![\sigma](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a2ab7d71a0f07f388ff823293c147d21.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\sigma)

- **Probability density function of Gaussian: [![g(x,\mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x-\mu)^2}{2\sigma^2}}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/34b687a1c9e129cffe08e3828c8f22dd.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=g(x%2C\mu%2C\sigma) %3D \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x-\mu)^2}{2\sigma^2}})** 

- Class conditional likelihood of [![k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8ce4b16b22b58894aa86c421e8759df3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k)th-continuous attribute given class [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i) is [![p(x_k|C_i) = g(x_k, \mu_{C_i}, \sigma_{C_i})](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261425858.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p(x_k|C_i) %3D g(x_k%2C \mu_{C_i}%2C \sigma_{C_i}))



To solve the equation for class conditional likelihood, we only need [![\mu_{C_i}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/81f66a68a096e7ae98c66759e8677688.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\mu_{C_i}) and [![\sigma_{C_i}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261425234.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\sigma_{C_i}), [which are calculated as given earlier](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2471993&chapterid=439266).



Example

Let's assume that the attribute *age* is not discretized in the following example:

| **age** | **credit_rating** | **buys_computer** |
| ------- | ----------------- | ----------------- |
| 22      | fair              | no                |
| 23      | fair              | yes               |
| 35      | excellent         | yes               |
| 31      | fair              | no                |
| 20      | excellent         | no                |
| 38      | excellent         | no                |
| 40      | fair              | yes               |



Let buys_computer be a class label, then [![C_1=yes](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e2a4777807e8d6be375dedd1481913ea.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_1%3Dyes) and [![C_2=no](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8e446f6ce5b9c8cd822c0100f8aa95b6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_2%3Dno).

The class conditional mean and variance of attribute age are:

- [![\mu_{C_1} = 32.67, \sigma^2 = 76.33](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261425376.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\mu_{C_1} %3D 32.67%2C \sigma^2 %3D 76.33)
- [![\mu_{C_2} = 27.75, \sigma^2 = 69.59](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3d0d589f00a478339cec667982c24d46.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\mu_{C_2} %3D 27.75%2C \sigma^2 %3D 69.59)

Let [![X=(30, fair)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261425708.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X%3D(30%2C fair)) be attributes of a future customer, the class conditional probability of this customer is:



[![p(age=30|buys\ computer=yes) = \frac{1}{\sqrt{2\pi}\sigma_{C_1}}e^{\frac{-(x_1-\mu_{C_1})^2}{2\sigma_{C_1}^2}} = 0.043579](https://wattlecourses.anu.edu.au/filter/tex/pix.php/974fbde66bbdc559a3a31c332231e3e3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p(age%3D30|buys\ computer%3Dyes) %3D \frac{1}{\sqrt{2\pi}\sigma_{C_1}}e^{\frac{-(x_1-\mu_{C_1})^2}{2\sigma_{C_1}^2}} %3D 0.043579) 
[![p(age=30|buys\ computer=no) = \frac{1}{\sqrt{2\pi}\sigma_{C_2}}e^{\frac{-(x_1-\mu_{C_2})^2}{2\sigma_{C_2}^2}} = 0.046115](https://wattlecourses.anu.edu.au/filter/tex/pix.php/edc040aa996643d1e22a7f1301d8a493.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p(age%3D30|buys\ computer%3Dno) %3D \frac{1}{\sqrt{2\pi}\sigma_{C_2}}e^{\frac{-(x_1-\mu_{C_2})^2}{2\sigma_{C_2}^2}} %3D 0.046115)
This likelihood for each continuous variable can be used directly in the calculation of class conditional likelihood for [Naive Bayes](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472057&chapterid=439349), combined with the likelihoods for discrete attributes. Via  [Bayes theorem](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472057&chapterid=439364), we can then predict the probability of the customer buying a computer.





### 4. Bayesian Belief Networks (Text: 9.1)

**Concept and Mechanism**

- Bayesian belief networks—probabilistic graphical models, which unlike naive Bayesian classifiers **allow the representation of dependencies** among subsets of attributes.

- The naive Bayesian classifier makes the assumption of class conditional independence, that is, given the class label of a tuple, the values of the attributes are assumed to be conditionally independent of one another.

- In practice, however, **dependencies can exist between variables** (attributes).

- Bayesian belief networks provide a graphical model of causal relationships between attributes.

- A belief network is defined by two components

- - a directed acyclic graph
    - Node: represents a random variable (attribute), can be discrete- or continuous-valued
    - Edge: represents a probabilistic dependence, If an arc is drawn from a node Y to a node Z, then Y is a parent or immediate predecessor of Z.
  - a set of conditional probability tables



**Example**

![belief_network](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261502031.png)

Simple Bayesian belief network with six boolean variables. (a) A proposed causal(graphical) model, represented by a directed acyclic graph. (b) The conditional probability table for the values of the variable *LungCancer* (LC) showing each possible combination of the values of its parent nodes, *FamilyHistory* (FH) and *Smoker* (S). 

**Causal relations**:

- having lung cancer is influenced by a person’s family history of lung cancer, as well as whether or not the person is a smoker. 
- Variable *PositiveXRay* is independent of whether the patient has a family history of lung cancer or is a smoker, given that we know the patient has lung cancer.
  - Once we know the outcome of the variable *LungCancer*, then the variables *FamilyHistory*and *Smoker* do not provide any additional information regarding *PositiveXRay.*
- Variable *LungCancer* is conditionally independent of *Emphysema*, given its parents, *FamilyHistory* and *Smoker.*

**Conditional probability table (CPT):**

The CPT for a variable [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) specifies the conditional distribution [![P(X|Parents(X))](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261502317.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(X|Parents(X))), where [![Parents(X)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261502541.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Parents(X)) are the parents of [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X). Figure (b) shows a CPT for the variable *LungCancer*. The conditional probability for each known value of *LungCancer* is given for each possible combination of the values of its parents. For instance, we can interpret the upper leftmost and bottom rightmost entries as

[![P(LungCancer = yes|FamilyHistory = yes, Smoker = yes) = 0.8](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c69e40dd52098ad30d61974fbccbb727.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(LungCancer %3D yes|FamilyHistory %3D yes%2C Smoker %3D yes) %3D 0.8)

[![P(LungCancer = no|FamilyHistory = no, Smoker = no)=0.9](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261502073.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(LungCancer %3D no|FamilyHistory %3D no%2C Smoker %3D no)%3D0.9)

More formally, let [![X = (x_1, ..., x_n)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/334ed5dfafbbc62c8c387c06a42d6323.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X %3D (x_1%2C ...%2C x_n)) be a data tuple described by the variables. Recall that <u>each variable is conditionally independent of its nondescendants in the network graph, given its parents.</u> This allows the network to provide a complete representation of the existing joint probability distribution with the following equation:

[![P(x_1, ..., x_n) = \prod_{i=1}^{n} P(x_i|Parents(x_i))](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3f4c1d47800ecfa23a18304f0ef3dbc2.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(x_1%2C ...%2C x_n) %3D \prod_{i%3D1}^{n} P(x_i|Parents(x_i))),

where [![P(x_1,..., x_n)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261502605.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(x_1%2C...%2C x_n)) is the probability of a particular combination of values of [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X), and the values for [![P(x_i|Parents(x_i))](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9ad01276c9311ed491506c1b00fba0c6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P(x_i|Parents(x_i))) correspond to the entries in the CPT for [![x_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1ba8aaab47179b3d3e24b0ccea9f4e30.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x_i).



#### 4.1. Training a Belief Network (Text 9.1.2)

**How to construct a directed network?**

- The network topology (or “layout” of nodes and arcs) may be constructed by human experts or alternatively inferred from the data.
- The network variables may be *observable* or *hidden* in all or some of the training tuples. The hidden data case is also referred to as *missing values* or i*ncomplete data*.
- Several algorithms exist for learning the network topology from the training data given observable variables. 
- Human experts usually have a good grasp of the direct conditional dependencies that hold in the domain under analysis, and can design the network topology. Typically, these conditional dependencies are thought of <u>causal relationships</u>, e.g. that Smoking *causes* LungCancer. Experts must specify conditional probabilities for some of the nodes that participate in these direct dependencies (some of the CPTs). These probabilities can then be used to compute the remaining probability values.

**How to learn the network?** 

- If the network topology is known and all the variables are observable in the training data
  - Computing the CPT entries is straightforward (very like naive Bayes)
- When the network topology is given and some of the variables are hidden
  - Several heuristic methods exist: many software packages provide solutions
  - The *<u>**gradient descent**</u> method* is well known: it works by treating each conditional probability as a *weight.* It **initialises the weights randomly** up front and then iteratively adjusts each one by a small amount to raise the product of the computed probabilites of each datapoint in the training set. It stops when it is not increasing the product any more.
  - This is computationally demanding, but it has the benefit that <u>human domain knowledge is employed</u> in the solution to design the network structure and thereby to <u>assign initial probability values.</u>