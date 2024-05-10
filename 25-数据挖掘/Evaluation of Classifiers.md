## Classification & Prediction: Evaluation of Classifiers - 6.4

### 1. Introduction

Most of this material is derived from the text, Han, Kamber and Pei, Chapter 8 and 9, or the corresponding powerpoint slides made available by the publisher. Where a source other than the text or its slides was used for the material, attribution is given. Unless otherwise stated, images are copyright of the publisher, Elsevier.

Here, we will discuss how to evaluate the performance of classifiers. When we have built a model using some learning algorithm or by fitting a statistical distribution, how do we know whether it is any good?



### 2. Model Evaluation and Selection (Text: 8.5)

Now that you may have built a classification model, there may be many questions going through your mind. For example, suppose you used data from previous sales (**training data**) to build a classifier to predict customer purchasing behaviour. You would like an estimate of how accurately the classifier can predict the purchasing behaviour of future customers (**test data**), that is, future customer data on which the classifier has not been trained. You may even have tried different methods to build more than one classifier and now you wish to compare their quality and choose the best one. For this, you will be most interested in the *accuracy* of the classifier. But

- What is accuracy?
- How can we estimate it? 
- Are some measures of a classifier’s accuracy more appropriate than others? 
- How can we obtain a reliable accuracy estimate?

These questions are addressed in this section. 



#### 2.1. Evaluation metrics for classification (Text 8.5.1)

*Accuracy,* defined to be the proportion of correctly labelled tuples, is not the only measure to evaluate performance of classification. To understand the other measures, we first need to look at the **confusion matrix**.

**Confusion Matrix** (also called **Error Matrix**)**



A confusion matrix is a useful tool for analysing how well a classifier can recognise tuples of different classes. Given a binary classification problem, a confusion matrix is a 2 by 2 matrix where each entry indicates the number of tuples categorised by the *actual* class (positive or negative label in training or testing data) vs *predicted* class (positive or negative predicted class suggested by the classifier).



| Actual class (rows) \ Predicted class (columns) | C1=Positive          | C2=Negative          |
| ----------------------------------------------- | -------------------- | -------------------- |
| C1=Positive                                     | True Positives (TP)  | False Negatives (FN) |
| C2=Negative                                     | False Positives (FP) | True Negatives (TN)  |



From the confusion matrix, we can define four important measures:

- **True Positive** (TP): Number of *positive* tuples that were *correctly* labelled positive by the classifier
- **True Negative** (TN): Number of *negative* tuples that were *correctly* labelled negative by the classifier
- **False Positive** (FP): Number of *negative* tuples that were *incorrectly* labelled as positive by the classifier 
- **False negative** (FN): A number of *positive* tuples that were *incorrectly* labelled as negative by the classifier 

TP + TN is the number of tuples *correctly* labelled by the classifier (hence called *True*).

FP + FN is the number of tuples *incorrectly* labelled by the classifier (hence called *False*). 

Note that all the FP tuples are *actually* N and all the FN tuples are *actually* P. That is, under this naming convention, the first character tells you if the classifier got it right (T) or wrong (F), and the second character tells you if the classifier predicts positive (P) or negative (N). The *actual*label for the tuple is not given in the name, but you can derive it.

**Beware:** There are several popular conventions for the layout of these matrices: sometimes postiveness is top and left as here but sometimes bottom and right; sometimes actuals are rows and predicted are columns, as here, but sometimes actuals are columns and predictions are rows. You need to pay attention to the table layout.



**Example of confusion matrix (actuals are rows and predicted are columns)**


![confusion_matrix_example](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270801202.png)

- Confusion matrix for the classes *buys_computer = yes* and *buys_computer = no*

- For example

- - 6954: The number of positive tuples classified as positive -- TP
  - 412: The number of negative tuples classified as positive -- FP
  - 46: The number of positive tuples classified as negative -- FN
  - 2588: The number of negative tuples classified as negative -- TN

 

**Various evaluation measures from a confusion matrix**

Let

- P = the number of tuples actually postive in the training data = TP + FN
- N = the number of tuples actually negative in the training data = TN + FP

With four primitive measures, we can define some important evaluation measures as follows:

![evaluation_measures](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270801520.png)

- This table shows some basic evaluation measures for classifications.
- accuracy = 1 - error rate

**Class Imbalance Problem: Beware
<!--类不平衡问题-->**

One may wonder why we need such a range of evaluation measures. At first, the accuracy seems to be enough for a classification task, bu*t t<u>he accuracy may not be a good way to show the performance of your classifier when the dataset is unbalanced.</u>*

An **unbalanced dataset is one where the classes are not evenly distributed in the data**, ie far from 50% each in the binary classification case. One class dominates the data; usually the positive class is rare.

Consider the following example that shows a confusion matrix for a cancer classification. Again, actuals are rows and predicted are columns in the table. 

![confusion_matrix_cancer](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270802866.png)

If we only care about the classifier's accuracy, then 96.4% appears to be a good result at first glance, quite close to 100%. **Wrong!**

Let's consider a classifier that we have learnt which classifies every patient as "cancer=no". Clearly we did not need a complex data mining algorithm to learn this ridiculously simple classifier, a majority vote. In this case, we have 97.7% accuracy (9770/10000). Not bad, huh? **Wrong!** Our new classifier is telling us nothing, only the distribution of the classes. And note our first classifier above performed even worse than this according to accuracy, so it was unacceptably poor. 

Clearly, an accuracy rate of 97% is not acceptable on this problem—a classifier with this accuracy could be correctly labelling only the noncancer tuples and misclassifying all the cancer tuples as our "cancer=no" classifier does. Instead, we need other measures, which can distinguish how well the classifier can recognise the positive tuples (cancer = yes) and how well it can recognise the negative tuples (cancer = no).

The **sensitivity** and **specificity** measures can be used, respectively, for this purpose.

For example, the sensitivity and specificity of the above example are:

- [![sensitivity = \frac{TP}{P} = \frac{90}{300}=0.30](https://wattlecourses.anu.edu.au/filter/tex/pix.php/064e9b8a59ac0702d8ed8f782fa77be7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=sensitivity %3D \frac{TP}{P} %3D \frac{90}{300}%3D0.30)
- [![specificity = \frac{TN}{N} = \frac{9560}{9700}=0.99](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204261821126.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=specificity %3D \frac{TN}{N} %3D \frac{9560}{9700}%3D0.99)

Thus, we note that although the classifier has a high accuracy, it’s ability to correctly label the positive (rare) class is poor as given by its low sensitivity. It has high specificity, meaning that it can recognise negative tuples quite well. The sensitivity is much more important than specificity in this case, due to the purpose of the classification task. 

But what if our classifier could deliver 100% sensitivity? Too easy: the classifier "cancer=yes" can do this. **Is this a good result?** No! Accuracy would be only 3%. Specificity would be 0%. Predicting all people have cancer is just as useless in practice as predicting no people have cancer.

Sensitivity and specificity are typically a **tradeoff,** you can maximise one by reducing the other: 100% for each is ideal (well, maybe not, due to potential *overfitting* discussed later), but the tradeoff between them, and whether some classifer is therefore *good enough*, is something for the expert to interpret with knowledge of the underlying purpose. 

The **precision** and **recall** measures, originally developed for information retrieval, are also widely used in classification as alternative tradeoff quality measures: 

- Precision can be thought of as a measure of exactness 

  - i.e., what percentage of tuples classified as positive are actually such

- Recall (equivalent to *sensitivity*, above) is a measure of completeness 

- - i.e., what percentage of positive tuples are classified as such

Often precision and recall are combined into an *F1-score*, which is the harmonic mean of the precision and recall. It might also be called simply *f-score*, or *f-measure*. See the table above for its formulation. Maximising f-score is useful because it provides a single measure, and takes account of potentially unbalanced data by focusing on the postive class, but it emphasises a particular relationship between right and wrong predictions. **Is this the right quality measure for your classification task? Or not?** 



**ACTION: Calculate precision, recall and f-measure for the cancer classification confusion matrix above as well as for the classifiers "cancer=yes" (which classifies every tuple as positive) and "cancer=no" (which classifies every tuple as negative). Comment on the interpretation of f-measure for this problem. A worked solution is here:**



## Solution to Exercise: Evaluation measures



![confusion_matrix_cancer-2](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270804154.png)



*Given the cancer prediction classifier with the confusion matrix above, (actuals are rows and predicted are columns in the table), calculate precision, recall and f-measure for the cancer classification confusion matrix above as well as for the classifiers "cancer = yes" and "cancer = no". Comment on the interpretation of f-measure for this problem.*

(a) *cancer prediction classifier* 

Precision = TP/ (TP+FP) = 90/(90+140) = 90/ 230 = 0.39

Recall = TP/(TP+FN) = 90/(90+210) = 90/ 300 = 0.30

F_1 = (2 x precision x recall)/ (precision + recall) = (2 x 0.39 x 0.30) / (0.39 + 0.30) = 0.234/0.69 = 0.33

For comparison, accuracy is 0.96



(b) *classifier is "cancer = yes"*

| Real \ Predict | Yes (Predict) | No (Predict) | Total |
| -------------- | ------------- | ------------ | ----- |
| Yes (Real)     | 300           | 0            | 300   |
| No (Real)      | 9700          | 0            | 9700  |
| Total          | 10000         | 0            | 10000 |

Precision = TP/(FP+TP) = 300/ 10000 = 0.03

Recall = TP/(TP+FN) = 300/300 = 1

F_1 = (2 x precision x recall)/ (precision + recall) = 2 x 0.03/1.03 = 0.06 

For comparison, accuracy is 0.03



(c) *classifer is "cancer = no"*

| Real \ Predict | Yes (Predict) | No (Predict) | Total |
| -------------- | ------------- | ------------ | ----- |
| Yes (Real)     | +1            | 300          | 300   |
| No (Real)      | 0             | 9700         | 9700  |
| Total          | +1            | 10000        | 10000 |

Precision = TP/(FP+TP) = 0/0 undefined. NB if we predicted just one positive tuple then precision would be 1 if we got it right and 0 if we got it wrong. --> 1 

Recall = TP/P = 0/300 = 0. NB if we predicted just one positive tuple then recall would be 1/300 if we got it right and 0 if we got it wrong. # --> 1/300

F_1 is undefined. NB if we predicted just one positive tuple and got it right F_1 would be (2 x 1/300) / (301/300) = 2/301 = 0.007 which is very small.

For comparison, accuracy is 0.98



Looking at all three classifiers here, we can see that F-measure prefers correct performance on the positive class over correct performance on the negative class, giving more meaningful results than accuracy in this example of an unbalanced dataset where the negatives strongly dominate. The first classifier gave F-measure of 0.33, being very far from the ideal 1, because it recognises that performance is poor on the rare positive class. By contrast, for the same classifier, accuracy at 0.96 is quite close to 1, which misleadingly looks quite good. Focus on the positive class makes sense for cancer screening prediction, because we do not mind predicting a few false positives (which reduce precision a bit) as long as we predict positive for nearly all of the positives in the data (to achieve high recall).

This positive-focus effect of F-measure is clear when comparing F_1 for the “cancer = yes” and “cancer = no” classifiers. Although “cancer = no” has a higher number of correct classifications, that is, much higher accuracy, it is far outperformed by “cancer = yes” according to F_measure. When a single correct classification is added to the “cancer = no” classifier (so that F_1 is defined), the F_1 is 0.007 which compares very badly to the “cancer = yes” classifier with 0.06.

Back to the drawing board -- we can see by f-measure that all three of these models are very poor, even if accuracy might look ok for two of them.



**Key message:** Always consider whether your measure of performance is appropriate for your problem. Choose an appropriate measure which might be influenced by the practice in the domain of application. Always consider how your performance compares to a dumb classifier which might be 50% accuracy for a balanced dataset but something else for unbalanced data.



#### 2.2. Estimating a classifier’s accuracy (Text: 8.5.2-8.5.4)

In this section, we will see how to report the performance of the models by various methods.

Accuracy (or error rate) on training data is not a good indicator of future performance because the model may be overly-tuned towards exactly the data on which it was trained. For example, consider the model which internally simply remembers the data it has seen and classifies that data as it was classified in the training data, and every new data item is classified arbitrarily. Would you expect this model to work well?

This situation is called **overfitting** and commonly leads to poor performance on unseen data.


**Holdout method**

- Given data is randomly partitioned into two independent sets
  - Training set (e.g., 2/3) for model construction
  - Test set (e.g., 1/3) for accuracy estimation
  - Overall flow of holdout method:
    ![holdout_method](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270805078.png)

**Training / Validation / Test**

Variation of holdout method when you you want to experiment with parameters of the model-building algorithm.

- <u>**Randomly partition**</u> the given data into three different sets:
- Training set
  - To construct (i.e *train*) a classification model
- Validation set
  - To find the best parameters for the model, likely to be done by a person studying the effect of various parameters on the accuracy (or other quality measures) of the training set.
- Test set
  - To measure the final performance of the model as it would be reported.





**Cross-validation**

- k-fold, where <u>**k = 10**</u> is most popular (due to low bias and variance)
- Randomly partition the data into k mutually exclusive subsets [![D_1, D_2, ..., D_k](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270804781.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D_1%2C D_2%2C ...%2C D_k), each approximately equal size
- At i-th iteration, use [![D_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a1384aa3ea4e1d4ca4707ed950caee26.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D_i) as test set and others as training set
- **Leave-one-out**: Special case for *small-sized dataset*s: Use k folds where k = number of tuples, 
- **Stratified cross-validation**: Special case where folds are not randomly selected but stratified so that the class distribution in each fold is approximately the same as that in the initial data (to achieve low bias).
- Overall accuracy is computed as the average accuracy of each model on its respective test set.



**Bootstrap**

- Works well for small data sets where, otherwise, the requirement to split the data into training and testing sets makes both sets too small for purpose.

- Samples the given training tuples uniformly **with replacement**

  - i.e., each time a tuple is selected, it is equally likely to be selected again and added to the training set again.

- Several bootstrap methods : a common one is .**632 bootstrap**

- - A data set with [![d](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8277e0910d750195b448797616e091ad.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=d) tuples is sampled [![d](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8277e0910d750195b448797616e091ad.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=d) times, with replacement, resulting in a training set of [![d](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8277e0910d750195b448797616e091ad.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=d)samples. The data tuples that did not make it into the training set end up forming the test set. About 63.2% of the original data end up in the training set, and the remaining 36.8% form the test set (since [![(1 - 1/d)d \approx e^{-1} = 0.368](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a2593a374d61bbfd1f0b7ff71d55a515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=(1 - 1%2Fd)d \approx e^{-1} %3D 0.368) if [![d](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8277e0910d750195b448797616e091ad.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=d) is very large.)
  - Repeat the sampling procedure [![k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8ce4b16b22b58894aa86c421e8759df3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k) times, overall accuracy of the model is:
    ![bootstrap_acc](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270805605.png)
    where [![Acc(M_i)_{test set}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/b4a10e8132956510a2c69ca8ef5c086e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Acc(M_i)_{test set}) is the accuracy of the model trained with training set [![i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/865c0c0b4ab0e063e5caa3387c1a8741.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=i) when it is applied to test set [![i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/865c0c0b4ab0e063e5caa3387c1a8741.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=i) and [![Acc(M_i)_{train set}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270804334.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Acc(M_i)_{train set}) is the accuracy of the model obtained with training set [![i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/865c0c0b4ab0e063e5caa3387c1a8741.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=i) when it is applied to the training set [![i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/865c0c0b4ab0e063e5caa3387c1a8741.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=i).



#### 2.3. ROC Curve (Text: 8.5.6)

![roc_example](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270806687.png)

Example of ROC curve (axes are labelled with percentages in this diagram). 
The dashed diagonal line indicates when TPR = FPR and has an area underneath it of 0.5

 

**ROC (Receiver Operating Characteristics) curves**: for visual comparison of probablistic classification models and selection of a decision threshold. Also for visually comparing tradeoffs in performance for alternative deterministic classifiers.



- **Basic idea**: a *probabilistic classifier* returns a probability of a tuple being in the positive class. What if we consider a probability threshold for positive classification being somewhere in the range [0,1] instead of using the simple 0.5 (majority vote for the class)? This is a way to recognise that the cost of errors (ie FP vs FN) may not be equal for each class.
- Shows the **trade-off** between the true positive rate (TPR) and the false positive rate (FPR)
  - TPR ( = sensitivity) is the proportion of positive tuples that are correctly labelled by the model: TP/P
  - FPR (= 1- specificity) is the proportion of negative tuples that are mislabelled as positive: FP/N
  - A deterministic classifier (which assigns classes without probabilities) can be plotted as a single point on the ROC chart (the point is (FPR, TPR)).
  - A probablistic classifer is plotted as a ROC curve on the chart (see below).
- Use the ROC curve to choose a decision threshold for your probablistic classifier that reflects the tradeoff you need, ideally the probability corresponding to an inflexion point where the curve turns from vertical to horizontal, so that you are getting the benefit of near-maximal TPs with near-minimal FPs. Selection and use of the decision threshold at this point turns your probablistic classifier into a deterministic one plotted at that point.
- The area under a ROC curve (**ROC-AUC**) is often used to measure the performance of a *probablistic* model.
- The area under a ROC curve (**ROC-AUC**) can also be computed for a *deterministic* model as the area under the curve constructed by drawing a line from (0,0) to (FPR,TPR) and another from (FPR,TPR) to (1,1). By geometric analysis, it is easy to see that this equates to the average of sensitivity and specificity, i.e. (TP/P + TN/N) /2 .
- The diagonal line on the graph represents a model that randomly labels the tuples according to the distribution of labels in the data. This line has AUC of 0.5. **<u>*A model better than random should appear above the diagonal. The closer a model is to random (i.e., the closer it's ROC-AUC is to 0.5), the poorer is the model. A model falling below the diagonal line is worse than random (which is very, very poor, but hopefully you are building better models than that!).*</u>** 
- Many deterministic models with distinct (FPR, TPR) points on the graph share the same ROC-AUC, falling on an isometric line parallel to the AUC=0.5 diagonal. While these models have different performance on P and N examples, ROC-AUC alone does not distingush them. A visual study of the chart might be helpful.
- For a probablistic model, the ROC curve may cross the diagonal line for some probabilities; but it may still be a good model if the ROC-AUC is high.
- A ROC-AUC of 1 indicates a perfect classifier for which all the actual P tuples have a higher probability of being labelled P than all the actual N tuples. A ROC-AUC of 0 is the reverse situation: all the actual Ps are less likely to to be labelled P than all the actual Ns, denoting a worst case model.
- The ROC-AUC represents the proportion of randomly drawn pairs (one from each of the two classes) for which the model correctly classifies both tuples in the random pair. In contrast to accuracy or error rate, ROC-AUC allows for unbalanced datasets by counting the performance over the subsets T (on the [![y](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270806902.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y) axis) and N (on the [![x](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9dd4e461268c8034f5c8564e155c67a6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x) axis) independently, valuing errors in each class of the dataset independently of the proportion of each class in the dataset as a whole.
- Instead of probabilities generated by a probablistic classifier, the ROC can also be used to choose a cost or risk function to be used with a deterministic classifier. 
- The ROC may be used together with cross-validation so it is not overly influenced by a particular training set. 



**Plotting ROC curve for a probablistic classifier
**

- ROC curve can be plotted with a probabilistic classifier (e.g. naive Bayes, some decision trees, neural nets)
- The vertical [![y](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270806902.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y) axis of an ROC curve represents TPR. The horizontal [![x](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9dd4e461268c8034f5c8564e155c67a6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x) axis represents FPR.

1. **Rank the test tuples in decreasing order**: the one that is most likely to belong to the positive class (highest probability) appears at the top of the list.
2. Starting at the bottom left corner (where TPR = FPR = 0), we check the tuple’s actual class label at the top of the list. If we have a **true positive** (i.e., a positive tuple that was correctly classified), then true positive (TP) and thus TPR increase.
   - On the graph, we **move up and plot a point**.
3. If, instead, the model classifies a **negative tuple as positive**, we have a false positive (FP), and so both FP and FPR increase.
   - On the graph, we **move right and plot a point**.
4. This process is repeated for each of the test tuples in ranked order, each time moving up on the graph for a true positive or toward the right for a false positive.

Example 1

The following table shows the probability value of being in the positive class that is returned by a probabilistic classifier (column 3), for each of the 10 tuples in a test set. Column 2 is the actual class label of the tuple. There are five positive tuples and five negative tuples, thus P = 5 and N = 5. As we examine the known class label of each tuple, we can determine the values of the remaining columns, TP, FP, TN, FN, TPR, and FPR. 

![roc_example_table](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270807866.png)

We start with tuple 1, which has the highest probability score and take that score as our threshold, that is, t = 0.9. Thus, the classifier considers tuple 1 to be positive, and all the other tuples are considered negative. Since the actual class label of tuple 1 is positive, we have a true positive, hence TP = 1 and FP = 0. Among the remaining nine tuples, which are all classified as negative, five actually are negative (thus, TN = 5). The remaining four are all actually positive, thus, FN = 4. We can therefore compute TPR = TP = 1 = 0.2, while FPR = 0. Thus, we have the point (0.2, 0) for the ROC curve.

Next, threshold *t* is set to 0.8, the probability value for tuple 2, so this tuple is now also considered positive, while tuples 3 through 10 are considered negative. The actual class label of tuple 2 is positive, thus now TP = 2. The rest of the row can easily be computed, resulting in the point (0.4, 0). Next, we examine the class label of tuple 3 and let *t* be 0.7, the probability value returned by the classifier for that tuple. Thus, tuple 3 is considered positive, yet its actual label is negative, and so it is a false positive. Thus, TP stays the same and FP increments so that FP = 1. The rest of the values in the row can also be easily computed, yielding the point (0.4,0.2). The resulting ROC graph, from examining each tuple, is the jagged line as follows. A convex hull curve is then fitted to the jagged line as shown.

![roc_example_plot](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270807609.png)

![roc_example_two_models](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270807457.png)

ROC curves of two probablistic classification models, M1 and M2. The diagonal shows where, for every true positive, we are equally likely to encounter a false positive. The closer a ROC curve is to the diagonal line, the less accurate the model is. Thus M1 is more accurate here. If the ROC curves for M1 and M2 cross over then varying the threshold selection will vary which is more accurate for binary classification.



#### 2.4. Comparing classifiers (Text: 8.5.5)

**Classifier Models M1 vs. M2**

- Suppose we have 2 classifiers, M1 and M2. Which one is better?

- Use 10-fold cross-validation to obtain mean error rates for M1 and M2, [![err(M1), err(M2)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/28424e494a72d3deeb0cba5f9f420bbd.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=err(M1)%2C err(M2))

- It may seem intuitive to choose the model with **<u>*the lowest error rate*</u>**

- **However**, these mean error rates are just **estimates** of error on the true population of future data cases

- What if the difference between the 2 error rates is just attributed to chance?

- - Use a test of statistical significance
  - Obtain confidence limits for our error estimates

**Estimating Confidence Intervals: Null Hypothesis**

- **Null Hypothesis: M1 & M2 are the same**
- Test the null hypothesis with **t-test**
  - Use t-distribution with k-1 degree of freedom
- If we can reject null hypothesis, then 
  - we conclude that the difference between M1 & M2 is statistically significant.
  - Chose model with lower error rate
- Perform 10-fold cross-validation (k=10)
  - For *i*-th round of 10-fold cross-validation, the same cross partitioning is used to obtain [![err(M1)_i](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270808565.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=err(M1)_i) and [![err(M2)_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/951a8d9071dbdb7a61ad7f5343120465.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=err(M2)_i).
  - Average over 10 rounds to get [![\overline{err}(M1) = \frac{1}{k} \sum\limits_{i=1}^{k} err(M1)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/684df909517d05b7fe6f81b6836a16db.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\overline{err}(M1) %3D \frac{1}{k} \sum\limits_{i%3D1}^{k} err(M1)) and similarly for [![\overline{err}(M2)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c2600c21772369d460332eec263955cc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\overline{err}(M2)) 
  - t-test computes **t-statistic** with **<u>*k-1 degrees of freedom*</u>**:
    ![t_statistic](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270808491.png)where (using the variance for the population, as given in the text):
    ![t_var](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270809157.png)
  - To determine whether M1 and M2 are significantly different, we compute t-statistic and select a significance level.
    - 5% significance levels: The difference between M1 and M2 is significantly different for 95% of population.
    - 1% significance levels: The difference between M1 and M2 is significantly different for 99% of population.
  - Based on t-statistics and significance level, we consult a table for the t-distribution.
    - We need to find the t -distribution value corresponding to k − 1 degrees of freedom (or 9 degrees of freedom for our example) from the table (Two-sided).
    - [T-distribution table](https://wattlecourses.anu.edu.au/mod/url/view.php?id=2472070)
      [Hidden from students:URL](https://wattlecourses.anu.edu.au/mod/url/view.php?id=2472070)[URL](https://wattlecourses.anu.edu.au/mod/url/view.php?id=2472070)
  - If the t-statistic we calculated above is *not* between the corresponding value in the table and its negative (i.e. the corresponding value in the table multiplied by -1), then we reject the null hypothesis and conclude that M1 and M2 are significantly different (at the significance level we chose above). 
  - Alternatively, if the t-statistic we calculated above *is between* the corresponding value in the table and its negative, we conclude that M1 and M2 are essentially the same and any difference is attributed to chance.



#### 2.5. Exercises

**ACTION:** Try out this exercise.

[Exercise: Model Selection using t-test](https://wattlecourses.anu.edu.au/mod/page/view.php?id=2472068)



## Exercise: Model Selection using t-test

Suppose that we want to select between two prediction models, M1 and M2. 

We have performed 10 rounds of 10-fold cross-validation on each model, where the same data partitioning in round *i* is used for both M1 and M2. 

The error rates obtained for M1 are 30.5, 32.2, 20.7, 20.6, 31.0, 41.0, 27.7, 26.0, 21.5, 26.0. 

The error rates for M2 are 22.4, 14.5, 22.4, 19.6, 20.7, 20.4, 22.1, 19.4, 16.2, 35.0. 

Comment on whether one model is significantly better than the other considering a significance level of 1%.



When you have had a go, you can check your answers against this worked answer:

[Solution to Exercise: Model Selection using t-test](https://wattlecourses.anu.edu.au/mod/page/view.php?id=2472069) 



## Solution to Exercise: Model Selection using t-test

We can do hypothesis testing to see if there is a significant difference in average error. Given that we used the same test data for each observation we can use the “paired observation” hypothesis test to compare two hypotheses:

[![H_0: \bar{x}_1 - \bar{x}_2 = 0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6d0d65c5da9029f1e16fbfb75b1acc33.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=H_0%3A \bar{x}_1 - \bar{x}_2 %3D 0)

[![H_1: \bar{x}_1 - \bar{x}_2 \neq 0 ](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270810622.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=H_1%3A \bar{x}_1 - \bar{x}_2 \neq 0 )
Where [![\bar{x}_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0e0257bb099669bb47ffe18e5096d737.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\bar{x}_1) is the mean error of model M1, and [![\bar{x}_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/937a7cafde5f0a5f6e55e15c8e662722.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\bar{x}_2) is the mean error of model M2. We compute the test statistic *t* using the formula:

[![t=\frac{\bar{d}}{s_d/\sqrt{n}}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9754de9c9476872ff3d99758fe4a5373.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=t%3D\frac{\bar{d}}{s_d%2F\sqrt{n}})

where [![\bar{d}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9df5e8b901876880c5cfc2a108072b8c.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\bar{d}) is the mean of the differences in error, [![s_d](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270810176.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=s_d) is the sample standard deviation of the differences in error, and *n* is the number of observations. In the follolwing we use the sample standard deviation (i.e. when calculating the standard deviation we divide by (n-1) instead of n because we are estimating the true standard deviation of the errors.

In this case the differences in the model errors are 8.1, 17.7, -1.7, 1, 10.3, 20.6, 5.6, 6.6, 5.3, -9.

Then [![\bar{d} = 6.45](https://wattlecourses.anu.edu.au/filter/tex/pix.php/74f68775c3bf9f7d400149c27aeae139.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\bar{d} %3D 6.45), [![s_d = 8.7](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1218ea40c6316a5d08ae6a334edcf1a1.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=s_d %3D 8.7), and [![n = 10](https://wattlecourses.anu.edu.au/filter/tex/pix.php/b992b47484ee3ffdc664c39c731c3bb0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=n %3D 10). Substituting these values in the equation we get [![t = 2.34](https://wattlecourses.anu.edu.au/filter/tex/pix.php/91eb2517318a61ac1ebb61588a2b8f7d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=t %3D 2.34). Using a *t* distribution table, we look a value for probability 0.01 (0.99) and 9 degrees of freedom, which is 3.25. Given that −3.25 < 2.34 < 3.25 we accept the null hypothesis, i.e., **t****he two models are not different at a significance level of 0.01**.

If you use the population standard deviation instead, you get [![t=2.47](https://wattlecourses.anu.edu.au/filter/tex/pix.php/75a40cf4ad97a2d2a2061d29e1d66750.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=t%3D2.47) and so −3.25 < 2.34 < 3.25 so we come to the same conclusion.



#### 2.6. Other issues affecting the quality of a model

A model may be preferred over another for a number of reasons. These reasons will influence the choice of a learning method as well as the selection of a particular classifier produced by the method.

- Accuracy
  - Classifier accuracy: predicting class label
- Speed
  - Time to construct the model (training time)
  - Time to use the model (classification/prediction time)
- Robustness: handling noise and missing values
- Scalability: efficiency in disk-resident databases
- Interpretability
  - Understanding and insight provided by the model. This is especially important in cases where the model is never intended to be put into practice over unseen data, but instead to influence systemic behaviours such as business rules and policies. It may also be critical to enable qualitative evaluation of the model for embedded bias. 
- Availability and Trust
  - Does the business environment have the technical infrastructure, skills and policy or governance framework to use it?
  - Will the business environment trust the results to be used for the intended purpose?

- Other measures specific to the method, e.g., goodness of rules, such as decision tree size or compactness of classification rules. 



Note well that these kind of factors are just as influential on the selection of a method and a model for mining problems other than classification and prediction. For example, association rules are great for interpretability and scalability, but may not be considered trustworthy.



### 3. Practical Exercises: Evaluation

**ACTION:** Attempt these practical exercises with Rattle. There is a video showing the mechanics to get you started, written instructions for you to work through, and separately some suggested solutions.



<iframe src="https://www.youtube.com/embed/hAaBckbCZ9E" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" width="560" height="315" frameborder="0" style="box-sizing: border-box;"></iframe>





[Practical Exercise: Evaluation](https://wattlecourses.anu.edu.au/mod/page/view.php?id=2472063)



**Evaluation in Rattle**





**Objectives**

The objectives of this lab are to experiment with the **evaluation** methods available in **R** and **Rattle**, in order to better understand the issues involved with evaluation in data mining; and to experiment with the different evaluation methods for supervised classification available in the Rattle tool. 

------

**Preliminaries**



 Read through the following section in the **Rattle** online documentation:



- [**Evaluating Models**](https://www.togaware.com/datamining/survivor/Evaluation1.html)



For more information about ROC please have a look at the paper 

- [**ROC Graphs: Notes and Practical Considerations**](https://wattlecourses.anu.edu.au/mod/resource/view.php?id=2472065)

------



For this lab, we will mainly use the **[weather.csv](http://rattle.togaware.com/weather.csv)** data set which you have used previously. If you want to use another data set to conduct more experiments at the end of the lab please do so.

 

------

**Tasks**

1. Start **Rattle** in the lab. Here is a quick description of the steps involved:

   a) Open a terminal window (Main menu -> Accessories).
   b) Start R by typing R (capitalised!) followed by 'Enter'.
   c) Type: library(rattle) followed by 'Enter'.
   d) Type: rattle() followed by 'Enter'.

2.  Go to the Data tab, click Library, and then select 
   **weather:rattle:Sample...

   **

3. Click Execute to load the data into **Rattle**. 

   

4. Training / Validation and Overfitting

   1. Now make sure the variable (attribute) **RainTomorrow** is selected as Target variable, and that you partition the data (e.g. leave the 70/15/15 percentage split in the Partition box - which must be ticked). This means that we will use 70% of all records in the **weather** data set for training, 15% for validation and 15% for testing. Rattle randomly allocates the appropriate share of the input data to each subset.

      

   2. Also make sure that the variable **Date** is set to role Ident(ifier). Click Execute if you have changed anything since the previous Execute.

      

   3. Now go to the Model tab and make sure the Tree type radio button is selected. Set Min Split, Min Bucket, Max Depth, and Complexity to 1, 1, 50, and 0.01, respectively. These parameters constrain the shape of the decision tree and may be used to trade off simplicity of structure against accuracy of classification. To generate the decision tree, click on Execute and inspect what is printed into the main **Rattle** output area. Look carefully at the structure of the tree.

      

   4. Now go to the Evaluate tab and examine the performance of the decision tree. At this time, we first examine the performance of the decision tree on **training set**. Check the Error matrix radio button. Click the Training radio button and hover to read it's tooltip hint. Then click on Execute. What is the overall error of the trained decision tree? Why do you think the model yields that error rate? Write down the error rate for this tree and each of the following trees you generate.

      

   5. Next check the Validation radio button, read it's tool tip, and again click on Execute. What is the error rate and what is the accuracy? Why are they different between the Training and Validation settings? Why is the training error rate nearly always better? And why is it considered unreliable for evaluation? 

      

   6. Now experiment with your tree building parameters to build a different model. Go back to the Modeltab. Set Min Split, Min Bucket, Max Depth, and Complexity to 20, 7, 30, and 0.01, respectively. Click on Execute, inspect the structure of the tree printed into the main **Rattle** output area, and move to the Evaluate tab again.

      

   7. Make sure both Error Matrix and Training radio buttons are selected. Run Execute again and check the overall error this time. What is the overall error of the model with different parameters? Can you tell which model is better?

      

   8. Now check the error rate for this model on the **validation** dataset by checking Validation in Evaluate tab. What are the overall errors on the **validation** data set? Now which of your two tree models do you think is better? Choose the best one, in your opinion. Finally, check its error rate on the **testing** data, reading the tool tip for Testing on the way.

      

   9. Do you understand the importance and roles of the distinct training, validation and testing data sets? Explain.

      

5. ROC Curve

   1. Now we will measure the performance of models with ROC-Curve and overall error.

      

   2. Go to the Model tab and make sure the Tree type radio button is selected. Set Min Split, Min Bucket, Max Depth, and Complexity to 15, 5, 30, and 0.01, respectively. To generate a decision tree, click on Execute and inspect what is printed into the main **Rattle** output area.

      

   3. Move to the Evaluate tab, and check the overall error and averaged class error. Make sure both Error Matrix and Validation radio buttons are selected. Make a note of the error rate.

      

   4. Now check ROC and Validation radio buttons, and click on Execute to see the ROC curve. What does the ROC curve look like? What are the axes of the graph? What is the value of Area under the ROC curve in the main Rattle output area?

      

   5. Go back to the Model tab, and change Min Split, Min Bucket, Max Depth, and Complexity to 13, 3, 30, and 0.01, respectively. Click on Execute and inspect the result in the main Rattle output area.

      

   6. Move to the Evaluate tab, and check the overall error and averaged class error. Make sure both Error Matrix and Validation radio buttons are selected. In terms of the overall error, does the new model perform better than the previously trained model? 

      

   7. Now check ROC and Validation radio buttons, and click on Execute to see the ROC curve. Inspect the output ROC-curve and the value of Area under the ROC curve.

      

   8. What is the relationship between the overall error and Area under the ROC curve? Which measure is more appropriate for predicting the **RainTomorrow** variable? Why do you think this? (Hint: check the distribution of the **RainTomorrow** variable)

      

6. **Training proportion** (Optional extension lab)

   1. In this section, we will examine an effect of training proportion for training a model.

      

   2. Choose whatever data you want to examine this time. It can be audit.csv, weather.csv, or any other data you are interested in.

      

   3. Make sure Partition box checked. Adjust training/validation/test proportions to 15/70/15.

      

   4.  Click on Execute, and go to Model section and train the decision tree with a proper parameter configuration.

      

   5. Now evaluate the trained model on Evaluate tab. Choose one of Error Matrix or ROC measures. Evaluate the performance of the model on the selected measure. Make sure your evaluation is performed on Testing data set.

      

   6. Go back to Data tab, and re-load the same data set. At this time, change the training/validation/test proportions to 30/55/15. Repeat the previous three steps (6.3-6.5). Is your performance increased or decreased?

      

   7. Increase the proportion assigned to training data, but keep the same testing proportion. Measure the performance with the adjusted proportions. When is the performance saturated? Does adding more training data always increase the performance?



[Solution to Practical Exercises: Evaluation ](https://wattlecourses.anu.edu.au/mod/page/view.php?id=2472064)



## Solution to Pratical Exercises: Evaluation

*N.B. The following results may not be identical across different platforms and software versions.*

**Q 4.4:**

Error on training set:

Overall error: 0%, Averaged class error: 0%

This means the model is 100% accurate!

By setting min-bucket and min-split to 1, the decision tree algorithm will generate enough branches that can perfectly classify the training data. That's why we obtain 0% error rate here. This is often called overfitting: https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472055&chapterid=439331





***\*Q 4.5:\***



Error on validation set:

Overall error: 20.3%. Accuracy : 100 - 20.3 = 79.7% . Averaged class error: 31.8%

You should expect a decision tree classifier to perform better on the data on which it was trained, but the critical question is how it will perform on *unseen* data, such as the validation dataset here. It is much more realistic to say that the accuracy of the model is 79% than 100%. **
**



**Q 4.7:**

Error on training set:

Overall error: 9.7%, Averaged class error: 20.65%

Although the error rate for this model increases over the previous case, we have a much simpler tree that makes some sense in the domain. We cannot properly determine which model is the better of the two because the performance was evaluated here on the **training dataset.** For a fair comparison, we need to evaluate both models on some unseen dataset (i.e. validation data).



**Q 4.8 and 4.9:**





Error on validation set (model 1 with parameters 1, 1, 50, and 0.01) from above:

Overall error: 20.3%. Averaged class error: 31.8%



Error on validation set (model 2 with parameters 20, 7, 30, and 0.01):

Overall error: 18.5%, Averaged class error: 30.7%





Evaluation based on the validation data set suggests that the second model performs better than the first. (NB. You may find that, because in your environment the data partitioning may be different, you have different results here and instead you find that the first model is better on the validation data. )

Since these evaluations are measured with the validation set, we can claim that the second model is better than the first because the first overfits the training data. For robustness in that claim, we may use the t-test to check the significance of the improvement -- is it just better by chance?

(N.B if you found the reverse situation, this is an unusual case because we conjecture that the first model overfits the training data, but it turns out the first model also performs well on the validation data. In general, we do not observe this behaviour frequently, but at least in this case, you may have found the first model seems better than the second.)

For example, if you train a 3rd decision tree with parameters 8, 4, 30, and 0.01, the error on the training set is:

Model 3: Overall error: 3.1%, Averaged class error: 6.8%

and on the validation set is:

Model 3: Overall error: 14.8%, Averaged class error: 20.7%

Although the error rate of this model on the training set is worse than the first model was, we can see the error rate on the validation set is better than both the first and second model. So we can show that the first model, which is overfitted to training set, has not generalised well to the unseen data to compared with the third model here.

So, having built several trees, which one are you going to choose to put in practice? After tuning the parameters, have you over-tuned and now effectively and unfairly fitted to the validation set as well? You have chosen the best performer for the validation set but how accurate is it really? You will want to know that before putting it into practice, so you can make business decisions based on the accuracy of your predictions. 

For example, if you choose to run with the first model in your business, you should use use the fresh unseen **testing** data to report its accuracy fairly. We find: Overall error: 14.3%, Averaged class error: 18.25%

While Rattle conveniently supports this 3-partitions approach to evaluation, please be aware that other cross-validation approaches as discussed in the course notes are widely used.

**Q 5.3:**

Overall error: 13%, Averaged class error: 23.4%

**Q 5.4:**

Area under the curve = 0.67. The chart shows true positive rate (TP/P = sensitivity = recall) versus false positive rate (FP/N = 1- specificity), when the data points are ordered by *decreasing probability* of being true, according to the model, from left to right.

**![5-4](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270814823.png)**

**Q 5.6:**

Overall error: 14.8%, Averaged class error: 20.7%

**Q 5.7**

**![5-7](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270815286.png)**



In terms of the overall error the first model performs better than the second model, however, in terms of the averaged class error and AUC score, the second model performs better than the first model. If you look at the distribution of the RainTomorrow variable, you see the distribution of the variable is highly skewed (Yes=41, No=215). In general, to measure the performance of class imbalanced classification (but only if we are using a probablistic classifier) we prefer to use ROC and averaged class error since, for this problem, it is more important to identify a rare class than a frequent class (because people tend to be more angry when they get caught in the rain unprepared.).

Again the choice of evaluation measure is subjective, and someone may have a different opinion. **You need to choose an evaluation measure that is aligned with your understanding of the problem.**

**Q6.6-7**

In my case:

15% training 15% testing: Overall error: 28.5%, Averaged class error: 48.7%

30% training 15% testing: Overall error: 23.2%, Averaged class error: 48.8%

45% training 15% testing: Overall error: 23.2%, Averaged class error: 40.65%

60% training 15% testing: Overall error: 21.4%, Averaged class error: 30.55%

75% training 15% testing: Overall error: 16%, Averaged class error: 24.4%

In general, the performance of your model increases as you increase the training proportion. In this example, we cannot find any saturation in performance. In other words, the performance increases as we increase the training proportion. However, if we are allowed to have a larger dataset, we may observe a point where adding more training data does not increase the model performance.