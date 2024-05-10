# Classification & Prediction: Linear Regression & Neural Nets - 7.1

## 1. Introduction

Most of this material is derived from the text, Han, Kamber and Pei, Chapter 9, or the corresponding powerpoint slides made available by the publisher. Where a source other than the text or its slides was used for the material, attribution is given. Unless otherwise stated, images are copyright of the publisher, Elsevier.

Here we continue the topic of classification and prediction, firstly introducing the classic technique of regression, or fitting a line, and then the bio-inspired technique of neural networks and a brief overview of some other "soft" techniques.

For in-depth learning on neural nets and other soft techniques, the courses *Neural Networks, Deep Learning and Bio-inspired Computing* [COMP8420](http://programsandcourses.anu.edu.au/course/COMP8420) and [COMP4660 ](http://programsandcourses.anu.edu.au/course/COMP4660 )are recommended.



**ACTION:** If you would like more information on the applications for neural networks, refer to this page https://www.smartsheet.com/neural-network-applications#real-world-and-industry-applications-of-neural-networks or watch this youtube video.

<iframe src="https://www.youtube.com/embed/V6qF33W2_bs" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" width="560" height="315" frameborder="0" style="box-sizing: border-box;"></iframe>

**ACTION:** You can watch this video lecture overview of the classsification and prediction topic if you find it helpful but all it covers is also in the written notes. Note that this recording covers topics in this e-book as well as topics to be addressed in the subsequent e-book.

[Prerecorded lecture on Regression, NN, SVM, LL](https://wattlecourses.anu.edu.au/mod/page/view.php?id=2472094)



## 2. Linear Regression (Not in textbook)

**Linear Regression**

- models relationship between one *dependent variable* (numerical target variable) and *explanatory variables* (other variables)

- - Input: independent, explanatory, predictor variables
  - Output: one dependent, target, response variable

- provides a way to **predict a numerical variable** given other variables. The other variables are typically numeric too. Ordinal variables can be mapped to numeric values if required. Other categorical variables can also be mapped to numeric values but be very careful to assess the effect of such variables on results as they may obscure more informative relationships.




Regression is different from classification

- Classification predicts categorical class labels
- Regression models continuous-valued functions and is a *numerical prediction* method because the values for explanatory variables of an unseen object can be plugged in to the model to predict the value of the dependent variable. 



**Types of relationships considered in regression**

**Deterministic relationship**

- Relationship between explanatory and dependent variables is deterministic
- If we know the value of an explanatory variable then we can predict dependent variable without error
- For example:
  - [![Circumference = \pi \times diameter](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830052.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Circumference %3D \pi \times diameter)
  - [![Fahrenheit = \frac{9}{5} \times Celsius + 32](https://wattlecourses.anu.edu.au/filter/tex/pix.php/bae23626dfd20b4568934a5a9efa2651.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=Fahrenheit %3D \frac{9}{5} \times Celsius %2B 32)

**Statistical relationship**

- Relationship between explanatory and dependent variable is not perfect
- For example:
  - Height and weight: as height increases, you'd expect weight to increase, but not perfectly
- Another example of a statistical relationship:
  ![img](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270834480.png)
  - Dependent variable is the mortality due to skin cancer (number of deaths per 10 million people)
  - Explanatory variable is the latitude of State (measured at the centre of the State in US)
  - Living in a northern area may reduce the mortality ratio but the relationship is not perfect.



### 2.1. Simple Linear Regression

Simple linear regression is a statistical method that allows us to study relationships between two continuous variables:

- One variable, denoted [![x](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830325.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x), is regarded as the **predictor**, explanatory, or independent variable
- The other variable, denoted [![y](https://wattlecourses.anu.edu.au/filter/tex/pix.php/415290769594460e2e485922904f345d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y), is regarded as the **response**, outcome, target, or dependent variable
- The goal is to predict the value of response variable when the value of predictor variable is given

Simple linear regression gets its adjective "simple," because it concerns the study of **only one predictor variable**. In contrast, multiple linear regression concerns the the general case of two or more predictor variables.



**Linear Regression Model**

Simple linear regression models the relation between predictor and response variables as follows:

[![\hat{y} = w_0 + w_1\times x](https://wattlecourses.anu.edu.au/filter/tex/pix.php/91f3c2809e1c09bc0b90aa2dae235ed3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\hat{y} %3D w_0 %2B w_1\times x)

where [![w_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ac1052c8c41fa0e8d67714e0723a068b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0) (**<u>*y-intercept*</u>**) and [![w_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/db007d6a923c2909d42c4292bffca5f0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_1) (<u>***slope***</u>) are **<u>*coefficient parameters*</u>** and [![\hat{y}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830824.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\hat{y}) is a predicted response variable based on the predictor variable [![x](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830325.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x). *If the value of coefficient parameter is positive then there is a positive relation between predictor and response variable* (the value of response variable increases as the value of predictor variable increases).

The relationship between the two variables can be plotted in x-y plane as follows:

![img](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270835190.jpeg)



This scatter plot (dots) represent an observed pair of weight and height. Here the height is the predictor variable, and the weight is the response variable. In addition, two linear relations are suggested in this graph. You may have two natural questions from above graph:

- which line is the best line? and
- how can we find it?





**What is the Best Fitting Line?**

Let's assume that we have a training data [![D = \{(x_1, y_1), (x_2,y_2), ... (x_n,y_n)\}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3115f8fae88e5479675aed77acf0c04e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D %3D \{(x_1%2C y_1)%2C (x_2%2Cy_2)%2C ... (x_n%2Cy_n)\}) which consists of n-pairs of [![x](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830325.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x) and [![y](https://wattlecourses.anu.edu.au/filter/tex/pix.php/415290769594460e2e485922904f345d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y). 

The goal of parameter estimation is to estimate proper values of [![w_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ac1052c8c41fa0e8d67714e0723a068b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0) and [![w_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/db007d6a923c2909d42c4292bffca5f0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_1). To estimate parameters, we first define **<u>*the sum of square error*</u>**:

[![E = \sum_{i=1}^{|D|} ( y_i - \hat{y}_i)^2 =\sum_{i=1}^{n} ( y_i - (w_0 + w_1\times x_i))^2](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830155.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E %3D \sum_{i%3D1}^{|D|} ( y_i - \hat{y}_i)^2 %3D\sum_{i%3D1}^{n} ( y_i - (w_0 %2B w_1\times x_i))^2),

where [![\hat{y}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830824.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\hat{y}) is the predicted response variable. The sum of square error computes, first, the difference between the true value [![y](https://wattlecourses.anu.edu.au/filter/tex/pix.php/415290769594460e2e485922904f345d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=y) and the predicted value [![\hat{y}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/5d28a7ba1a44a73b8c2ed21321697c59.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\hat{y}), second, square of the difference, and then summation of the squared difference across all training data points.

Parameter estimation finds the values for [![w_0](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ac1052c8c41fa0e8d67714e0723a068b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0) and [![w_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/db007d6a923c2909d42c4292bffca5f0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_1) that minimise [![E](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3a3ea00cfc35332cedf6e5e9a32e94da.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E) and so this technique is called the **least square method**.

**<u>*Due to the effect of squaring the differences above, linear regression is very senstitive to outliers*</u>** that fall well away from the line--that is the whole fitted line will be pulled away from the non-outlying values.



**How to evaluate performance?**

The parameter estimation technique suggests an evaluation technique. Accuracy does not make sense here as there are no classes; accuracy requires the independent variable value to fall either inside the class or outside the class.

The usual methods for assessing performance are

- **MAE, Mean Absolute Error.** It is the average of the individual errors, each one computed as the absolute value of the difference between the actual value and the predicted value. MAE of zero is a perfect fit. As there is no squared term **<u>*in MAE, it does not penalise long-distance outliers as much as the regression fitting does.*</u>**
- **RMSE, RMSD, root mean square error.** **<u>*It is the square root of the mean of squared errors*</u>** for each value of the dependent variable, where error is the simple difference of the actual value and predicted value. It is equal to the standard deviation of the errors, and so tells you how spread out from the prediction line the actual observations are. **<u>An RMSE value of zero represents a perfect fit and there is no upper bound.</u>**
- **R-squared**: the goodness of fit or coefficient of determination. It is the proportion of the dependent variable variance that is explained by a linear model. R-squared = Explained variance / Total variance. More precisely, it is 1 minus (the ratio of the sum of squared prediction error for each value of the dependent variable to the sum of squared difference from the mean for each value of the dependent variable). <u>**A value of zero says the model is only just as good as simply predicting the mean value every time, and a value of one represents a perfect fit.**</u> Minor variations include pseudo-R-squared methods and adjusted or predicted R-squared. ![image-20220428下午50332527](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204281703092.png)
- **Plot** the predictions vs observations on a Cartesian plane and visually assess if the points appear to line up. It is a good idea to *do this every time* for linear regression, in combination with quantitative measures. **<u>*Are the points close enough to the theoretical prediction = observation line?*</u>** Do they fall somewhat symmetrically around that line? Are there too many points a long way off? Would a line fitted to the predictions vs observations fall close to the *prediction = observation* line? If it doesn't look good, it probably isn't.

Obviously these approaches should be applied to **<u>*test data*</u>** rather than training data for a proper assessment, and indeed **multifold cross-validation** may be applied using the quantitative performance measures above in place of accuracy.

### 2.2. Multiple Linear Regression

In many applications, it is more general to have more than one predictor variable.

Multiple linear regression is a direct extension of the linear regression model. 

Let's assume that we have [![n](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7b8b965ad4bca0e41ab51de7b31363a1.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=n) predictor variables for each response variable, i.e., [![(x_1,x_2,...,x_n, y)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830329.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=(x_1%2Cx_2%2C...%2Cx_n%2C y)).

In the multiple linear regression model, the relation between a response variable and predictor variables are modelled as:

[![\hat{y} = w_0 + w_1\times x_1 + w_2 \times x_2 ... + w_n \times x_n](https://wattlecourses.anu.edu.au/filter/tex/pix.php/595626ed8ae7fde6fb4f61994258258b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\hat{y} %3D w_0 %2B w_1\times x_1 %2B w_2 \times x_2 ... %2B w_n \times x_n)

Again, parameters [![w_0,.., w_n](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830665.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_0%2C..%2C w_n) can be estimated in least square fashion:

[![E = \sum_{i=1}^{|D|} (y_i - \hat{y}_i)^2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6dfd1c88bc28655dfbcd4caaace42c4b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E %3D \sum_{i%3D1}^{|D|} (y_i - \hat{y}_i)^2)

A least square method will find optimal parameters to minimise the squared error [![E](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3a3ea00cfc35332cedf6e5e9a32e94da.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E).



**Significance of the weight**

Most regression libraries or programs support significance testing for each weight. Usually, significant weights are denoted by \*** (or ** or * depending on the significance level).

The significant attributes indicate that the effect (or influence) of the attribute is not just random! In other words, those significant **attributes have high influence on predicting your response variable**. This alone may be valuable information, even if you discard the linear model itself.



**Performance evaluation**

The methods for evaluation of simple linear regression apply straightforwardly to the multiple variable case.

## 3. Classification or Prediction by Neural Networks (Text: 9.2)

**Neural Network Learning**

- Neural networks were first investigated by psychologists and neurobiologists to develop and test computational analogues of neurons
- A neural network: a set of connected nodes where each node is an *input* (corresponding to an independent variable), *output* (corresponding to a dependent variable for numeric prediction or a class for classification; there may be several output nodes) or *hidden* and where each connection has a **weight** associated with it.
- During the learning phase, the **network learns by adjusting the weights** so as to be able to output the correct class label or predicted value for the input tuples.
- Also referred to as ***<u>connectionist learning</u>*** due to the connection between units
- The classic training algorithm is called *backpropagation.*
- *Convolutional Neural Networks* (CNNs) are neural networks where there are specific roles and connection architectures for hidden nodes and these have been applied very sucessfuly to image recognition and other problems. 
- The term *deep learning* often refers to training CNNs. They are deep because there are many layers of hidden nodes.



**Strength/Weakness of neural network as a classifier**

**Strength**

- High tolerance to noisy data
- Ability to classify untrained patterns
- Well-suited for continuous-valued inputs and outputs
- Successful on an array of real-world data, e.g., hand-written letters, images, voice
- Algorithms are inherently parallel
- Techniques have recently been developed for the extraction of rules from trained neural networks

**Weakness**

- *Long training time*
- Require ***<u>a number of parameters</u>*** typically best determined empirically, e.g., network topology or structure
- Poor interpretability: ***<u>difficult to interpret the symbolic meaning behind</u>*** the learned weights and of hidden units in network, that is, behaves as a "black box" model.



### 3.1. Multi-Layer Feed-Forward Neural Network

![Multilayer feed-forward neural network](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270836366.png)



**Structure of Multilayer neural network**

- The **inputs** to the network correspond to the attributes measured for each training tuple
- Inputs are fed simultaneously into the units making up the **input layer**
- They are then weighted and fed simultaneously to a **hidden layer**
  - The number of hidden layers is arbitrary (1 hidden layer in the example above).
  - The number of hidden units is arbitrary (3 hidden units in the example above).
- The weighted outputs of the last hidden layer are input to units making up the **output layer**, which emits the network's prediction
- The network is **feed-forward**: none of the weights cycles back to an input unit or to an output unit of a previous layer
- From a statistical point of view, networks perform **nonlinear regression**: *given enough hidden units and enough training samples, they can closely approximate any function* (output)



**Defining a network topology**

- Decide the **network topology**:
  - Specify the number of units in the input layer: usually *one input unit per attribute* in the data (but *nominal* attributes can have one input per value).
  - the number of hidden layers (at least one, and commonly only one)
    - the number of units in each hidden layer
  - and the number of units in the output layer
- **Output**, one unit per response variable. In a typical binary classification problem only one output unit is used and a threshold is applied to the output value to select the class label. The value can be interpreted as a probability of belonging to the positive class, and in this way a neural network can be used as a *probablistic model*. For classification of more than two classes, one output unit per class is used and the highest-valued class is selected for the label.
- Choose an **activation function** for each hidden and output unit (explained later)
- Usually *randomly*, determine initial values for the weights.
- Once a network has been trained, if its accuracy is unacceptable, try a different network topology or a different set of initial weights or maybe different activation functions. Use trial-and-error or there are methods that systematically search for a high-performing topology. 



### 3.2. Prediction with Neural Network

![Multilayer neural network](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270837652.png)

We first discuss how to predict an output variable using a feed-forward neural network model *that has already been trained*.

**1. Initialise the weights by training
**

- The weights in the network are set to small random numbers (e.g., ranging from −1.0 to 1.0, or −0.5 to 0.5) by training.
- Each unit has a bias associated with it, as explained later. The biases are similarly set to small random numbers by training.

Each testing tuple, is first *normalised* to [0.0 ~ 1.0]. Consider a normalised tuple [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X), processed by the following steps.

**2. Propagate the inputs forward**

1. First, the testing tuple is *normalised* to [0.0 ~ 1.0] and the normalised tuple [![X](https://wattlecourses.anu.edu.au/filter/tex/pix.php/02129bb861061d1a052c592e2dc6b383.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=X) is fed to the network’s input layer. The inputs pass through the input units, unchanged. 
2. Hidden layer
   - Input of hidden unit: all outputs of the previous layer, e.g. if the hidden unit is in the first hidden layer, then the inputs are [![x_1, x_2, ... , x_n](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830879.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x_1%2C x_2%2C ... %2C x_n).
   - Output of hidden unit [![j](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830139.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j): weighted linear combination of its input followed by an activation function
     - [![O_j = f(\sum_{i=1}^{n}w_{ij}\cdot x_i + \theta_j)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/2ab05edf0bd7d99eb89fbe8ccbea049f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O_j %3D f(\sum_{i%3D1}^{n}w_{ij}\cdot x_i %2B \theta_j))
     - where [![w_{ij}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/322b227441558f0ed48aba836d0caf6f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_{ij}) is the weight of the connection from input [![i](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830365.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=i) in the previous layer to unit [![j](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830139.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j)
     - [![\theta_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/cb0e17d96e58d55d1eb06dc1b14b7a7b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\theta_j) is the bias variable of unit [![j](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830139.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j)
     - [![f(\cdot)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830645.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f(\cdot)) is a non-linear activation function which will be described later.
   - If there is more than one hidden layer, the above procedure will be repeated until the final hidden layer will result outputs.
3. Output layer
   - The outputs of the final hidden layer will be used as inputs of the output layer.
   - The number of output units are determined by a task
     - If our goal is to predict a single numerical variable, then one output unit is enough
   - Final output [![O_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/993186a770ae8b3ab8f2619c125ba298.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O_k):
     - [![O_k = \sum_{j=1}^{h}w_{jk}\cdot O_j + \theta_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1a319af585c5b508be52d401dbfb9794.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O_k %3D \sum_{j%3D1}^{h}w_{jk}\cdot O_j %2B \theta_k) or [![O_k = f(\sum_{j=1}^{h}w_{jk}\cdot O_j + \theta_k)](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830857.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O_k %3D f(\sum_{j%3D1}^{h}w_{jk}\cdot O_j %2B \theta_k))
     - [![O_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/993186a770ae8b3ab8f2619c125ba298.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O_k) is the final predicted value given [![x_1, x_2,...,x_n](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e2e8149aea2cc391bb69364f6723c5bc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=x_1%2C x_2%2C...%2Cx_n).

![computation graph of hidden unit j](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270837972.png)

\> Graphical illustration of the computational flow of hidden unit [![j](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830139.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j): The inputs to unit [![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j) are outputs from the previous layer. These are multiplied by their corresponding weights to form a weighted sum, which is added to the bias [![\theta_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/cb0e17d96e58d55d1eb06dc1b14b7a7b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\theta_j) associated with unit [![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j). A nonlinear activation function [![f](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8fa14cdd754f91cc6554c9e71929cce7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f) is applied to the net input. Again the output of this unit will be used as an input of the successive layer.



**Non-linear activation function**

An activation function imposes a certain non-linearity in a neural network. There are several possible choices of activation functions, but **sigmoid** (or logistic) function is the most widely used activation function.

- Sigmoid function
  - Sigmoid functions have domain of all real numbers, with return value monotonically increasing most often from 0 to 1. 
  - [![f(x) = \frac{1}{1+e^{-x}}](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830162.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=f(x) %3D \frac{1}{1%2Be^{-x}})
    **![Sigmod function](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270837872.png)**
  - Also referred to as a *squashing function*, because it maps the input domain onto the range of 0 to 1
- Other activation functions
  - Hyperbolic tangent function (tanh)
  - Softmax function (use for output nodes for classification problems to push the output value towards binary 0 or 1)
  - ReLU
  - etc.



### 3.3. Training a Neural Network

**Backpropagation**



The backpropagation algorithm, based on the **gradient descent** algorithm, is designed to *train the weights and biases of a neural network*.



**How does backpropagation work?**

For training, each training tuple,is first *normalised* to [0.0 ~ 1.0].**
**

The error is **propagated backward** by **updating the weights and biases** to reflect the error of the network’s prediction for the dependent variable for a training tuple.

- For unit [![k](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830362.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k) in the output layer, its error [![E_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/08f5af0d3874b9d9388108ed70fa0102.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E_k) is computed by
  - [![E_k = O_k(1-O_k)(T_k-O_k)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/5aa2003f72bf27eb676ef9c0614d41fb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E_k %3D O_k(1-O_k)(T_k-O_k)),
  - where [![O_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/993186a770ae8b3ab8f2619c125ba298.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O_k) is the actual output of unit [![k](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830362.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k), and [![T_k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8b0f55b15c05710c97a6c3df6649d3a7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=T_k) is the known target value of the given training tuple.
- To compute the error of a hidden layer unit [![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j), the weighted sum of the errors of the units connected to unit [![k](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830362.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k) in the next layer are considered.
  - [![E_j = O_j(1-O_j) \sum_k E_kw_{jk},](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830858.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E_j %3D O_j(1-O_j) \sum_k E_kw_{jk}%2C)
  - where [![w_{jk}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a87b437b0c2403b57f841d01c1b1b94f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_{jk}) is the weight of the connection from unit [![j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/363b122c528f54df4a0446b6bab05515.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=j) to a unit [![k](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830362.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k) in the next higher layer.
- The weights and biases are **updated to reflect the propagated errors**.
  - Each weight [![w_{jk}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a87b437b0c2403b57f841d01c1b1b94f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_{jk}) in the network is updated by the following equations, where [![\Delta w_{jk}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/5ac2f85a14b70294aed293d2bae5a8c6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\Delta w_{jk}) is the change in weight [![w_{jk}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a87b437b0c2403b57f841d01c1b1b94f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_{jk}):
    - [![\Delta w_{jk} =\ell \times E_kO_j](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830134.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\Delta w_{jk} %3D\ell \times E_kO_j)
    - [![w_{jk} = w_{jk} + \Delta w_{jk}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c613978ae099cb2eb9f22b3cbc7199f4.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=w_{jk} %3D w_{jk} %2B \Delta w_{jk})
  - Each bias [![\theta_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/cb0e17d96e58d55d1eb06dc1b14b7a7b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\theta_j) in the network is updated by the following equations, where [![\Delta\theta_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/352bc1f32c9de0ec5cd22189c3d832cd.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\Delta\theta_j) is the change in bias [![\theta_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/cb0e17d96e58d55d1eb06dc1b14b7a7b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\theta_j):
    - [![\Delta\theta_j = \ell \times E_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3b3b95331c376279c6454fbc89665100.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\Delta\theta_j %3D \ell \times E_j) 
    - [![\theta_j = \theta_j + \Delta\theta_j](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830520.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\theta_j %3D \theta_j %2B \Delta\theta_j)


**Learning rate**: The parameter [![\ell](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ee5e5c003694e7cd5ae404923c665edb.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\ell) is the learning rate, typically, a value between 0.0 and 1.0. If the learning rate is too small, then learning (ie reduction in the Error) will occur at a very slow pace. If the learning rate is too large, then oscillation between inadequate solutions may occur. A rule of thumb is to set the learning rate to [![1/t](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8a8b4e979e6a0c57ef2282f0a8d3c3d8.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=1%2Ft), where [![t](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830710.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=t) is the number of iterations through the training set so far. In principle the network should *converge* towards a minimal Error, but if this is not happening, try reducing this parameter value and retrain.



**Updating schedules**

- **Case updating**: The backpropagation algorithm given here updates the weights and biases immediately after the presentation of each tuple.
- **Epoch updating**: Alternatively, the weight and bias increments could be accumulated in variables, so that the weights and biases are updated after all the tuples in the training set have been presented.
- Either way, we keep updating until we have a satisfactory error; presenting each tuple of the training set many times over
- In theory, the mathematical derivation of backpropagation employs epoch updating, yet in practice, case updating is more common because it tends to yield more accurate results.

![Back propagation algorithm from textbook](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270838051.png)

\> Backpropagation algorithm from the textbook

**How do you decide to terminate training**?

- When your weights and biases are changing very little , ie. all the ![](file:///Users/guangmingzeng/Downloads/Classification%20&%20Prediction_%20Linear%20Regression%20&%20Neural%20Nets_files/e2de579d082cac364b363458bde14910.png)s are small; or
- Accuracy on the training set is good enough; or
- A prespecified number of epochs have passed.

### 3.4. Performance Evaluation



**Objective performance evaluation
**

For numerical prediction tasks, the methods used for evaluating linear regression may also be used for evaluating neural nets.

For classification tasks, any of the methods used for evaluating classification tasks may also be used for neural nets.

In addition, as a neural net can be used as a probablistic classifier, methods for probablistic classifiers, such as ROC charts may be used.



**Interpretability of neural network**

*"Neural networks are like a black box. How can I ‘understand’ what the backpropagation network has learned?"*

- A major disadvantage of neural networks lies in their knowledge representation.
- Acquired knowledge in the form of a network of units connected by weighted links is difficult for humans to interpret.
- Easier-to-interpret rules may be extracted from the network by pruning or by sensitivity analysis.

**Rule extraction by \**network pruning for interpretability
\****

- Simplify the network structure by removing weighted links that have the least effect on the trained network
- The set of input and activation values are studied to derive rules describing the relationship between the input and hidden unit layers

**Sensitivity analysis for interpretability
**

- Assess the impact that a given input variable has on a network output.
- The input to the variable is varied while the remaining input variables are fixed at some value.
- The knowledge gained from this analysis can be represented in rules
  - Such as “IF X decreases 5% THEN Y increases 8%.”





### 3.5. Other Neural Net Architectures (Not in text)

There are many adaptations of the basic Neural Network architecture presented here.

Here is a convenient graphical summary from [http://www.asimovinstitute.org/neural-network-zoo/](http://www.asimovinstitute.org/neural-network-zoo/ )

![neuralnetworks from http://www.asimovinstitute.org/neural-network-zoo/ ](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830749.png)

## 4. Other "Soft" Classification Algorithms (Text 9.6)

**Genetic Algorithm**

- Genetic Algorithm: based on an analogy to biological evolution
- An initial population [![P](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830769.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P) is created consisting of randomly generated rules
  - Each rule is represented by a string of bits
  - E.g., Consider Boolean (binary) attributes A1 and A2 to learn (binary) class C. The rule "if A1 and not A2 then not C" can be encoded as (binary string) "100" reading from the left. Similarly, the rule "if not A1 and not A2 the N" is encoded as "001".
  - If an attribute has k > 2 values, k bits can be used
- Based on the notion of survival of the fittest, a new population is formed to consist of the fittest rules and their offspring
  - The fitness of a rule is represented by its classification accuracy on a set of training examples
- Offspring are generated by crossover and mutation
  - **Crossover**: sub-strings from pairs of rules are swapped to form new pairs of rules
  - **Mutation**: randomly selected bits in a rule's string are inverted
- The process continues until a population [![P](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270830769.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P) evolves when each rule in [![P](https://wattlecourses.anu.edu.au/filter/tex/pix.php/44c29edb103a2872f519ad0c9a0fdaaa.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=P) satisfies a prespecified threshold
- Slow but easily parallelisable

 

**Fuzzy-Set Approach**

- Fuzzy logic uses truth values between 0.0 and 1.0 to represent the degree of membership (such as in a fuzzy membership graph)
- Attribute values are converted to fuzzy values:
  - Income is assigned a fuzzy membership value to each of the discrete categories {low, medium, high}, without fuzzy logic $49K income will be categorized as "medium income"
  - In fuzzy logic:
    - $49K belongs to “medium income” with fuzzy value 0.15
    - but belongs to “high income” with fuzzy value 0.96
      ![Fuzzy truth values for income](https://cdn.jsdelivr.net/gh/California3/Imagelib/default/202204270840309.png)
  - Fuzzy membership values do not have to sum to 1.
- Each applicable rule contributes a vote for membership in the categories
- Typically, the truth values for each predicted category are summed, and these sums are combined

ACTION: You can watch Pouya's lecture on this booklet content. He walks you through this e-book if you find it helpful but all it covers is also in the written notes.



## 5. Practical Exercises: Neural Network and Linear Regression in R

**ACTION:** Attempt these practical exercises with Rattle. There is a video showing the mechanics to get you started, and written instructions for you to work through.

<iframe src="https://www.youtube.com/embed/CvMJ2dfaTmg" title="YouTube video player" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" width="560" height="315" frameborder="0" style="box-sizing: border-box;"></iframe>



[![img](https://wattlecourses.anu.edu.au/theme/image.php/anu/page/1617192474/icon)Neural Networks and Linear Regression in R](https://wattlecourses.anu.edu.au/mod/page/view.php?id=2472097)