## Cluster Analysis

### 1. Introduction (Text:10.1)

Most of this material is derived from the text, Han, Kamber and Pei, Chapter 10, or the corresponding powerpoint slides made available by the publisher. Where a source other than the text or its slides was used for the material, attribution is given. Unless otherwise stated, images are copyright of the publisher, Elsevier. 

Clustering is the process of grouping a set of data objects into multiple groups or clusters so that objects within a cluster have high similarity, but are very dissimilar to objects in other clusters. Clustering is usually used to understand the structure of a dataset, to inform more in-depth analysis and understanding later. 

**ACTION:** Check out this video if you would like to see an example of how clustering might be applied to solve the problem of customer segmentation. 



### 2. Clustering: Basic Concepts

**What is Cluster Analysis?**



- Cluster: A collection of data objects

  - similar (or related) to one another within the same group
  - dissimilar (or unrelated) to the objects in other groups

- Cluster analysis (or clustering, data segmentation, …)

  - Finding similarities between data according to the characteristics found in the data and grouping similar data objects into clusters; discovering groups within the data

- **Unsupervised learning**: no predefined classes

- Typical applications

- - As a **stand-alone tool** to get insight into data distribution 
  - As a **preprocessing step** for other algorithms



**Clustering as a Preprocessing Tool**

- Summarisation: 
  - Preprocessing for regression, principal components analysis, classification, and association analysis
- Compression:
  - Image processing: vector quantisation
- Finding K-nearest Neighbours
  - Localising search to one or a small number of clusters
- [Outlier detection](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472118)
  - Outliers are often viewed as those “far away” from any cluster



**Clustering for Data Understanding and Applications**

- Biology: taxonomy of living things: kingdom, phylum, class, order, family, genus and species
- Information retrieval: document clustering
- Land use: Identification of areas of similar land use in an earth observation database
- Marketing: Help marketers discover distinct groups in their customer bases, and then use this knowledge to develop targeted marketing programs
- City-planning: Identifying groups of houses according to their house type, value, and geographical location
- Earth-quake studies: Observed earth quake epicenters should be clustered along continent faults
- Climate: understanding earth climate, find patterns of atmospheric and ocean
- Economic Science: market research

#### 2.1. Quality of Clustering

**What is Good Clustering?**

- A good clustering method will produce high quality clusters

  - high intra-class similarity: cohesive within clusters
  - low inter-class similarity: distinctive between clusters

- The quality of a clustering method depends on

- - the similarity measure used by the method 
  - its implementation, and
  - Its ability to discover some or all of the hidden patterns



**Measure the Quality of Clustering**

- Dissimilarity/Similarity metric
  - Similarity is expressed in terms of a **distance function**, typically metric: d(i, j)
  - The definitions of distance functions are usually rather different for interval-scaled, boolean, categorical, ordinal ratio, and vector variables
  - Weights should be associated with different variables based on applications and data semantics
- Quality of clustering:
  - There is usually a separate "quality" function that measures the "goodness" of a cluster.
  - It is hard to define "similar enough" or "good enough"
    - The answer is typically highly subjective

#### 2.2. Considerations

**Algorithmic Considerations**

- Partitioning criteria

  - Single level vs. hierarchical partitioning (often, multi-level hierarchical partitioning is desirable)

- Separation of clusters

  - Exclusive (e.g., one customer belongs to only one region) vs. non-exclusive (e.g., one document may belong to more than one class)

- Similarity measure

  - Distance-based (e.g., Euclidean, road network, vector) vs. connectivity-based (e.g., density or contiguity)

- Clustering space

- - Full space (often when low dimensional) vs. subspaces (often in high-dimensional clustering)



**Requirements and Challenges**



- Scalability

  - Clustering all the data instead of only on samples

- Ability to deal with different types of attributes

  - Numerical, binary, categorical, ordinal, linked, and mixture of these 

- Constraint-based clustering

  - User may give inputs on constraints
  - Use domain knowledge to determine input parameters

- Interpretability and usability

- Others 

- - Discovery of clusters with arbitrary shape
  - Ability to deal with noisy data
  - Incremental clustering and insensitivity to input order
  - High dimensionality

#### 2.3. Major Approaches

Based on different approaches we can categorise known clustering algorithms into:

- Partitioning approach: 

  - Construct various partitions and then evaluate them by some criterion, e.g., minimising the sum of square errors
  - Typical methods: k-means, k-medoids, CLARANS

- Hierarchical approach: 

  - Create a hierarchical decomposition of the set of data (or objects) using some criterion
  - Typical methods: Diana, Agnes, BIRCH, CAMELEON

- Density-based approach: 

  - Based on connectivity and density functions
  - Typical methods: DBSCAN (Density-based spatial clustering of applications with noise), OPTICS, DenClue

- Grid-based approach: 

- - based on a multiple-level granularity structure
  - Typical methods: STING, WaveCluster, CLIQUE

- Model-based: 

  - A model is hypothesised for each of the clusters and tries to find the best fit of that model to each other
  - Typical methods: EM, SOM, COBWEB

- Frequent pattern-based:

  - Based on the analysis of frequent patterns
  - Typical methods: p-Cluster

- User-guided or constraint-based: 

  - Clustering by considering user-specified or application-specific constraints
  - Typical methods: COD (obstacles), constrained clustering

- Link-based clustering:

- - Objects are often linked together in various ways
  - Massive links can be used to cluster objects: SimRank, LinkClus

We will discuss some of major approaches in detail in the following.

### 3. Partitioning Methods (K-means) (Text:10.2)

**Partitioning method**

Partitioning a database [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) of [![n](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7b8b965ad4bca0e41ab51de7b31363a1.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=n) objects into a set of [![k](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/8ce4b16b22b58894aa86c421e8759df3_No.00033516514138151651413815133.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k) clusters. The quality of cluster [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i) can be measured by the within-cluster variation, which is the sum of squared distances between all objects in [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i) and the centroid [![c_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/96fafac0c054b9eb47d3f630ed02c289.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=c_i), defined as:

[![E = \sum_{i=1}^{k}\sum_{p\in C_i}(p-c_i)^2](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/fec260a07f423724f63cd8d6ae3e699c_No.00033516514138151651413815364.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E %3D \sum_{i%3D1}^{k}\sum_{p\in C_i}(p-c_i)^2)



- Given [![k](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/8ce4b16b22b58894aa86c421e8759df3_No.00033516514138151651413815133.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k), find a partition of [![k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8ce4b16b22b58894aa86c421e8759df3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k) clusters that optimises the chosen partitioning criterion

- - Globally optimal: exhaustively enumerate all partitions
  - Heuristic methods: k-means and k-medoids algorithms
  - k-means: Each cluster is represented by the centre of the cluster
  - k-medoids or PAM (Partition around medoids): Each cluster is represented by one of the objects in the cluster



**K-means**



- Given [![k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8ce4b16b22b58894aa86c421e8759df3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k), the k-means algorithm is implemented in four steps:

- 1. Arbitrarily choose a centre of [![k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8ce4b16b22b58894aa86c421e8759df3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k) clusters as the initial cluster centres
  2. Assign each object to the cluster to which the object is the most similar
  3. Update the cluster means, that is, calculate the mean value of the objects for each cluster
  4. Go back to Step 2, stop when the assignment does not change

 

**Illustration of the K-means algorithm** (from Pattern Recognition & Machine Learning, Bishop). 

![image](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/I4Kab4_No.00063316514139931651413993591.jpg)



(a) Green points denote the data set in a two-dimensional Euclidean space. The initial choices for centres for [![C_1](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/4fa71d007c094ac3c858919aec515277_No.00033516514138151651413815517.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_1) and [![C_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f0350e5818b058dbcfd95f155e417f6a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_2) are shown by the red and blue crosses, respectively. 

(b) Each data point is assigned either to the red cluster or to the blue cluster, according to which cluster centre is nearer. This is equivalent to classifying the points according to which side of the perpendicular bisector of the two cluster centres, shown by the magenta line, they lie on. 

(c) In the subsequent step, each cluster centre is re-computed to be the mean of the points assigned to the corresponding cluster. 

(d)–(i) show successive steps through to final convergence of the algorithm.

### 3. Partitioning Methods (K-means) (Text:10.2)

#### 3.1. Strength and Weakness

- **Strength**: 
  - Computationally Efficient: time complexity is [![O(tkn)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/380fe54d483e3f6df5c28453280be5d8.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O(tkn)), where [![n](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/7b8b965ad4bca0e41ab51de7b31363a1-20220502%25E4%25B8%258A%25E5%258D%2588120943657_No.00094316514141831651414183727.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=n) is the number of objects, [![k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8ce4b16b22b58894aa86c421e8759df3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k) is the number of clusters, and [![t](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e358efa489f58062f10dd7316b65649e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=t) is the number of iterations. Normally, [![k, t ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c95d3de59bddc3a75cceb3e9edc39ca6.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k%2C t ).
    - Comparing: PAM: [![O(k(n-k)^2 )](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/4667c2d5e08f762abace2493995adc37-20220502%25E4%25B8%258A%25E5%258D%2588120944084_No.00094416514141841651414184158.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O(k(n-k)^2 )), CLARA: [![O(ks^2 + k(n-k))](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/9119b7a711cfad1e3e9a4b74769140e0-20220502%25E4%25B8%258A%25E5%258D%2588120944127_No.00094416514141841651414184202.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=O(ks^2 %2B k(n-k)))
- **Weakness**
  - Need to specify k, the number of clusters, in advance
  - Applicable only to objects in a continuous n-dimensional space 
    - k-modes variant method for categorical data: replaces the mean value by the *mode* of a nominal attribute.
    - In comparison, k-medoids can be applied to a wide range of data
  - Sensitive to noisy data and outliers
  - Non deterministic algorithm. The final result depends on the first initialisation.
  - Often terminates at a local optimum, rather than a global optimum.
  - Not suitable for clusters with non-convex shapes



**Noisy data point example**

Consider six points in 1-D space having the values 1,2,3,8,9,10, and 25, respectively. Intuitively, by visual inspection we may imagine the points partitioned into the clusters {1,2,3} and {8,9,10}, **where point 25 is excluded** because it appears to be an **outlier**. How would k-means partition the values? If we apply k-means using k = 2, 

- Case 1: partition values into {{1, 2, 3}, {8, 9, 10, 25}}. Within-cluster variation is
  [![(1-2)^2 +(2-2)^2 +(3-2)^2 +(8-13)^2 +(9-13)^2 +(10-13)^2 +(25-13)^2=196](https://wattlecourses.anu.edu.au/filter/tex/pix.php/41c1a3b8cae54995a4e6db205bc124ab.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=(1-2)^2 %2B(2-2)^2 %2B(3-2)^2 %2B(8-13)^2 %2B(9-13)^2 %2B(10-13)^2 %2B(25-13)^2%3D196),
  given that the mean of cluster {1,2,3} is 2 and the mean of {8,9,10,25} is 13.
- Case 2: partition values into {{1, 2, 3, 8}, {9, 10, 25}}. Within-cluster variation is
  [![(1-3.5)^2 +(2-3.5)^2 +(3-3.5)^2 +(8-3.5)^2 +(9-14.67)^2 + (10-14.67)^2 + (25-14.67)^2 = 189.67](https://wattlecourses.anu.edu.au/filter/tex/pix.php/310dcfd79c0783ab0308f569c2ab0642.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=(1-3.5)^2 %2B(2-3.5)^2 %2B(3-3.5)^2 %2B(8-3.5)^2 %2B(9-14.67)^2 %2B (10-14.67)^2 %2B (25-14.67)^2 %3D 189.67),
  given that 3.5 is the mean of cluster {1, 2, 3, 8} and 14.67 is the mean of cluster {9, 10, 25}.

The latter partitioning has the lowest within-cluster variation; therefore, the k-means method assigns the value 8 to a cluster different from that containing 9 and 10 due to the outlier point 25. Moreover, the centre of the second cluster, 14.67, is substantially far from all the members in the cluster. 



#### 3.2. K-Medoids (PAM)

*"How can we modify the k-means algorithm to diminish sensitivity to outliers?"*



**K-medoids**

Instead of taking the mean value of the object in a cluster as a reference point, we can pick actual objects to represent the clusters.



The k-medoids method is more robust than k-means in the presence of noise and outliers because a medoid is less influenced by outliers or other extreme values than a mean.





**Partitioning Around Medoids (PAM)**

PAM algorithm is a popular realisation of k-medoids clustering.

Starts from an initial set of medoids and iteratively replaces one of the medoids by one of the non-medoids if it improves the total distance of the resulting clustering.

The quality of clustering can be measured by an absolute-error criterion (total cost):

[![E = \sum_{i=1}^{k}\sum_{p\in C_i} dist(p, o_i)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6ea5613a6def72b0e853b5167b699789.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E %3D \sum_{i%3D1}^{k}\sum_{p\in C_i} dist(p%2C o_i)),

where [![E](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3a3ea00cfc35332cedf6e5e9a32e94da.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=E) is the sum of the absolute error for all objects [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p) in the dataset, and [![o_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/c02c4a71b77353b6618d5fb75c880ad7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o_i) is the representative object of [![C_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4bd1241d43b60e0e4190660b97d2f686.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i).

PAM works effectively for small data sets, but does not scale well for large data sets (due to the computational complexity)





**Algorithm**

1. Arbitrarily choose k objects in D as the initial representative objects

2. Assign each remaining object to the cluster with the nearest representative object

3. Randomly select a non-representative object, [![o_{random}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/247fef39c8e6aa22cb8cece450028f4b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o_{random})

4. For each representative object [![o_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e72a575160511241fdc51b485cea7d1e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o_j),

5. 1. Compute the total cost of swapping representative object, [![o_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e72a575160511241fdc51b485cea7d1e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o_j), with [![o_{random}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/247fef39c8e6aa22cb8cece450028f4b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o_{random})
   2. If the swapping reduces the total cost, then swap [![o_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/e72a575160511241fdc51b485cea7d1e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o_j) with [![o_{random}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/247fef39c8e6aa22cb8cece450028f4b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o_{random}) to form the new set of [![k](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8ce4b16b22b58894aa86c421e8759df3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=k)representative objects

6. Repeat 2-4 until there is no change.



**Illustration of k-medoids algorithm**

![xEm1jU_No.00262616514151861651415186350](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/xEm1jU_No.00262616514151861651415186350_No.00290616514153461651415346300.jpg)

#### 3.3. Exercise 📝

######  Exercise: K-means clustering

Suppose that the data mining task is to cluster points (with (x,y) representing location) into three clusters, where the points are



[![A_1(2,10), A_2(2,5), A_3(8,4), B_1(5,8), B_2(7,5), B_3(6,4), C_1(1,2), C_2(4,9)](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/1eb69ea4cb49291ce0fcf0fb2930ab38-20220502%E4%B8%8A%E5%8D%88122322949_No.00242016514150601651415060853.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_1(2%2C10)%2C A_2(2%2C5)%2C A_3(8%2C4)%2C B_1(5%2C8)%2C B_2(7%2C5)%2C B_3(6%2C4)%2C C_1(1%2C2)%2C C_2(4%2C9)).



The distance function is Euclidean distance. Suppose initially we assign [![A_1, B_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9150d8496b222d049545696ccc01a3e7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=A_1%2C B_1), and [![C_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4fa71d007c094ac3c858919aec515277.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_1) as the center of each cluster, respectively. Use the k-means algorithm to show three clusters and their centres after the first round of execution.



###### Solution to Exercise: K-means clustering

Find the three cluster centres after the first round of execution.

Step 1: Centres are (1) A1 (2,10) , (2) B1(5,8), (3) C1(1,2) as given.

Step 2: Assign each object to nearest cluster by Euclidean distance from cluster centre

Object A2(2,5) is 5 from A1, sqrt(9+9) from B1, sqrt(1+9) from C1 so is assigned to cluster 3.

Object A3(8,4) is sqrt(36+36) from A1, sqrt(9+16) from B1, sqrt( 49+4) from C1 so is assigned to cluster 2.

Object B2(7,5) is sqrt(25+25) from A1, sqrt(4+9) from B1, sqrt(36+9) from C1 so is assigned to cluster 2.

Object B3(6,4) is sqrt(16+36) from A1, sqrt(1+16) from B1, sqrt(25+4) from C1 so is assigned to cluster 2.

Object C2(4,9) is sqrt(4+1) from A1, sqrt(2) from B1, sqrt(9+49) from C1 so is assigned to cluster 2.



Answer:

After the first round, the three new clusters are:

(1) {A1}, (2) {B1, A3, B2, B3, C2}, (3) {C1, A2},

and their new centres (step 3) are

(1) (2, 10)

(2) ((5+8+7+6+4)/5 = 6, ((8+4+5+4+9)/5 =6) = (6, 6)

(3) ((1+2) /2 = 1.5, (2+5) / 2=3.5) = (1.5, 3.5)



And something to ponder: *When we have finished the k-means algorithm, what is cluster 1 going to look like? And what does this say for the effectiveness of k-means?*



### 4. Hierarchical Clustering (AGNES and DIANA) (Text: 10.3)

Hierarchical clustering is a method of cluster analysis which seeks to **build a hierarchy of clusters**. Strategies for hierarchical clustering generally fall into two types:

- **Agglomerative**: This is a "bottom up" approach: each observation starts in its own cluster, and a pair of clusters is merged in each step of moving up the hierarchy.
- **Divisive**: This is a "top down" approach: all observations start in one cluster, and a cluster is split into two at each step of moving down the hierarchy.

In general, the merges and splits are determined in a greedy manner. The results of hierarchical clustering are usually presented in a [dendrogram](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472104&chapterid=439415).

Here is an example of the agglomerative and divisive hierarchical clustering approaches on data objects *{a,b,c,d,e}*.

![ZDYg9f_No.00350916514157091651415709546](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/ZDYg9f_No.00350916514157091651415709546.jpg)

Initially, the agglomerative method places each object into a cluster of its own. The clusters are then merged step-by-step according to some criterion. The merging process is repeated until all the objects are eventually merged to one cluster.

The divisive method proceeds in the oppostive direction. All the objects are used to form one initial cluster. The cluster is split according to some principle. The splitting process repeats until each new cluster contains only a single object.





AGNES (AGglomerative NESting)

- Uses the single-link method for determining the distance (dissimilarity) between clusters. Other methods can instead be applied, see [Distance between clusters](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2472104&chapterid=439423), together with a [dissimilarity matrix](https://wattlecourses.anu.edu.au/mod/book/view.php?id=2471993&chapterid=439251)
- Merges nodes that have the least dissimilarity
- Go on until all nodes are in the same cluster

DIANA (DIvisive ANAlysis)

- Inverse order of AGNES
- Go on until each distinct data object forms its own cluster



#### 4.1. Distance between Clusters

Whether using an agglomerative method or a divisive method, a core need is to measure the distance between two clusters, [![C_i, C_j](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f5f9a20c979d79e91632cae88dce6c7e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C_i%2C C_j) where each cluster is a set of objects.



- **Single link (minimum distance, nearest-neighbour clustering)**: smallest distance between an element in one cluster and an element in the other

  - i.e., [![dist(C_i, C_j) = min_{p \in C_i, q \in C_j}(|p-q|)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/3258a73dfeb7efc80a493d1967c4df27.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=dist(C_i%2C C_j) %3D min_{p \in C_i%2C q \in C_j}(|p-q|))

- **Complete link (maximum distance)**: largest distance between an element in one cluster and an element in the other

  - i.e., [![dist(C_i, C_j) = max_{p \in C_i, q \in C_j}(|p-q|)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/17c95d9c5f75e8bd0d7e3a43efd4bb6d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=dist(C_i%2C C_j) %3D max_{p \in C_i%2C q \in C_j}(|p-q|))

- **Average (average distance)**: average distance between an element in one cluster and an element in the other

  - i.e., [![dist(C_i, C_j) = \frac{1}{|C_i| |C_j|}\sum_{p \in C_i, q \in C_j}(|p-q|)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/54c8647b9991517b07652abc0fcd2e27.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=dist(C_i%2C C_j) %3D \frac{1}{|C_i| |C_j|}\sum_{p \in C_i%2C q \in C_j}(|p-q|))

- **Centroid**: distance between the centroids of two clusters

  - i.e., [![dist(C_i, C_j) = (|c_i-c_j|)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8e0fa8c77e9525b45d5178dd67d1613e.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=dist(C_i%2C C_j) %3D (|c_i-c_j|))

- **Medoid**: distance between the medoids of two clusters

- - i.e., [![dist(C_i, C_j) = (|o_i-o_j|)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/14e4cc1c43cb18b8fb25fa6838c36476.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=dist(C_i%2C C_j) %3D (|o_i-o_j|))

#### 4.2. Dendrogram

A binary-tree-structured diagram, called a dendrogram, is commonly used to represent the process of hierarchical clustering. It shows how objects are grouped together (in an agglomerative method) or partitioned (in a divisive method) step-by-step. The similarity of the cluster pairs selected at the step of their agglomeration or division may be shown on a similarity scale.

A final clustering of the data objects is obtained by cutting the dendrogram at the desired level, then each connected component at that level forms a cluster. 

The desired level is usually determined by selecting a threshold for similarity amongst clusters, but the desired number of clusters could be a factor too.

Here's an example of a dendrogram on data objects *{a,b,c,d,e}*

![p6TNbw_No.00363616514157961651415796678](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/p6TNbw_No.00363616514157961651415796678.jpg)

For example, by setting the similarity threshold to 0.5, one can obtain 3 clusters (a,b), (c), (d,e) from the dendogram.

### 5. Practical Exercises: K-means and Hierarchical Clustering ⌨️

**ACTION:** Have a go yourself with these R exercises!  There is a video showing the mechanics to get you started, written instructions for you to work through, and separately some suggested solutions.





<iframe src="https://www.youtube.com/embed/wtvl69TnCTk" title="YouTube video player" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" width="560" height="315" frameborder="0" style="box-sizing: border-box;"></iframe>

**Objectives**

The objectives of this lab are to experiment with the **clustering** package available in **R** and **Rattle**, in order to better understand k-means and hierarchical clustering algorithms. 

------

**Preliminaries**

For this exercise, we will mainly use "protein.csv" data set which contains 25 European countries and their protein intakes from nine major food sources. Click the following link to download the dataset:

- [![ ](https://wattlecourses.anu.edu.au/theme/image.php/anumore/core/1498660086/f/spreadsheet-24)European Protein Consumption dataset](https://wattlecourses.anu.edu.au/mod/resource/view.php?id=2472114)

We will use clustering library in R. To install the library type:

- install.packages("cluster")

You can get help on packages used in this lab by typing the following three commands into the **R** console. Specifically for the k-means algorithm, type:



- help(kmeans)**[
  ](http://www.togaware.com/datamining/survivor/Building_Model.html)**

And for Hierarchical clustering (AGNES) type:

- library(cluster)
- help(agnes)

------

**Tasks**

1. For this exercise, we need to load the data set first. Change the current working directory to the directory which contains "protein.csv" file. For example:

   setwd("/user/xxx/download/")

   will change the current working directory to "/user/xxx/download/"

   

2. To load csv file from the folder, type command:

   protein_df <- read.csv("protein.csv")

   

3. To check the details of the dataset, type

   head(protein_df)

   How many attributes are in the data set? An attribute value indicates a relative amount of protein source (as a percentage of all protein sources).

   

4. Cluster countries with given protein sources. First we apply k-means clustering using "RedMeat" and "Fish" attributes. 

   

   set.seed(123456789) ## to fix the random starting clusters
   grpMeat <- kmeans(protein_df[,c("RedMeat","Fish")], centers=5, nstart=10)

   where centers argument indicates the pre-specified number of clusters. Type:

   grpMeat

   to check the output of the clustering. What are the cluster means? Which cluster consumes the red meat most? Which cluster consumes the fish most?

   

5. To see the actual assignment for each country, type:

   o=order(grpMeat$cluster)
   data.frame(protein_df$Country[o],grpMeat$cluster[o])

   

6. The following commands will plot the clustering result on two-dimensional(red-meat, fish) space.

   
   plot(protein_df$Red, protein_df$Fish, type="p", xlim=c(3,19), ylim=c(0,15), xlab="Red Meat", ylab="Fish", col=grpMeat$cluster+1)
   text(x=protein_df$Red, y=protein_df$Fish-.6, labels=protein_df$Country, col=grpMeat$cluster+1)

   Can you find a relationship between geographical locations of countries and clustering assignments? Cross check the result with the clustering means. Is there any geographical relation between cluster means and assigned countries?

   

7. Cluster countries using all attributes. Let the number of clusters be 5.

   set.seed(123456789)
   grpProtein <- kmeans(protein_df[,-1], centers=5, nstart=10)
   o=order(grpProtein$cluster)
   data.frame(protein_df$Country[o],grpProtein$cluster[o])

   Did you find any difference in clustering?

8. Alternatively, we can apply a hierarchical clustering approach. Use the AGNES algorithm for clustering. For AGNES clustering, type:

   library(cluster)
   agg<-agnes(protein_df[,-1], diss=FALSE, metric="euclidian", method="average")

   method determines the distance metric between clusters, and metric determines the basic distance measure between data points.

9. Plot dendrogram using command (enter twice)

   

   plot(agg, main='Dendrogram', labels=protein_df$Country)

   To extract 5 clusters from the dendrogram, type:

   rect.hclust(agg, k=5, border="red")

   Again, do you think countries in close geographic proximity tend to be clustered into the same group?

   

10. Use *single link (minimum distance)* hierarchical clustering.

    agg<-agnes(protein_df[,-1], diss=FALSE, metric="euclidian", method="single")

    Plot the result and extract 5 clusters from the tree:

    plot(agg, main='Dendrogram', labels=protein_df$Country)
    rect.hclust(agg, k=5, border="red")

    Which countries comprise the smallest cluster? What does the smallest cluster mean?



\3. How many attributes are in the data set? 

\> 1)RedMeat 2)WhiteMeat 3)Eggs 4)Milk 5)Fish 6)Cereals 7)Starch 8)Nuts 9)Fr.Veg

\> Except country name, the data contains 9 attributes.



4.What are the cluster means? 

![F8dP7z_No.00395616514159961651415996229](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/F8dP7z_No.00395616514159961651415996229.jpg)

Which cluster consumes the red meat most? Cluster 1

Which cluster consumes the fish most? Cluster 5



\6. Can you find a relationship between geographical locations of countries and clustering assignments? 

\> Countries in close geographic proximity tend to be clustered into the same group.

Cross check the result with the clustering means. Is there any geographical relation between cluster means and assigned countries?

\> Except a few cases, countries near the ocean consumes a relatively large mount of fish and clustered together (cluster 3 and cluster 5). 

Cluster 4 consists of eastern Europe countries whose fish and red meat consumption are relatively less than the other countries.



\7. Did you find the difference in clustering?

\> The clustering results are different. For example, Portugal does not represent a single cluster any more.



\9. Again, do you think countries in close geographic proximity tend to be clustered into the same group? > In general, yes.

![Nshr2P_No.00401416514160141651416014107](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/Nshr2P_No.00401416514160141651416014107.jpg)

\10. Which countries consist the smallest cluster? What does the smallest cluster mean?

\> Albania and Finland. Single link approach agglomerate clusters based on the smallest distance between clusters. Therefore, the clustering results indicate that the protein consumption patterns of Albania and Finland are quite different from the other countries (in terms of Euclidean distance).

![Abytqx_No.00403216514160321651416032173](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/Abytqx_No.00403216514160321651416032173.jpg)

### 6. Density-Based Methods (DBSCAN) 🎥 (Text: 10.4)



**Density-Based Clustering**

Model clusters as dense regions in the data space, separated by sparse regions. Does not attempt to assign every object to a cluster; many may be left out as "noise". 

- Major features:
  - Discovers clusters of arbitrary shape
    - Partitioning and hierarchical methods are designed to find spherical-shaped (convex) clusters
  - Handles noise
  - One scan through the data only
  - Needs parameters to define threshold dense-ness (but not for the number of clusters)



**DBSCAN** (**D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise)

- **Density** of an object [![o](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d95679752134a2d9eb61dbd7b91c4bcc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o): the number of objects close to [![o](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d95679752134a2d9eb61dbd7b91c4bcc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o)

- **Core** objects: Objects that have a dense neighbourhood

- DBSCAN: connects core objects and their neighbourhoods to form dense regions as cluster

- Two parameters:

  - [![\epsilon](https://wattlecourses.anu.edu.au/filter/tex/pix.php/92e4da341fe8f4cd46192f21b6ff3aa7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\epsilon): Maximum radius of the neighbourhood
  - *MinPts*: Minimum number of points in an [![\epsilon](https://wattlecourses.anu.edu.au/filter/tex/pix.php/92e4da341fe8f4cd46192f21b6ff3aa7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\epsilon)-neighbourhood of that point

- [![N_{\epsilon}(p)](https://wattlecourses.anu.edu.au/filter/tex/pix.php/1399b106d979361f8a577c389610e2b0.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=N_{\epsilon}(p)): {[![q \in D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/016a1e049f1811725dac283d6cee07d2.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q \in D) | [![dist(p,q) \leq \epsilon](https://wattlecourses.anu.edu.au/filter/tex/pix.php/9a62098fb4a3bd7fe91f410818a9cade.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=dist(p%2Cq) \leq \epsilon)}

  - Number of neighbourhood objects including [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p)
  - If [![N_{\epsilon}(p) >= ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4814bb8015606939d27b920c78fc5fda.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=N_{\epsilon}(p) >%3D ) *MinPts*, then [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p) is core object
  - [![D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/f623e75af30e62bbd73d6df5b50bb7b5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=D) is a data set.

- **Directly density-reachable**: A point [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p) is directly density-reachable from a **core** point [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) if [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p) is within the [![\epsilon](https://wattlecourses.anu.edu.au/filter/tex/pix.php/92e4da341fe8f4cd46192f21b6ff3aa7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\epsilon)-neighbourhood of [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q)By definition, no points are *directly density-reachable* from a non-core point.

- **Density-reachable**: [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p) is density-reachable from a **core** point [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) if


  there is a chain of objects [![p_1, p_2, ... p_n](https://wattlecourses.anu.edu.au/filter/tex/pix.php/ef955ecced7db294cb3ad6f006c64f24.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_1%2C p_2%2C ... p_n) such that [![p_1 = q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/dced535763c38ac6e26849974a86e80f.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_1 %3D q), [![p_n=p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/270636db9838e20650bd09c00c3cc4ec.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_n%3Dp) and [![p_{i+1}](https://wattlecourses.anu.edu.au/filter/tex/pix.php/a717fc999f5afcb96a9a04da0a46f592.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_{i%2B1}) is *directly density-reachable* from [![p_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/eca91c83a74a2373ca5f796700e99fd3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_i) with respect to [![\epsilon](https://wattlecourses.anu.edu.au/filter/tex/pix.php/92e4da341fe8f4cd46192f21b6ff3aa7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\epsilon) and *MinPts.* 

  

  - By definition, all the [![p_i](https://wattlecourses.anu.edu.au/filter/tex/pix.php/eca91c83a74a2373ca5f796700e99fd3.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_i)s other than [![p_n = p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d592b5609c7bade3635887fa151d9235.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_n %3D p) are core points

- **Density-connected**: Two objects [![p_1, p_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/8654f8626086b448deda0be9c36cd451.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_1%2C p_2) are density-connected if 

- - there is an object [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) such that both [![p_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/03b632315ee5bee654b60a6bd902a249.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_1) and [![p_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6fe97b358b528edc477ba63d50b652af.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_2) are *density-reachable* from [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) with respect to [![\epsilon](https://wattlecourses.anu.edu.au/filter/tex/pix.php/92e4da341fe8f4cd46192f21b6ff3aa7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\epsilon) and *MinPts*.
  - By definition, [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) must be a core point, and [![p_1](https://wattlecourses.anu.edu.au/filter/tex/pix.php/03b632315ee5bee654b60a6bd902a249.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_1) and [![p_2](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6fe97b358b528edc477ba63d50b652af.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p_2) must be in the neighbourhood of a core point, but may not be core points themselves.



**Definition of Cluster in DBSCAN**

A subset [![C \subseteq D](https://wattlecourses.anu.edu.au/filter/tex/pix.php/33e49359251b0355d7a2537e28417fb5.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C \subseteq D) is a cluster if

- All points within the cluster [![C](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0d61f8370cad1d412f80b84d143e1257.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C) are **mutually density-connected**, and
- There is no point outside [![C](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0d61f8370cad1d412f80b84d143e1257.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C) that is **density-connected** to a point inside [![C](https://wattlecourses.anu.edu.au/filter/tex/pix.php/0d61f8370cad1d412f80b84d143e1257.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=C).



**Example of density-reachable and density-connected:**

![wLVsyj_No.00413416514160941651416094568](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/wLVsyj_No.00413416514160941651416094568.jpg)

\> Let [![\epsilon](https://wattlecourses.anu.edu.au/filter/tex/pix.php/92e4da341fe8f4cd46192f21b6ff3aa7.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=\epsilon) be the radius of the circles and *MinPts* 3.  

\> [![m,p,o, r](https://wattlecourses.anu.edu.au/filter/tex/pix.php/b8eee1b9225a6a40bb120a0a1b259e75.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m%2Cp%2Co%2C r) are core objects. 

\> Object [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) is directly density-reachable from [![m](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6f8f57715090da2632453988d9a1501b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m). 

\> Object [![m](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6f8f57715090da2632453988d9a1501b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m) is directly density-reachable from [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p) and vice versa.  

\> Object [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) is density-reachable from [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p) because [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) is directly density reachable from [![m](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6f8f57715090da2632453988d9a1501b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m) and [![m](https://wattlecourses.anu.edu.au/filter/tex/pix.php/6f8f57715090da2632453988d9a1501b.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=m) is directly density-reachable from [![p](https://wattlecourses.anu.edu.au/filter/tex/pix.php/83878c91171338902e0fe0fb97a8c47a.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p). However, [![p ](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d572956b265c891bdb3bacbcca08e1fd.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=p ) is not density reachable from [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) because [![q](https://wattlecourses.anu.edu.au/filter/tex/pix.php/7694f4a66316e53c8cdd9d9954bd611d.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=q) is not a core object. 

\> [![r](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4b43b0aee35624cd95b910189b3dc231.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=r) and [![s](https://wattlecourses.anu.edu.au/filter/tex/pix.php/03c7c0ace395d80182db07ae2c30f034.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=s) are density-reachable from [![o](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d95679752134a2d9eb61dbd7b91c4bcc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o)

\> [![o](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d95679752134a2d9eb61dbd7b91c4bcc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o) is density-reachable from [![r](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4b43b0aee35624cd95b910189b3dc231.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=r). 

\> [![o](https://wattlecourses.anu.edu.au/filter/tex/pix.php/d95679752134a2d9eb61dbd7b91c4bcc.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=o), [![r](https://wattlecourses.anu.edu.au/filter/tex/pix.php/4b43b0aee35624cd95b910189b3dc231.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=r), and [![s](https://wattlecourses.anu.edu.au/filter/tex/pix.php/03c7c0ace395d80182db07ae2c30f034.png)](https://wattlecourses.anu.edu.au/filter/tex/displaytex.php?texexp=s) are all density-connected.





**DBSCAN algorithm**

![VHRrrE_No.00421516514161351651416135312](https://cdn.jsdelivr.net/gh/California3/Imagelib@master/uPic/2022/05/02/VHRrrE_No.00421516514161351651416135312.jpg)


**ACTION:** Watch this video that shows how the DBSCAN algorithm works through to build the clusters in the diagram above.  