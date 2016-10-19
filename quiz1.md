Ryan Tillis - Machine Learning - Data Science - Quiz 1 - Coursera
================
<a href="http://www.ryantillis.com"> Ryan Tillis </a>
June 3, 2016

Machine Leearning Quiz 1
------------------------

This is Quiz 1 from the Machine Learning course within the Data Science Specialization. This publication is intended as a learning resource, all answers are documented and explained.

### Questions

<hr>
<font size="+2">1. </font> Which of the following are components in building a machine learning algorithm?

<hr>
-   <font size="+1">**Training and test sets**

</font>

<hr>
##### Explanation:

Self-Explanatory

Self-explanatory

<hr>
<font size="+2">2. </font> Suppose we build a prediction algorithm on a data set and it is 100% accurate on that data set. Why might the algorithm not work well if we collect a new data set?

<hr>
-   <font size="+1">**Our algorithm may be overfitting the training data, predicting both the signal and the noise.**</font>

<hr>
##### Explanation:

If your algorithm is too specific to the peculiarities of an individual set, it may lead to false classification and bad predictions.

<hr>
<font size="+2">3. </font> What are typical sizes for the training and test sets?

<hr>
-   <font size="+1"> **60% in the training set, 40% in the testing set.**

</font>

<hr>
##### Explanation:

Self-explanatory

<hr>
<font size="+2">4. </font> What are some common error rates for predicting binary variables (i.e. variables with two possible values like yes/no, disease/normal, clicked/didn't click)? Check the correct answer(s).

<hr>
-   <font size="+1">**Predictive value of a positive** </font>

<hr>
##### Explanation:

Self-explanatory

<hr>
<font size="+2">5. </font> Suppose that we have created a machine learning algorithm that predicts whether a link will be clicked with 99% sensitivity and 99% specificity. The rate the link is clicked is 1/1000 of visits to a website. If we predict the link will be clicked on a specific visit, what is the probability it will actually be clicked?

<hr>
-   <font size="+1"> **9%** </font>

<hr>
##### Explanation:

``` r
n <- 1000000
prevalence = 1/1000
sens <- .99
spec <- .99
clicked <- prevalence * n
TP <- sens * clicked
Notclicked <- n - clicked
TN <- spec * Notclicked
FP <- Notclicked * sens
FP<- Notclicked-TN
PPV <- TP/ (TP+FP)

PPV
```

    ## [1] 0.09016393

<hr>
Check out my website at: <http://www.ryantillis.com/>

<hr>
