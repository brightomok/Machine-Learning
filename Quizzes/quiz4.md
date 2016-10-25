Ryan Tillis - Machine Learning - Data Science - Quiz 4 - Coursera
================
<a href="http://www.ryantillis.com"> Ryan Tillis </a>
June 3, 2016

Machine Learning Quiz 4
-----------------------

This is Quiz 4 from the Machine Learning course within the Data Science Specialization. This publication is intended as a learning resource, all answers are documented and explained.

### Questions

<hr>
<font size="+2">1. </font> For this quiz we will be using several R packages. R package versions change over time, the right answers have been checked using the following versions of the packages.

AppliedPredictiveModeling: v1.1.6

caret: v6.0.47

ElemStatLearn: v2012.04-0

pgmm: v1.1

rpart: v4.1.8

gbm: v2.1

lubridate: v1.3.3

forecast: v5.6

e1071: v1.6.4

If you aren't using these versions of the packages, your answers may not exactly match the right answer, but hopefully should be close.

Load the vowel.train and vowel.test data sets:

``` r
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
```

Set the variable y to be a factor variable in both the training and test set. Then set the seed to 33833. Fit (1) a random forest predictor relating the factor variable y to the remaining variables and (2) a boosted predictor using the "gbm" method. Fit these both with the train() command in the caret package.

What are the accuracies for the two approaches on the test data set? What is the accuracy among the test set samples where the two methods agree?

<hr>
<font size="+1"><b>

-   RF Accuracy = 0.6082

-   GBM Accuracy = 0.5152

-   Agreement Accuracy = 0.6361

</b> </font>

<hr>
##### Explanation:

RF is slightly more accurate than the boosted gbm model, where they agree the accuracy is even higher.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(gbm)
```

    ## Loading required package: survival

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:caret':
    ## 
    ##     cluster

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.1

``` r
vowel.train$y <- factor(vowel.train$y)
vowel.test$y <- factor(vowel.test$y)
```

``` r
rf <- train(y~., method = "rf",data =vowel.train)
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
#Predicting
pred_rf <- predict(rf, vowel.test)
pred_boost <- predict(boost, vowel.test)

#Accuracies
confusionMatrix(pred_rf, vowel.test$y)$overall[1]
```

    ##  Accuracy 
    ## 0.5974026

``` r
confusionMatrix(pred_boost, vowel.test$y)$overall[1]
```

    ## Accuracy 
    ## 0.530303

``` r
#Matched Accuracy
match <- (pred_boost == pred_rf)
confusionMatrix(vowel.test$y[match], pred_boost[match])$overall[1]
```

    ##  Accuracy 
    ## 0.6269592

<hr>
<font size="+2">2. </font> Load the Alzheimer's data using the following commands

``` r
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
```

Set the seed to 62433 and predict diagnosis with all the other variables using a random forest ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model. Stack the predictions together using random forests ("rf"). What is the resulting accuracy on the test set? Is it better or worse than each of the individual predictions?
<hr>
<font size="+1"> <b>

-   Stacked Accuracy: 0.80 is better than random forests and lda and the same as boosting.

</b> </font>

##### Explanation:

Combining all three models (boosting, random forest, linear discriminant analysis) result in a higher accuracy.

``` r
set.seed(62433)
#Training Random Forest, Boosting, and Linear Discriminant Analysis
rf <- train(diagnosis~., method = "rf",data =training)
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
lda <- train(diagnosis~., method = "lda",data =training)
```

    ## Loading required package: MASS

    ## Warning in lda.default(x, grouping, ...): variables are collinear

    ## Warning in lda.default(x, grouping, ...): variables are collinear

``` r
#Predicting the Model
pred_rf <- predict(rf, testing)
pred_boost <- predict(boost, testing)
```

    ## Loading required package: plyr

    ## 
    ## Attaching package: 'plyr'

    ## The following object is masked from 'package:ElemStatLearn':
    ## 
    ##     ozone

``` r
pred_lda <- predict(lda, testing)

#Combining Prediction Sets, training against diagnosis and predicting 
all_pred <- data.frame(pred_rf,pred_lda,pred_boost, diagnosis = testing$diagnosis)
combinedMod <- train(diagnosis~.,method="rf", data = all_pred)
```

    ## note: only 2 unique complexity parameters in default grid. Truncating the grid to 2 .

``` r
combinedPred <- predict(combinedMod,all_pred)

#Accuracies
confusionMatrix(testing$diagnosis, pred_rf)$overall[1]
```

    ##  Accuracy 
    ## 0.7682927

``` r
confusionMatrix(testing$diagnosis, pred_lda)$overall[1]
```

    ##  Accuracy 
    ## 0.7682927

``` r
confusionMatrix(testing$diagnosis, pred_boost)$overall[1]
```

    ##  Accuracy 
    ## 0.7926829

``` r
confusionMatrix(testing$diagnosis, combinedPred)$overall[1]
```

    ## Accuracy 
    ## 0.804878

<hr>
<font size="+2">3. </font> Load the concrete data with the commands:

``` r
set.seed(3523)
library(caret)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
```

<hr>
Set the seed to 233 and fit a lasso model to predict Compressive Strength. Which variable is the last coefficient to be set to zero as the penalty increases? (Hint: it may be useful to look up ?plot.enet).

<font size="+1"> <b>

-   Cement

</b> </font>

<hr>
##### Explanation:

Regularizing the coefficients is like a stepwise approach toward least squares. The enet chart shows which coefficients are the last to go.

``` r
set.seed(233)
lasso <- train(CompressiveStrength~., method = "lasso", data = training)
```

    ## Loading required package: elasticnet

    ## Loading required package: lars

    ## Loaded lars 1.2

``` r
plot.enet(lasso$finalModel, xvar = "penalty")
```

![](quiz4_files/figure-markdown_github/Question%203-1.png)

<hr>
<font size="+2">4. </font> Load the data on the number of visitors to the instructors blog from here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv>

Using the commands:

``` r
library(lubridate) # For year() function below
```

    ## 
    ## Attaching package: 'lubridate'

    ## The following object is masked from 'package:plyr':
    ## 
    ##     here

    ## The following object is masked from 'package:base':
    ## 
    ##     date

``` r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv", destfile = "gaData.csv")
dat = read.csv("~/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)
```

<hr>
-   <font size="+1">**96%** </font>

<hr>
##### Explanation:

Fitting the bats models and forecasting gives us the upper and lower bounds.

``` r
library(forecast)
```

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

    ## Loading required package: timeDate

    ## This is forecast 7.3

``` r
bats <- bats(training$visitsTumblr)
fcast <- forecast(bats, level = 95, h = dim(testing)[1])

sum(fcast$lower < testing$visitsTumblr &  testing$visitsTumblr < fcast$upper)/nrow(testing)
```

    ## [1] 0.9617021

<hr>
<font size="+2">5. </font> Load the concrete data with the commands:

``` r
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
```

Set the seed to 325 and fit a support vector machine using the e1071 package to predict Compressive Strength using the default settings. Predict on the testing set. What is the RMSE?
<hr>
-   <font size="+1"> **6.72** </font>

<hr>
##### Explanation:

Support Vector machines are a supervised learning classification and regression method

``` r
set.seed(325)
library(e1071)
```

    ## 
    ## Attaching package: 'e1071'

    ## The following objects are masked from 'package:timeDate':
    ## 
    ##     kurtosis, skewness

``` r
svm <- svm(CompressiveStrength ~ ., data = training)
pred <- predict(svm, testing)
accuracy(pred, testing$CompressiveStrength)
```

    ##                 ME     RMSE      MAE       MPE     MAPE
    ## Test set 0.1682863 6.715009 5.120835 -7.102348 19.27739

<hr>
Check out my website at: <http://www.ryantillis.com/>

<hr>
