Ryan Tillis - Machine Learning - Data Science - Quiz 2 - Coursera
================
<a href="http://www.ryantillis.com"> Ryan Tillis </a>
July 3, 2016

Machine Learning Quiz 2
-----------------------

This is Quiz 2 from the Machine Learning course within the Data Science Specialization. Contents include Principle component analysis, variable transformations, and the caret package. This is intended as a learning resource.

### Questions

<hr>
<font size="+2">1. </font> Load the Alzheimer's disease data using the commands:

``` r
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
```

Which of the following commands will create non-overlapping training and test sets with about 50% of the observations assigned to each?
<hr>
<font size="+1">**Answer:** </font>

``` r
adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]
```

</font>

<hr>
##### Explanation:

The createDataPartition function creates an index that splits on a given proporition p.

<hr>
<font size="+2">2. </font> Load the cement data using the commands:

``` r
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
```

Make a plot of the outcome (CompressiveStrength) versus the index of the samples. Color by each of the variables in the data set (you may find the cut2() function in the Hmisc package useful for turning continuous covariates into factors). What do you notice in these plots?
<hr>
-   <font size="+1">**There is a non-random pattern in the plot of the outcome versus index that does not appear to be perfectly explained by any predictor suggesting a variable may be missing.**</font>

<hr>
##### Explanation:

![](quiz2_files/figure-markdown_github/Question%202-1.png)

<hr>
<font size="+2">3. </font> Load the cement data using the commands:

``` r
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
```

Make a histogram and confirm the SuperPlasticizer variable is skewed. Normally you might use the log transform to try to make the data more symmetric. Why would that be a poor choice for this variable?

<hr>
-   <font size="+1"> **There are a large number of values that are the same and even if you took the log(SuperPlasticizer + 1) they would still all be identical so the distribution would not be symmetric.**

</font>

##### Explanation:

<hr>
``` r
hist(training$Superplasticizer+1)
```

![](quiz2_files/figure-markdown_github/Question%203-1.png)

``` r
hist(log(training$Superplasticizer+1))
```

![](quiz2_files/figure-markdown_github/Question%203-2.png)

<hr>
<font size="+2">4. </font> Load the Alzheimer's disease data using the commands:

``` r
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
```

Find all the predictor variables in the training set that begin with IL. Perform principal components on these variables with the preProcess() function from the caret package. Calculate the number of principal components needed to capture 80% of the variance. How many are there?
<hr>
-   <font size="+1">**7** </font>

<hr>
##### Explanation:

Preprocessing with a threshold of .8 shows that 7 principle components are needed to capture 80% of the variance.

``` r
train <- training[,c(58:69)]
names(train)
```

    ##  [1] "IL_11"         "IL_13"         "IL_16"         "IL_17E"       
    ##  [5] "IL_1alpha"     "IL_3"          "IL_4"          "IL_5"         
    ##  [9] "IL_6"          "IL_6_Receptor" "IL_7"          "IL_8"

``` r
preP <- preProcess(train,method = "pca", thresh = .8)
preP
```

    ## Created from 251 samples and 12 variables
    ## 
    ## Pre-processing:
    ##   - centered (12)
    ##   - ignored (0)
    ##   - principal component signal extraction (12)
    ##   - scaled (12)
    ## 
    ## PCA needed 7 components to capture 80 percent of the variance

<hr>
<font size="+2">5. </font> Load the Alzheimer's disease data using the commands:

``` r
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
```

Create a training data set consisting of only the predictors with variable names beginning with IL and the diagnosis. Build two predictive models, one using the predictors as they are and one using PCA with principal components explaining 80% of the variance in the predictors. Use method="glm" in the train function.

<hr>
-   <font size="+1"> **Non-PCA Accuracy: 0.65** **PCA Accuracy: 0.72** </font>

<hr>
##### Explanation:

``` r
traindex <- training[,c(1, 58:69)]

#Data without outcome for prediction
train <- training[,c(58:69)]
test <- testing[, c(58:69)]

#Fitting model without PCA
fit <- train(diagnosis~., data = traindex,method="glm")
pred <- predict(fit,test)
confusionMatrix(data = pred, testing$diagnosis)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Impaired Control
    ##   Impaired        2       9
    ##   Control        20      51
    ##                                          
    ##                Accuracy : 0.6463         
    ##                  95% CI : (0.533, 0.7488)
    ##     No Information Rate : 0.7317         
    ##     P-Value [Acc > NIR] : 0.96637        
    ##                                          
    ##                   Kappa : -0.0702        
    ##  Mcnemar's Test P-Value : 0.06332        
    ##                                          
    ##             Sensitivity : 0.09091        
    ##             Specificity : 0.85000        
    ##          Pos Pred Value : 0.18182        
    ##          Neg Pred Value : 0.71831        
    ##              Prevalence : 0.26829        
    ##          Detection Rate : 0.02439        
    ##    Detection Prevalence : 0.13415        
    ##       Balanced Accuracy : 0.47045        
    ##                                          
    ##        'Positive' Class : Impaired       
    ## 

``` r
ctrl <- trainControl(preProcOptions = list(thresh = 0.95))

#Fitting Model with PCA 
fit2 <- train(diagnosis~., data=traindex,method = "glm",preProcess = c("pca"), trControl = ctrl)

pred <- predict(fit2,test)
confusionMatrix(data = pred, testing$diagnosis)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Impaired Control
    ##   Impaired        3       5
    ##   Control        19      55
    ##                                           
    ##                Accuracy : 0.7073          
    ##                  95% CI : (0.5965, 0.8026)
    ##     No Information Rate : 0.7317          
    ##     P-Value [Acc > NIR] : 0.737155        
    ##                                           
    ##                   Kappa : 0.0664          
    ##  Mcnemar's Test P-Value : 0.007963        
    ##                                           
    ##             Sensitivity : 0.13636         
    ##             Specificity : 0.91667         
    ##          Pos Pred Value : 0.37500         
    ##          Neg Pred Value : 0.74324         
    ##              Prevalence : 0.26829         
    ##          Detection Rate : 0.03659         
    ##    Detection Prevalence : 0.09756         
    ##       Balanced Accuracy : 0.52652         
    ##                                           
    ##        'Positive' Class : Impaired        
    ## 

<hr>
Check out my website at: <http://www.ryantillis.com/>

<hr>
