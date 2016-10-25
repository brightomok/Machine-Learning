Maching Learning Project - Random Forest - Sensor Data
================
<a href="http://www.ryantillis.com"> Ryan Tillis </a>
September 4, 2016

Human Activity Recognition
--------------------------

With today's sensor technology phones can teach us to salsa and watches can teach us how to eat salsa. As you might expect, applications of motion sensing data is not only limited to salsa-related activities.

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. The goal of this project is to predict how well a bicep curl was performed using the data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Data for this project comes from <http://groupware.les.inf.puc-rio.br/har>

Six participants performed 10 bicep curls in five different fashions: exactly according to the correct specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

### Data Processing

The data appears at first to have a lot of NA values. The data is provided with aggregated statistical metrics across each window of observation. The columns that contain these aggregated values are assigned NA while the data is collected. For this analysis I chose to separate the aggregated data and the raw data into 2 data frames and build models off of both. The section of code below includes creating a training/testing partition and separating the aggregated data columns from the raw data and finally the removal of the NA values from the summarized data.

``` r
#Setting seed for reproducibility
set.seed(22)

#Download the data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "training.csv")

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "testing.csv")

#Full data for partitioning
data <- read.csv("training.csv")

#20 Cases for Validation
validation <- read.csv("testing.csv")

class(data$classe)
```

    ## [1] "factor"

``` r
levels(data$classe)
```

    ## [1] "A" "B" "C" "D" "E"

``` r
dim(data)
```

    ## [1] 19622   160

``` r
#Identifying NA level
NA_levels <- unique(apply(data, 2,function(x){sum(is.na(x))}))
number_NA <- dim(data)[1]-NA_levels[2]
non_NA <- number_NA/dim(data)[1]
sprintf("%1.2f%%", 100*non_NA)
```

    ## [1] "2.07%"

``` r
#Setting empty spaces and div0 to be NA
data[data == ""] <- NA
data[data=="#DIV/0!"] <- NA
data[data=="<NA>"] <- NA

#Splitting the data for test/train
set.seed(22)
traindex <- createDataPartition(data$classe,p = 0.8,list = FALSE)
train <- data[traindex,]
test <- data[-traindex,]

#Selecting non-aggregated RAW sensor data
train_raw <- train[which(train$new_window == "no"),]

#Raw sensor data without NA columns(summary data)
train_raw <- train[!colSums(is.na(train)) > 0]

#Testing NA purity
sum(is.na(train_raw))
```

    ## [1] 0

``` r
#Splitting data to new window rows (aggregated data)
train_sum <- train[which(train$new_window == "yes"),]
test_sum <- test[which(test$new_window == "yes"),]

#Removing full NA columns
train_sum_clean <- subset(train_sum,
                          select=-c(kurtosis_picth_belt,kurtosis_yaw_belt,kurtosis_picth_arm,kurtosis_yaw_arm,skewness_pitch_arm,kurtosis_yaw_dumbbell,skewness_yaw_dumbbell,skewness_yaw_forearm,kurtosis_yaw_forearm,skewness_yaw_belt,skewness_roll_belt.1))

test_sum_clean <- subset(test_sum,
                          select=-c(kurtosis_picth_belt,kurtosis_yaw_belt,kurtosis_picth_arm,kurtosis_yaw_arm,skewness_pitch_arm,kurtosis_yaw_dumbbell,skewness_yaw_dumbbell,skewness_yaw_forearm,kurtosis_yaw_forearm,skewness_yaw_belt,skewness_roll_belt.1))

#Removing NA rows
train_done <- train_sum_clean[complete.cases(train_sum_clean),]
sum(is.na(train_done))
```

    ## [1] 0

``` r
test_done <- test_sum_clean[complete.cases(test_sum_clean),]
sum(is.na(test_done))
```

    ## [1] 0

### Model Fitting

Due to the charactertics of the noise in the sensor data, a random forest model is best for this task. The first model (model1) achieves an estimated out of sample **error rate of 0.43%.** The model uses bootstrap resampling with the training set partition given above to crossvalidate against the test set. Since the fit uses all the possible (59) clean predictor variables, k-fold cross validation would be computationally intensive.

``` r
#Important to not include the X row in the dataset because it is an index and the data is organized alphabetically by class outcome.
model1 <- randomForest(classe ~. , data=train_raw[,-c(1:7)], method="class")
pred_test1 <- predict(model1, test)
pred_train1 <- predict(model1, train)

confusionMatrix(pred_test1, test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1115    4    0    0    0
    ##          B    1  755    3    0    0
    ##          C    0    0  681    2    0
    ##          D    0    0    0  640    3
    ##          E    0    0    0    1  718
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.9964        
    ##                  95% CI : (0.994, 0.998)
    ##     No Information Rate : 0.2845        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.9955        
    ##  Mcnemar's Test P-Value : NA            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9991   0.9947   0.9956   0.9953   0.9958
    ## Specificity            0.9986   0.9987   0.9994   0.9991   0.9997
    ## Pos Pred Value         0.9964   0.9947   0.9971   0.9953   0.9986
    ## Neg Pred Value         0.9996   0.9987   0.9991   0.9991   0.9991
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2842   0.1925   0.1736   0.1631   0.1830
    ## Detection Prevalence   0.2852   0.1935   0.1741   0.1639   0.1833
    ## Balanced Accuracy      0.9988   0.9967   0.9975   0.9972   0.9978

``` r
confusionMatrix(pred_train1, train$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 4464    0    0    0    0
    ##          B    0 3038    0    0    0
    ##          C    0    0 2738    0    0
    ##          D    0    0    0 2573    0
    ##          E    0    0    0    0 2886
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9998, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

The second model (model2) uses feature selection to narrow down the 59 predictors to only 7 chosen Variables: classe ~ roll\_belt + pitch\_belt + yaw\_belt + magnet\_arm\_x + gyros\_dumbbell\_y + magnet\_dumbbell\_y + pitch\_forearm. This model achieves a 98.16% accuracy with an **expected error of 1.81%.** The expected error is higher, but is still very successful considering this model uses 52 fewer predictors. To create this model, 3-fold crossvalidation was implemented with the caret package.

``` r
#Using Correlation based feature selection and best-first algorithm
features <- cfs(classe~.,train_raw[,-c(1:7)])
f <- as.simple.formula(features, "classe")

fitControl <- trainControl(## 10-fold CV
                           method = "cv",
                           number = 3,
                           ## repeated ten times
                           repeats = 3)

model2 <- train(f, method = "rf", data =train_raw, trControl = fitControl)

model2
```

    ## Random Forest 
    ## 
    ## 15699 samples
    ##     7 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 10466, 10465, 10467 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##   2     0.9761127  0.9697912
    ##   4     0.9738197  0.9668918
    ##   7     0.9661761  0.9572249
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 2.

``` r
pred_test2 <- predict(model2, test)
pred_train2 <- predict(model2, train)

confusionMatrix(pred_test2, test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1098    8    0    2    0
    ##          B    7  732    3    0    5
    ##          C    4   15  679    5    1
    ##          D    6    4    2  636    4
    ##          E    1    0    0    0  711
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9829          
    ##                  95% CI : (0.9784, 0.9867)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2e-16         
    ##                                           
    ##                   Kappa : 0.9784          
    ##  Mcnemar's Test P-Value : 0.00075         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9839   0.9644   0.9927   0.9891   0.9861
    ## Specificity            0.9964   0.9953   0.9923   0.9951   0.9997
    ## Pos Pred Value         0.9910   0.9799   0.9645   0.9755   0.9986
    ## Neg Pred Value         0.9936   0.9915   0.9984   0.9979   0.9969
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2799   0.1866   0.1731   0.1621   0.1812
    ## Detection Prevalence   0.2824   0.1904   0.1795   0.1662   0.1815
    ## Balanced Accuracy      0.9902   0.9798   0.9925   0.9921   0.9929

``` r
confusionMatrix(pred_train2, train$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 4464    0    0    0    0
    ##          B    0 3038    0    0    0
    ##          C    0    0 2738    0    0
    ##          D    0    0    0 2573    0
    ##          E    0    0    0    0 2886
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9998, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

The final model (model3) is fit using the provided summary data. This model achieves an accuracy of only 71.74%, or a **28.26% expected error** rate against the test validation set. To create this model 3-fold crossvalidation was implemented with the caret package.

``` r
#Predicting off of summary statistics
features3 <- cfs(classe~.,train_done[,-c(1:7)])
z <- as.simple.formula(features3, "classe")

model3 <- train(z, method = "rf", data =train_done, trControl = fitControl)

model3
```

    ## Random Forest 
    ## 
    ## 187 samples
    ##  11 predictor
    ##   5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 124, 126, 124 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##      2  0.2567439  0.0000000
    ##     53  0.6462833  0.5482757
    ##   1456  0.6782028  0.5940114
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 1456.

``` r
pred_test3 <- predict(model3, test_done)
pred_train3 <- predict(model3, train_done)

confusionMatrix(pred_test3, test_done$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction A B C D E
    ##          A 9 3 2 1 0
    ##          B 1 6 1 1 1
    ##          C 0 0 6 0 0
    ##          D 0 0 0 6 1
    ##          E 0 1 0 0 7
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7391          
    ##                  95% CI : (0.5887, 0.8573)
    ##     No Information Rate : 0.2174          
    ##     P-Value [Acc > NIR] : 6.634e-14       
    ##                                           
    ##                   Kappa : 0.6722          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9000   0.6000   0.6667   0.7500   0.7778
    ## Specificity            0.8333   0.8889   1.0000   0.9737   0.9730
    ## Pos Pred Value         0.6000   0.6000   1.0000   0.8571   0.8750
    ## Neg Pred Value         0.9677   0.8889   0.9250   0.9487   0.9474
    ## Prevalence             0.2174   0.2174   0.1957   0.1739   0.1957
    ## Detection Rate         0.1957   0.1304   0.1304   0.1304   0.1522
    ## Detection Prevalence   0.3261   0.2174   0.1304   0.1522   0.1739
    ## Balanced Accuracy      0.8667   0.7444   0.8333   0.8618   0.8754

``` r
confusionMatrix(pred_train3, train_done$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  A  B  C  D  E
    ##          A 48  0  0  0  0
    ##          B  0 39  0  0  0
    ##          C  0  0 31  0  0
    ##          D  0  0  0 34  0
    ##          E  0  0  0  0 35
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9805, 1)
    ##     No Information Rate : 0.2567     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2567   0.2086   0.1658   0.1818   0.1872
    ## Detection Rate         0.2567   0.2086   0.1658   0.1818   0.1872
    ## Detection Prevalence   0.2567   0.2086   0.1658   0.1818   0.1872
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Conclusions
-----------

Predicting off of the summary statistics is much less accurate. The most differentiating variable is the pitch of the belt sensor. This quickly distinguishes a lot of cases where the individual performs a Class E mistake ("throwing the hips to the front"). The first model performed the best overall, and will be used to predict the validation set.

``` r
predict(model1,validation)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Appendix
--------

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = train_raw[, -c(1:7)],      method = "class") 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 7
    ## 
    ##         OOB estimate of  error rate: 0.43%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 4461    1    0    1    1 0.000672043
    ## B   13 3019    6    0    0 0.006254115
    ## C    0   11 2725    2    0 0.004747991
    ## D    0    0   27 2544    2 0.011270890
    ## E    0    0    1    3 2882 0.001386001

    ## Random Forest 
    ## 
    ## 15699 samples
    ##     7 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 10466, 10465, 10467 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##   2     0.9761127  0.9697912
    ##   4     0.9738197  0.9668918
    ##   7     0.9661761  0.9572249
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 2.

    ## Random Forest 
    ## 
    ## 187 samples
    ##  11 predictor
    ##   5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 124, 126, 124 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##      2  0.2567439  0.0000000
    ##     53  0.6462833  0.5482757
    ##   1456  0.6782028  0.5940114
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 1456.

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6\_6. Cited by 2 (Google Scholar)

L. B. Statistics and L. Breiman. Random forests. In Machine Learning, pages 5–32, 2001.

Read more: [http://groupware.les.inf.puc-rio.br/har\#sbia\_paper\_section\#ixzz4NpZpLz5s](http://groupware.les.inf.puc-rio.br/har#sbia_paper_section#ixzz4NpZpLz5s)

<hr>
Check out my website at: <http://www.ryantillis.com/>

<hr>
