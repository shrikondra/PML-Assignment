``` r
knitr::opts_chunk$set(
  fig.path = "figure/"
)
```

Executive Summary
-----------------

In this project, we will use data recorded from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information is available from the website <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

The goal of this project is to predict the manner in which the participants did the exercise. This is the classe variable of the training set, which classifies the correct and incorrect outcomes into A, B, C, D, and E categories.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(rpart)
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

Loading and preprocessing the data
----------------------------------

``` r
training <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
testing  <- read.csv("pml-testing.csv",  na.strings = c("NA", "#DIV/0!", ""))
```

Exploratory Data Analysis
-------------------------

``` r
dim(training)
```

    ## [1] 19622   160

``` r
dim(testing)
```

    ## [1]  20 160

``` r
#str(training)
str(training[,1:20]) # displaying only first 20 cols for the assignment
```

    ## 'data.frame':    19622 obs. of  20 variables:
    ##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ kurtosis_roll_belt  : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_picth_belt : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_yaw_belt   : logi  NA NA NA NA NA NA ...
    ##  $ skewness_roll_belt  : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_roll_belt.1: num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_yaw_belt   : logi  NA NA NA NA NA NA ...
    ##  $ max_roll_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_belt      : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_belt        : num  NA NA NA NA NA NA NA NA NA NA ...

The data has cols with many NA values and also first 7 cols seems to be user related info. We will remove this cols

Cleaning the Data
-----------------

``` r
CleanedTrainData <- training[, colSums(is.na(training)) == 0] # remove cols with NA values
CleanedTrainData <- CleanedTrainData[, -c(1:7)]
d <- dim(CleanedTrainData)
str(CleanedTrainData[,-c(1:(d[2]-10))]) # displaying last 10 cols
```

    ## 'data.frame':    19622 obs. of  10 variables:
    ##  $ gyros_forearm_x : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
    ##  $ gyros_forearm_y : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
    ##  $ gyros_forearm_z : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
    ##  $ accel_forearm_x : int  192 192 196 189 189 193 195 193 193 190 ...
    ##  $ accel_forearm_y : int  203 203 204 206 206 203 205 205 204 205 ...
    ##  $ accel_forearm_z : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
    ##  $ magnet_forearm_x: int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
    ##  $ magnet_forearm_y: num  654 661 658 658 655 660 659 660 653 656 ...
    ##  $ magnet_forearm_z: num  476 473 469 469 473 478 470 474 476 473 ...
    ##  $ classe          : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...

Data Partitioning for Training
------------------------------

``` r
set.seed(1234)
inTrain<-createDataPartition(y=CleanedTrainData$classe, p=0.7,list=F)
TrainSubset<-CleanedTrainData[inTrain,] 
TestSubset<-CleanedTrainData[-inTrain,] 
d1 <- dim(TrainSubset)
d2 <- dim(TestSubset)
print(paste("No of Training Samples : ", d1[1], ", No of Validation Samples : ", d2[1] ))
```

    ## [1] "No of Training Samples :  13737 , No of Validation Samples :  5885"

Model Prediction
----------------

We will try Random Forest and SVM, two well known methods which gives high accuracies

### Method 1 Random Forests

``` r
set.seed(1234)
fitControl1<-trainControl(method="cv", number=2, allowParallel=T, verbose=T)
rffit<-train(classe~.,data=TrainSubset, method="rf", trControl=fitControl1, verbose=F)
```

    ## + Fold1: mtry= 2 
    ## - Fold1: mtry= 2 
    ## + Fold1: mtry=27 
    ## - Fold1: mtry=27 
    ## + Fold1: mtry=52 
    ## - Fold1: mtry=52 
    ## + Fold2: mtry= 2 
    ## - Fold2: mtry= 2 
    ## + Fold2: mtry=27 
    ## - Fold2: mtry=27 
    ## + Fold2: mtry=52 
    ## - Fold2: mtry=52 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting mtry = 27 on full training set

``` r
rffit$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry, verbose = ..1) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.6%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3901    3    1    0    1 0.001280082
    ## B   19 2634    5    0    0 0.009029345
    ## C    0   11 2381    4    0 0.006260434
    ## D    0    1   26 2223    2 0.012877442
    ## E    0    0    3    7 2515 0.003960396

``` r
predrf<-predict(rffit, newdata=TestSubset)
confusionMatrix(predrf, TestSubset$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674   10    0    0    0
    ##          B    0 1129    4    1    0
    ##          C    0    0 1018    5    2
    ##          D    0    0    4  957    4
    ##          E    0    0    0    1 1076
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9947          
    ##                  95% CI : (0.9925, 0.9964)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9933          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9912   0.9922   0.9927   0.9945
    ## Specificity            0.9976   0.9989   0.9986   0.9984   0.9998
    ## Pos Pred Value         0.9941   0.9956   0.9932   0.9917   0.9991
    ## Neg Pred Value         1.0000   0.9979   0.9984   0.9986   0.9988
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1918   0.1730   0.1626   0.1828
    ## Detection Prevalence   0.2862   0.1927   0.1742   0.1640   0.1830
    ## Balanced Accuracy      0.9988   0.9951   0.9954   0.9956   0.9971

``` r
pred20<-predict(rffit, newdata=testing)
# Output for the prediction of the 20 cases provided
pred20
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

### Method 2 SVM

``` r
set.seed(1234)
fitControl2<-trainControl(method="cv", number=2, allowParallel=T, verbose=T)
svmfit<-train(classe~.,data=TrainSubset, method="svmRadial", trControl=fitControl2, verbose=F)
```

    ## Loading required package: kernlab

    ## 
    ## Attaching package: 'kernlab'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     alpha

    ## + Fold1: sigma=0.01376, C=0.25 
    ## - Fold1: sigma=0.01376, C=0.25 
    ## + Fold1: sigma=0.01376, C=0.50 
    ## - Fold1: sigma=0.01376, C=0.50 
    ## + Fold1: sigma=0.01376, C=1.00 
    ## - Fold1: sigma=0.01376, C=1.00 
    ## + Fold2: sigma=0.01376, C=0.25 
    ## - Fold2: sigma=0.01376, C=0.25 
    ## + Fold2: sigma=0.01376, C=0.50 
    ## - Fold2: sigma=0.01376, C=0.50 
    ## + Fold2: sigma=0.01376, C=1.00 
    ## - Fold2: sigma=0.01376, C=1.00 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting sigma = 0.0138, C = 1 on full training set

``` r
svmfit$finalModel
```

    ## Support Vector Machine object of class "ksvm" 
    ## 
    ## SV type: C-svc  (classification) 
    ##  parameter : cost C = 1 
    ## 
    ## Gaussian Radial Basis kernel function. 
    ##  Hyperparameter : sigma =  0.0137558234122932 
    ## 
    ## Number of Support Vectors : 7006 
    ## 
    ## Objective Function Value : -1130.27 -844.447 -742.4992 -445.0451 -1088.381 -593.5092 -771.4257 -1030.911 -732.7647 -618.6445 
    ## Training error : 0.072432

``` r
predSVM<-predict(svmfit, newdata=TestSubset)
confusionMatrix(predrf, TestSubset$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674   10    0    0    0
    ##          B    0 1129    4    1    0
    ##          C    0    0 1018    5    2
    ##          D    0    0    4  957    4
    ##          E    0    0    0    1 1076
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9947          
    ##                  95% CI : (0.9925, 0.9964)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9933          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9912   0.9922   0.9927   0.9945
    ## Specificity            0.9976   0.9989   0.9986   0.9984   0.9998
    ## Pos Pred Value         0.9941   0.9956   0.9932   0.9917   0.9991
    ## Neg Pred Value         1.0000   0.9979   0.9984   0.9986   0.9988
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1918   0.1730   0.1626   0.1828
    ## Detection Prevalence   0.2862   0.1927   0.1742   0.1640   0.1830
    ## Balanced Accuracy      0.9988   0.9951   0.9954   0.9956   0.9971

``` r
pred20_svm <- predict(svmfit, newdata=testing)
pred20_svm
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Conclusion
----------

Both Random Forests and SVM gave same results.
