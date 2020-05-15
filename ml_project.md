---
title: "Machine Learning Project"
output:
  html_document:
    keep_md: true 
fontsize: 8pt
geometry: margin=0.75in
header-includes:
       - \usepackage{setspace}\singlespacing
---



## Executive Summary
Data was obtained from the Human Activity Recognition dataset. It includes sensor data collected when six participants performed one set of 10 repetitions of Unilateral Dumbell Biceps curl in 5 ways. Sensor locations were belt, glove, arm, and dumbbell. The classes are as follows: Class A =  exactly as specified, Class B = throwing elbows to the front, Class C = lifting the dumbbell only halfway, Class D = lowering the dumbbell only halfway, Class E = throwing the hips to the front. The purpose of this exercise was to create a model to predict the class from the remaining data.


## Getting and Cleaning Data

```r
training_data_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing_data_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(training_data_url, destfile="train_dataset.csv")
download.file(testing_data_url, destfile="test_dataset.csv")

train_data <- read.csv("train_dataset.csv", na.strings=c('#DIV/0!', '', 'NA'))
test_data <- read.csv("test_dataset.csv", na.strings=c('#DIV/0!', '', 'NA'))

# start with test data to see what data is available
not_all_na <- function(x) {!all(is.na(x))}
clean_test_data <- test_data %>% select_if(not_all_na)

# remove first 7 columns (not accelerometer data)
clean_test_data <- clean_test_data[,-c(1:7)]

# choose columns 1:52 for use in training rf model ('problem_id' is not relevant)
columns_to_use <- names(clean_test_data)[1:52]

# select columns from training data 
clean_train_data <- train_data[, c(columns_to_use,"classe")]
```
# Building the models

```r
# Test with 50% training 50% testing data

inTrain <- createDataPartition(y=clean_train_data$classe, p=0.5, list=FALSE)
training <- clean_train_data[inTrain,]
testing <- clean_train_data[-inTrain,]

# try a simple model with naive bayes
modelFit_nb <- train(classe ~ ., method="naive_bayes", data=training)
test_predictions_nb <- predict(modelFit_nb, testing[,-53])
confusionMatrix(testing$classe, test_predictions_nb)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2040  122  273  347    8
##          B  168 1319  231  161   19
##          C   98  125 1371  112    5
##          D   95    7  286 1122   98
##          E   49  204   85   79 1386
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7378         
##                  95% CI : (0.729, 0.7465)
##     No Information Rate : 0.2497         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6703         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8327   0.7423   0.6104   0.6161   0.9142
## Specificity            0.8981   0.9279   0.9551   0.9392   0.9497
## Pos Pred Value         0.7312   0.6949   0.8013   0.6978   0.7687
## Neg Pred Value         0.9416   0.9421   0.8920   0.9148   0.9838
## Prevalence             0.2497   0.1811   0.2290   0.1856   0.1545
## Detection Rate         0.2080   0.1345   0.1398   0.1144   0.1413
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8654   0.8351   0.7827   0.7777   0.9320
```

```r
# Accuracy is not great, try gbm method to see if it improves accuracy
ctrl <- trainControl(method = "cv", number=10)
modelFit_gbm <- train(classe ~ ., method="gbm",data=training, trControl=ctrl, verbose=FALSE)
test_predictions_gbm <- predict(modelFit_gbm, testing[,-53])
confusionMatrix(testing$classe, test_predictions_gbm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2743   31   10    4    2
##          B   79 1770   47    1    1
##          C    0   56 1637   13    5
##          D    2    7   61 1521   17
##          E    4   23   14   29 1733
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9586          
##                  95% CI : (0.9545, 0.9625)
##     No Information Rate : 0.2883          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9476          
##                                           
##  Mcnemar's Test P-Value : 3.045e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9699   0.9380   0.9254   0.9700   0.9858
## Specificity            0.9933   0.9838   0.9908   0.9894   0.9913
## Pos Pred Value         0.9832   0.9326   0.9568   0.9459   0.9612
## Neg Pred Value         0.9879   0.9852   0.9837   0.9943   0.9969
## Prevalence             0.2883   0.1924   0.1803   0.1598   0.1792
## Detection Rate         0.2796   0.1804   0.1669   0.1550   0.1767
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9816   0.9609   0.9581   0.9797   0.9885
```

```r
# Output from gbm looks promising, proceed with predicting 20 new classes for the test data set.
print(modelFit_gbm)
```

```
## Stochastic Gradient Boosting 
## 
## 9812 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 8831, 8831, 8831, 8831, 8831, 8831, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7559135  0.6905359
##   1                  100      0.8208342  0.7732634
##   1                  150      0.8540567  0.8153304
##   2                   50      0.8512031  0.8114638
##   2                  100      0.9045042  0.8791693
##   2                  150      0.9324292  0.9144978
##   3                   50      0.8970649  0.8697144
##   3                  100      0.9410921  0.9254672
##   3                  150      0.9601502  0.9495813
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```
# Predictions

```r
# select columns from test that were used in training 
clean_test_data <- test_data[, c(columns_to_use, 'problem_id')]

test_predictions_gbm <- predict(modelFit_gbm, clean_test_data[,-53])

test_predictions_gbm
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
