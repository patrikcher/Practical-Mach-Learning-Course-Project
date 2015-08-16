library(ggplot2)
library(caret)
library(randomForest)
library(doMC)

# use 4 cores
registerDoMC(cores=4)

setwd("/Users/patrickcher/Dropbox/Learning/Data Science/JHU Data Science/08 Practical Machine Learning/Course Project/Practical Machine Learning")

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
# load data from both training and testing set
# replace #DIV/0! values with NA
trainingData <- read.csv("pml-training.csv", row.names = 1, na.strings = c("#DIV/0!"))
testingData <- read.csv("pml-testing.csv", row.names = 1, na.strings = c("#DIV/0!"))

# remove near zero covariates
nearZeroCovar <- nearZeroVar(trainingData, saveMetrics = T)
trainingData <- trainingData[, !nearZeroCovar$nzv]

# remove variables with more than 80% missing values
naValues <- sapply(colnames(trainingData), 
              function(x) 
                if(sum(is.na(trainingData[, x])) > 0.8*nrow(trainingData)) {
                  return(T)
                }
                else{
                  return(F)
                  }
              )

trainingData <- trainingData[, !naValues]

# calculate correlations
correlations <- abs(sapply(colnames(trainingData[, -ncol(trainingData)]), 
                           function(x) 
                             cor(as.numeric(trainingData[, x]), 
                                 as.numeric(trainingData$classe), 
                                 method="spearman")
                           )
                    )

# plot predictors 
summary(correlations)

plot(trainingData[, names(which.max(correlations))], 
     trainingData[, names(which.max(correlations[-which.max(correlations)]))], 
     col = trainingData$classe, pch = 19, cex = 0.1, 
     xlab = names(which.max(correlations)), 
     ylab = names(which.max(correlations[-which.max(correlations)])))


# boosting model
set.seed(888)
# Fit model with boosting algorithm and 10-fold cross validation 
# to predict classes with other predictors
boostFit <- train(classe ~ ., method="gbm", data = trainingData, 
                  verbose = F, trControl = trainControl(method = "cv", number = 10))



# Fit model with random forests algorithm and 10-fold cross validation 
# to predict classe with all other predictors.
set.seed(888)
rfFit <- train(classe ~ ., method = "rf", data = trainingData, 
               importance = T, trControl = trainControl(method = "cv", number = 10))

rfFit
# Plot accuracy of the model on the same scale as boosting model.
plot(rfFit, ylim = c(0.9, 1))

impt <- varImp(rfFit)$importance
impt$max <- apply(impt, 1, max)
impt <- impt[order(impt$max, decreasing = T), ]


# final model
rfFit$finalModel
# prediction
(prediction <- as.character(predict(rfFit, testingData)))

# write prediction files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./prediction/problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}

pml_write_files(prediction)
