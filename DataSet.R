# Load necessary libraries
library(e1071)       # For SVM
library(caret)       # For model evaluation
library(datasets)    # For iris dataset
library(rpart)       # For decision tree
library(cluster)     # For hierarchical clustering
library(dplyr)       # For data manipulation
library(ggplot2)     # For plotting

# Load the iris dataset
data(iris)
head(iris)

# Split data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
trainData <- iris[trainIndex,]
testData <- iris[-trainIndex,]

### 1. Support Vector Machine (SVM) - Classification ###
svm_model <- svm(Species ~ ., data = trainData, kernel = "linear")
svm_predictions <- predict(svm_model, testData)
svm_conf_matrix <- confusionMatrix(svm_predictions, testData$Species)
print("SVM Model Accuracy:")
print(svm_conf_matrix$overall['Accuracy'])

### 2. Multiple Linear Regression ###
# Using Sepal.Length as the dependent variable, predicting it based on other features
linear_model <- lm(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, data = trainData)
linear_predictions <- predict(linear_model, testData)
linear_rmse <- sqrt(mean((linear_predictions - testData$Sepal.Length)^2))
print("Multiple Linear Regression RMSE:")
print(linear_rmse)

### 3. Hierarchical Clustering - Unsupervised Learning ###
# Prepare the data by removing the Species column
clustering_data <- scale(iris[, -5])
distance_matrix <- dist(clustering_data, method = "euclidean")
hc <- hclust(distance_matrix, method = "ward.D2")
plot(hc, main = "Hierarchical Clustering Dendrogram", xlab = "", sub = "")

# Cut the dendrogram into 3 clusters and compare with actual species
cluster_groups <- cutree(hc, k = 3)
table(cluster_groups, iris$Species)

### 4. Decision Tree Classification ###
tree_model <- rpart(Species ~ ., data = trainData, method = "class")
tree_predictions <- predict(tree_model, testData, type = "class")
tree_conf_matrix <- confusionMatrix(tree_predictions, testData$Species)
print("Decision Tree Model Accuracy:")
print(tree_conf_matrix$overall['Accuracy'])

# Visualize the decision tree
plot(tree_model)
text(tree_model, use.n = TRUE)

# Summary of models' performance
cat("\nSummary of Model Performance:\n")
cat("1. SVM Accuracy:", svm_conf_matrix$overall['Accuracy'], "\n")
cat("2. Multiple Linear Regression RMSE:", linear_rmse, "\n")
cat("3. Hierarchical Clustering groups:\n")
print(table(cluster_groups, iris$Species))
cat("4. Decision Tree Accuracy:", tree_conf_matrix$overall['Accuracy'], "\n")
