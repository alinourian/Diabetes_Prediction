## ======================= Import Libraries & Load Data ========================

library(data.table)
library(ggplot2)
library(glmnet)
library(caret)
library(randomForest)
library(class)
library(e1071)
library(MASS)

dataset = fread('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

dataset$Diabetes_binary = as.factor(dataset$Diabetes_binary)
head(dataset, 5)

## ================================ Question 1 =================================


ggplot(dataset, aes(BMI, fill = Diabetes_binary))+
  geom_density(alpha = .75)

dataset$one = 1
ds = dataset[, .(n = sum(one)), .(Diabetes_binary, NoDocbcCost)]
ds[, n_total := sum(n), .(NoDocbcCost)]
ds[, n_percent := n / n_total]

ggplot(ds, aes(as.factor(NoDocbcCost), n_percent, fill = Diabetes_binary))+
  geom_bar(stat = 'identity', )

summary(dataset)

pairs(dataset[, c("GenHlth", "MentHlth", "PhysHlth", "Income", "BMI", "Age", "Education", "Diabetes_binary")])

# Histogram of HighBP
ggplot(dataset, aes(x = HighBP)) +
  geom_histogram(fill = "blue", binwidth = 0.5, alpha = 0.8)

# Histogram of HighChol
ggplot(dataset, aes(x = HighChol)) +
  geom_histogram(fill = "orange", binwidth = 0.5, alpha = 0.8)

# Histogram of CholCheck
ggplot(dataset, aes(x = CholCheck)) +
  geom_histogram(fill = "green", binwidth = 0.5, alpha = 0.8)

# Histogram of BMI
ggplot(dataset, aes(x = BMI)) +
  geom_histogram(fill = "red", binwidth = 1, alpha = 0.8)

# Histogram of CholCheck
ggplot(dataset, aes(x = Smoker)) +
  geom_histogram(fill = "yellow", binwidth = 0.5, alpha = 0.8)

# Histogram of PhysActivity
ggplot(dataset, aes(x = PhysActivity)) +
  geom_histogram(fill = "purple", binwidth = 0.5, alpha = 0.8)

# Histogram of HvyAlcoholConsump
ggplot(dataset, aes(x = HvyAlcoholConsump)) +
  geom_histogram(fill = "pink", binwidth = 0.5, alpha = 0.8)

# Histogram of GenHlth
ggplot(dataset, aes(x = GenHlth)) +
  geom_histogram(fill = "blue", binwidth = 0.5, alpha = 0.8)

# Histogram of MentHlth
ggplot(dataset, aes(x = MentHlth)) +
  geom_histogram(fill = "orange", binwidth = 0.8, alpha = 0.8)

# Histogram of PhysHlth
ggplot(dataset, aes(x = PhysHlth)) +
  geom_histogram(fill = "green", binwidth = 0.8, alpha = 0.8)

# Histogram of Sex
ggplot(dataset, aes(x = Sex)) +
  geom_histogram(fill = "red", binwidth = 0.5, alpha = 0.8)

# Histogram of Age
ggplot(dataset, aes(x = Age)) +
  geom_histogram(fill = "yellow", binwidth = 0.5, alpha = 0.8)

# Histogram of Education
ggplot(dataset, aes(x = Education)) +
  geom_histogram(fill = "purple", binwidth = 0.5, alpha = 0.8)

# Histogram of Income
ggplot(dataset, aes(x = Income)) +
  geom_histogram(fill = "pink", binwidth = 0.5, alpha = 0.8)


color = c("blue", "red")

mosaicplot(HighBP ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(HighChol ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(CholCheck ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(Stroke ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(HeartDiseaseorAttack ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(Sex ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(Smoker ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(PhysActivity ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(Veggies ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(Fruits ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(HvyAlcoholConsump ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(AnyHealthcare ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(NoDocbcCost ~ Diabetes_binary, data = dataset, color=color)
mosaicplot(DiffWalk ~ Diabetes_binary, data = dataset, color=color)

# Boxplot of BMI by diabetes status
ggplot(dataset, aes(x = BMI, y = Diabetes_binary, color = Diabetes_binary, group = Diabetes_binary)) +
  geom_boxplot(alpha = 0.75)

# Boxplot of GenHlth by diabetes status
ggplot(dataset, aes(x = GenHlth, y = Diabetes_binary, color = Diabetes_binary, group = Diabetes_binary)) +
  geom_boxplot(alpha = 0.75)

# Boxplot of MentHlth by diabetes status
ggplot(dataset, aes(x = MentHlth, y = Diabetes_binary, color = Diabetes_binary, group = Diabetes_binary)) +
  geom_boxplot(alpha = 0.75)

# Boxplot of PhysHlth by diabetes status
ggplot(dataset, aes(x = PhysHlth, y = Diabetes_binary, color = Diabetes_binary, group = Diabetes_binary)) +
  geom_boxplot(alpha = 0.75)

# Boxplot of Age by diabetes status
ggplot(dataset, aes(x = Age, y = Diabetes_binary, color = Diabetes_binary, group = Diabetes_binary)) +
  geom_boxplot(alpha = 0.75)

# Boxplot of Education by diabetes status
ggplot(dataset, aes(x = Education, y = Diabetes_binary, color = Diabetes_binary, group = Diabetes_binary)) +
  geom_boxplot(alpha = 0.75)

# Boxplot of Income by diabetes status
ggplot(dataset, aes(x = Income, y = Diabetes_binary, color = Diabetes_binary, group = Diabetes_binary)) +
  geom_boxplot(alpha = 0.75)

##  ============================== Question 2 & 3 & 4 =============================


# Load dataset
data <- read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# Shuffle the dataset
set.seed(123)
shuffled_data <- data[sample(nrow(data)), ]

# Split the shuffled data into training and testing sets
train_ratio <- 0.7  # Set the desired training set ratio
train_indices <- createDataPartition(shuffled_data$Diabetes_binary, p = train_ratio, list = FALSE)
train_data <- shuffled_data[train_indices, ]
test_data <- shuffled_data[-train_indices, ]


# Create a logistic regression model
logistic_model <- glm(Diabetes_binary ~ ., data = train_data, family = binomial)
logistic_predictions <- predict(logistic_model, newdata = test_data, type = "response")
summary_data <- summary(logistic_model)

# Extract p-values for each feature
p_values <- summary_data$coefficients[, "Pr(>|z|)"]

# Extract feature coefficients and their names
coefficients <- coef(logistic_model)[-1]  # Exclude the intercept
feature_names <- names(coefficients)

# Sort the coefficients by absolute value in descending order
sorted_indices <- order(abs(coefficients), decreasing = TRUE)
sorted_coefficients <- coefficients[sorted_indices]
sorted_feature_names <- feature_names[sorted_indices]

# Print the feature names and their corresponding coefficients
for (i in seq_along(sorted_coefficients)) {
  feature <- sorted_feature_names[i]
  coefficient <- sorted_coefficients[i]
  p_value <- p_values[feature]
  cat("Feature:", feature, ",\tCoefficient:", round(coefficient, 5), ",\tp_value:", p_value, "\n")
}



# Create a logistic regression model with stepwise feature selection
model <- step(glm(Diabetes_binary ~ ., data = train_data), direction = "both", trace = TRUE)
summary(model)

# Make predictions on the test data
predictions <- predict(model, newdata = test_data, type = "response")
logistic_accuracy <- sum(predictions == test_data$Diabetes_binary) / length(test_data$Diabetes_binary)
cat("Accuracy:", logistic_accuracy, "\n")


# Using random forest instead of stepwise selection:

# Split the data into features and target variable
features <- data[, -1]  # Exclude the target variable
target <- data$Diabetes_binary

# Create a Random Forest model
rf_model <- randomForest(features, target, ntree=100, importance=TRUE, do.trace=10) # ntree=250 comes into a good result
rf_predictions <- predict(rf_model, newdata = test_data, type = "class")


# Get feature importances
importances <- importance(rf_model)

print(rf_model)
print(importances)

# Plot the variable importance for the model
varImpPlot(rf_model, main = "Variable Importance")

rf_df <- as.data.frame(rf_model$importance)

# Plot the variable importance for the model using all the variables
ggplot(rf_df, aes(x = reorder(rownames(rf_df), IncNodePurity), y = IncNodePurity)) +
  geom_col(fill = "red") +
  coord_flip() +
  labs(x = "Variables", y = "Mean Decrease in Gini", title = "Variable Importance")



# Using KNN:
# Train and evaluate the K-nearest neighbors model
knn_model <- knn(train_data[, -1], test_data[, -1], train_data$Diabetes_binary, k = 5)
knn_predictions <- as.factor(knn_model)
knn_accuracy <- sum(knn_predictions == test_data$Diabetes_binary) / length(test_data$Diabetes_binary)
cat("Accuracy:", knn_accuracy, "\n")


best_features = c("BMI", "HighBP", "GenHlth", "Age", "PhysHlth", "HighChol", "DiffWalk", "CholCheck", "HvyAlcoholConsump", "Diabetes_binary")

new_data = data[, best_features]
shuffled_data <- new_data[sample(nrow(data)), ]

# Split the shuffled data into training and testing sets
train_ratio <- 0.7  # Set the desired training set ratio
train_indices <- createDataPartition(shuffled_data$Diabetes_binary, p = train_ratio, list = FALSE)
train_data <- shuffled_data[train_indices, ]
test_data <- shuffled_data[-train_indices, ]


# logistic
# Create a logistic regression model
logistic_model <- glm(Diabetes_binary ~ ., data = train_data, family = binomial)
logistic_predictions <- predict(logistic_model, newdata = test_data, type = "response")
summary_data <- summary(logistic_model)
print(summary_data)

# LDA
# Extract the predictor variables and the target variable from the train and test data
train_x <- train_data[, -1]
train_y <- train_data[, 1]
test_x <- test_data[, -1]
test_y <- test_data[, 1]


library(MASS)
library(gmodels)
# Apply Linear Discriminant Analysis (LDA)
lda_model <- lda(train_x, train_y)

# Predict using LDA
lda_pred <- predict(lda_model, test_x)

# Calculate accuracy for LDA
lda_accuracy = sum(lda_pred$class == test_y) / length(test_y)

# Print the accuracies
cat("LDA Accuracy:", lda_accuracy, "\n")
summary(lda_model)


# KNN
# Train and evaluate the K-nearest neighbors model
knn_model <- knn(train_data[, -1], test_data[, -1], train_data$Diabetes_binary, k = 5)
knn_predictions <- as.factor(knn_model)
knn_accuracy <- sum(knn_predictions == test_data$Diabetes_binary) / length(test_data$Diabetes_binary)
print(knn_accuracy)

