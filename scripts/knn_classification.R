# ===================== Credit Card Data - KNN Classification =====================

# Clear environment
rm(list = ls())
set.seed(1)

# -------------------- Libraries --------------------
library(kknn)
library(ggplot2)

# -------------------- Load & Combine Data --------------------
# Read files: one with headers, one without
data_with_headers <- read.delim("data/credit_card_data-headers.txt")
data_no_headers   <- read.delim("data/credit_card_data.txt", header = FALSE)

# Assign column names to no-header dataset
colnames(data_no_headers) <- colnames(data_with_headers)

# Combine both datasets
data <- rbind(data_with_headers, data_no_headers)

# Check the dataset
cat("Combined dataset dimensions:", dim(data), "\n")
head(data)

# -------------------- Identify target --------------------
target_col <- ncol(data)                   # Last column is target
target_name <- colnames(data)[target_col]  # Column name of target
predictors <- colnames(data)[-target_col]  # All other columns

cat("Target column detected:", target_name, "\n")

# -------------------- Accuracy Function --------------------
check_knn_accuracy <- function(k_value) {
  predicted <- numeric(nrow(data))
  
  # Leave-one-out cross-validation
  for (i in 1:nrow(data)) {
    model <- kknn(
      formula = as.formula(paste(target_name, "~ .")),
      train = data[-i, ],
      test  = data[i, ],
      k = k_value,
      scale = TRUE
    )
    # Round prediction to 0/1
    predicted[i] <- as.integer(fitted(model) + 0.5)
  }
  
  # Accuracy fraction
  sum(predicted == data[[target_name]]) / nrow(data)
}

# -------------------- Evaluate K values --------------------
k_max <- 20
accuracies <- numeric(k_max)

for (k in 1:k_max) {
  accuracies[k] <- check_knn_accuracy(k)
}

# -------------------- Report & Plot --------------------
# Print accuracy table
accuracy_table <- data.frame(k = 1:k_max, Accuracy = accuracies)
print(accuracy_table)

# Plot accuracy vs k
ggplot(accuracy_table, aes(x = k, y = Accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "KNN Accuracy vs Number of Neighbors",
       x = "Number of Neighbors (k)",
       y = "Accuracy") +
  theme_minimal()

# -------------------- Best K --------------------
best_k <- accuracy_table$k[which.max(accuracy_table$Accuracy)]
best_acc <- max(accuracy_table$Accuracy)
cat(sprintf("\nBest k: %d with accuracy: %.4f (%.2f%%)\n", best_k, best_acc, best_acc * 100))
