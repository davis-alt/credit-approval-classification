# ===================== Credit Card Data - SVM Classification =====================

# Clear environment
rm(list = ls())
set.seed(1) # reproducibility

# -------------------- Libraries --------------------
library(kernlab)

# -------------------- Load & Combine Data --------------------
# One file has headers, the other does not
data_with_headers <- read.delim("data/credit_card_data-headers.txt")
data_no_headers   <- read.delim("data/credit_card_data.txt", header = FALSE)

# Assign column names to no-header dataset
colnames(data_no_headers) <- colnames(data_with_headers)

# Combine both (unified dataset)
data <- rbind(data_with_headers, data_no_headers)

# Quick check
cat("Combined dataset dimensions:", dim(data), "\n")
head(data)

# -------------------- Identify target --------------------
target_col <- ncol(data)                   # Last column is target
target_name <- colnames(data)[target_col]  # Column name of target
formula <- as.formula(paste(target_name, "~ ."))

cat("Target column detected:", target_name, "\n")

# -------------------- Helper Functions --------------------

# Extract linear coefficients (a, a0) from a ksvm model
extract_coefficients <- function(model) {
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])
  a0 <- -model@b
  list(a = a, a0 = a0)
}

# Manual prediction function
manual_predict <- function(model, a, a0, data, scaled = TRUE) {
  n <- nrow(data)
  preds <- numeric(n)
  
  for (i in 1:n) {
    x <- as.numeric(data[i, -target_col])  # all predictors
    
    if (scaled && !is.null(model@scaling)) {
      x <- (x - model@scaling$x.scale$`scaled:center`) /
        model@scaling$x.scale$`scaled:scale`
    }
    
    preds[i] <- ifelse(sum(a * x) + a0 >= 0, 1, 0)
  }
  preds
}

# Accuracy function
accuracy <- function(preds, truth) {
  mean(preds == truth)
}

# -------------------- Train Models --------------------

# Scaled model
model_scaled <- ksvm(
  formula, data = data,
  type = "C-svc",
  kernel = "vanilladot",
  C = 100,
  scaled = TRUE
)

# Unscaled model
model_unscaled <- ksvm(
  formula, data = data,
  type = "C-svc",
  kernel = "vanilladot",
  C = 100,
  scaled = FALSE
)

# -------------------- Coefficients & Predictions --------------------

truth <- data[[target_name]]

# Scaled
coef_scaled <- extract_coefficients(model_scaled)
pred_scaled_manual <- manual_predict(model_scaled, coef_scaled$a, coef_scaled$a0, data, scaled = TRUE)
pred_scaled_builtin <- predict(model_scaled, data[, -target_col])

cat("\nScaled model accuracy (manual):", accuracy(pred_scaled_manual, truth))
cat("\nScaled model accuracy (predict()):", accuracy(pred_scaled_builtin, truth))

# Unscaled
coef_unscaled <- extract_coefficients(model_unscaled)
pred_unscaled_manual <- manual_predict(model_unscaled, coef_unscaled$a, coef_unscaled$a0, data, scaled = FALSE)
pred_unscaled_builtin <- predict(model_unscaled, data[, -target_col])

cat("\n\nUnscaled model accuracy (manual):", accuracy(pred_unscaled_manual, truth))
cat("\nUnscaled model accuracy (predict()):", accuracy(pred_unscaled_builtin, truth), "\n")

# -------------------- Display classifier equations --------------------

cat("\nScaled model classifier equation:\n")
cat("f(x) = sign(", round(coef_scaled$a0, 6), sep = "")
for (i in 1:length(coef_scaled$a)) {
  cat(" + ", round(coef_scaled$a[i], 6), "*x", i, sep = "")
}
cat(")\n")

cat("\nUnscaled model classifier equation:\n")
cat("f(x) = sign(", round(coef_unscaled$a0, 6), sep = "")
for (i in 1:length(coef_unscaled$a)) {
  cat(" + ", round(coef_unscaled$a[i], 6), "*x", i, sep = "")
}
cat(")\n")

# -------------------- Generate Confusion Matrix --------------------
library(caret)  # for confusionMatrix

# Predictions from scaled model
pred_scaled <- predict(model_scaled, data[, -target_col])

# Convert to factors to avoid warnings
truth_factor <- factor(data[[target_col]])
pred_factor  <- factor(pred_scaled, levels = levels(truth_factor))

# Confusion matrix
cm <- confusionMatrix(pred_factor, truth_factor)
print(cm)


fourfoldplot(cm$table, color = c("skyblue", "pink"),
             conf.level = 0, margin = 1,
             main = "SVM Confusion Matrix")
