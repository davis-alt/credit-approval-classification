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

# Combine both (in case you need a unified dataset)
data <- rbind(data_with_headers, data_no_headers)

# Quick check
cat("Combined dataset dimensions:", dim(data), "\n")
head(data)

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
    x <- data[i, 1:10]
    
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
  V11 ~ ., data = data,
  type = "C-svc",
  kernel = "vanilladot",
  C = 100,
  scaled = TRUE
)

# Unscaled model
model_unscaled <- ksvm(
  V11 ~ ., data = data,
  type = "C-svc",
  kernel = "vanilladot",
  C = 100,
  scaled = FALSE
)

# -------------------- Coefficients & Predictions --------------------

# Scaled
coef_scaled <- extract_coefficients(model_scaled)
pred_scaled_manual <- manual_predict(model_scaled, coef_scaled$a, coef_scaled$a0, data, scaled = TRUE)
pred_scaled_builtin <- predict(model_scaled, data[, 1:10])

cat("\nScaled model accuracy (manual):", accuracy(pred_scaled_manual, data$V11))
cat("\nScaled model accuracy (predict()):", accuracy(pred_scaled_builtin, data$V11))

# Unscaled
coef_unscaled <- extract_coefficients(model_unscaled)
pred_unscaled_manual <- manual_predict(model_unscaled, coef_unscaled$a, coef_unscaled$a0, data, scaled = FALSE)
pred_unscaled_builtin <- predict(model_unscaled, data[, 1:10])

cat("\n\nUnscaled model accuracy (manual):", accuracy(pred_unscaled_manual, data$V11))
cat("\nUnscaled model accuracy (predict()):", accuracy(pred_unscaled_builtin, data$V11), "\n")
