

# 00_setup.R
# Load data, create time features, define train/test indices
# Dataset: reife_tb_cases_2008_2025.csv
# Train: Jan 2001 - Dec 2020 240
# Test: Jan 2021 - Dec 2025 60
# nrow(df) = 300 rows


# Load libraries ----------------------------------------------------------

library(glarma)
library(MASS)
library(neuralnet)
library(e1071)
library(randomForest)
library(xgboost)
library(lightgbm)
library(keras3)
library(forecast)

set.seed(2026)
tensorflow::set_random_seed(2026)


# Load data ---------------------------------------------------------------


df <- read.csv("data/recife_tb_2001_2025.csv",
               stringsAsFactors = FALSE)
head(df)
df$date <- as.Date(df$date)


stopifnot(all(c("date", "cases", "split") %in% names(df)))
cat("Data Loaded:", nrow(df), "rows\n")
cat("Tain: ", sum(df$split=="train"), "| Test: ", sum(df$split=="test"), "\n")


# Time featuers -----------------------------------------------------------

df$month <- as.integer(format(df$date, "%m"))
df$intercept <- 1L

df$sin1 <- sin(2*pi*df$month/12)
df$cos1 <- cos(2*pi*df$month/12)
df$sin2 <- sin(4*pi*df$month/12)
df$cos2 <- cos(4*pi*df$month/12)



# Indices -----------------------------------------------------------------

train_idx <- which(df$split=="train")
test_idx <- which(df$split=="test")

y_train <- df$cases[train_idx]
y_test <- df$cases[test_idx]

cat(sprintf("Train %s to %s (n=%d)\n",
            min(df$date[train_idx]),
            max(df$date[train_idx]),
            length(train_idx)))
cat(sprintf("Test %s to %s (n=%d)\n",
            min(df$date[test_idx]),
            max(df$date[test_idx]),
            length(test_idx)))
cat(sprintf("Train: mean=%.1f min=%d max=%d",
            mean(y_train), min(y_train), max(y_train)))
cat(sprintf("Test: mean=%.1f min=%d max=%d",
            mean(y_test), min(y_test), max(y_test)))






cat("\n=== 00_setup.R complete ===\n")
