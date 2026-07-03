

# Pure ML models (no GLARMA component)

# Dataset:  recife_tb_cases_2001_2025.csv
# Features: rho=6 lagged ACTUAL case counts (NOT Pearson residuals)
# Target:   actual case count y_t (NOT a correction)
# Purpose:  benchmark - can ML alone match GLARMA+ML hybrid?
# Train:    Jan 2001 - Dec 2020 (n=240)
# Test:     Jan 2021 - Dec 2025 (n=60)



# Setup -------------------------------------------------------------------


set.seed(2026)
tensorflow::set_random_seed(2026)

rho <- 6
LAG_NAMES <- paste0("lag", 1:rho)

cat("=== 04_pure_ml.R ===\n")
cat(sprintf("rho = %d lagged case counts as features \n", rho))
cat(sprintf("Train n = %d | Test n = %d\n\n", 
            length(train_idx), length(test_idx)))



# Build feature matrix from lagged case counts ----------------------------


y_full <- df$cases 
n_tr <- length(train_idx)

lag_mat <- matrix(NA_real_, nrow = n_tr, ncol = rho)
for (k in 1:rho) {
  for (i in 1:n_tr) {
    t_i <- train_idx[i]
    lag_idx <- t_i - k
    if(lag_idx >=1){
      lag_mat[i, k] <- y_full[lag_idx]
    }
  }
}

colnames(lag_mat) <- LAG_NAMES

ok_rows <- complete.cases(lag_mat)
feats <- lag_mat[ok_rows, ,drop=FALSE]
target <- y_full[train_idx][ok_rows]

cat(sprintf("Feature matrix: %d rows x %d cols (dropped %d for lag init)\n",
            nrow(feats), ncol(feats), sum(!ok_rows)))
cat(sprintf("Target (case counts): mean=%.1f  min=%d  max=%d\n\n",
            mean(target), min(target), max(target)))



# Min-max scaling (train set only) ----------------------------------------


mins <- apply(feats, 2, min)
maxs <- apply(feats, 2, max)
rngs <- pmax(maxs - mins, 1e-8)
fs <-  sweep(sweep(feats,2,mins,"-"), 2, rngs, "/")
colnames(fs) <- LAG_NAMES


tmin <- min(target)
tmax <- max(target)
ts <- (target-tmin)/ max(tmax - tmin, 1e-8)


safe_scale <- function(v){
  v[!is.finite(v)] <- 0
  (v-mins)/rngs
}
unscale <- function(s){
  s * (tmax - tmin) + tmin
}

formula_ml <-  as.formula(paste("target ~",
                                paste(LAG_NAMES, collapse = " + ")))
train_df <- as.data.frame(fs)
train_df$target <-ts



# Fitting ML models -------------------------------------------------------

# Fit ANN

cat("Fitting ANN...\n")

fit_ann <- neuralnet(formula_ml, data=train_df,
                     hidden=1, linear.output = TRUE,
                     stepmax=1e6, threshold=0.01)


# Fit SVR

cat("Fitting SVR...\n")

fit_svr <- svm(formula_ml, data = train_df,
               kernel = "radial", cost = 1,
               eps = 0.01, scale = FALSE)


# Fit RandomForest

cat("Fitting RandomForest...\n")

fit_rf <- randomForest(formula_ml, data = train_df,
                       ntree = 500, nodesize = 5)


# Fit XGBoost

cat("Fitting XGBoost...\n")

dtrain_xgb <- xgb.DMatrix(data= as.matrix(fs), label = ts)

fit_xgb <- xgboost::xgb.train(
  params = list(max_depth=3, eta=0.1, objective= "reg:squarederror"),
  data = dtrain_xgb, nrounds = 100, verbose = 0
)

# Fit LightGBM

cat("Fiting LightGBM...\n")

dtrain_lgb <- lgb.Dataset(data = as.matrix(fs), label = ts)

fit_lgb <- lgb.train(
  params = list(objective= "regression", metric = "mse",
                num_leaves = 31, learning_rate = 0.05,
                min_data_in_leaf=3, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=5,
                verbose=-1),
  data = dtrain_lgb, nrounds = 300, verbose = -1
)



# Fit LSTM

cat("Fitting LSTM...\n")

x_rnn <- array(as.matrix(fs), dim = c(nrow(fs), rho, 1))

lstm_model <- keras3::keras_model_sequential() |>
  keras3::layer_lstm(units = 32, input_shape = c(rho, 1)) |>
  keras3::layer_dense(units = 16, activation = "relu") |>
  keras3::layer_dense(units = 1, activation = "linear")
lstm_model |> keras3::compile(
  optimizer=keras3::optimizer_adam(learning_rate = 0.001),
  loss="mse")
suppressWarnings(lstm_model |> keras3::fit(
  x=x_rnn, y=ts, epochs=100, batch_size=16, 
  validation_split = 0.1, verbose= 0
))

# Fit GRU

cat("Fitting GRU...\n")

gru_model <- keras3::keras_model_sequential() |>
  keras3::layer_gru(units = 32, input_shape = c(rho, 1)) |>
  keras3::layer_dense(units = 16, activation = 'relu') |>
  keras3::layer_dense(units = 1, activation = 'linear')
gru_model |> keras3::compile(
  optimizer=keras3::optimizer_adam(learning_rate = 0.001),
  loss="mse"
)
suppressWarnings(gru_model |> keras3::fit(
  x=x_rnn, y=ts, epochs=100, batch_size=16,
  validation_split = 0.1, verbose = 0 
))



cat("All models trained.\n\n")



# Fixed-origin one-step-ahead prediction loop -----------------------------

cat("Running predictions...\n")

n_test <- length(test_idx)
ann_preds <- numeric(n_test)
svr_preds <- numeric(n_test)
rf_preds <- numeric(n_test)
xgb_preds <- numeric(n_test)
lgb_preds <- numeric(n_test)
lstm_preds <- numeric(n_test)
gru_preds <- numeric(n_test)

for (i in seq_along(test_idx)) {
  t_i <- test_idx[i]
  
  # Feature vector: last rho Actual case counts
  last_y <- numeric(rho)
  for (k in 1:rho) {
    lag_idx <- t_i - k
    if(lag_idx >= 1)
      last_y[k] <- y_full[lag_idx]
  }
  
  # scale using training stats
  feat_row <- matrix(safe_scale(last_y), nrow = 1)
  colnames(feat_row) <- LAG_NAMES
  feat_df <- as.data.frame(feat_row)
  
  # Predict and unscale back to case counts
  ann_preds[i] <- pmax(unscale(as.numeric(
    predict(fit_ann, feat_df[, LAG_NAMES])
  )), 0)
  
  svr_preds[i] <- pmax(unscale(as.numeric(
    predict(fit_svr, feat_df)
  )), 0)
  
  rf_preds[i] <- pmax(unscale(as.numeric(
    predict(fit_rf, feat_df)
  )), 0)
  
  xgb_preds[i] <- pmax(unscale(
    predict(fit_xgb, xgboost::xgb.DMatrix(data=as.matrix(feat_row)))
  ), 0)
  
  lgb_preds[i] <- pmax(unscale(as.numeric(
    predict(fit_lgb, as.matrix(feat_row))
  )), 0)
  
  
  lstm_preds[i] <- pmax(unscale({
    x3d <- array(as.numeric(feat_row), dim = c(1,rho,1))
    as.numeric(predict(lstm_model, x3d, verbose = 0))
  }), 0)
  
  
  gru_preds[i]  <- pmax(unscale({
    x3d <- array(as.numeric(feat_row), dim=c(1,rho,1))
    as.numeric(predict(gru_model, x3d, verbose=0))}), 0)
  
}

cat("Done. \n\n")


# Save Predictions --------------------------------------------------------


pure_df <-  data.frame(
  date = df$date[test_idx],
  actual = y_test,
  pure_ann = round(ann_preds, 3),
  pure_svr = round(svr_preds, 3),
  pure_rf = round(rf_preds, 3),
  pure_xgb = round(xgb_preds, 3),
  pure_lgbm = round(lgb_preds, 3),
  pure_lstm = round(lstm_preds, 3),
  pure_gru = round(gru_preds, 3)
)

write.csv(pure_df, "out/pure_ml_preds.csv", row.names = FALSE)
cat(sprintf("Saved out/pure_ml_preds.csv (n = %d)\n\n", nrow(pure_df)))



# Metrics -----------------------------------------------------------------

mape_fn <- function(a,p) mean(abs(a-p)/pmax(a,1)*100, na.rm=TRUE)
rmse_fn <- function(a,p) sqrt(mean((a-p)^2, na.rm=TRUE))
mae_fn  <- function(a,p) mean(abs(a-p), na.rm=TRUE)
cor_fn  <- function(a,p) cor(a,p, use="complete.obs")

models <- c("pure_ann","pure_svr","pure_rf",
            "pure_xgb","pure_lgbm","pure_lstm","pure_gru")

cat(sprintf("%-15s %8s %8s %8s %8s\n",
            "Model","RMSE","MAE","MAPE%","COR"))
cat(strrep("-", 55), "\n")
for (m in models) {
  p <- pure_df[[m]]
  cat(sprintf("%-15s %8.3f %8.3f %8.3f %8.4f\n", m,
              rmse_fn(y_test,p), mae_fn(y_test,p),
              mape_fn(y_test,p), cor_fn(y_test,p)))
}



# Save models -------------------------------------------------------------

saveRDS(list(
  fit_ann    = fit_ann,
  fit_svr    = fit_svr,
  fit_rf     = fit_rf,
  fit_xgb    = fit_xgb,
  fit_lgb    = fit_lgb,
  lstm_model = lstm_model,
  gru_model  = gru_model,
  mins       = mins,
  maxs       = maxs,
  rngs       = rngs,
  tmin       = tmin,
  tmax       = tmax,
  rho        = rho
), "out/pure_ml_models.rds")
cat("Saved out/pure_ml_models.rds\n")

cat("\n=== 04_pure_ml.R complete ===\n")
