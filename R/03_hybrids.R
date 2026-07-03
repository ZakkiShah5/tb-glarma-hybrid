
# 03_hybrids.R -- Hybrid GLARMA-NegBin + ML models

# Dataset:  recife_tb_cases_2008_2025.csv
# Model:    GLARMA(1,0) NegBin K=2
# Hybrid:   ŷ_t = μ_g + ê_ML × √V(μ_g)  [Zhang 2003]
# Features: rho=6 lagged Pearson residuals
# Train:    Jan 2008 - Dec 2020 (n=156)
# Test:     Jan 2021 - Dec 2025 (n=60)
# =====================================================================


# Reproducibility ---------------------------------------------------------

set.seed(2026)
tensorflow::set_random_seed(2026)



# Confirmed GLARMA(1,0) NegBin K=2 parameters -----------------------------

# Source: 02_glarma_nb.R output

X_cols   <- c("intercept","sin1","cos1","sin2","cos2")
beta_est <- c(5.0558, -0.0095, -0.0135, -0.0043, -0.0396)
phi_est  <- 0.0342
THETA    <- 101.7225
ETA_CAP  <- log(max(df$cases) * 10)


# Define Lags -------------------------------------------------------------


rho       <- 6
LAG_NAMES <- paste0("lag", 1:rho)


# Load training residuals from RDS ----------------------------------------

g        <- readRDS("out/glarma_nb_fit.rds")
resid_tr <- g$resid_train

cat("=== 03_hybrids.R ===\n")
cat(sprintf("rho=%d lagged Pearson residuals as ML features\n", rho))
cat(sprintf("Train n=%d | Test n=%d\n\n",
            length(train_idx), length(test_idx)))


# Build training feature matrix -------------------------------------------

n_tr <- length(resid_tr)
lag_mat <- matrix(NA_real_, nrow = n_tr, ncol = rho)

for (k in (1:rho)) {
  if(k+1 <= n_tr)
   lag_mat[(k+1):n_tr,k] <- resid_tr[1:(n_tr-k)]
colnames(lag_mat) <- LAG_NAMES
}

ok_rows <- complete.cases(lag_mat) & is.finite(resid_tr)
feats <- lag_mat[ok_rows, , drop=FALSE]
target <- resid_tr[ok_rows]


cat(sprintf("Feature matrix: %d rows x %d cols (dropped %d for lag init)\n",
            nrow(feats), ncol(feats), sum(!ok_rows)))
cat(sprintf("Target: mean=%.4f  SD=%.4f\n\n", mean(target), sd(target)))


# Min-max scaling (train only -- no leakage) ------------------------------


mins <- apply(feats, 2, min)
maxs <- apply(feats, 2, max)
rngs <- pmax(maxs - mins, 1e-8)
fs   <- sweep(sweep(feats, 2, mins, "-"), 2, rngs, "/")
colnames(fs) <- LAG_NAMES

tmin <- min(target); tmax <- max(target)
ts   <- (target - tmin) / max(tmax - tmin, 1e-8)

safe_scale <- function(v) {
  v[!is.finite(v)] <- 0
  (v - mins) / rngs
}
unscale <- function(s) s * (tmax - tmin) + tmin

formula_ml <- as.formula(paste("target ~",
                               paste(LAG_NAMES, collapse=" + ")))
train_df   <- as.data.frame(fs)
train_df$target <- ts


# Fit ML models -----------------------------------------------------------

# ── Fit ANN ───────────────────────────────────────────────────────────
cat("Fitting ANN (1 hidden neuron)...\n")
fit_ann <- neuralnet(formula_ml, data=train_df,
                     hidden=1, linear.output=TRUE,
                     stepmax=1e6, threshold=0.01)

# ── Fit SVR ───────────────────────────────────────────────────────────
cat("Fitting SVR (RBF kernel)...\n")
fit_svr <- svm(formula_ml, data=train_df,
               kernel="radial", cost=1, epsilon=0.1, scale=FALSE)

# ── Fit RF ────────────────────────────────────────────────────────────
cat("Fitting RF (500 trees)...\n")
fit_rf <- randomForest(formula_ml, data=train_df,
                       ntree=500, nodesize=5)

# ── Fit XGBoost ───────────────────────────────────────────────────────
cat("Fitting XGBoost (100 rounds)...\n")
dtrain_xgb <- xgboost::xgb.DMatrix(data=as.matrix(fs), label=ts)
fit_xgb    <- xgboost::xgb.train(
  params  = list(max_depth=3, eta=0.1, objective="reg:squarederror"),
  data    = dtrain_xgb, nrounds=100, verbose=0)

# ── Fit LightGBM ──────────────────────────────────────────────────────
cat("Fitting LightGBM (300 rounds)...\n")
dtrain_lgb <- lgb.Dataset(data=as.matrix(fs), label=ts)
fit_lgb    <- lgb.train(
  params = list(objective="regression", metric="mse",
                num_leaves=31, learning_rate=0.05,
                min_data_in_leaf=3, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=5, verbose=-1),
  data=dtrain_lgb, nrounds=300, verbose=-1)

# ── Fit LSTM ──────────────────────────────────────────────────────────
cat("Fitting LSTM (100 epochs)...\n")
x_rnn <- array(as.matrix(fs), dim=c(nrow(fs), rho, 1))
  
lstm_model <- keras3::keras_model_sequential() |>
  keras3::layer_lstm(units=32, input_shape=c(rho,1)) |>
  keras3::layer_dense(units=16, activation="relu") |>
  keras3::layer_dense(units=1,  activation="linear")
lstm_model |> keras3::compile(
  optimizer=keras3::optimizer_adam(learning_rate=0.001), loss="mse")
suppressWarnings(lstm_model |> keras3::fit(
  x=x_rnn, y=ts, epochs=100, batch_size=16,
  validation_split=0.1, verbose=0))

# ── Fit GRU ───────────────────────────────────────────────────────────
cat("Fitting GRU (100 epochs)...\n")
gru_model <- keras3::keras_model_sequential() |>
  keras3::layer_gru(units=32, input_shape=c(rho,1)) |>
  keras3::layer_dense(units=16, activation="relu") |>
  keras3::layer_dense(units=1,  activation="linear")
gru_model |> keras3::compile(
  optimizer=keras3::optimizer_adam(learning_rate=0.001), loss="mse")
suppressWarnings(gru_model |> keras3::fit(
  x=x_rnn, y=ts, epochs=100, batch_size=16,
  validation_split=0.1, verbose=0))

cat("All models trained.\n\n")

# Fixed-origin one-step-ahead prediction loop -----------------------------


cat("Running predictions...\n")

n_test       <- length(test_idx)
glarma_preds <- numeric(n_test)
ann_preds    <- numeric(n_test)
svr_preds    <- numeric(n_test)
rf_preds     <- numeric(n_test)
xgb_preds    <- numeric(n_test)
lgb_preds    <- numeric(n_test)
lstm_preds   <- numeric(n_test)
gru_preds    <- numeric(n_test)

ext <- resid_tr   # grows with each test step

for (i in seq_along(test_idx)) {
  t_i <- test_idx[i]
  
  # GLARMA prediction
  X_t   <- matrix(as.numeric(df[t_i, X_cols]), nrow=1)
  eta_t <- as.numeric(X_t %*% beta_est)
  for (j in seq_along(phi_est)) {
    il <- length(ext) - j + 1
    if (il >= 1 && is.finite(ext[il]))
      eta_t <- eta_t + phi_est[j] * ext[il]
  }
  eta_t <- min(eta_t, ETA_CAP)
  mu_g  <- exp(eta_t)
  glarma_preds[i] <- mu_g
  
  # NegBin variance: V(mu) = mu + mu^2/theta
  v_t  <- mu_g + mu_g^2 / max(THETA, 0.01)
  sd_t <- sqrt(max(v_t, 1e-6))
  
  # Feature vector: last rho Pearson residuals scaled
  last_r   <- tail(ext, rho)
  
  feat_row <- matrix(safe_scale(last_r), nrow=1)
  colnames(feat_row) <- LAG_NAMES
  feat_df  <- as.data.frame(feat_row)
  
  # ML corrections (dimensionless Pearson residual predictions)
  corr_ann  <- unscale(as.numeric(
    predict(fit_ann, feat_df[, LAG_NAMES])))
  corr_svr  <- unscale(as.numeric(
    predict(fit_svr, feat_df)))
  corr_rf   <- unscale(as.numeric(
    predict(fit_rf, feat_df)))
  corr_xgb  <- unscale(predict(fit_xgb,
                               xgboost::xgb.DMatrix(data=as.matrix(feat_row))))
  corr_lgb  <- unscale(as.numeric(
    predict(fit_lgb, as.matrix(feat_row))))
  corr_lstm <- unscale({
    x3d <- array(as.numeric(feat_row), dim=c(1,rho,1))
    as.numeric(predict(lstm_model, x3d, verbose=0))})
  corr_gru  <- unscale({
    x3d <- array(as.numeric(feat_row), dim=c(1,rho,1))
    as.numeric(predict(gru_model, x3d, verbose=0))})
  
  # Hybrid = mu_g + correction * sd_t  [Zhang 2003]
  ann_preds[i]  <- pmax(mu_g + corr_ann  * sd_t, 0)
  svr_preds[i]  <- pmax(mu_g + corr_svr  * sd_t, 0)
  rf_preds[i]   <- pmax(mu_g + corr_rf   * sd_t, 0)
  xgb_preds[i]  <- pmax(mu_g + corr_xgb  * sd_t, 0)
  lgb_preds[i]  <- pmax(mu_g + corr_lgb  * sd_t, 0)
  lstm_preds[i] <- pmax(mu_g + corr_lstm * sd_t, 0)
  gru_preds[i]  <- pmax(mu_g + corr_gru  * sd_t, 0)
  
  # Update residuals with ACTUAL y(t) -- never the prediction
  r_t <- pmax(pmin((df$cases[t_i] - mu_g) / sd_t, 10), -10)
  ext <- c(ext, r_t)
}

cat("Done.\n\n")



# Save predictions CSV ----------------------------------------------------



preds_df <- data.frame(
  date        = df$date[test_idx],
  actual      = y_test,
  glarma      = round(glarma_preds, 3),
  hybrid_ann  = round(ann_preds,    3),
  hybrid_svr  = round(svr_preds,    3),
  hybrid_rf   = round(rf_preds,     3),
  hybrid_xgb  = round(xgb_preds,    3),
  hybrid_lgbm = round(lgb_preds,    3),
  hybrid_lstm = round(lstm_preds,   3),
  hybrid_gru  = round(gru_preds,    3)
)
write.csv(preds_df, "out/hybrid_preds.csv", row.names=FALSE)
cat(sprintf("Saved out/hybrid_preds.csv (n=%d)\n\n", nrow(preds_df)))


# Metrics summary ---------------------------------------------------------



mape_fn <- function(a,p) mean(abs(a-p)/pmax(a,1)*100, na.rm=TRUE)
rmse_fn <- function(a,p) sqrt(mean((a-p)^2, na.rm=TRUE))
mae_fn  <- function(a,p) mean(abs(a-p), na.rm=TRUE)
cor_fn  <- function(a,p) cor(a,p, use="complete.obs")

models <- c("glarma","hybrid_ann","hybrid_svr","hybrid_rf",
            "hybrid_xgb","hybrid_lgbm","hybrid_lstm","hybrid_gru")

cat(sprintf("%-15s %8s %8s %8s %8s\n",
            "Model","RMSE","MAE","MAPE%","COR"))
cat(strrep("-", 55), "\n")
for (m in models) {
  p <- preds_df[[m]]
  cat(sprintf("%-15s %8.3f %8.3f %8.3f %8.4f\n", m,
              rmse_fn(y_test,p), mae_fn(y_test,p),
              mape_fn(y_test,p), cor_fn(y_test,p)))
}



# Save model objects ------------------------------------------------------



saveRDS(list(
  fit_ann=fit_ann, fit_svr=fit_svr, fit_rf=fit_rf,
  fit_xgb=fit_xgb, fit_lgb=fit_lgb,
  lstm_model=lstm_model, gru_model=gru_model,
  mins=mins, maxs=maxs, rngs=rngs,
  tmin=tmin, tmax=tmax, rho=rho
), "out/hybrid_models.rds")
cat("Saved out/hybrid_models.rds\n")

cat("\n=== 03_hybrids.R complete ===\n")


