

# 02_glarma_nb.R
# Fit GLARMA NegBin to TB confirmed cases


# NB-GLM warm start -------------------------------------------------------
# Used to get initial beta and alpha for GLARMA fitting
mean(y_train)
var(y_train)
# warm start K=2

nb_glm_k2 <- glm.nb(y_train ~ -1 + as.matrix(
  df[train_idx, c("intercept", "sin1", "cos1", "sin2", "cos2")]),
  control = glm.control(maxit = 200)
  )

beta_init_K2 <- unname(coef(nb_glm_k2))
alpha_init_K2 <- nb_glm_k2$theta

cat(sprintf("NB-GLM K=2 warm start: alpha=%.4f  AIC=%.2f\n\n",
            alpha_init_K2, AIC(nb_glm_k2)))

# warm start K= 1

nb_glm_k1 <- glm.nb(y_train ~ -1 + as.matrix(
  df[train_idx, c("intercept", "sin1", "cos1")]),
  control = glm.control(maxit = 200)
  )
beta_init_K1 <- unname(coef(nb_glm_k1))
alpha_init_K1 <- nb_glm_k1$theta

cat(sprintf("NB-GLM K=1 warm start: alpha=%.4f  AIC=%.2f\n\n",
            alpha_init_K1, AIC(nb_glm_k1)))



# Design martix  -----------------------------------------------------------

# K=2

X_cols_K2 <- c("intercept", "sin1", "cos1", "sin2", "cos2")
X_train_K2 <-as.matrix(df[train_idx, X_cols_K2])

# K=1

X_cols_K1 <- c("intercept", "sin1", "cos1")
X_train_K1 <-as.matrix(df[train_idx, X_cols_K1])


# Manual AIC calculator: includes alpha in k (package excludes it) --------

aic_manual <- function(fit, p, q, n_beta){
  ll <- as.numeric(logLik(fit))
  k  <- n_beta + p + q + 1L
  -2 * ll + 2 * k
}


# Models (K=2) (one by one)  -----------------------------------------------------


cat("Model comparison (K=2 Fourier: sin1,cos1,sin2,cos2):\n")
cat(sprintf("%-12s %6s %10s %10s\n", "Model", "iter", "logLik", "AIC"))
cat(strrep("-", 42), "\n")


# GLARMA (0,0) (No ARMA)
fit_00 <- glarma(y=y_train, X=X_train_K2, type = "NegBin",
                 residuals = "Pearson", method = "NR",
                 beta = beta_init_K2, alphaInit = alpha_init_K2,
                 maxit = 500, grad = 1e-6
                 )
cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(0,0)",
            fit_00$iter, logLik(fit_00), aic_manual(fit_00,0,0, length(X_cols_K2))))

# GLARMA (1,0) --- AR(1)
fit_10 <- glarma(y=y_train, X=X_train_K2, type = "NegBin",
                 residuals = "Pearson", method = "FS",
                 phiLags = 1, phiInit = 0,
                 beta = beta_init_K2, alphaInit = alpha_init_K2,
                 maxit = 500, grad = 1e-6
)
cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(1,0)",
            fit_10$iter, logLik(fit_10), aic_manual(fit_10,1,0, length(X_cols_K2))))

# GLARMA (2,0) --- AR(2)
fit_20 <- glarma(y=y_train, X=X_train_K2, type = "NegBin",
                 residuals = "Pearson", method = "FS",
                 phiLags = 1:2, phiInit = c(0,0),
                 beta = beta_init_K2, alphaInit = alpha_init_K2,
                 maxit = 500, grad = 1e-6
)
cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(2,0)",
            fit_20$iter, logLik(fit_20), aic_manual(fit_20,2,0, length(X_cols_K2))))

# GLARMA(0,1) -- MA(1)
fit_01 <- glarma(y=y_train, X=X_train_K2, type="NegBin",
                 residuals="Pearson", method="FS",
                 thetaLags=1, thetaInit=0,
                 beta=beta_init_K2, alphaInit=alpha_init_K2,
                 maxit=5000, grad=1e-6)

cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(0,1)",
            fit_01$iter, logLik(fit_01), aic_manual(fit_01,0,1, length(X_cols_K2))))

# GLARMA(0,2) -- MA(2)

fit_02 <- glarma(y=y_train, X=X_train_K2, type="NegBin",
                 residuals="Pearson", method="FS",
                 thetaLags=1:2, thetaInit=c(0,0),
                 beta=beta_init_K2, alphaInit=alpha_init_K2,
                 maxit=5000, grad=1e-6)
cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(0,2)",
            fit_02$iter, logLik(fit_02), aic_manual(fit_02,0,2, length(X_cols_K2))))

cat(strrep("-", 42), "\n")
cat("Burnham-Anderson: |dAIC|<=2 indistinguishable -> prefer simpler\n")


# Models (K=1) (one by one)  -----------------------------------------------------


cat("Model comparison (K=1 Fourier: sin1,cos1):\n")
cat(sprintf("%-12s %6s %10s %10s\n", "Model", "iter", "logLik", "AIC"))
cat(strrep("-", 42), "\n")


# GLARMA (0,0) (No ARMA)
fit_00_k1 <- glarma(y=y_train, X=X_train_K1, type = "NegBin",
                 residuals = "Pearson", method = "NR",
                 beta = beta_init_K1, alphaInit = alpha_init_K1,
                 maxit = 500, grad = 1e-6
                 )
cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(0,0)",
            fit_00_k1$iter, logLik(fit_00_k1), aic_manual(fit_00_k1,0,0, length(X_cols_K1))))

# GLARMA (1,0) --- AR(1)
fit_10_k1 <- glarma(y=y_train, X=X_train_K1, type = "NegBin",
                 residuals = "Pearson", method = "FS",
                 phiLags = 1, phiInit = 0,
                 beta = beta_init_K1, alphaInit = alpha_init_K1,
                 maxit = 500, grad = 1e-6
)
cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(1,0)",
            fit_10_k1$iter, logLik(fit_10_k1), aic_manual(fit_10_k1,1,0, length(X_cols_K1))))
# GLARMA (2,0) --- AR(2)
fit_20_k1 <- glarma(y=y_train, X=X_train_K1, type = "NegBin",
                 residuals = "Pearson", method = "FS",
                 phiLags = 1:2, phiInit = c(0,0),
                 beta = beta_init_K1, alphaInit = alpha_init_K1,
                 maxit = 500, grad = 1e-6
)
cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(2,0)",
            fit_20_k1$iter, logLik(fit_20_k1), aic_manual(fit_20_k1,2,0, length(X_cols_K1))))

# GLARMA(0,1) -- MA(1)
fit_01_k1 <- glarma(y=y_train, X=X_train_K1, type="NegBin",
                 residuals="Pearson", method="FS",
                 thetaLags=1, thetaInit=0,
                 beta=beta_init_K1, alphaInit=alpha_init_K1,
                 maxit=5000, grad=1e-6)

cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(0,1)",
            fit_01_k1$iter, logLik(fit_01_k1), aic_manual(fit_01_k1,0,1, length(X_cols_K1))))

# GLARMA(0,2) -- MA(2)

fit_02_k1 <- glarma(y=y_train, X=X_train_K1, type="NegBin",
                 residuals="Pearson", method="FS",
                 thetaLags=1:2, thetaInit=c(0,0),
                 beta=beta_init_K1, alphaInit=alpha_init_K1,
                 maxit=5000, grad=1e-6)
cat(sprintf("%-12s %6d %10.3f %10.2f\n", "GLARMA(0,2)",
            fit_02_k1$iter, logLik(fit_02_k1), aic_manual(fit_02_k1,0,2, length(X_cols_K1))))



# Models Comparison K=1 vs K=1 --------------------------------------------



cat(strrep("-", 42), "\n")
cat("Burnham-Anderson: |dAIC|<=2 indistinguishable -> prefer simpler\n")

aic_k1 <- aic_manual(fit_10_k1, 1, 0, length(X_cols_K1))
aic_k2 <- aic_manual(fit_10, 1, 0, length(X_cols_K2))

cat(sprintf("K=1 AR(1): AIC=%.2f\n", aic_k1))
cat(sprintf("K=2 AR(1): AIC=%.2f\n", aic_k2))
cat(sprintf("Delta AIC (K2-K1) = %+.2f\n\n", aic_k2 - aic_k1))

# Selected model is GLARMA(1,0) K=2 



# Hardcoded Final model (After reviewing the AIC tables above)) -----------

p_final = 1L
q_final = 0L
X_cols_fin <- X_cols_K2

X_train_fin <- as.matrix(df[train_idx, X_cols_fin])
nb_fin      <- glm.nb(y_train ~ -1 + X_train_fin,
                      control = glm.control(maxit=200))


fit_final <- glarma(
  y = y_train,
  X = X_train_fin,
  type = "NegBin",
  residuals = "Pearson",
  method = "FS",
  phiLags = p_final,
  phiInit = 0,
  beta = unname(coef(nb_fin)),
  alphaInit = nb_fin$theta,
  maxit = 500,
  grad = 1e-6
)

cat(sprintf("\n=== Final: GLARMA(%d,%d) NegBin K=2  iter=%d ===\n",
            p_final, q_final, fit_final$iter))

# Extract coefficients

coef_list <- coef(fit_final, type="all")
beta_est  <- as.numeric(coef_list$beta)
names(beta_est) <- X_cols_fin

phi_est   <- as.numeric(coef_list$ARMA)
phi_est   <- phi_est[is.finite(phi_est)]

THETA     <- as.numeric(coef_list$NB["alpha"])
if (!is.finite(THETA) || THETA <= 0) THETA <- nb_fin$theta

ETA_CAP   <- log(max(df$cases) * 10)


cat(sprintf("beta:  %s\n", paste(round(beta_est,4), collapse=", ")))
cat(sprintf("phi:   %s\n", paste(round(phi_est,4), collapse=", ")))
cat(sprintf("theta: %.4f\n", THETA))
cat(sprintf("AIC:   %.2f\n",
            aic_manual(fit_final, p_final, q_final, length(X_cols_fin))))



# Ljung-Box test ----------------------------------------------------------


resid_fin <- as.numeric(residuals(fit_final, type="pearson"))
resid_fin[!is.finite(resid_fin)] <- 0
lb12 <- Box.test(resid_fin, lag=12, type="Ljung-Box")
lb24 <- Box.test(resid_fin, lag=24, type="Ljung-Box")
cat(sprintf("LB p12=%.4f  LB p24=%.4f\n", lb12$p.value, lb24$p.value))


# Fixed-origin one-step-ahead forecast ------------------------------------


cat("\n=== Fixed-origin forecast on test set ===\n")
glarma_preds <- numeric(length(test_idx))
ext          <- resid_fin

for (i in seq_along(test_idx)) {
  t_i   <- test_idx[i]
  X_t   <- matrix(as.numeric(df[t_i, X_cols_fin]), nrow=1)
  eta_t <- as.numeric(X_t %*% beta_est)
  for (j in seq_along(phi_est)) {
    il <- length(ext) - j + 1
    if (il >= 1 && is.finite(ext[il]))
      eta_t <- eta_t + phi_est[j] * ext[il]
  }
  eta_t <- min(eta_t, ETA_CAP)
  mu_t  <- exp(eta_t)
  glarma_preds[i] <- mu_t
  v_t <- mu_t + mu_t^2 / max(THETA, 0.01)
  r_t <- pmax(pmin((df$cases[t_i] - mu_t) / sqrt(v_t), 10), -10)
  ext <- c(ext, r_t)
}


# Metrics Calculation -----------------------------------------------------


mape <- mean(abs(y_test - glarma_preds) / pmax(y_test,1) * 100)
rmse <- sqrt(mean((y_test - glarma_preds)^2))
mae  <- mean(abs(y_test - glarma_preds))
cor_ <- cor(y_test, glarma_preds)

cat(sprintf("MAPE = %.3f%%\n", mape))
cat(sprintf("RMSE = %.3f\n",   rmse))
cat(sprintf("MAE  = %.3f\n",   mae))
cat(sprintf("COR  = %.4f\n",   cor_))

# Save RDS for 03_hybrids.R
saveRDS(list(
  X_cols      = X_cols_fin,
  beta_est    = beta_est,
  phi_est     = phi_est,
  theta       = THETA,
  ETA_CAP     = ETA_CAP,
  resid_train = resid_fin,
  lb12_p      = lb12$p.value,
  lb24_p      = lb24$p.value
), "out/glarma_nb_fit.rds")
cat("Saved out/glarma_nb_fit.rds\n")

# Save predictions
write.csv(data.frame(
  date   = df$date[test_idx],
  actual = y_test,
  glarma = round(glarma_preds, 3)
), "out/glarma_nb_preds.csv", row.names=FALSE)
cat("Saved out/glarma_nb_preds.csv\n")


