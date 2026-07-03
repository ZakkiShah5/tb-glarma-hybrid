

# 05_results.R
# Final results: metrics tables, DM tests, plots



# Load predictions --------------------------------------------------------

hybrid_df <- read.csv("out/hybrid_preds.csv", stringsAsFactors = FALSE)
pure_df <- read.csv("out/pure_ml_preds.csv", stringsAsFactors = FALSE)

stopifnot(nrow(hybrid_df)==60, nrow(pure_df)==60)
stopifnot(all(hybrid_df$actual == pure_df$actual))

actual <- hybrid_df$actual
dates <- as.Date(hybrid_df$date)

cat("Predictions Loaded n = 60 \n\n")



# All predictions in one list ---------------------------------------------


preds <- list(
  glarma = hybrid_df$glarma,
  hybrid_ann = hybrid_df$hybrid_ann,
  hybrid_svr = hybrid_df$hybrid_svr,
  hybrid_rf = hybrid_df$hybrid_rf,
  hybrid_xgb = hybrid_df$hybrid_xgb,
  hybrid_lgbm = hybrid_df$hybrid_lgbm,
  hybrid_lstm = hybrid_df$hybrid_lstm,
  hybrid_gru = hybrid_df$hybrid_gru,
  pure_ann = pure_df$pure_ann,
  pure_svr = pure_df$pure_svr,
  pure_rf = pure_df$pure_rf,
  pure_xgb = pure_df$pure_xgb,
  pure_lgbm = pure_df$pure_lgbm,
  pure_lstm = pure_df$pure_lstm,
  pure_gru = pure_df$pure_gru
)



# Metric Functions --------------------------------------------------------

rmse_fn <- function(a, p) {
  sqrt(mean((a - p)^2, na.rm=TRUE))
}

mae_fn <- function(a, p) {
  mean(abs(a - p), na.rm=TRUE)
}

mape_fn <- function(a, p) {
  mean(abs(a - p)/pmax(a,1)*100, na.rm=TRUE)
}

cor_fn <- function(a, p) {
  cor(a,p, use = "complete.obs")
}


# GLARMA baselie

gl_rmse <- rmse_fn(actual, preds$glarma)
gl_mae <- mae_fn(actual, preds$glarma)
gl_mape <- mape_fn(actual, preds$glarma)
gl_cor <- cor_fn(actual, preds$glarma)




# Table 1: Metrics + % Difference vs GLARMA ---------------------------------

cat("===Table 1: Metrics + % Difference vs GLARMA ===\n\n")
cat(sprintf("%-16s %8s %8s %8s %7s %8s %8s %8s %8s\n",
            "Model","RMSE","MAE","MAPE%","COR", "%Diff_RMSE","%Diff_MAE","%Diff_MAPE", "%Diff_COR"))
cat(strrep("-", 93), "\n")

table1 <- data.frame()

for (nm in names(preds)) {
  p <- preds[[nm]]
  rmse <-  rmse_fn(actual, p)
  mae <-  mae_fn(actual, p)
  mape <-  mape_fn(actual, p)
  cor <-  cor_fn(actual, p)
  
  d_rmse <- (rmse - gl_rmse)/gl_rmse*100
  d_mae <- (mae - gl_mae)/gl_mae*100
  d_mape <- (mape - gl_mape)/gl_mape*100
  d_cor <- (cor - gl_cor)/gl_cor*100
  
  cat(sprintf("%-16s %8.3f %8.3f %8.3f %7.4f  %8.2f %8.2f %8.2f %8.2f\n",
              nm, rmse, mae, mape, cor_, d_rmse, d_mae, d_mape, d_cor))
  
  table1 <- rbind(table1, data.frame(
    model = nm, 
    RMSE = round(rmse, 3),
    MAE = round(mae, 3),
    MAPE = round(mape, 3),
    COR = round(cor, 3),
    d_RMSE = round(d_rmse, 3),
    d_MAE = round(d_mae, 3),
    d_MAPE = round(d_mape, 3),
    d_COR = round(d_cor, 3)
  ))
}


write.csv(table1, "out/table1_NegBin.csv", row.names=FALSE)
cat("Saved out/table1_NegBin.csv\n\n")

# Table 2 : Win/Loss Matrix -----------------------------------------------

cat("===Table 2: Win/Loss Matrix ===\n")
cat(sprintf("%-16s %6s %6s %6s %6s %8s\n",
            "Model","RMSE","MAE","MAPE","COR","Total"))
cat(strrep("-", 52), "\n")

table2 <- data.frame()
ml_models <- names(preds)[names(preds) != "glarma"]

for (nm in ml_models) {
  p <- preds[[nm]]
  w1 <- as.integer(rmse_fn(actual,p) < gl_rmse)
  w2 <- as.integer(mae_fn(actual,p) < gl_mae)
  w3 <- as.integer(mape_fn(actual,p) < gl_mape)
  w4 <- as.integer(cor_fn(actual,p) > gl_cor)
  
  tot <- w1 + w2 + w3 + w4
  
  cat(sprintf("%-16s %6d %6d %6d %6d %8d\n",
              nm, w1, w2, w3, w4, tot))
  table2 <- rbind(table10, data.frame(
    model=nm, RMSE=w1, MAE=w2, MAPE=w3, COR=w4, Total=tot))
}


write.csv(table2, "out/table2_NegBin.csv", row.names=FALSE)
cat("Saved out/table1_NegBin.csv\n\n")



# DIEBOLD-MARIANO TESTS ---------------------------------------------------


cat("=== DIEBOLD-MARIANO TESTS vs GLARMA ===\n")
cat("h=1, power=1 (MAE loss), alternative='less'\n")
cat(sprintf("%-16s %10s %8s %6s\n","Model","DM stat","p-value","Sig."))
cat(strrep("-", 44), "\n")



err_glarma <- actual - preds$glarma
dm_table <- data.frame()


for (nm in ml_models) {
  err_m <- actual - preds[[nm]]
  
  dm    <- tryCatch(
    dm.test(err_m, err_glarma, h=1, power=1, alternative="less"),
    error=function(e) NULL)
  
  if (is.null(dm)) next
  sig <- ifelse(dm$p.value<0.001,"***",
                ifelse(dm$p.value<0.01, "**",
                       ifelse(dm$p.value<0.05, "*",
                              ifelse(dm$p.value<0.10, ".", ""))))
  cat(sprintf("%-16s %10.4f %8.4f %6s\n",
              nm, dm$statistic, dm$p.value, sig))
  
  dm_table <- rbind(dm_table, data.frame(
    model=nm, DM_stat=round(dm$statistic,4),
    p_value=round(dm$p.value,4), sig=sig))
}

cat("Signif: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10\n")
write.csv(dm_table, "out/dm_tests_task1.csv", row.names=FALSE)
cat("Saved out/dm_tests_task1.csv\n\n")





#  PLOT 1 — Full observed time series with train/test split ---------------

png("plots/ts_full_series.png", width=1600, height=550, res=150)
par(mar=c(4,4,3,2))

all_dates <- as.Date(df$date)
all_cases <- df$cases
split_date <- as.Date("2021-01-01")

plot(all_dates, all_cases,
     type="l", lwd=1.3, col="black",
     xlab="Year", ylab="Monthly TB Cases",
     main="Monthly Confirmed TB Cases — Recife PE (2001–2025)",
     ylim=c(0, max(all_cases)*1.12))

# Vertical line at train/test split
abline(v=as.numeric(split_date), lty=2, col="grey30", lwd=1.5)

# Labels
text(split_date - 180, max(all_cases)*1.06,
     "Training", cex=0.8, col="grey30", pos=2, font=2)
text(split_date + 180, max(all_cases)*1.06,
     "Test", cex=0.8, col="grey30", pos=4, font=2)

dev.off()
cat("Saved plots/ts_full_series.png\n")



# PLOT 2 — Test period: Observed vs GLARMA vs Best Hybrid (GRU) -----------

png("plots/ts_best_hybrid.png", width=1400, height=550, res=150)
par(mar=c(4,4,3,2))

ylim <- range(c(actual, preds$glarma, preds$hybrid_gru)) * c(0.9, 1.1)

plot(dates, actual,
     type="l", lwd=2, col="black",
     ylim=ylim,
     xlab="Year", ylab="Monthly TB Cases",
     main="Test Period 2021–2025: Observed vs GLARMA vs Hybrid GRU")
lines(dates, preds$glarma,
      lwd=1.5, col="steelblue", lty=2)
lines(dates, preds$hybrid_gru,
      lwd=1.5, col="firebrick", lty=1)

legend("topright",
       legend=c(
         "Observed",
         sprintf("GLARMA(2,0)  MAPE=%.2f%%", gl_mape),
         sprintf("Hybrid GRU   MAPE=%.2f%%",
                 mape_fn(actual, preds$hybrid_gru))
       ),
       col=c("black","steelblue","firebrick"),
       lwd=c(2,1.5,1.5), lty=c(1,2,1),
       bty="n", cex=0.85)

dev.off()
cat("Saved plots/ts_best_hybrid.png\n")



# PLOT 3 — Test period: All models vs Observed ----------------------------

png("plots/ts_all_models.png", width=1400, height=650, res=150)
par(mar=c(4,4,3,2))

# Colour scheme
hybrid_cols <- c("dodgerblue","royalblue","steelblue",
                 "lightblue3","cyan4","blue3","navy")
pure_cols   <- c("tomato","firebrick","salmon",
                 "coral3","orangered","red4","darkred")

hybrid_nms <- names(preds)[grepl("hybrid", names(preds))]
pure_nms   <- names(preds)[grepl("pure",   names(preds))]

ylim <- range(c(actual, unlist(preds[c(hybrid_nms,pure_nms)]))) * c(0.88,1.1)

plot(dates, actual,
     type="l", lwd=2.5, col="black",
     ylim=ylim,
     xlab="Year", ylab="Monthly TB Cases",
     main="Test Period 2021–2025: All Models vs Observed")

for (j in seq_along(hybrid_nms))
  lines(dates, preds[[hybrid_nms[j]]], col=hybrid_cols[j], lwd=1, lty=2)
for (j in seq_along(pure_nms))
  lines(dates, preds[[pure_nms[j]]], col=pure_cols[j],   lwd=1, lty=3)

legend("topright",
       legend=c("Observed", hybrid_nms, pure_nms),
       col   =c("black", hybrid_cols, pure_cols),
       lwd   =c(2.5, rep(1,7), rep(1,7)),
       lty   =c(1,   rep(2,7), rep(3,7)),
       bty="n", cex=0.6, ncol=2)

dev.off()
cat("Saved plots/ts_all_models.png\n")

cat("\n=== 05_results.R complete ===\n")







