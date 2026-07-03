# Hybrid GLARMA + ML Forecasting of TB Count Time Series

Masters Dissertation | PPGMat UFSM | 2026
Supervisor: Prof. Fernando Arturo Peña Ramirez

## Overview
Hybrid forecasting framework combining GLARMA count time series
models with machine learning for monthly TB case prediction in
Recife PE, Brazil (2001-2025).

## Methodology
- Baseline: GLARMA(2,0) Negative Binomial, K=2 Fourier
- Hybrid formula (Zhang 2003): y_hat = mu_GLARMA + e_ML * sqrt(V(mu))
- 7 hybrid models: ANN, SVR, RF, XGBoost, LightGBM, LSTM, GRU
- 7 pure ML benchmarks: same architectures on lagged counts
- Evaluation: Fixed-origin one-step-ahead (Guerra et al. 2024)

## Key Results - Task 1 (TB Confirmed Cases, Recife PE)
| Model       | MAPE%  | RMSE   | COR    |
|-------------|--------|--------|--------|
| GLARMA(2,0) | 14.541 | 39.018 | 0.1971 |
| Hybrid GRU  | 11.011 | 31.448 | 0.3326 |
| Hybrid LSTM | 11.716 | 32.773 | 0.3110 |
| Pure GRU    | 11.418 | 32.252 | 0.3846 |

6/7 hybrid models outperform GLARMA baseline.
Best hybrid GRU: -24.3% relative MAPE improvement.

## Pipeline
R/
  00_setup.R       - Data loading, feature engineering
  02_glarma_nb.R   - GLARMA(2,0) NegBin model selection
  03_hybrids.R     - 7 hybrid GLARMA+ML models
  04_pure_ml.R     - 7 pure ML benchmarks
  05_results.R     - DM tests, tables, plots

## Data Sources
SINAN/DATASUS, municipality 261160 Recife PE
Period: January 2001 - December 2025 (n=300)
Train: 2001-2020 (n=240) | Test: 2021-2025 (n=60)

## Tasks
Task 1 DONE  - GLARMA NegBin + Hybrids (TB confirmed cases)
Task 2       - GLARMA Poisson + Hybrids (sensitivity)
Task 3       - ARIMA + Hybrids (benchmark)
Task 4       - GLARMA Binomial + Hybrids (hospitalisation rate)

## References
Zhang (2003) Neurocomputing 50:159-175
Guerra et al. (2024) Computer Networks 243:110258
Dunsmuir & Scott (2015) J Statistical Software 67(7)
Diebold & Mariano (1995) JBES 13(3):253-263
