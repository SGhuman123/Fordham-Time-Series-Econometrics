# Variational Inference for GARCH-Family Models

**Time Series Econometrics project, Fordham University**

This project replicates and extends the Microsoft case study of
[Magris & Iosifidis (2023), *Variational Inference for GARCH-family Models*](https://arxiv.org/abs/2310.03435).

**Research question:** can Gaussian Variational Inference (VI) give Bayesian GARCH estimates and forecasts that are as good as
MCMC and Quasi-Maximum Likelihood (QML)? And what does it cost to compute them?

<a href="https://colab.research.google.com/github/SGhuman123/Fordham-Time-Series-Econometrics/blob/main/VI_GARCH.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## What is in the repo

| Path | Contents |
|---|---|
| [`VI_GARCH.ipynb`](VI_GARCH.ipynb) | The full analysis in small, commented cells: data collection, cleaning, EDA, model, QML, MCMC, BBVI, full-covariance VI, results, Student-t extension, multi-stock experiment |
| [`data/`](data) | Cached Yahoo Finance prices (MSFT and the Dow 30, 2018-01-01 to 2023-06-30) so results are reproducible |
| [`figures/`](figures) | Figures saved by the notebook (used in the slides) |
| [`slides/`](slides) | Presentation: *Bayesian Volatility at Scale* (`.pptx` and `.pdf`) |
| [`docs/`](docs) | Project proposal and presentation script (`.docx` and `.pdf`) |

## Design

- **Data:** daily close-to-close log returns (×100) for Microsoft, 1 Jan 2018 – 30 Jun 2023 (1,383 trading days). This is the paper's own case-study window.
  Chronological split: 75% train / 25% test.
- **Model:** zero-mean GARCH(1,1). Constraints are handled with the paper's transforms (Prop. 2): ω = exp(θω), α = σ(θα), β = σ(θβ)(1 − α).
- **Estimators:**
  1. QML (checked against the `arch` package)
  2. Random-Walk Metropolis MCMC
  3. BBVI, the paper's Algorithm 1 (score-function gradients, diagonal covariance, the paper's hyperparameters)
  4. Full-covariance Gaussian VI in PyTorch (reparameterisation trick)
- **Extension:** Student-t innovations.
- **Scale check:** QML / MCMC / BBVI on 28 Dow 30 stocks with identical settings.

## Key results

| | ω | α | β |
|---|---|---|---|
| QML (ours / paper) | 0.160 / 0.15 | 0.186 / 0.19 | 0.769 / 0.77 |
| MCMC (ours / paper) | 0.217 ± 0.049 / 0.200 ± 0.046 | 0.203 ± 0.031 / 0.203 ± 0.031 | 0.731 ± 0.036 / 0.737 ± 0.036 |
| BBVI (ours / paper) | 0.214 ± 0.022 / 0.196 ± 0.021 | 0.201 ± 0.023 / 0.199 ± 0.023 | 0.732 ± 0.023 / 0.740 ± 0.023 |
| Full-cov VI (ours / paper's MGVB) | 0.220 ± 0.050 / 0.205 ± 0.047 | 0.200 ± 0.029 / 0.201 ± 0.030 | 0.732 ± 0.035 / 0.736 ± 0.035 |

1. **Replication:** all estimates are within about 0.02 of the paper's Table II.
2. **Accuracy:** out of sample, all Bayesian methods are within 1% of QML on NLL, RMSE, MAD and QLIK. Across 28 Dow stocks the average gap is about 0.1–0.3%.
3. **Uncertainty:** full-covariance VI reproduces the MCMC posterior. Diagonal BBVI is over-confident (posterior SD of ω is about 55% too small).
4. **Fat tails:** Student-t innovations (ν ≈ 7.4) improve the ELBO by about 11 and the test log-likelihood by about 12. The volatility path barely changes.
5. **Cost:** for a 3-parameter GARCH, MCMC (≈ 0.3 s) is cheaper than BBVI (≈ 1 s) and PyTorch VI (≈ 16 s). VI's advantages are generic tuning,
   simple convergence monitoring, and scalability to high-dimensional models, not raw speed here.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook VI_GARCH.ipynb
```

The whole notebook runs in about 2 minutes on a laptop. In Colab, the first cell installs any missing packages. Prices are read from
`data/` when present and otherwise downloaded from Yahoo Finance.

## References

- Magris, M., & Iosifidis, A. (2023). Variational Inference for GARCH-family Models. arXiv:2310.03435.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3).
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. *JASA*, 112(518).
- Ranganath, R., Gerrish, S., & Blei, D. (2014). Black Box Variational Inference. *AISTATS*.
