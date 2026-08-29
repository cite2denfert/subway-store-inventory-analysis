# 🚇 Subway Station Convenience Store Inventory & Display Optimization Analysis

**역사 인근 편의점 시간대별 재고 및 진열 최적화 분석**

A statistics-driven analysis of how ridership patterns at Seoul subway stations
relate to time-of-day demand at nearby convenience stores — and what that means
for inventory and shelf-display strategy.

> ⚠️ **Correlation, not causation.** This project deliberately frames every
> result as a correlational finding. Observational ridership data cannot
> establish that morning ridership *causes* evening ridership; a control
> variable (mid-day total ridership) is used to reduce, not eliminate,
> confounding. See [Methodology](#methodology--why-a-control-variable) below.

---

## Why this project

Convenience stores near subway stations see very different traffic depending
on the hour — but not every station follows the same rhythm. Some are busiest
in the morning rush, others in the evening. If a store's inventory and shelf
layout don't match that rhythm, it's either under-stocked at peak demand or
stuck holding stale product.

This project asks a narrow, testable question:

> Do stations with heavy **morning alighting** (people getting off, 07:00–09:00
> — the commute-in crowd) also see heavy **evening boarding** (people getting
> on, 18:00–20:00 — the commute-home crowd)? And can that relationship be used
> to classify stations into two operationally useful groups?

## Data

| Dataset | Description | Included in repo? |
|---|---|---|
| Seoul subway hourly ridership (2021-07-05 snapshot) | Boarding/alighting counts per station, per hour-of-day | ❌ (15.9 MB — see [Data access](#data-access) below) |
| Subway station coordinates | Station name ↔ lat/lon lookup used to merge/validate station identity | ❌ (see [Data access](#data-access) below) |

Both source files are excluded from version control to keep the repository
small and reviewable — this repo is meant to showcase the analysis, not host
a data mirror. See below for how to obtain them.

### Data access

- **Seoul subway ridership data**: published by Seoul Metro / Seoul Open Data
  Plaza ([data.seoul.go.kr](https://data.seoul.go.kr)) under the 지하철 시간대별
  승하차인원 dataset series. Search "지하철 시간대별 승하차인원" on the portal
  and export a CSV for the date you want to reproduce this analysis with.
- **Station coordinates**: station-level latitude/longitude lookup, also
  sourced from Seoul's public transit open-data listings (station master
  data). Any up-to-date 서울교통공사 역위치 dataset will work as a drop-in
  replacement.
- To reproduce the analysis, place the files at the repo root as
  `Seoul_subway_data_20210705.csv` (encoding: `cp949`) and
  `subway_location_data.csv` (encoding: `utf-8-sig`), matching the paths
  read by [`src/subway_regression_analysis.py`](src/subway_regression_analysis.py).

## Methodology — why a control variable?

A naive approach would just regress evening boarding (Y) on morning alighting
(X) and call a high R² "proof" that morning commuters become evening
commuters at the same station. That's a weak claim on its own — both
variables are also driven by a station's overall size/importance, which
inflates the correlation without telling us anything about the *daily rhythm*
specifically.

So the analysis runs two models side by side:

1. **Simple OLS regression** — `Y ~ X` (baseline, reported for comparison only)
2. **Multiple regression with a control variable** — `Y ~ X + C`, where `C` is
   each station's mean **mid-day (10:00–16:00) total ridership** — a proxy for
   "how busy this station generally is," independent of the morning/evening
   commute peaks.

If X remains a statistically significant predictor of Y *after* controlling
for C, that's a materially stronger claim than the naive baseline. The script
also runs:

- **Residual diagnostics**: Shapiro–Wilk normality test on model residuals
- **Multicollinearity check**: correlation between X and C is reported
  explicitly, with a note that VIF should be checked before over-interpreting
  coefficient magnitudes
- **Heteroscedasticity check**: residuals-vs-fitted plot

## Key findings

Across **570 stations**:

| Model | R² | Notes |
|---|---|---|
| Simple OLS (Y ~ X) | 0.9422 | slope = 0.9907, p ≈ 0 |
| Controlled (Y ~ X + C) | 0.9815 | X coefficient β_x = 0.7469 (p ≈ 2.5×10⁻³¹⁶); control coefficient β_c = 0.119 (p ≈ 1.6×10⁻¹⁴²) |

- Morning alighting (X) remains a statistically significant (p ≪ 0.05)
  positive predictor of evening boarding (Y) **even after controlling for**
  mid-day total ridership — suggesting the morning↔evening relationship isn't
  purely an artifact of "big stations are big all day."
- X and the control variable C are meaningfully correlated (r = 0.7693), so
  coefficient magnitudes should be read with caution (multicollinearity).
- Using the control-model residuals, each station is classified into one of
  two operational profiles:
  - **퇴근집중형 (Evening-skewed, 303 stations)** — residual > 0. Evening
    boarding is higher than the model predicts. → lean shelf space toward
    **alcohol / late-night snacks**.
  - **출근집중형 (Morning-skewed, 267 stations)** — residual ≤ 0. → lean
    shelf space toward **grab-and-go food / coffee** for the morning rush.

Full statistics (R², coefficients, p-values, RMSE, Shapiro–Wilk) are printed
by the script at runtime and reproduced in the [analysis report](docs/report.md).

## Repository structure

```
.
├── README.md                              ← you are here
├── src/
│   └── subway_regression_analysis.py      ← end-to-end analysis script
├── notebooks/
│   └── analysis.ipynb                     ← exploratory notebook version (Colab)
├── docs/
│   └── report.md                          ← full written report
└── (fig1_regression_analysis_v2.png,      ← generated by the script; not
    fig2_hourly_pattern_heatmap.png)          committed — regenerate locally
```

## How to run

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels

# Place Seoul_subway_data_20210705.csv and subway_location_data.csv
# in the repo root (see "Data access" above), then:
python src/subway_regression_analysis.py
```

This regenerates:
- `fig1_regression_analysis_v2.png` — 4-panel regression dashboard (baseline
  scatter, control-variable scatter, statistics table, residuals-vs-fitted)
- `fig2_hourly_pattern_heatmap.png` — hourly ridership-share heatmap by
  station type

## Limitations

- **Single-day snapshot**: the ridership data reflects one date
  (2021-07-05). Day-of-week and seasonal effects aren't modeled.
- **Correlational, not causal**: as emphasized throughout, this analysis
  cannot show that morning ridership *causes* evening ridership at a given
  station — only that the two are related beyond what general station
  "busyness" explains.
- **No store-level sales data**: inventory/display recommendations are
  inferred from ridership proxies, not validated against actual convenience
  store sales.

## Tech stack

`Python` · `pandas` / `numpy` · `statsmodels` (OLS) · `scipy.stats`
(Shapiro–Wilk, linregress) · `matplotlib` / `seaborn` (visualization,
Claus Wilke visualization principles — proportional ink, meaningful color,
data-driven ordering)

---

*This was originally built as a data-analysis coursework project and has
been reorganized here as a portfolio piece.*
