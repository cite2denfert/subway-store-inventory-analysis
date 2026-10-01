# 🚇 Subway Station Convenience Store Inventory & Display Optimization

**역사 인근 편의점 시간대별 재고 및 진열 최적화 분석**

Seoul subway ridership by hour → which stations lean toward the **morning
commute** vs the **evening commute** → what a nearby convenience store should
stock and put up front at each time of day.

> ⚠️ **Correlation, not causation.** Every result here is a correlational
> finding from observational data. A control variable (mid-day ridership)
> reduces, but does not eliminate, confounding by station size.

---

## Question

> Do stations with heavy **morning alighting** (07–09h, people arriving for
> work) also have heavy **evening boarding** (18–20h, people heading home) —
> *beyond* what the station's general busyness explains? And can the
> leftover difference sort stations into two operationally useful groups?

## Approach

| Step | What | Why |
|---|---|---|
| 1. Aggregate | Per station: X = 07–09h alighting, Y = 18–20h boarding, C = 10–16h total ridership. Transfer stations are **summed across lines**; if the file has a `사용월` column, months are summed per station then **averaged** (unit: monthly average riders). | One row per physical station. |
| 2. Baseline | `Y ~ X` (OLS) | Shown for comparison only — big stations are big at every hour, so R² here is mostly a size effect. |
| 3. Controlled model | `Y ~ X + C` (OLS, **HC3 robust SE**) | Tests whether X still tracks Y once general busyness (C) is held fixed. Residual spread grows with station size, which makes classical SEs too optimistic — Breusch–Pagan is reported to check this. |
| 4. Diagnostics | VIF, Breusch–Pagan, Shapiro–Wilk, standardized coefficients, 95% CI | With n≈570, every p-value is tiny; effect size and intervals carry the information. |
| 5. Classify | Residual of step 3 > 0 → **퇴근집중형 (evening-skewed)**, ≤ 0 → **출근집중형 (morning-skewed)** | Evening boarding higher / lower than the station's own morning & mid-day traffic predict. |
| 6. Robustness | Re-fit as `log Y ~ log X + log C` and report how often the two classifications agree; flag stations with \|standardized residual\| < 0.5 as weak-signal | The sign of a residual near zero is noise, not a store strategy. |

## Results (v1 run)

Numbers below are from the original notebook run on 570 stations. The v2
script fixes station de-duplication and transfer-station aggregation (see
[Changelog](#changelog)), so the exact values will shift slightly when re-run;
`outputs/results.json` is the source of truth after a run.

| Model | R² | Key coefficient |
|---|---|---|
| Baseline `Y ~ X` | 0.942 | slope 0.991 |
| Controlled `Y ~ X + C` | 0.982 | β_x = 0.747, β_c = 0.119 (both p < 1e-100) |

- **X stays a strong positive predictor after controlling for C.** Holding
  mid-day ridership fixed, one additional morning alighting is associated
  with ~0.75 additional evening boardings — consistent with "people who
  arrive for work leave from the same station."
- **Multicollinearity is moderate, not severe.** corr(X, C) = 0.769 ⇒
  VIF = 1 / (1 − 0.769²) ≈ **2.4**, well under the common threshold of 5.
- **Classification:** 303 evening-skewed vs 267 morning-skewed stations.
  - 퇴근집중형 → widen evening shelf space for **beer / snacks / late-night food**
  - 출근집중형 → put **grab-and-go food and coffee** front-and-center 06–10h
- These store actions are **hypotheses derived from ridership**, not tested
  against sales (see Limitations).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── src/
│   └── subway_regression_analysis.py   ← end-to-end analysis (CLI)
├── tests/
│   └── test_pipeline.py                ← smoke test on synthetic data (same schema)
├── notebooks/
│   └── analysis.ipynb                  ← original Colab exploration (v1, kept for history)
├── docs/
│   └── report.md                       ← written report (Korean)
├── data/                               ← put the CSVs here (git-ignored)
└── outputs/                            ← generated figures / CSV / JSON (git-ignored)
```

## How to run

```bash
pip install -r requirements.txt

# 1) put the two CSVs in data/  (see "Data access")
# 2) run
python src/subway_regression_analysis.py
python src/subway_regression_analysis.py --last-months 12   # optional: recent 12 months only

# tests (no real data needed)
pytest -q
```

Outputs in `outputs/`:

| File | Content |
|---|---|
| `fig1_regression_analysis.png` | Baseline scatter (colored by station type), control-vs-Y scatter, stats table, residuals-vs-fitted |
| `fig2_hourly_pattern_heatmap.png` | Hour-of-day ridership share (%) by station type |
| `fig3_top20_station_heatmap.png` | Hour-of-day share for the 20 busiest stations, ordered by volume |
| `station_types.csv` | Per-station X, Y, C, fitted value, residual, standardized residual, type |
| `results.json` | All headline statistics |

Korean labels need a Korean font (Noto Sans CJK, Nanum Gothic, Malgun
Gothic or AppleGothic are detected automatically; on Ubuntu/Colab:
`apt install fonts-nanum`).

## Data

| File (place in `data/`) | Encoding | Source |
|---|---|---|
| `Seoul_subway_data_20210705.csv` | cp949 | Seoul Open Data Plaza ([data.seoul.go.kr](https://data.seoul.go.kr)) — *서울시 지하철 호선별 역별 시간대별 승하차 인원 정보*. The `20210705` in the file name appears to be the export date; the script prints the `사용월` range it actually finds. |
| `subway_location_data.csv` | utf-8-sig | Station name ↔ coordinates (서울교통공사 역 위치 data). Used as the station universe; coordinates are reserved for the spatial extension in Future work. |

Raw data is not committed (15.9 MB, and it is public). Station names are
normalized identically on both files (`서울역` → `서울`, `신촌(지하)` → `신촌`);
the script prints any unmatched names so you can check coverage.

## Limitations

- **Ridership is a proxy for store demand.** No convenience-store sales data
  was used; the inventory/display actions are untested hypotheses.
- **Correlational.** Nothing here shows that morning ridership *causes*
  evening ridership.
- **Binary split at residual = 0** is a simplification; stations close to
  zero are reported as weak-signal rather than forced into a strategy.
- **Residuals are not normal** (Shapiro–Wilk). Inference uses HC3 robust
  SEs and a log-log robustness model, but the very largest stations still
  carry a lot of weight in the level model.
- **No line/area effects.** All stations are treated as one population.

## Future work

- Validate the two profiles against actual store POS data (A/B on shelf layout)
- Line- or district-level differences (ANOVA / mixed models)
- Spatial autocorrelation between neighboring stations (Moran's I) using the coordinates

## Changelog

**v2 (2026-10)** — code quality & correctness pass
- Fixed: transfer stations were **averaged** across lines instead of summed
- Fixed: duplicate station rows in the coordinate file could duplicate
  stations in the regression (inflating n); now one row per station, asserted
- Fixed: `서울역`-style names failed to match because `역` was stripped on one side only
- Added: HC3 robust SEs, VIF, Breusch–Pagan, standardized coefficients, 95% CI,
  log-log robustness check, weak-signal flag, `station_types.csv` / `results.json`
- Added: Fig 3 (top-20 stations), legends, zero-based axes, portable Korean font
  detection, CLI paths, `requirements.txt`, synthetic-data smoke test

**v1** — original bootcamp project (notebook)

---

*Built as a data-analysis project during the 이어드림스쿨 bootcamp and
reorganized as a portfolio piece.*
