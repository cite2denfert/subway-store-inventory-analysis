"""
역사 인근 편의점 시간대별 재고 및 진열 최적화 분석
──────────────────────────────────────────────────
오전 출근 하차(X, 07-09시) ↔ 오후 퇴근 승차(Y, 18-20시)의 관계를
낮 시간대(10-16시) 총이용객(C)을 통제변수로 둔 다중회귀로 검증하고,
통제모델 잔차의 부호로 역을 '퇴근집중형 / 출근집중형'으로 분류한다.

※ 관측 데이터 기반 회귀분석이므로 결과는 '상관관계'이며 인과관계를 뜻하지 않는다.

실행:
    python src/subway_regression_analysis.py                 # data/ 에서 읽고 outputs/ 에 저장
    python src/subway_regression_analysis.py --last-months 12 # 최근 12개월만 사용

입력 (기본 위치: data/):
    Seoul_subway_data_20210705.csv  역별·시간대별 승하차인원 (cp949)
    subway_location_data.csv        역별 좌표 (utf-8-sig) — 역 목록 기준으로 사용

산출 (기본 위치: outputs/):
    fig1_regression_analysis.png    회귀 대시보드
    fig2_hourly_pattern_heatmap.png 역 유형별 시간대 이용 비중
    fig3_top20_station_heatmap.png  이용량 상위 20개 역 시간대 이용 비중
    station_types.csv               역별 변수·예측값·잔차·유형
    results.json                    핵심 통계량
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402
from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 상수 ────────────────────────────────────────────────────────
# 04시-05시 … 23시-24시, 00시-01시 … 03시-04시 (운행일 기준 순서)
HOURS = [f"{h:02d}시-{h + 1:02d}시" for h in [*range(4, 24), *range(0, 4)]]
MORNING_ALIGHT = ["07시-08시 하차인원", "08시-09시 하차인원"]
EVENING_BOARD = ["18시-19시 승차인원", "19시-20시 승차인원"]
MIDDAY_HOURS = ["10시-11시", "11시-12시", "12시-13시", "13시-14시", "14시-15시", "15시-16시"]

X_COL, Y_COL, C_COL = "오전_출근_하차", "오후_퇴근_승차", "낮시간대_총이용객"
EVENING, MORNING = "퇴근집중형", "출근집중형"
COLOR = {EVENING: "#2549D8", MORNING: "#F4A261"}  # 퇴근=네이비, 출근=오렌지 (전 그림 공통)
ACCENT, NEUTRAL = "#E63946", "#457B9D"

KOREAN_FONT_CANDIDATES = [
    "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "Malgun Gothic", "Noto Sans CJK JP",
    "AppleGothic", "Apple SD Gothic Neo",
]


# ── 0. 환경 설정 ─────────────────────────────────────────────────
def setup_korean_font() -> str | None:
    """설치된 한글 폰트를 찾아 matplotlib 기본 폰트로 지정 (OS 무관)."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in KOREAN_FONT_CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        name = None
        print(" [경고] 한글 폰트를 찾지 못했습니다. 그림의 한글이 깨질 수 있습니다.\n"
              "        (예: Ubuntu `apt install fonts-nanum`, Colab 동일)")
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "white",
    })
    return name


def section(title: str) -> None:
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def thousands(ax, axis: str = "both") -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


# ── 1. 데이터 로드 및 전처리 ─────────────────────────────────────
def normalize_station(name: str) -> str:
    """역명 표기 통일: 공백·괄호 부기 제거, 끝의 '역' 제거.
    예) '서울역' → '서울', '신촌(지하)' → '신촌', '강남역 ' → '강남'
    두 데이터에 *같은 함수*를 적용해야 '서울역'처럼 이름에 '역'이 포함된 역도 매칭된다."""
    s = re.sub(r"\(.*?\)", "", str(name)).strip()
    s = re.sub(r"\s+", "", s)
    return s[:-1] if len(s) > 1 and s.endswith("역") else s


def load_ridership(path: Path, last_months: int | None) -> pd.DataFrame:
    """원본 → 역(정규화 키) 단위 시간대별 승하차 테이블.

    - 한 역이 여러 노선에 걸쳐 있으면(환승역) 노선별 행을 '합산'한다.
      (원본 코드는 평균을 내서 환승역 이용량이 노선 수만큼 과소평가됐다.)
    - '사용월' 컬럼이 있으면 월별로 합산한 뒤 월 평균을 낸다 → 단위: 월평균 인원.
    """
    df = pd.read_csv(path, encoding="cp949")
    df.columns = [c.strip() for c in df.columns]
    count_cols = [c for c in df.columns if re.match(r"\d{2}시-\d{2}시 (승차|하차)인원$", c)]
    missing = [c for c in MORNING_ALIGHT + EVENING_BOARD if c not in count_cols]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    df[count_cols] = df[count_cols].apply(
        lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    ).fillna(0)
    df["역_키"] = df["지하철역"].map(normalize_station)

    month_col = "사용월" if "사용월" in df.columns else None
    if month_col:
        months = sorted(df[month_col].unique())
        if last_months:
            months = months[-last_months:]
            df = df[df[month_col].isin(months)]
        print(f" 사용월 범위: {months[0]} ~ {months[-1]} ({len(months)}개월) → 월평균으로 집계")
        per_month = df.groupby([month_col, "역_키"])[count_cols].sum()
        station = per_month.groupby(level="역_키").mean()
    else:
        print(" '사용월' 컬럼 없음 → 노선별 행을 역 단위로 합산")
        station = df.groupby("역_키")[count_cols].sum()
    print(f" 원본 행 수: {len(df):,} → 역 수: {len(station):,}")
    return station


def load_locations(path: Path) -> pd.DataFrame:
    loc = pd.read_csv(path, encoding="utf-8-sig")
    loc["역_키"] = loc["지하철역"].map(normalize_station)
    n_before = len(loc)
    # 환승역이 노선별로 여러 행인 경우 → 병합 시 행이 복제되어 n과 유의성이 부풀려짐. 역당 1행으로.
    loc = loc.drop_duplicates("역_키")
    if n_before != len(loc):
        print(f" 좌표 데이터 중복 역 {n_before - len(loc)}행 제거 (환승역 등)")
    return loc


def build_features(station: pd.DataFrame, loc: pd.DataFrame | None) -> pd.DataFrame:
    feat = pd.DataFrame(index=station.index)
    feat[X_COL] = station[MORNING_ALIGHT].sum(axis=1)
    feat[Y_COL] = station[EVENING_BOARD].sum(axis=1)
    mid_cols = [c for c in station.columns if c.split(" ")[0] in MIDDAY_HOURS]
    assert len(mid_cols) == 2 * len(MIDDAY_HOURS), f"낮 시간대 컬럼 수 이상: {mid_cols}"
    feat[C_COL] = station[mid_cols].sum(axis=1)
    feat = feat.reset_index()

    if loc is not None:
        keys = set(loc["역_키"])
        unmatched = sorted(set(feat["역_키"]) - keys)
        print(f" 좌표 데이터와 매칭: {len(feat) - len(unmatched)}/{len(feat)}개 역")
        if unmatched:
            preview = ", ".join(unmatched[:15]) + (" …" if len(unmatched) > 15 else "")
            print(f"  미매칭 {len(unmatched)}개: {preview}")
        feat = feat.merge(loc[["역_키"]], on="역_키", how="inner")

    # 0 승하차 역(운영 중단·신규 개통 등)은 회귀·로그 변환에서 제외
    zero = (feat[[X_COL, Y_COL, C_COL]] <= 0).any(axis=1)
    if zero.any():
        print(f" 승하차 0인 역 {zero.sum()}개 제외: {', '.join(feat.loc[zero, '역_키'][:10])}")
    feat = feat[~zero].rename(columns={"역_키": "지하철역"}).reset_index(drop=True)
    assert feat["지하철역"].is_unique, "역 단위 중복 행이 남아 있습니다."
    return feat


# ── 2~3. 통계 분석 ──────────────────────────────────────────────
def describe(feat: pd.DataFrame) -> None:
    section("STEP 2. 기술 통계")
    cols = [X_COL, Y_COL, C_COL]
    print(feat[cols].describe().round(1).to_string())
    print("\n 왜도:", ", ".join(f"{c} {feat[c].skew():.2f}" for c in cols))
    print(" 첨도:", ", ".join(f"{c} {feat[c].kurtosis():.2f}" for c in cols))


def fit_models(feat: pd.DataFrame) -> dict:
    section("STEP 3. 단순회귀 vs 통제변수 반영 다중회귀 (상관관계 분석)")
    x, y, c = (feat[k].to_numpy(float) for k in (X_COL, Y_COL, C_COL))
    n = len(feat)

    simple = sm.OLS(y, sm.add_constant(x)).fit()

    X_multi = sm.add_constant(np.column_stack([x, c]))
    multi = sm.OLS(y, X_multi).fit()
    # 잔차 vs 적합값이 깔때기형(이분산) → 계수 추론은 HC3 강건 표준오차로
    multi_hc3 = sm.OLS(y, X_multi).fit(cov_type="HC3")
    bp_stat, bp_p, _, _ = het_breuschpagan(multi.resid, X_multi)
    vif_x = variance_inflation_factor(X_multi, 1)
    vif_c = variance_inflation_factor(X_multi, 2)
    sw_stat, sw_p = stats.shapiro(multi.resid)

    # 표준화 계수: 단위(명)가 아닌 표준편차 기준 효과 크기
    beta_std_x = multi.params[1] * x.std() / y.std()
    beta_std_c = multi.params[2] * c.std() / y.std()

    # 강건성 검증: 로그-로그 모델 (역 규모에 따른 이분산 완화, 계수=탄력성)
    X_log = sm.add_constant(np.column_stack([np.log(x), np.log(c)]))
    log_model = sm.OLS(np.log(y), X_log).fit(cov_type="HC3")

    res = {
        "n_stations": n,
        "simple": {"r2": simple.rsquared, "slope": simple.params[1],
                   "intercept": simple.params[0], "p": simple.pvalues[1]},
        "multi": {
            "r2": multi.rsquared, "adj_r2": multi.rsquared_adj,
            "intercept": multi.params[0],
            "beta_x": multi.params[1], "beta_c": multi.params[2],
            "beta_std_x": beta_std_x, "beta_std_c": beta_std_c,
            "p_x_hc3": multi_hc3.pvalues[1], "p_c_hc3": multi_hc3.pvalues[2],
            "ci95_x_hc3": multi_hc3.conf_int()[1].tolist(),
            "rmse": float(np.sqrt(multi.mse_resid)),
        },
        "diagnostics": {
            "corr_x_c": float(np.corrcoef(x, c)[0, 1]),
            "vif_x": vif_x, "vif_c": vif_c,
            "breusch_pagan_p": bp_p, "shapiro_p": sw_p,
        },
        "log_model": {"r2": log_model.rsquared, "elasticity_x": log_model.params[1],
                      "elasticity_c": log_model.params[2]},
    }
    m, d = res["multi"], res["diagnostics"]

    print("\n [단순회귀 Y ~ X — 비교용 baseline]")
    print(f"  Ŷ = {res['simple']['slope']:.4f}·X + {res['simple']['intercept']:,.0f}   "
          f"R² = {res['simple']['r2']:.4f}")
    print("\n [다중회귀 Y ~ X + C]  (p값·신뢰구간은 HC3 강건 표준오차 기준)")
    print(f"  Ŷ = {m['beta_x']:.4f}·X + {m['beta_c']:.4f}·C + {m['intercept']:,.0f}")
    print(f"  ├ R² = {m['r2']:.4f} (adj. {m['adj_r2']:.4f}), RMSE = {m['rmse']:,.0f}")
    print(f"  ├ β_x = {m['beta_x']:.4f}  95% CI [{m['ci95_x_hc3'][0]:.3f}, {m['ci95_x_hc3'][1]:.3f}]"
          f"  p = {m['p_x_hc3']:.2e}  (표준화 β = {beta_std_x:.3f})")
    print(f"  ├ β_c = {m['beta_c']:.4f}  p = {m['p_c_hc3']:.2e}  (표준화 β = {beta_std_c:.3f})")
    print(f"  ├ corr(X, C) = {d['corr_x_c']:.4f}, VIF = {vif_x:.2f} (5 미만이면 다중공선성 우려 낮음)")
    print(f"  ├ Breusch-Pagan p = {bp_p:.2e} (작을수록 이분산 → HC3 사용 근거)")
    print(f"  └ Shapiro-Wilk p = {sw_p:.2e} (작을수록 잔차 비정규)")
    print(f"\n [강건성: log Y ~ log X + log C]  R² = {log_model.rsquared:.4f}, "
          f"X 탄력성 = {log_model.params[1]:.3f}")
    print("\n [해석] 관측 데이터의 상관관계이며, '오전 하차가 오후 승차를 유발한다'는 인과 해석은 불가.")
    print("        n이 크면 p값은 쉽게 작아지므로 계수 크기·신뢰구간·표준화 계수를 함께 볼 것.")

    feat = feat.assign(
        예측_오후승차=multi.fittedvalues,
        잔차=multi.resid,
        표준화잔차=multi.get_influence().resid_studentized_internal,
        로그모델_잔차=log_model.resid,
    )
    feat["유형"] = np.where(feat["잔차"] > 0, EVENING, MORNING)
    feat["로그모델_유형"] = np.where(feat["로그모델_잔차"] > 0, EVENING, MORNING)
    agree = (feat["유형"] == feat["로그모델_유형"]).mean()
    res["classification"] = {
        "n_evening": int((feat["유형"] == EVENING).sum()),
        "n_morning": int((feat["유형"] == MORNING).sum()),
        "agreement_with_log_model": float(agree),
        "n_weak_signal_abs_std_resid_lt_0_5": int((feat["표준화잔차"].abs() < 0.5).sum()),
    }
    cl = res["classification"]
    section("STEP 4. 역 유형 분류 (다중회귀 잔차 부호 기준)")
    print(f"  ├ {EVENING}: {cl['n_evening']}개 역 (잔차 > 0, 예측보다 퇴근 승차 많음)")
    print(f"  ├ {MORNING}: {cl['n_morning']}개 역 (잔차 ≤ 0)")
    print(f"  ├ 로그모델 분류와 일치율: {agree:.1%}")
    print(f"  └ |표준화잔차| < 0.5 인 경계 역: {cl['n_weak_signal_abs_std_resid_lt_0_5']}개 "
          f"(분류 신호가 약함 — 현장 판단 병행 권장)")
    top_e = feat.nlargest(10, "표준화잔차")["지하철역"]
    top_m = feat.nsmallest(10, "표준화잔차")["지하철역"]
    print(f"\n  {EVENING} 상위 10개 역 (표준화잔차 큰 순): {', '.join(top_e)}")
    print(f"  {MORNING} 상위 10개 역 (표준화잔차 작은 순): {', '.join(top_m)}")
    return {"results": res, "feat": feat, "simple": simple}


# ── 4. 시각화 ───────────────────────────────────────────────────
def plot_regression(feat: pd.DataFrame, fit: dict, out: Path) -> None:
    res, simple = fit["results"], fit["simple"]
    m, d = res["multi"], res["diagnostics"]
    x, y, c = feat[X_COL], feat[Y_COL], feat[C_COL]
    colors = feat["유형"].map(COLOR)

    fig = plt.figure(figsize=(18, 11))
    fig.text(0.5, 0.965, "오전 하차 vs 오후 승차 상관관계 분석 (통제변수 반영)",
             fontsize=15, fontweight="bold", ha="center", va="top")
    fig.text(0.5, 0.935, f"통제변수: 낮 시간대(10-16시) 총이용객 · 상관관계 기반 분석 — 인과관계 아님 "
             f"(n={res['n_stations']}개 역, 단위: 월평균 인원)",
             fontsize=10, ha="center", va="top", color="#555")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28, top=0.89, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x, y, c=colors, alpha=0.55, s=22, zorder=3, linewidths=0)
    xs = np.linspace(0, x.max(), 200)
    ax1.plot(xs, simple.params[1] * xs + simple.params[0], color=ACCENT, lw=2, zorder=4)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"단순회귀 Y ~ X  (R²={res['simple']['r2']:.3f}) · 점 색 = 다중회귀 잔차 기준 유형",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("오전 출근 하차인원 (07-09시)")
    ax1.set_ylabel("오후 퇴근 승차인원 (18-20시)")
    ax1.legend(handles=[
        Line2D([], [], marker="o", ls="", color=COLOR[EVENING], label=f"{EVENING} (잔차>0)"),
        Line2D([], [], marker="o", ls="", color=COLOR[MORNING], label=f"{MORNING} (잔차≤0)"),
        Line2D([], [], color=ACCENT, lw=2, label="단순회귀선"),
    ], loc="upper left", frameon=False, fontsize=9)
    thousands(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(c, y, alpha=0.35, s=18, color=NEUTRAL, zorder=3, linewidths=0)
    s_cy = stats.linregress(c, y)
    xs2 = np.linspace(0, c.max(), 200)
    ax2.plot(xs2, s_cy.slope * xs2 + s_cy.intercept, color=ACCENT, lw=2, zorder=4)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.set_title(f"통제변수(낮시간대 총이용객) vs Y  (R²={s_cy.rvalue ** 2:.3f})",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("낮 시간대(10-16시) 총이용객")
    ax2.set_ylabel("오후 퇴근 승차인원 (18-20시)")
    thousands(ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    rows = [
        ["단순회귀 R²", f"{res['simple']['r2']:.4f}"],
        ["다중회귀 R² (X + C)", f"{m['r2']:.4f}"],
        ["β_x (통제 후) [95% CI]", f"{m['beta_x']:.3f} [{m['ci95_x_hc3'][0]:.3f}, {m['ci95_x_hc3'][1]:.3f}]"],
        ["β_x p-value (HC3)", f"{m['p_x_hc3']:.2e}"],
        ["β_c (통제변수)", f"{m['beta_c']:.4f}"],
        ["β_c p-value (HC3)", f"{m['p_c_hc3']:.2e}"],
        ["corr(X, C) / VIF", f"{d['corr_x_c']:.3f} / {d['vif_x']:.2f}"],
        ["Breusch-Pagan p (이분산)", f"{d['breusch_pagan_p']:.2e}"],
        ["Shapiro-Wilk p (정규성)", f"{d['shapiro_p']:.2e}"],
        ["로그모델 분류 일치율", f"{res['classification']['agreement_with_log_model']:.1%}"],
    ]
    tbl = ax3.table(cellText=rows, colLabels=["통계량", "값"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r_i, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDD")
        if r_i == 0:
            cell.set_facecolor(COLOR[EVENING])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#EEF2FF" if r_i % 2 == 0 else "white")
    tbl.scale(1, 1.5)
    ax3.set_title("다중회귀 통계 요약", fontsize=11, fontweight="bold", pad=12)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(feat["예측_오후승차"], feat["잔차"], c=colors, alpha=0.5, s=18, zorder=3, linewidths=0)
    ax4.axhline(0, color=ACCENT, lw=1.8, ls="--", zorder=4)
    ax4.set_title("다중회귀 잔차 vs 적합값 (등분산성 점검)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("적합값 (Fitted values)")
    ax4.set_ylabel("잔차 (Residuals)")
    thousands(ax4)

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" → {out.name}")


def hourly_share(station: pd.DataFrame) -> pd.DataFrame:
    """역별 시간대 승+하차 합계를 '해당 역 하루 이용량 대비 비중(%)'으로 변환."""
    total = pd.DataFrame({h: station[f"{h} 승차인원"] + station[f"{h} 하차인원"] for h in HOURS})
    return total.div(total.sum(axis=1), axis=0) * 100


def draw_heatmap(ax, data: pd.DataFrame, title: str) -> plt.cm.ScalarMappable:
    im = ax.imshow(data.values, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(HOURS)))
    ax.set_xticklabels([h[:3] for h in HOURS], fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.grid(False)
    vmax = np.nanmax(data.values)
    for yi in range(data.shape[0]):
        for xi in range(data.shape[1]):
            val = data.values[yi, xi]
            ax.text(xi, yi, f"{val:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if val > vmax * 0.6 else "black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("시간대 (시작 시각, 04시 → 익일 03시)")
    return im


def plot_heatmaps(station: pd.DataFrame, feat: pd.DataFrame, out2: Path, out3: Path) -> None:
    share = hourly_share(station).loc[feat["지하철역"]]
    share.index.name = "지하철역"
    types = feat.set_index("지하철역")["유형"]

    # Fig 2: 유형별 평균 — 역마다 비중을 먼저 구한 뒤 평균(대형 역이 결과를 좌우하지 않도록)
    by_type = share.groupby(types).mean().reindex([EVENING, MORNING])
    by_type.index = [f"{t} ({(types == t).sum()}개 역)" for t in by_type.index]
    fig, ax = plt.subplots(figsize=(16, 3.6))
    im = draw_heatmap(ax, by_type, "역 유형별 시간대 이용 비중(%) — 역별 비중의 평균, 서술적 비교")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("하루 이용량 대비 비중(%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" → {out2.name}")

    # Fig 3: 이용량 상위 20개 역 — 이용량 순 정렬(가나다·노선순 아님), 라벨에 유형 표기
    volume = station.loc[feat["지하철역"]].sum(axis=1).sort_values(ascending=False)
    top = volume.index[:20]
    top_share = share.loc[top]
    top_share.index = [f"{s} · {types[s]}" for s in top]
    fig, ax = plt.subplots(figsize=(16, 8.5))
    im = draw_heatmap(ax, top_share, "이용량 상위 20개 역의 시간대 이용 비중(%) — 이용량 내림차순")
    for lbl, s in zip(ax.get_yticklabels(), top):
        lbl.set_color(COLOR[types[s]])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("하루 이용량 대비 비중(%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" → {out3.name}")


# ── main ─────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ridership", type=Path, default=REPO_ROOT / "data" / "Seoul_subway_data_20210705.csv")
    p.add_argument("--locations", type=Path, default=REPO_ROOT / "data" / "subway_location_data.csv",
                   help="좌표 파일. 없으면 역 목록 필터링 없이 진행")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--last-months", type=int, default=None,
                   help="'사용월'이 있을 때 최근 N개월만 사용 (기본: 전체 기간)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if not args.ridership.exists():
        raise SystemExit(f"승하차 데이터가 없습니다: {args.ridership}\n"
                         "README의 'Data access'를 참고해 data/ 폴더에 넣어주세요.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = setup_korean_font()

    section("STEP 1. 데이터 로드 및 전처리")
    print(f" 한글 폰트: {font}")
    station = load_ridership(args.ridership, args.last_months)
    loc = load_locations(args.locations) if args.locations.exists() else None
    if loc is None:
        print(f" 좌표 파일 없음({args.locations.name}) → 전체 역 사용")
    feat = build_features(station, loc)
    print(f" 분석 대상: {len(feat)}개 역")

    describe(feat)
    fit = fit_models(feat)
    feat = fit["feat"]

    section("STEP 5. 시각화 및 결과 저장")
    plot_regression(feat, fit, args.out_dir / "fig1_regression_analysis.png")
    plot_heatmaps(station, feat, args.out_dir / "fig2_hourly_pattern_heatmap.png",
                  args.out_dir / "fig3_top20_station_heatmap.png")
    feat.sort_values("표준화잔차", ascending=False).to_csv(
        args.out_dir / "station_types.csv", index=False, encoding="utf-8-sig")
    print(" → station_types.csv")
    with open(args.out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(fit["results"], f, ensure_ascii=False, indent=2, default=float)
    print(" → results.json")
    print(f"\n완료: {args.out_dir}")
    return fit["results"]


if __name__ == "__main__":
    main()
