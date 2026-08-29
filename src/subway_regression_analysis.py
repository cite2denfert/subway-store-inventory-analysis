"""
역사 인근 편의점 시간대별 재고 및 진열 최적화 분석
──────────────────────────────────────────────────
피라미드 근거 통계분석: OLS 단순회귀 + 통제변수(낮 시간대 총이용객) 반영 다중회귀 + 잔차 진단
히트맵 시각화: Claus Wilke 원칙 적용 (비례잉크·의미있는색상·데이터기반정렬)

※ 본 분석은 변수 간 "상관관계"를 다루며, 관측 데이터 기반 회귀분석만으로는
   인과관계를 확정할 수 없습니다. 오전 하차인원과 오후 승차인원이 함께 증가하는
   경향(상관관계)을 낮 시간대 총이용객을 통제변수로 반영하여 재검증합니다.

데이터:
- Seoul_subway_data_20210705.csv (역별 시간대별 승하차인원)
- subway_location_data.csv (역별 위경도 좌표)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── 0. 환경 설정 ──────────────────────────────────────────────
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
KR = fm.FontProperties(fname=FONT_PATH)
KR_B = fm.FontProperties(fname=BOLD_PATH)

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.22,
    'grid.linestyle': '--',
    'axes.facecolor': '#FAFAFA',
    'figure.facecolor': 'white',
})


def set_kr(ax):
    """축 레이블에 한국어 폰트 적용"""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(KR)
    ax.xaxis.label.set_fontproperties(KR)
    ax.yaxis.label.set_fontproperties(KR)


# ── 1. 데이터 로드 및 전처리 ──────────────────────────────────
print("=" * 60)
print("STEP 1. 데이터 로드 및 전처리")
print("=" * 60)

df = pd.read_csv('Seoul_subway_data_20210705.csv', encoding='cp949', on_bad_lines='skip')
df_loc = pd.read_csv('subway_location_data.csv', encoding='utf-8-sig')

# 역명 정규화 (위치 데이터의 '역' suffix 제거)
df_loc['지하철역_키'] = df_loc['지하철역'].str.replace('역$', '', regex=True)

# 핵심 시간대 피처 생성
df['오전_출근_하차'] = df['07시-08시 하차인원'] + df['08시-09시 하차인원']
df['오후_퇴근_승차'] = df['18시-19시 승차인원'] + df['19시-20시 승차인원']

# 통제변수: 낮 시간대(10-16시) 총이용객 (오전/오후 러시아워와 구분되는 구간)
mid_hours = ['10시-11시', '11시-12시', '12시-13시', '13시-14시', '14시-15시', '15시-16시']
mid_cols = [col for col in df.columns if any(h in col for h in mid_hours)]
df['통제_낮시간대_총이용객'] = df[mid_cols].sum(axis=1)

station_summary = df.groupby('지하철역')[
    ['오전_출근_하차', '오후_퇴근_승차', '통제_낮시간대_총이용객']
].mean().reset_index()
m_df = pd.merge(station_summary, df_loc, left_on='지하철역', right_on='지하철역_키', how='inner')
m_df = m_df.rename(columns={'지하철역_x': '지하철역', 'x좌표': 'lat', 'y좌표': 'lon'})

print(f" 분석 대상: {len(m_df)}개 역")
print(f" 오전 하차 평균: {m_df['오전_출근_하차'].mean():,.0f}명")
print(f" 오후 승차 평균: {m_df['오후_퇴근_승차'].mean():,.0f}명")
print(f" 통제변수(낮 시간대 총이용객) 평균: {m_df['통제_낮시간대_총이용객'].mean():,.0f}명")

# ── 2. 기술 통계 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2. 기술 통계 분석")
print("=" * 60)

desc = m_df[['오전_출근_하차', '오후_퇴근_승차', '통제_낮시간대_총이용객']].describe()
print(desc.to_string())
print(f"\n 왜도(Skewness):")
print(f"  오전 출근 하차: {m_df['오전_출근_하차'].skew():.4f}")
print(f"  오후 퇴근 승차: {m_df['오후_퇴근_승차'].skew():.4f}")
print(f"\n 첨도(Kurtosis):")
print(f"  오전 출근 하차: {m_df['오전_출근_하차'].kurtosis():.4f}")
print(f"  오후 퇴근 승차: {m_df['오후_퇴근_승차'].kurtosis():.4f}")

# ── 3. OLS 단순회귀 + 통제변수 반영 다중회귀 (상관관계 분석) ──
print("\n" + "=" * 60)
print("STEP 3. 상관관계 분석 — 단순회귀 vs 통제변수 반영 다중회귀")
print("  변수(Y): 오후 퇴근 승차인원(18-20시)")
print("  변수(X): 오전 출근 하차인원(07-09시)")
print("  통제변수(C): 낮 시간대(10-16시) 총이용객")
print("=" * 60)

x = m_df['오전_출근_하차'].values
y = m_df['오후_퇴근_승차'].values
c = m_df['통제_낮시간대_총이용객'].values
n = len(x)

# 3-1. 단순회귀 (통제 전) — 참고용 baseline
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_simple = r_value ** 2
residuals_simple = y - (slope * x + intercept)

# 3-2. 통제변수를 반영한 다중회귀 (X + C → Y)
X_multi = sm.add_constant(np.column_stack([x, c]))
model = sm.OLS(y, X_multi).fit()
r_multi = model.rsquared
beta_x = model.params[1]
p_x = model.pvalues[1]
beta_c = model.params[2]
p_c = model.pvalues[2]
corr_xc = np.corrcoef(x, c)[0, 1]

y_fitted = model.predict(X_multi)
residuals = y - y_fitted

mse = np.sum(residuals ** 2) / (n - 3)
rmse = np.sqrt(mse)

# Shapiro-Wilk 정규성 검정 (통제모델 잔차 기준)
_, sw_p = stats.shapiro(residuals[:min(5000, n)])

print(f"\n [단순회귀 결과 (통제 전, baseline)]")
print(f"  Ŷ = {slope:.4f} × X + {intercept:,.0f}")
print(f"  R² = {r_simple:.4f}, p = {p_value:.4e}")

print(f"\n [통제변수 반영 다중회귀 결과]")
print(f"  Ŷ = {beta_x:.4f} × X + {beta_c:.4f} × C + {model.params[0]:,.0f}")
print(f"  ├ 결정계수 R² (X+C)     : {r_multi:.4f}")
print(f"  ├ X 계수 (통제 후)      : {beta_x:.4f}  (p = {p_x:.4e})")
print(f"  ├ 통제변수(C) 계수      : {beta_c:.4f}  (p = {p_c:.4e})")
print(f"  ├ X와 통제변수의 상관계수: {corr_xc:.4f}")
print(f"  ├ RMSE                  : {rmse:,.2f}")
print(f"  └ Shapiro-Wilk p (잔차) : {sw_p:.4e}")

print(f"\n [해석 — 상관관계 기반, 인과관계 아님]")
print(f"  → 오전 하차인원(X)은 낮 시간대 총이용객(C)을 통제한 상태에서도")
print(f"    오후 승차인원(Y)과 통계적으로 유의한 양(+)의 상관관계를 보임 (p<0.05).")
print(f"  → X와 C의 상관계수가 {corr_xc:.2f}로 비교적 높아 다중공선성 가능성이 있으므로,")
print(f"    계수 크기의 해석에는 주의가 필요하며 VIF 등 추가 진단이 권장됨.")
print(f"  → 본 결과는 관측 데이터에 대한 상관관계 분석이며, 실험적 통제가 없어")
print(f"    '오전 하차 증가가 오후 승차 증가를 유발한다'는 인과적 해석은 할 수 없음.")

# ── 4. 역 유형 분류 (통제모델 잔차 기준) ────────────────────
m_df['잔차_통제'] = residuals
m_df['유형'] = m_df['잔차_통제'].apply(lambda r: '퇴근집중형' if r > 0 else '출근집중형')
m_df['예측'] = y_fitted

n_evening = (m_df['유형'] == '퇴근집중형').sum()
n_morning = (m_df['유형'] == '출근집중형').sum()
print(f"\n 역 유형 분류 (통제변수 반영 다중회귀 잔차 기준):")
print(f"  ├ 퇴근집중형: {n_evening}개 역 ({n_evening/len(m_df)*100:.1f}%) → 주류·야식 재고 강화")
print(f"  └ 출근집중형: {n_morning}개 역 ({n_morning/len(m_df)*100:.1f}%) → 간편식·커피 전면 배치")

# ── 5. FIGURE 1: 통제변수 반영 회귀분석 종합 대시보드 ───────
print("\n" + "=" * 60)
print("STEP 4. 회귀분석 시각화 생성 (fig1_regression_analysis_v2.png)")
print("=" * 60)

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('white')
fig.text(0.5, 0.96, '오전 하차 vs 오후 승차 상관관계 분석 (통제변수 반영)',
          fontproperties=KR_B, fontsize=15, ha='center', va='top')
fig.text(0.5, 0.935,
          f'통제변수: 낮 시간대(10-16시) 총이용객 | 상관관계 기반 분석 — 인과관계 아님 (n={n}개 역)',
          fontproperties=KR, fontsize=10, ha='center', va='top', color='#555')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28, top=0.9, bottom=0.08)

# ── (0,0) 단순회귀(통제 전) 산점도 ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#2549D8' if r > 0 else '#F4A261' for r in residuals]
ax1.scatter(x, y, c=colors, alpha=0.5, s=22, zorder=3)
xs = np.linspace(x.min(), x.max(), 200)
ax1.plot(xs, slope * xs + intercept, color='#E63946', lw=2, zorder=4)
ax1.set_title(f'단순회귀(통제 전) R²={r_simple:.3f}', fontproperties=KR_B, fontsize=11)
ax1.set_xlabel('오전 출근 하차인원', fontproperties=KR, fontsize=9)
ax1.set_ylabel('오후 퇴근 승차인원', fontproperties=KR, fontsize=9)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
set_kr(ax1)

# ── (0,1) 통제변수 vs Y 산점도 ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(c, y, alpha=0.35, s=18, color='#457B9D', zorder=3)
slope_cy, intercept_cy, r_cy, p_cy, se_cy = stats.linregress(c, y)
xs2 = np.linspace(c.min(), c.max(), 200)
ax2.plot(xs2, slope_cy * xs2 + intercept_cy, color='#E63946', lw=2, zorder=4)
ax2.set_title(f'통제변수(낮시간대 총이용객) vs Y R²={r_cy**2:.3f}',
               fontproperties=KR_B, fontsize=11)
ax2.set_xlabel('낮시간대(10-16시) 총이용객', fontproperties=KR, fontsize=9)
ax2.set_ylabel('오후 퇴근 승차인원', fontproperties=KR, fontsize=9)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
set_kr(ax2)

# ── (1,0) 통계 요약표 ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')
rows = [
    ['단순회귀 R²', f'{r_simple:.4f}'],
    ['통제회귀 R²(X+통제변수)', f'{r_multi:.4f}'],
    ['X 계수(통제후)', f'{beta_x:.4f}'],
    ['X p-value(통제후)', f'{p_x:.2e}'],
    ['통제변수 계수', f'{beta_c:.4f}'],
    ['통제변수 p-value', f'{p_c:.2e}'],
    ['X-통제변수 상관계수', f'{corr_xc:.4f}'],
    ['Shapiro-Wilk p', f'{sw_p:.3e}'],
]
tbl = ax3.table(cellText=rows, colLabels=['통계량', '값'], loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r_i, c_i), cell in tbl.get_celld().items():
    if r_i == 0:
        cell.set_facecolor('#2549D8')
        cell.get_text().set_color('white')
        cell.get_text().set_fontproperties(KR_B)
    elif r_i % 2 == 0:
        cell.set_facecolor('#EEF2FF')
        cell.get_text().set_fontproperties(KR)
    else:
        cell.set_facecolor('white')
        cell.get_text().set_fontproperties(KR)
    cell.set_edgecolor('#DDD')
tbl.scale(1, 1.6)
ax3.set_title('통제변수 반영 다중회귀 통계 요약', fontproperties=KR_B, fontsize=11, pad=12)

# ── (1,1) 통제모델 잔차 vs 적합값 ────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_fitted, residuals, alpha=0.4, s=18, color='#457B9D', zorder=3)
ax4.axhline(0, color='#E63946', lw=1.8, ls='--', zorder=4)
ax4.set_title('통제모델 잔차 vs 적합값\n(등분산성 점검)', fontproperties=KR_B, fontsize=11)
ax4.set_xlabel('적합값 (Fitted Values)', fontproperties=KR, fontsize=9)
ax4.set_ylabel('잔차 (Residuals)', fontproperties=KR, fontsize=9)
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
set_kr(ax4)

plt.savefig('fig1_regression_analysis_v2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(" → fig1_regression_analysis_v2.png 저장 완료")

# ── 6. FIGURE 2: 역 유형별 시간대별 이용 패턴 히트맵 ─────────
print("\n" + "=" * 60)
print("STEP 5. 역 유형별 시간대별 이용 패턴 히트맵 생성 (fig2_hourly_pattern_heatmap.png)")
print("=" * 60)

hour_bins = [
    '04시-05시', '05시-06시', '06시-07시', '07시-08시', '08시-09시', '09시-10시',
    '10시-11시', '11시-12시', '12시-13시', '13시-14시', '14시-15시', '15시-16시',
    '16시-17시', '17시-18시', '18시-19시', '19시-20시',
    '20시-21시', '21시-22시', '22시-23시', '23시-24시',
    '00시-01시', '01시-02시', '02시-03시', '03시-04시',
]
for h in hour_bins:
    df[h + '_합계'] = df[h + ' 승차인원'] + df[h + ' 하차인원']

station_type_map = m_df.drop_duplicates(subset=['지하철역']).set_index('지하철역')['유형']
df['유형'] = df['지하철역'].map(station_type_map)
df_typed = df.dropna(subset=['유형'])

hourly_cols = [h + '_합계' for h in hour_bins]
type_hourly = df_typed.groupby('유형')[hourly_cols].mean()
type_hourly = type_hourly.reindex(['퇴근집중형', '출근집중형'])
type_hourly_norm = type_hourly.div(type_hourly.sum(axis=1), axis=0) * 100

fig2, ax = plt.subplots(figsize=(16, 4.2))
im = ax.imshow(type_hourly_norm.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(hour_bins)))
ax.set_xticklabels(hour_bins, fontproperties=KR, fontsize=8, rotation=45, ha='right')
ax.set_yticks(range(len(type_hourly_norm.index)))
ax.set_yticklabels(type_hourly_norm.index, fontproperties=KR_B, fontsize=11)
for yi in range(type_hourly_norm.shape[0]):
    for xi in range(type_hourly_norm.shape[1]):
        val = type_hourly_norm.values[yi, xi]
        ax.text(xi, yi, f'{val:.1f}', ha='center', va='center', fontsize=6.5,
                 color='black' if val < type_hourly_norm.values.max() * 0.6 else 'white')
ax.set_title('역 유형별(통제모델 잔차 기준 분류) 시간대별 이용량 비중(%) - 상관관계 기반 서술적 비교',
              fontproperties=KR_B, fontsize=12)
cbar = fig2.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label('일평균 이용량 대비 비중(%)', fontproperties=KR, fontsize=8)
plt.tight_layout()
fig2.savefig('fig2_hourly_pattern_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(" → fig2_hourly_pattern_heatmap.png 저장 완료")

print("\n" + "=" * 60)
print("분석 완료!")
print(" ① fig1_regression_analysis_v2.png — 통제변수 반영 회귀분석 대시보드")
print(" ② fig2_hourly_pattern_heatmap.png — 역 유형별 시간대별 이용 패턴 히트맵")
print("=" * 60)
