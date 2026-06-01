import numpy as np
import matplotlib.pyplot as plt

# 데이터 설정 (터미널 출력값 기반)
beta_exp2 = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80])
log_inv_T = np.array([-8.319, -8.392, -8.395, -8.160, -8.318, -8.830, -9.274, -9.714, -9.665])

beta_exp3 = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00])
success_rate = np.array([90, 86, 82, 92, 86, 68, 60, 74, 50, 48, 30, 18, 8])

beta_c = 0.4244
beta_star = 0.5500

# 폰트 및 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

# 두 개의 그래프를 한 장에 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ─────────────────────────────────────────────────────────────────
# [EXP-2 그래프] log(1/T) vs beta
# ─────────────────────────────────────────────────────────────────
# 실제 실험 데이터 플롯
ax1.scatter(beta_exp2, log_inv_T, color='royalblue', s=60, zorder=3, label='Empirical Data (●)')

# 선형 회귀 추세선 표현 (y = -2.7508*x - 7.5321)
x_range = np.linspace(0.15, 0.85, 100)
y_trend = -2.7508 * x_range - 7.5321
ax1.plot(x_range, y_trend, color='darkorange', linestyle='--', linewidth=2, 
         label='Linear Fit (Slope: -2.751)\nTheory ($-H^*$: -2.356)')

ax1.set_title('EXP-2: Convergence Rate Scaling [Theorem 5.1]', weight='bold', pad=15)
ax1.set_xlabel(r'Inverse Temperature ($\beta$)')
ax1.set_ylabel(r'Convergence Speed $\log(1/T_{cov})$')
ax1.legend(frameon=True, facecolor='white', edgecolor='gainsboro')

# ─────────────────────────────────────────────────────────────────
# [EXP-3 그래프] 전역 커버리지 성공률 vs beta
# ─────────────────────────────────────────────────────────────────
# 성공률 바 차트 (임계 구역 기준 색상 분기)
colors = ['#52be80' if b < beta_c else '#e67e22' if b <= beta_star else '#e74c3c' for b in beta_exp3]
bars = ax2.bar(beta_exp3, success_rate, width=0.04, color=colors, edgecolor='grey', alpha=0.85, zorder=2)

# 임계선 강조 (이론적 beta_c 및 실험적 beta_star)
ax2.axvline(beta_c, color='darkgreen', linestyle=':', linewidth=2, label=r'Theory $\beta_c = 0.4244$')
ax2.axvline(beta_star, color='darkred', linestyle='-.', linewidth=2, label=r'Measured $\beta^* = 0.5500$')

# 텍스트 주석 추가 (영역별 거동 의미 설명)
ax2.text(0.20, 45, 'Global\nExploration', color='darkgreen', weight='bold', ha='center')
ax2.text(0.75, 45, 'Exponential\nTrapping', color='darkred', weight='bold', ha='center')

# 데이터 레이블 추가
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{int(height)}%', 
             ha='center', va='bottom', fontsize=9, weight='semibold')

ax2.set_title('EXP-3: Phase Transition Analysis [Section 7]', weight='bold', pad=15)
ax2.set_xlabel(r'Inverse Temperature ($\beta$)')
ax2.set_ylabel('Coverage Success Rate (%)')
ax2.set_ylim(0, 105)
ax2.legend(frameon=True, facecolor='white', edgecolor='gainsboro', loc='upper right')

# 레이아웃 조정 및 저장
plt.tight_layout()
plt.savefig('langevin_mcmc_verification.png', dpi=300)
plt.show()
print("[알림] 시각화 그래프가 'langevin_mcmc_verification.png' 파일로 고해상도 저장되었습니다.")
