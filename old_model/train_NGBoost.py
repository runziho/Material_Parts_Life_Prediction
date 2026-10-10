"""
NGBoost 크립 수명 예측 모델 학습 스크립트
==========================================
실행 방법:
    python train_ngboost.py

출력 파일:
    ngboost_model.pkl   — 학습된 NGBoost 모델
    scaler_X_ngb.pkl    — 입력 스케일러

의존 패키지:
    pip install ngboost scikit-learn pandas openpyxl joblib matplotlib
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. 설정 ───────────────────────────────────────────────────────────────────
DATA_PATH    = "creep_austenite.xlsx"
MODEL_PATH   = "ngboost_model.pkl"
SCALER_PATH  = "scaler_X_ngb.pkl"

N_ESTIMATORS  = 800
LEARNING_RATE = 0.05
MAX_DEPTH     = 4
MIN_SAMPLES   = 10
EARLY_STOP    = 50
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# ── 2. 데이터 로드 ────────────────────────────────────────────────────────────
print("=" * 55)
print("NGBoost 크립 수명 예측 모델 학습")
print("=" * 55)
print("\n[1/5] 데이터 로드 중...")

df = pd.read_excel(DATA_PATH)
print(f"  데이터: {df.shape[0]}행 × {df.shape[1]}열")

# 파생 변수 (Stabilisation ratio)
epsilon = 1e-7
df["Stab_ratio"] = (df["Nb"] / 8 + df["Ti"] / 4) / (df["C"] + df["N"] + epsilon)

X = df.drop(columns=["log10(t_r / h)"])
y = df["log10(t_r / h)"].values

# ── 3. 분할 및 스케일링 ───────────────────────────────────────────────────────
print("\n[2/5] 데이터 분할 및 스케일링 중...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train).astype(np.float64)
X_test_s  = scaler_X.transform(X_test).astype(np.float64)
y_train   = y_train.astype(np.float64)
y_test    = y_test.astype(np.float64)

print(f"  학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
print(f"  입력 변수: {X_train_s.shape[1]}개")

# ── 4. 모델 학습 ──────────────────────────────────────────────────────────────
print("\n[3/5] NGBoost 학습 중...")
print(f"  n_estimators={N_ESTIMATORS}, lr={LEARNING_RATE}, depth={MAX_DEPTH}")

ngb = NGBRegressor(
    Dist             = Normal,
    Score            = MLE,
    Base             = DecisionTreeRegressor(
                           max_depth=MAX_DEPTH,
                           min_samples_leaf=MIN_SAMPLES
                       ),
    n_estimators     = N_ESTIMATORS,
    learning_rate    = LEARNING_RATE,
    natural_gradient = True,
    verbose          = True,
    verbose_eval     = 100,
    random_state     = RANDOM_STATE,
)

ngb.fit(
    X_train_s, y_train,
    X_val=X_test_s, Y_val=y_test,
    early_stopping_rounds=EARLY_STOP,
)

best_iter = getattr(ngb, "best_val_loss_itr", N_ESTIMATORS)
print(f"\n  최적 iteration: {best_iter}")

# ── 5. 성능 평가 ──────────────────────────────────────────────────────────────
print("\n[4/5] 성능 평가 중...")

dist       = ngb.pred_dist(X_test_s)
y_pred     = dist.loc
y_std      = dist.scale
y_ci95     = 1.96 * y_std

rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
r2        = r2_score(y_test, y_pred)
within_95 = np.mean(np.abs(y_test - y_pred) <= y_ci95) * 100

print("-" * 45)
print(f"  RMSE             : {rmse:.4f}")
print(f"  R² Score         : {r2:.4f}")
print(f"  95% CI Coverage  : {within_95:.1f}%")
print(f"  평균 σ           : {y_std.mean():.4f}")
print("-" * 45)

# ── 6. 모델 저장 ──────────────────────────────────────────────────────────────
print("\n[5/5] 모델 저장 중...")

joblib.dump(ngb,      MODEL_PATH)
joblib.dump(scaler_X, SCALER_PATH)

print(f"  저장 완료!")
print(f"  └ {MODEL_PATH}")
print(f"  └ {SCALER_PATH}")

# ── 7. 시각화 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 예측 vs 실제
ax = axes[0]
ax.errorbar(y_test, y_pred, yerr=y_ci95,
            fmt='o', alpha=0.4, ecolor='lightcoral',
            elinewidth=1.0, capsize=2, markersize=4)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='y=x')
ax.set_xlabel("실제값 [log10(t_r / h)]")
ax.set_ylabel("예측값 [log10(t_r / h)]")
ax.set_title(f"예측 vs 실제\nRMSE={rmse:.4f}  R²={r2:.4f}")
ax.legend(); ax.grid(True, linestyle='--', alpha=0.4)

# 불확실성 vs 오차
ax = axes[1]
abs_err = np.abs(y_test - y_pred)
sc = ax.scatter(y_std, abs_err, alpha=0.4, s=12, c=abs_err, cmap='YlOrRd')
plt.colorbar(sc, ax=ax, label='|오차|')
ax.set_xlabel("불확실성 σ")
ax.set_ylabel("절대 오차")
ax.set_title(f"불확실성 vs 오차\n95% Coverage={within_95:.1f}%")
ax.grid(True, linestyle='--', alpha=0.4)

# 불확실성 분포
ax = axes[2]
ax.hist(y_std, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(y_std.mean(), color='red', linestyle='--', lw=1.5,
           label=f'평균 σ={y_std.mean():.4f}')
ax.set_xlabel("불확실성 σ")
ax.set_ylabel("샘플 수")
ax.set_title("불확실성 분포")
ax.legend(); ax.grid(True, linestyle='--', alpha=0.4)

plt.suptitle("NGBoost 크립 수명 예측 결과", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("ngboost_train_result.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n완료! 결과 그래프: ngboost_train_result.png")
print("=" * 55)
