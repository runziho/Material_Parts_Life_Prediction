"""
BNN v2 학습: MC Dropout + 노이즈(분산) 동시 예측 (Heteroscedastic)

v1 대비 바뀐 점
  1) 출력이 2개: [예측 평균 μ, 로그분산 log σ²]  → 데이터 노이즈(aleatoric)까지 학습
  2) 손실함수: Gaussian NLL (먼저 MSE로 워밍업 후 NLL로 전환해서 학습 안정화)
  3) 학습률 0.01 → 0.001, ReduceLROnPlateau, EarlyStopping(best 가중치 복원)
  4) 네트워크 64-32 → 128-64-32
  5) 학습 끝나면 테스트셋 R², RMSE, 95% Coverage를 바로 출력

저장 파일 (v1 파일은 건드리지 않음)
  bnn_v2_model.keras, bnn_v2_scaler_X.pkl, bnn_v2_scaler_y.pkl, bnn_v2_features.pkl

실행: py -3.11 train_BNN_v2.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import keras
from keras import layers, ops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 설정 ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "creep_austenite.xlsx"
TARGET_COL  = "log10(t_r / h)"
SEED        = 42                # v1, ANN, NGBoost와 같은 테스트셋을 쓰려면 42 유지
TEST_SIZE   = 0.2
VAL_SIZE    = 0.1               # 학습셋에서 검증용으로 떼어내는 비율
DROPOUT     = 0.1
WARMUP_EPOCHS = 40              # 1단계: MSE로 평균만 먼저 학습
MAX_EPOCHS    = 400             # 2단계: NLL 학습 (EarlyStopping이 알아서 멈춤)
BATCH_SIZE  = 64
N_MC_EVAL   = 100               # 테스트 평가용 MC 샘플 수
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0   # 분산이 폭주하지 않도록 제한 (스케일된 y 기준)

PREFIX = "bnn_v2"

keras.utils.set_random_seed(SEED)

print("=" * 60)
print(" BNN v2 (MC Dropout + 노이즈 예측) 학습")
print("=" * 60)

# ── 1. 데이터 ────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"[오류] 데이터 파일({DATA_PATH})을 찾을 수 없습니다.")

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()
feature_cols = [c for c in df.columns if c != TARGET_COL]

X = df[feature_cols].values.astype(np.float64)
y = df[TARGET_COL].values.astype(np.float64).reshape(-1, 1)

# v1과 완전히 같은 테스트셋 (random_state=42, 20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED)
# 학습셋에서 검증셋 분리 (조기종료 판단용)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED)

scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_tr_s = scaler_X.fit_transform(X_tr)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)
y_tr_s = scaler_y.fit_transform(y_tr)
y_val_s = scaler_y.transform(y_val)

joblib.dump(scaler_X, f"{PREFIX}_scaler_X.pkl")
joblib.dump(scaler_y, f"{PREFIX}_scaler_y.pkl")
joblib.dump(feature_cols, f"{PREFIX}_features.pkl")
print(f"[1/4] 데이터 준비 완료: 학습 {len(X_tr)} / 검증 {len(X_val)} / 테스트 {len(X_test)}, feature {len(feature_cols)}개")


# ── 2. 모델 ──────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="Custom", name="MCDropout")
class MCDropout(layers.Dropout):
    """추론 때도 드롭아웃을 켜두는 레이어 (Monte Carlo Dropout)"""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def build_model(n_features):
    inp = keras.Input(shape=(n_features,))
    h = layers.Dense(128, activation="relu")(inp)
    h = MCDropout(DROPOUT)(h)
    h = layers.Dense(64, activation="relu")(h)
    h = MCDropout(DROPOUT)(h)
    h = layers.Dense(32, activation="relu")(h)
    h = MCDropout(DROPOUT)(h)
    mu = layers.Dense(1, name="mu")(h)          # 예측 평균 (스케일된 y)
    logvar = layers.Dense(1, name="logvar")(h)  # 로그분산 (스케일된 y)
    out = layers.Concatenate(name="mu_logvar")([mu, logvar])
    return keras.Model(inp, out, name="bnn_v2")


@keras.saving.register_keras_serializable(package="Custom")
def mse_on_mu(y_true, y_pred):
    """워밍업용: 평균 출력만 MSE로 학습"""
    return ops.mean(ops.square(y_true - y_pred[:, :1]))


@keras.saving.register_keras_serializable(package="Custom")
def gaussian_nll(y_true, y_pred):
    """평균과 분산을 함께 학습하는 Gaussian negative log-likelihood"""
    mu = y_pred[:, :1]
    logvar = ops.clip(y_pred[:, 1:2], LOGVAR_MIN, LOGVAR_MAX)
    return ops.mean(0.5 * (logvar + ops.square(y_true - mu) * ops.exp(-logvar)))


@keras.saving.register_keras_serializable(package="Custom")
def rmse_mu(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(y_true - y_pred[:, :1])))


model = build_model(X_tr_s.shape[1])
model.summary()

# ── 3. 학습 ──────────────────────────────────────────────────────────────────
print(f"\n[2/4] 1단계: MSE 워밍업 ({WARMUP_EPOCHS} epochs)")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=mse_on_mu, metrics=[rmse_mu])
model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s),
          epochs=WARMUP_EPOCHS, batch_size=BATCH_SIZE, verbose=2)

print(f"\n[3/4] 2단계: NLL 학습 (최대 {MAX_EPOCHS} epochs, 조기종료 적용)")
model.compile(optimizer=keras.optimizers.Adam(5e-4), loss=gaussian_nll, metrics=[rmse_mu])
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=12,
                                      min_lr=1e-5, verbose=1),
]
model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s),
          epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

model.save(f"{PREFIX}_model.keras")
print(f"\n✅ 모델 저장 완료: {PREFIX}_model.keras")


# ── 4. 테스트셋 평가 ─────────────────────────────────────────────────────────
def mc_predict(model, X_scaled, n_mc=N_MC_EVAL):
    """MC 샘플링 예측. 반환: 평균, 총 Std, 모델 불확실성 Std, 노이즈 Std, 예측 샘플"""
    x = np.asarray(X_scaled, dtype=np.float32)
    s = float(scaler_y.scale_[0])
    mus, vars_ = [], []
    for _ in range(n_mc):
        out = np.asarray(model(x, training=False), dtype=np.float64)
        mu = scaler_y.inverse_transform(out[:, :1]).flatten()
        lv = np.clip(out[:, 1], LOGVAR_MIN, LOGVAR_MAX)
        mus.append(mu)
        vars_.append(np.exp(lv) * s ** 2)       # 원래 단위(log10 h)의 분산
    mus, vars_ = np.array(mus), np.array(vars_)
    mean = mus.mean(0)
    epi_var = mus.var(0)                         # 모델 불확실성 (epistemic)
    ale_var = vars_.mean(0)                      # 데이터 노이즈 (aleatoric)
    rng = np.random.default_rng(SEED)
    samples = mus + np.sqrt(vars_) * rng.standard_normal(mus.shape)
    return mean, np.sqrt(epi_var + ale_var), np.sqrt(epi_var), np.sqrt(ale_var), samples


print("\n[4/4] 테스트셋 평가")
mean, std, epi, ale, samples = mc_predict(model, X_test_s)
yt = y_test.flatten()
lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)

err = mean - yt
r2 = 1 - np.sum(err ** 2) / np.sum((yt - yt.mean()) ** 2)
rmse = np.sqrt(np.mean(err ** 2))
mae = np.mean(np.abs(err))
cov = np.mean((yt >= lo) & (yt <= hi)) * 100

print("-" * 60)
print(f" 테스트셋 (n={len(yt)})")
print(f"   R²              : {r2:.4f}")
print(f"   RMSE            : {rmse:.4f}   (v1: 0.3142, NGBoost: 0.2678, ANN: 0.2725)")
print(f"   MAE             : {mae:.4f}")
print(f"   95% Coverage    : {cov:.1f}%   (목표 95%, v1: 67.4%, NGBoost: 87.7%)")
print(f"   평균 총 Std     : {std.mean():.4f}")
print(f"   └ 모델 불확실성 : {epi.mean():.4f}")
print(f"   └ 데이터 노이즈 : {ale.mean():.4f}")
print("-" * 60)
print("\n다음 단계: py -3.11 eval_compare.py  (v1, v2, ANN, NGBoost 한 번에 비교)")