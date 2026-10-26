"""
BNN v2 학습: 하이브리드 손실함수(NLL + MSE) 적용으로 정확도 올리기
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
SEED        = 42
TEST_SIZE   = 0.2
VAL_SIZE    = 0.1
DROPOUT     = 0.05
WARMUP_EPOCHS = 60
MAX_EPOCHS    = 600
BATCH_SIZE  = 32
N_MC_EVAL   = 100
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0

PREFIX = "bnn_v2"

keras.utils.set_random_seed(SEED)

print("=" * 60)
print(" BNN v2 (최종 버전: 하이브리드 손실함수 적용)")
print("=" * 60)

# ── 1. 데이터 로드 및 피처 공학 (Stab_ratio 포함) ──────────────────────────────
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"[오류] 데이터 파일({DATA_PATH})을 찾을 수 없습니다.")

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

if all(col in df.columns for col in ['Nb', 'Ti', 'C', 'N']):
    df['Stab_ratio'] = (df['Nb'] / 8.0 + df['Ti'] / 4.0) / (df['C'] + df['N'] + 1e-7)
    print("도메인 지식 반영: 'Stab_ratio' 특성 변수 생성 및 추가 완료")

feature_cols = [c for c in df.columns if c != TARGET_COL]

X = df[feature_cols].values.astype(np.float64)
y = df[TARGET_COL].values.astype(np.float64).reshape(-1, 1)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED)
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


# ── 2. 모델 아키텍처 ────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="Custom", name="MCDropout")
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def build_model(n_features):
    inp = keras.Input(shape=(n_features,))
    h = layers.Dense(256, activation="relu")(inp)
    h = MCDropout(DROPOUT)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = MCDropout(DROPOUT)(h)
    h = layers.Dense(64, activation="relu")(h)
    h = MCDropout(DROPOUT)(h)
    
    mu = layers.Dense(1, name="mu")(h)
    logvar = layers.Dense(1, name="logvar")(h)
    out = layers.Concatenate(name="mu_logvar")([mu, logvar])
    return keras.Model(inp, out, name="bnn_v2_hybrid")


@keras.saving.register_keras_serializable(package="Custom")
def mse_on_mu(y_true, y_pred):
    return ops.mean(ops.square(y_true - y_pred[:, :1]))


@keras.saving.register_keras_serializable(package="Custom")
def hybrid_loss(y_true, y_pred):
    """NLL 손실에 MSE 항을 가중 결합하여 평균 예측 정확도(R²)와 불확실성을 동시에 잡음"""
    mu = y_pred[:, :1]
    logvar = ops.clip(y_pred[:, 1:2], LOGVAR_MIN, LOGVAR_MAX)
    nll = ops.mean(0.5 * (logvar + ops.square(y_true - mu) * ops.exp(-logvar)))
    mse = ops.mean(ops.square(y_true - mu))
    return nll + 1.2 * mse  # MSE 비중을 주어 R² 극대화 유도


@keras.saving.register_keras_serializable(package="Custom")
def rmse_mu(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(y_true - y_pred[:, :1])))


model = build_model(X_tr_s.shape[1])
model.summary()


# ── 3. 학습 루프 ─────────────────────────────────────────────────────────────
print(f"\n[2/4] 1단계: MSE 워밍업 학습 ({WARMUP_EPOCHS} epochs)")
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=mse_on_mu, metrics=[rmse_mu])
model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s),
          epochs=WARMUP_EPOCHS, batch_size=BATCH_SIZE, verbose=2)

print(f"\n[3/4] 2단계: 하이브리드 손실(NLL + MSE) 학습 (최대 {MAX_EPOCHS} epochs)")
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=hybrid_loss, metrics=[rmse_mu])
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=60,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20,
                                      min_lr=1e-6, verbose=1),
]
model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s),
          epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

model.save(f"{PREFIX}_model.keras")
print(f"\n✅ 하이브리드 최적화 모델 저장 완료: {PREFIX}_model.keras")


# ── 4. 테스트셋 평가 및 성능 검증 ──────────────────────────────────────────
def mc_predict(model, X_scaled, n_mc=N_MC_EVAL):
    x = np.asarray(X_scaled, dtype=np.float32)
    s = float(scaler_y.scale_[0])
    mus, vars_ = [], []
    for _ in range(n_mc):
        out = np.asarray(model(x, training=False), dtype=np.float64)
        mu = scaler_y.inverse_transform(out[:, :1]).flatten()
        lv = np.clip(out[:, 1], LOGVAR_MIN, LOGVAR_MAX)
        mus.append(mu)
        vars_.append(np.exp(lv) * s ** 2)
    mus, vars_ = np.array(mus), np.array(vars_)
    mean = mus.mean(0)
    epi_var = mus.var(0)
    ale_var = vars_.mean(0)
    rng = np.random.default_rng(SEED)
    samples = mus + np.sqrt(vars_) * rng.standard_normal(mus.shape)
    return mean, np.sqrt(epi_var + ale_var), np.sqrt(epi_var), np.sqrt(ale_var), samples


print("\n[4/4] 테스트셋 최종 성능 평가")
mean, std, epi, ale, samples = mc_predict(model, X_test_s)
yt = y_test.flatten()
lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)

err = mean - yt
r2 = 1 - np.sum(err ** 2) / np.sum((yt - yt.mean()) ** 2)
rmse = np.sqrt(np.mean(err ** 2))
mae = np.mean(np.abs(err))
cov = np.mean((yt >= lo) & (yt <= hi)) * 100

print("-" * 60)
print(f" 🎯 테스트셋 최종 결과 (n={len(yt)})")
print(f"    R² Score       : {r2:.4f}")
print(f"    RMSE           : {rmse:.4f}")
print(f"    MAE            : {mae:.4f}")
print(f"    95% Coverage   : {cov:.1f}%  (목표 95% 신뢰구간)")
print(f"    평균 총 Std    : {std.mean():.4f}")
print("-" * 60)
