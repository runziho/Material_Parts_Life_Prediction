"""
건강·생활습관 데이터 BNN 모델 재학습 (피처 10개 고정 버전)
실행: py -3.11 train_health_bnn.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import keras
from keras import layers, ops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH   = "health_data.xlsx"  # 본인의 건강 엑셀 파일명
TARGET_COL  = "bmi"               # 예측할 타겟
SEED        = 42
TEST_SIZE   = 0.2
VAL_SIZE    = 0.1
DROPOUT     = 0.05
WARMUP_EPOCHS = 40
MAX_EPOCHS    = 300
BATCH_SIZE  = 32
N_MC_EVAL   = 100
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0
PREFIX = "health_bnn"

keras.utils.set_random_seed(SEED)

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

# 범주형 변수 인코딩
for col in df.select_dtypes(include=['object']).columns:
    if col != TARGET_COL:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

if df[TARGET_COL].dtype == 'object':
    df[TARGET_COL] = LabelEncoder().fit_transform(df[TARGET_COL].astype(str))

# 10개의 입력 피처 명시적 정의 (웹 입력 폼과 100% 일치)
FEATURE_COLS = ['age', 'weight', 'height', 'exercise', 'sleep', 'sugar_intake', 'smoking', 'alcohol', 'married', 'profession']

# 데이터에 혹시 빠진 컬럼이 있는지 확인 후 0으로 채우기
for c in FEATURE_COLS:
    if c not in df.columns:
        df[c] = 0.0

X = df[FEATURE_COLS].values.astype(np.float64)
y = df[TARGET_COL].values.astype(np.float64).reshape(-1, 1)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED)

scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_tr_s = scaler_X.fit_transform(X_tr)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)
y_tr_s = scaler_y.fit_transform(y_tr)
y_val_s = scaler_y.transform(y_val)

# 명시적인 피처 리스트 저장
joblib.dump(scaler_X, f"{PREFIX}_scaler_X.pkl")
joblib.dump(scaler_y, f"{PREFIX}_scaler_y.pkl")
joblib.dump(FEATURE_COLS, f"{PREFIX}_features.pkl")

print(f"✅ 학습 데이터 준비 완료: 피처 {len(FEATURE_COLS)}개")

# 모델 아키텍처 및 학습
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
    return keras.Model(inp, out, name="health_bnn_hybrid")

@keras.saving.register_keras_serializable(package="Custom")
def mse_on_mu(y_true, y_pred):
    return ops.mean(ops.square(y_true - y_pred[:, :1]))

@keras.saving.register_keras_serializable(package="Custom")
def hybrid_loss(y_true, y_pred):
    mu = y_pred[:, :1]
    logvar = ops.clip(y_pred[:, 1:2], LOGVAR_MIN, LOGVAR_MAX)
    nll = ops.mean(0.5 * (logvar + ops.square(y_true - mu) * ops.exp(-logvar)))
    mse = ops.mean(ops.square(y_true - mu))
    return nll + 1.2 * mse

model = build_model(X_tr_s.shape[1])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=mse_on_mu)
model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s), epochs=WARMUP_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=hybrid_loss)
model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s), epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

model.save(f"{PREFIX}_model.keras")
print(f"✅ 학습 및 모델 저장 완료: {PREFIX}_model.keras")