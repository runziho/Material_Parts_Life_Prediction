import os
# TensorFlow / Keras Import 전 레거시 설정 적용
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
import keras as keras3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 0. 설정 및 파라미터 정의
# ==============================================================================
DATA_PATH = "Data_Traffic.csv"   # DSRC 공공교통데이터 경로
PREFIX = "bnn_traffic"                   # Traffic 전용 저장 파일 접두사
TARGET_COL = "평균속도"                   # 예측 타깃 (통행속도 km/h)
SEED = 42                                # 재현성을 위한 시드
TEST_SIZE = 0.2                          # 테스트셋 비율
VAL_SIZE = 0.1                           # 검증셋 비율
DROPOUT_RATE = 0.1                       # MC Dropout 비율
WARMUP_EPOCHS = 30                       # 1단계 MSE 워밍업 에포크
MAX_EPOCHS = 200                         # 2단계 Gaussian NLL 학습 에포크
BATCH_SIZE = 256                         # 교통 데이터 대용량 처리를 위한 바치 크기
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0       # 분산 폭주 방지 핑

keras3.utils.set_random_seed(SEED)

print("=" * 60)
print(" 🚗 Traffic BNN (교통 통행속도 및 정체 불확실성 예측) 모델 학습")
print("=" * 60)

# ==============================================================================
# 1. 커스텀 MCDropout 레이어 & Gaussian NLL 손실함수
# ==============================================================================
@keras3.saving.register_keras_serializable(package="Custom", name="MCDropout")
class MCDropout(keras3.layers.Dropout):
    """추론(Inference)시에도 무작위 드롭아웃을 활성화하는 MC Dropout 레이어"""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

@keras3.saving.register_keras_serializable(package="Custom", name="gaussian_nll")
def gaussian_nll(y_true, y_pred):
    """Gaussian Negative Log-Likelihood 손실함수 (평균 μ 및 로그분산 log σ² 학습)"""
    mu = y_pred[:, :1]
    logvar = keras3.ops.clip(y_pred[:, 1:2], LOGVAR_MIN, LOGVAR_MAX)
    return keras3.ops.mean(0.5 * (logvar + keras3.ops.square(y_true - mu) * keras3.ops.exp(-logvar)))

# ==============================================================================
# 2. 교통 데이터 로드 및 피처 엔지니어링 전처리
# ==============================================================================
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"[오류] 교통 데이터 파일({DATA_PATH})을 찾을 수 없습니다.")

print(f"📥 교통 데이터 로드 중: {DATA_PATH}...")
df_raw = pd.read_csv(DATA_PATH, encoding='euc-kr')

# 2-1. 통행차량 없음 및 이상치(-1.0 이하 속도) 정제
df = df_raw[df_raw[TARGET_COL] > 0].copy()

# 2-2. 날짜/시간 피처 생성 (시간대 Hour, 요일 Weekday)
df['datetime'] = pd.to_datetime(df['집계일자'] + ' ' + df['집계시분'], errors='coerce')
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday

# 2-3. BNN 학습용 특성 컬럼 정의
feature_cols = ['hour', 'weekday', 'DSRC차종유형구분코드']

# 카테고리형 도로 링크 ID 인코딩
if 'DSRC링크ID' in df.columns:
    df['link_code'] = df['DSRC링크ID'].astype('category').cat.codes
    feature_cols.append('link_code')

X = df[feature_cols].values.astype(np.float64)
y = df[TARGET_COL].values.astype(np.float64).reshape(-1, 1)

print(f"📊 정제 데이터 크기: {len(df):,}개 | 사용 입력 피처 ({len(feature_cols)}개): {feature_cols}")

# ==============================================================================
# 3. 데이터 분할 및 스케일링
# ==============================================================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED
)

scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_tr_s = scaler_X.fit_transform(X_tr)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)
y_tr_s = scaler_y.fit_transform(y_tr)
y_val_s = scaler_y.transform(y_val)

# 파일 저장 (app_gradio_traffic.py에서 불러와 활용)
joblib.dump(scaler_X, f"{PREFIX}_scaler_X.pkl")
joblib.dump(scaler_y, f"{PREFIX}_scaler_y.pkl")
joblib.dump(feature_cols, f"{PREFIX}_features.pkl")
print(f"✅ 스케일러 및 피처 구성 저장 완료 ({PREFIX}_scaler_X.pkl 등)")

# ==============================================================================
# 4. BNN 모델 네트워크 구축 및 2단계 학습 (MSE -> Gaussian NLL)
# ==============================================================================
inputs = keras3.Input(shape=(len(feature_cols),))
x = keras3.layers.Dense(128, activation="relu")(inputs)
x = MCDropout(DROPOUT_RATE)(x)
x = keras3.layers.Dense(64, activation="relu")(x)
x = MCDropout(DROPOUT_RATE)(x)
x = keras3.layers.Dense(32, activation="relu")(x)
x = MCDropout(DROPOUT_RATE)(x)

out_mu = keras3.layers.Dense(1, name="mu")(x)
out_logvar = keras3.layers.Dense(1, name="logvar")(x)
outputs = keras3.layers.Concatenate(axis=-1, name="bnn_output")([out_mu, out_logvar])

model = keras3.Model(inputs=inputs, outputs=outputs, name="Traffic_BNN_v2")

# --- 1단계: MSE 워밍업 (평균 예측 초기화) ---
print("\n🔥 [1/2 단계] MSE 손실함수로 평균(μ) 워밍업 학습...")
def mse_warmup(y_true, y_pred):
    return keras3.ops.mean(keras3.ops.square(y_true - y_pred[:, :1]))

model.compile(optimizer=keras3.optimizers.Adam(learning_rate=1e-3), loss=mse_warmup)
model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s),
          epochs=WARMUP_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# --- 2단계: Gaussian NLL 본 학습 (평균 + 불확실성 동시 습득) ---
print("\n🎯 [2/2 단계] Gaussian NLL 손실함수로 정체 불확실성 동시 학습...")
callbacks = [
    keras3.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    keras3.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
]

model.compile(optimizer=keras3.optimizers.Adam(learning_rate=5e-4), loss=gaussian_nll)
model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s),
          epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

# ==============================================================================
# 5. 모델 저장 및 성능 평가
# ==============================================================================
model_path = f"{PREFIX}_model.keras"
model.save(model_path)
print(f"\n💾 Traffic BNN 모델 저장 완료: {model_path}")

# 테스트셋 평가
y_pred_s = model.predict(X_test_s, batch_size=BATCH_SIZE)[:, :1]
y_pred = scaler_y.inverse_transform(y_pred_s)

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print("=" * 60)
print(f"🎉 Traffic BNN 학습 완료!")
print(f"📈 Test RMSE: {rmse:.2f} km/h")
print(f"📈 Test R² Score: {r2:.4f}")
print("=" * 60)