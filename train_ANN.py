# ---------------- 1. 환경 설정 ----------------
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ---------------- 2. 라이브러리 ----------------
import pandas as pd
import numpy as np
import tensorflow as tf
import tf_keras as tfk
import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import joblib

tfd = tfp.distributions
tfpl = tfp.layers

# ---------------- 3. 데이터 로드 ----------------
df = pd.read_excel('creep_austenite.xlsx')

# ---------------- 4. 파생 변수 ----------------
epsilon = 1e-7
df['Stab_ratio'] = (df['Nb']/8 + df['Ti']/4) / (df['C'] + df['N'] + epsilon)

# ---------------- 5. X / y ----------------
X = df.drop(columns=['log10(t_r / h)'])
y = df['log10(t_r / h)']

# ---------------- 6. train/test 분리 ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=3000, random_state=42
)

# ---------------- 7. 스케일링 ----------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# 👉 스케일러 저장 (중요)
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

# ---------------- 8. 앙상블 학습 ----------------
print("위원회(Committee) 모델 학습 시작")

n_models = 10
models = []

callback = tfk.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

for i in range(n_models):
    print(f"\n--- Model {i+1}/{n_models} ---")

    model = tfk.Sequential([
        tfk.layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),
        tfk.layers.Dense(64, activation='relu'),
        tfk.layers.Dense(32, activation='relu'),
        tfk.layers.Dense(16, activation='relu'),
        tfk.layers.Dense(1)
    ])

    model.compile(
        optimizer=tfk.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )

    model.fit(
        X_train_scaled, y_train_scaled,
        epochs=150,
        batch_size=64,
        validation_split=0.2,
        callbacks=[callback],
        verbose=0
    )

    # 👉 모델 저장 (핵심 추가)
    model.save(f"model_{i}.keras")

    models.append(model)

print("\n모델 학습 완료 및 저장 완료")

# ---------------- 9. 테스트 평가 ----------------
print("테스트 데이터 평가 시작")

y_preds_list = []

for model in models:
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_preds_list.append(y_pred_scaled.flatten())

y_preds_array = np.array(y_preds_list)

y_pred_mean_scaled = y_preds_array.mean(axis=0)
y_pred_std_scaled = y_preds_array.std(axis=0)

y_pred_mean = scaler_y.inverse_transform(
    y_pred_mean_scaled.reshape(-1, 1)
).flatten()

y_pred_std = y_pred_std_scaled * scaler_y.scale_[0]

# ---------------- 10. 성능 ----------------
mse = mean_squared_error(y_test, y_pred_mean)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_mean)

print("-" * 30)
print(f"RMSE : {rmse:.4f}")
print(f"R2 Score : {r2:.4f}")
print("-" * 30)

# ---------------- 11. 그래프 ----------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.errorbar(
    x=y_test,
    y=y_pred_mean,
    yerr=2 * y_pred_std,
    fmt='o',
    alpha=0.6,
    ecolor='lightgray',
    elinewidth=1.5,
    capsize=3
)

min_val = min(y_test.min(), y_pred_mean.min())
max_val = max(y_test.max(), y_pred_mean.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Committee Model')

plt.show()