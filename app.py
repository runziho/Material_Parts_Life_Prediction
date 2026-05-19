import streamlit as st
import pandas as pd
import numpy as np
import os

# ---------------- 환경 설정 ----------------
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tf_keras as tfk
import joblib
import matplotlib.pyplot as plt

# ---------------- 상태 ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "df" not in st.session_state:
    st.session_state.df = None

if "result_df" not in st.session_state:
    st.session_state.result_df = None

# ---------------- 모델 로드 ----------------
@st.cache_resource
def load_models():
    models = [tfk.models.load_model(f"model_{i}.keras") for i in range(10)]
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return models, scaler_X, scaler_y

models, scaler_X, scaler_y = load_models()

# ---------------- 사이드바 ----------------
st.sidebar.title("📌 메뉴")

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"

if st.sidebar.button("📂 Data Upload"):
    st.session_state.page = "Data Upload"

if st.sidebar.button("⚙️ Prediction"):
    st.session_state.page = "Prediction"

if st.sidebar.button("📊 Result"):
    st.session_state.page = "Result"

page = st.session_state.page

# ---------------- HOME ----------------
if page == "Home":
    st.title("소재부품 수명 예측 시스템")
    st.write("왼쪽 메뉴에서 데이터 업로드 후 예측을 진행하세요")

# ---------------- DATA ----------------
elif page == "Data Upload":

    st.header("📂 데이터 업로드")

    uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df

        st.success("업로드 완료")
        st.dataframe(df.head())

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.header("⚙️ 예측 실행")

    df = st.session_state.df

    if df is None:
        st.warning("먼저 데이터 업로드 하세요")
    else:
        if st.button("🚀 예측 시작"):

            epsilon = 1e-7
            df['Stab_ratio'] = (df['Nb']/8 + df['Ti']/4) / (df['C'] + df['N'] + epsilon)

            if 'log10(t_r / h)' in df.columns:
                X = df.drop(columns=['log10(t_r / h)'])
                y_true = df['log10(t_r / h)']
            else:
                X = df.copy()
                y_true = None

            X_scaled = scaler_X.transform(X)

            preds = []

            for model in models:
                pred_scaled = model.predict(X_scaled, verbose=0)
                pred = scaler_y.inverse_transform(pred_scaled)
                preds.append(pred.flatten())

            preds = np.array(preds)

            mean_pred = preds.mean(axis=0)
            std_pred = preds.std(axis=0)

            result_df = df.copy()
            result_df["Predicted Life"] = mean_pred
            result_df["Uncertainty"] = std_pred

            st.session_state.result_df = result_df
            st.session_state.y_true = y_true
            st.session_state.y_pred = mean_pred

            st.success("예측 완료")

# ---------------- RESULT ----------------
elif page == "Result":

    st.header("📊 결과")

    result_df = st.session_state.result_df
    y_true = st.session_state.get("y_true", None)
    y_pred = st.session_state.get("y_pred", None)

    if result_df is None:
        st.info("아직 예측 결과 없음")
    else:
        st.subheader("📋 개별 예측 결과")
        st.dataframe(result_df)

        # ---------------- 분포 그래프 ----------------
        st.subheader("📈 예측 분포")

        fig, ax = plt.subplots()
        ax.scatter(range(len(y_pred)), y_pred, alpha=0.6)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Predicted Life")
        ax.set_title("Prediction Distribution")

        st.pyplot(fig)

        # ---------------- 실제값 비교 ----------------
        if y_true is not None:
            st.subheader("📊 실제값 vs 예측값")

            fig2, ax2 = plt.subplots()

            ax2.scatter(y_true, y_pred, alpha=0.6)

            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))

            ax2.plot([min_val, max_val], [min_val, max_val], 'r--')

            ax2.set_xlabel("True Value")
            ax2.set_ylabel("Predicted Value")

            st.pyplot(fig2)

        # ---------------- 다운로드 ----------------
        csv = result_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 결과 다운로드",
            data=csv,
            file_name="prediction_result.csv",
            mime="text/csv"
        )