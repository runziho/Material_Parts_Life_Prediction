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

if st.sidebar.button("🤖 AI Assistant"):
    st.session_state.page = "Chatbot"

page = st.session_state.page

# ---------------- HOME ----------------
if page == "Home":
    st.title("소재부품 수명 예측 시스템")
    st.write("왼쪽 메뉴에서 기능을 선택하세요")

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

    if result_df is None:
        st.info("아직 예측 결과 없음")
    else:
        # 🔥 실제 시간 변환
        result_df["Predicted Life (hours)"] = 10 ** result_df["Predicted Life"]

        st.subheader("📋 개별 예측 결과")
        st.dataframe(result_df)

        # ---------------- 자동 해석 ----------------
        st.subheader("🤖 결과 해석")

        avg_life = result_df["Predicted Life"].mean()

        if avg_life > 4:
            st.write("👉 전반적으로 수명이 매우 긴 고내열 합금 조건입니다.")
        elif avg_life > 3:
            st.write("👉 일반적인 고온 환경에서 사용 가능한 수준입니다.")
        else:
            st.write("👉 수명이 짧아 조건이 가혹한 상태입니다.")

        # ---------------- 시뮬레이션 ----------------
        st.subheader("🧪 합금 성분 시뮬레이션")

        selected_idx = st.selectbox("샘플 선택", result_df.index)
        selected_row = result_df.loc[selected_idx].copy()

        # 🔥 기존 값
        original_pred = selected_row["Predicted Life"]
        original_hours = 10 ** original_pred

        exclude_cols = ["Predicted Life", "Uncertainty", "Predicted Life (hours)", "log10(t_r / h)"]
        available_cols = [col for col in selected_row.index if col not in exclude_cols]

        selected_features = st.multiselect(
            "조절할 성분 선택",
            available_cols,
            default=available_cols[:3]
        )

        modified_row = selected_row.copy()

        for feature in selected_features:
            val = float(selected_row[feature])

            new_val = st.slider(
                feature,
                float(val * 0.5),
                float(val * 1.5),
                val
            )

            modified_row[feature] = new_val

        # 🔥 입력 생성
        feature_cols = [col for col in modified_row.index if col not in exclude_cols]
        X_new = pd.DataFrame([modified_row[feature_cols]])

        X_scaled = scaler_X.transform(X_new)

        preds = []
        for model in models:
            pred_scaled = model.predict(X_scaled, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)
            preds.append(pred[0][0])

        new_pred = np.mean(preds)
        new_pred_hours = 10 ** new_pred

        # ---------------- 수명 비교 ----------------
        st.subheader("📊 수명 비교")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("기존 수명 (log)", round(original_pred, 3))
            st.metric("기존 수명 (hours)", f"{original_hours:,.0f}")

        with col2:
            st.metric("변경 수명 (log)", round(new_pred, 3))
            st.metric("변경 수명 (hours)", f"{new_pred_hours:,.0f}")

        # 🔥 변화량 표시
        diff = new_pred - original_pred
        percent = (diff / original_pred) * 100

        st.write(f"📈 변화량: {diff:.4f}")
        st.write(f"📊 변화율: {percent:.2f}%")

        # ---------------- 성분 비교 ----------------
        if selected_features:
            st.subheader("🧾 성분 변화 비교")

            compare_df = pd.DataFrame({
                "Original": selected_row[selected_features],
                "Modified": modified_row[selected_features]
            })

            st.dataframe(compare_df)

        # ---------------- 🔥 간지 그래프 ----------------
        st.subheader("📈 수명 변화 그래프")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))

        x = [0, 1]
        y = [original_pred, new_pred]

        # 🔥 선 + 점
        ax.plot(x, y, marker='o', linewidth=3)

        # 🔥 점 강조
        ax.scatter(x, y, s=100)

        # 🔥 값 표시
        for i, v in enumerate(y):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Original", "Modified"])

        # 🔥 핵심: 확대
        min_val = min(y)
        max_val = max(y)
        ax.set_ylim(min_val - 0.05, max_val + 0.05)

        ax.set_ylabel("Predicted Life (log)")
        ax.set_title("Life Change (Before → After)")

        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)
# ---------------- CHATBOT ----------------
elif page == "Chatbot":

    st.header("🤖 AI Assistant")

    user_input = st.text_input("질문 입력")

    if st.button("질문하기"):

        if user_input:
            user_input = user_input.lower()

            if "크리프" in user_input:
                answer = "크리프는 고온에서 재료가 변형되는 현상입니다."
            elif "합금" in user_input:
                answer = "합금 성분은 수명에 큰 영향을 줍니다."
            elif "온도" in user_input:
                answer = "온도가 높을수록 수명은 감소합니다."
            elif "응력" in user_input:
                answer = "응력이 증가하면 수명이 감소합니다."
            else:
                answer = "질문을 이해하지 못했습니다."

            st.write(f"🤖 {answer}")
