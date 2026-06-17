"""
소재부품 크립 수명 예측 시스템
================================
Committee ANN  → 정확한 수명 예측
NGBoost        → 불확실성 정량화
AI Assistant   → 키워드 기반 Q&A

실행 방법:
    py -3.11 -m streamlit run app.py

필요 파일 (같은 폴더):
    model_0~9.keras, scaler_X.pkl, scaler_y.pkl  ← train.py 결과
    ngboost_model.pkl, scaler_X_ngb.pkl          ← train_ngboost.py 결과
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tf_keras as tfk

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="크립 수명 예측 시스템",
    page_icon="⚙️",
    layout="wide"
)

# ── 세션 상태 ─────────────────────────────────────────────────────────────────
for key, val in {
    "page":         "Home",
    "df":           None,
    "result_df":    None,
    "y_true":       None,
    "y_pred":       None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ann():
    models   = [tfk.models.load_model(f"model_{i}.keras") for i in range(10)]
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return models, scaler_X, scaler_y

@st.cache_resource
def load_ngb():
    ngb      = joblib.load("ngboost_model.pkl")
    scaler_X = joblib.load("scaler_X_ngb.pkl")
    return ngb, scaler_X

ann_models, ann_scaler_X, ann_scaler_y = load_ann()
ngb_model,  ngb_scaler_X               = load_ngb()

# ── 변수 정의 ─────────────────────────────────────────────────────────────────
ALLOY_COLS = ['Cr','Ni','Mo','Mn','Si','Nb','Ti','V','W','Cu','N','C','B','P','S','Co','Al','Sn']
COND_COLS  = ['S_temp / oC', 'log10(S_time / s)', 'WQ', 'temp / oC', 'stress / Mpa']
ALL_COLS   = ALLOY_COLS + COND_COLS + ['Stab_ratio']

# 컬럼 순서 + 이름 매핑(설정된 순서와 다르면 매핑)
def align_input_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    aligned = pd.DataFrame()
    missing_cols = []

    for target in ALL_COLS:
        found = None

        for c in df.columns:
            if c == target:
                found = c
                break

        if not found:
            for c in df.columns:
                if c.lower() == target.lower():
                    found = c
                    break

        if not found:
            for c in df.columns:
                if target.lower() in c.lower() or c.lower() in target.lower():
                    found = c
                    break

        if found:
            aligned[target] = df[found]
        else:
            aligned[target] = 0
            missing_cols.append(target)

    return aligned, missing_cols

META = {
    'Cr':                (10.0, 26.0,  18.0,  0.1,   "Cr (wt%)"),
    'Ni':                (4.0,  35.0,  12.0,  0.1,   "Ni (wt%)"),
    'Mo':                (0.0,  3.1,   0.1,   0.01,  "Mo (wt%)"),
    'Mn':                (0.0,  15.0,  1.5,   0.1,   "Mn (wt%)"),
    'Si':                (0.0,  1.3,   0.5,   0.01,  "Si (wt%)"),
    'Nb':                (0.0,  5.0,   0.0,   0.01,  "Nb (wt%)"),
    'Ti':                (0.0,  0.56,  0.03,  0.01,  "Ti (wt%)"),
    'V':                 (0.0,  0.5,   0.0,   0.01,  "V (wt%)"),
    'W':                 (0.0,  1.03,  0.0,   0.01,  "W (wt%)"),
    'Cu':                (0.0,  3.1,   0.1,   0.01,  "Cu (wt%)"),
    'N':                 (0.0,  0.3,   0.05,  0.001, "N (wt%)"),
    'C':                 (0.0,  0.15,  0.06,  0.001, "C (wt%)"),
    'B':                 (0.0,  0.01,  0.0,   0.0001,"B (wt%)"),
    'P':                 (0.0,  0.05,  0.03,  0.001, "P (wt%)"),
    'S':                 (0.0,  0.03,  0.008, 0.001, "S (wt%)"),
    'Co':                (0.0,  0.54,  0.0,   0.01,  "Co (wt%)"),
    'Al':                (0.0,  3.93,  0.005, 0.001, "Al (wt%)"),
    'Sn':                (0.0,  0.02,  0.0,   0.001, "Sn (wt%)"),
    'S_temp / oC':       (1000, 1350,  1150,  10,    "고용화 온도 (°C)"),
    'log10(S_time / s)': (2.08, 4.26,  2.78,  0.01,  "고용화 시간 log10(s)"),
    'WQ':                (0,    1,     1,     1,     "수냉 여부"),
    'temp / oC':         (500,  1100,  650,   10,    "크립 시험 온도 (°C)"),
    'stress / Mpa':      (5.0,  471.0, 130.0, 1.0,   "응력 (MPa)"),
}

def stab_ratio(nb, ti, c, n):
    return (nb / 8 + ti / 4) / (c + n + 1e-7)

# ── 예측 함수 ─────────────────────────────────────────────────────────────────
def predict_ann(X_scaled):
    preds = []
    for m in ann_models:
        p = m.predict(X_scaled, verbose=0)
        preds.append(ann_scaler_y.inverse_transform(p).flatten())
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)

def predict_ngb(X_scaled):
    dist = ngb_model.pred_dist(X_scaled.astype(np.float64))
    return dist.loc, dist.scale

# ── AI Assistant 응답 ─────────────────────────────────────────────────────────
def get_answer(text: str) -> str:
    t = text.lower()


    if any(k in t for k in ["이게 뭐야", "뭐하는거야", "이 시스템", "무슨 프로그램"]):
        return "이 시스템은 합금 성분과 온도, 응력 조건을 입력하면 크립 수명을 예측해주는 프로그램입니다."
    elif any(k in t for k in ["어떻게 써", "사용법", "사용 방법"]):
        return "엑셀 데이터를 업로드하면 자동으로 정리된 후 모델이 수명을 예측합니다. Result에서 결과를 확인하고 Simulation으로 성분 변화도 테스트할 수 있습니다."
    elif any(k in t for k in ["엑셀", "파일", "순서", "달라도", "업로드", "데이터 넣기"]):
        return "엑셀 파일을 업로드하면 컬럼 순서나 일부 값이 달라도 자동으로 정리되어 모델 입력 형태로 변환됩니다."
    elif any(k in t for k in ["컬럼", "순서", "값","없으면","없는 값", "빈 값", "빈 데이터", "비어있는","데이터 정리", "자동 정리"]):
        return "입력 데이터의 컬럼 순서가 달라도 자동으로 정렬되며, 필요한 컬럼이 없으면 기본값으로 보정됩니다."
    elif any(k in t for k in ["왜 필요", "왜 쓰는", "필요한 이유"]):
        return "크립 수명은 실험으로 측정하려면 시간이 오래 걸립니다. 이 시스템은 AI로 빠르게 예측하여 시간과 비용을 줄이기 위해 사용됩니다."
    elif any(k in t for k in ["정확해", "믿을 수", "신뢰", "성능 어때", "잘 맞아"]):
        return "ANN 모델은 R² 약 0.92 수준의 정확도를 보이며, NGBoost를 통해 예측의 신뢰구간도 함께 제공합니다."
    if any(k in t for k in ["크리프", "크립", "creep"]):
        return "크립(Creep)은 고온·고응력 환경에서 재료가 시간에 따라 서서히 변형되는 현상입니다. 오스테나이트계 내열강에서는 온도와 응력이 수명에 가장 큰 영향을 미칩니다."
    elif any(k in t for k in ["합금", "성분", "alloy"]):
        return "합금 성분 중 Cr은 내산화성, Ni은 고온 안정성, Nb·Ti는 석출 강화에 기여합니다. 본 모델에서는 Stab_ratio = (Nb/8 + Ti/4) / (C + N) 파생변수도 활용합니다."
    elif any(k in t for k in ["온도", "temperature", "temp"]):
        return "크립 시험 온도가 높을수록 수명은 급격히 감소합니다. 변수 중요도 분석 결과, 온도(temp / °C)는 응력 다음으로 수명에 큰 영향을 미치는 변수입니다."
    elif any(k in t for k in ["응력", "stress", "mpa"]):
        return "응력(stress / MPa)은 수명 예측에 가장 중요한 변수입니다. 변수 중요도 분석 결과 약 28%를 차지하며, 응력이 증가할수록 크립 수명은 감소합니다."
    elif any(k in t for k in ["불확실성", "uncertainty", "sigma", "신뢰구간", "ci"]):
        return "NGBoost는 예측값(μ)과 불확실성(σ)을 동시에 출력합니다. σ가 클수록 예측이 어려운 조건입니다. 95% 신뢰구간은 μ ± 1.96σ로 계산됩니다."
    elif any(k in t for k in ["ann", "committee", "앙상블", "ensemble"]):
        return "Committee ANN은 동일 구조의 ANN 10개를 독립적으로 학습한 앙상블 모델입니다. 10개 예측의 평균이 최종 예측값, 표준편차가 모델 간 불확실성이 됩니다. R²=0.9236을 달성했습니다."
    elif any(k in t for k in ["ngboost", "부스팅", "boosting"]):
        return "NGBoost는 Natural Gradient Boosting 모델로, 출력이 확률 분포(Normal)입니다. 예측값과 불확실성을 이론적으로 동시에 추정할 수 있으며 테이블 데이터에 강합니다. R²=0.9155, 95% Coverage=88.1%를 달성했습니다."
    elif any(k in t for k in ["rmse", "r2", "r²", "성능", "정확도"]):
        return "모델 성능: Committee ANN — RMSE=0.2570, R²=0.9236 / NGBoost — RMSE=0.2694, R²=0.9155, 95% CI Coverage=88.1%. 예측 정확도는 ANN이, 불확실성 정량화는 NGBoost가 우수합니다."
    elif any(k in t for k in ["stab", "안정화", "stabilisation"]):
        return "Stab_ratio = (Nb/8 + Ti/4) / (C + N + ε) 로 계산되는 파생변수입니다. 합금 안정화 원소(Nb, Ti)와 탄질화물 형성 원소(C, N)의 비율을 나타내며 수명 예측에 유효한 변수임이 확인됐습니다."
    elif any(k in t for k in ["안녕", "hello", "hi", "반가워"]):
        return "안녕하세요! 소재부품 크립 수명 예측 시스템입니다. 크립, 합금 성분, 온도, 응력, 불확실성, 모델 성능 등에 대해 질문해보세요."
    else:
        return "죄송합니다, 질문을 이해하지 못했습니다. '크립', '합금', '온도', '응력', '불확실성', 'NGBoost', 'ANN', 'RMSE' 등의 키워드로 질문해보세요."

# ── 사이드바 ──────────────────────────────────────────────────────────────────
st.sidebar.title("📌 메뉴")
menu = {
    "🏠 Home":           "Home",
    "📂 Data Upload":    "Data Upload",
    "⚙️ Prediction":     "Prediction",
    "📊 Result":         "Result",
    "🔬 단일 샘플 예측":  "Single",
    "🤖 AI Assistant":   "Chatbot",
}
for label, key in menu.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.page = key

st.sidebar.divider()
st.sidebar.caption("Committee ANN — 정확한 예측\nNGBoost — 불확실성 정량화")

page = st.session_state.page

# ════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("⚙️ 소재부품 크립 수명 예측 시스템")

    st.markdown("""
    ### 🔍 시스템 소개
    본 시스템은 합금 성분과 온도, 응력 조건을 기반으로  
    **크립 수명(Creep Life)**을 예측하는 인공지능 모델입니다.

    ✔ **ANN Ensemble** → 높은 정확도의 수명 예측  
    ✔ **NGBoost** → 예측 불확실성(신뢰도) 제공  

    """)

    st.divider()

    st.markdown("### 📘 기초 개념")

    st.info("""
    **크립(Creep)**은 고온·고응력 환경에서 재료가 시간에 따라  
    서서히 변형되는 현상입니다.

    👉 온도(temp) ↑ → 수명 ↓  
    👉 응력(stress) ↑ → 수명 ↓  

    따라서 합금 성분 + 온도 + 응력이 수명에 큰 영향을 줍니다.
    """)

    st.divider()


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 Committee ANN")
        st.markdown("- **역할**: 정확한 수명 점 추정")
        st.markdown("- **구조**: ANN 10개 앙상블")
        m1, m2 = st.columns(2)
        m1.metric("RMSE", "0.2570")
        m2.metric("R²",   "0.9236")
    with col2:
        st.subheader("📊 NGBoost")
        st.markdown("- **역할**: 이론적 불확실성 정량화")
        st.markdown("- **구조**: Natural Gradient Boosting")
        m3, m4 = st.columns(2)
        m3.metric("RMSE", "0.2694")
        m4.metric("95% Coverage", "88.1%")

    st.divider()
    st.subheader("사용 방법")
    st.markdown("""
    1. **📂 Data Upload** — 엑셀 파일 업로드
    2. **⚙️ Prediction** — 배치 예측 실행
    3. **📊 Result** — 결과 확인 및 합금 성분 시뮬레이션
    4. **🔬 단일 샘플 예측** — 합금 성분 직접 입력 후 예측
    5. **🤖 AI Assistant** — 크립/모델 관련 질문
    """)

    st.divider()
    st.markdown("### ⚠️ 입력 데이터 주의사항")
    st.warning("""
    - 엑셀 파일에는 합금 성분 및 온도/응력 정보가 포함되어야 합니다.  
    - 컬럼 이름이 일부 다르거나 순서가 달라도 자동으로 보정됩니다.  
    - 없는 컬럼은 0으로 자동 처리됩니다.  
    """)

    st.divider()

# ════════════════════════════════════════════════════════════════════
# DATA UPLOAD
# ════════════════════════════════════════════════════════════════════
elif page == "Data Upload":
    st.header("📂 데이터 업로드")

    st.markdown("""
    ### 📌 업로드 안내
    - 엑셀(.xlsx) 파일 업로드
    - 컬럼 순서 상관 없음
    - 일부 컬럼 없어도 자동 보정됨
    """)

    uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)

            if df.empty:
                st.error("❌ 파일이 비어있습니다")
            else:
                st.session_state.df = df

                st.success("✅ 업로드 완료")

                st.subheader("📊 원본 데이터")
                st.dataframe(df.head())

                st.caption(f"총 {len(df)}행 × {df.shape[1]}열")

                # 🔥 여기 핵심 (자동 정렬)
                df_fixed, missing_cols = align_input_dataframe(df)

                st.divider()

                st.subheader("🛠 자동 정리된 데이터 (모델 입력용)")
                st.dataframe(df_fixed.head())

                # 🔥 사용자 안내
                if missing_cols:
                    st.warning(f"자동 생성된 컬럼: {missing_cols}")
                else:
                    st.info("모든 컬럼 정상 인식됨")

                st.caption("👉 위 형태로 자동 변환되어 모델에 입력됩니다")

        except Exception as e:
            st.error("❌ 파일 처리 중 오류 발생")
            st.error(str(e))
# ════════════════════════════════════════════════════════════════════
# PREDICTION (배치)
# ════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.header("⚙️ 예측 실행")
    df = st.session_state.df

    if df is None:
        st.warning("먼저 📂 Data Upload에서 데이터를 업로드하세요")
    else:
        st.markdown("""
        ### 📌 예측 안내
        - 업로드된 데이터를 기반으로 수명 예측 수행
        - 컬럼 순서 및 일부 누락 자동 보정
        """)

        st.dataframe(df.head())

        if st.button("🚀 예측 시작", type="primary"):
            try:
                with st.spinner("예측 중..."):

                    epsilon = 1e-7

                    X_fixed, missing_cols = align_input_dataframe(df)

                    X_fixed['Stab_ratio'] = (
                        (X_fixed['Nb'] / 8 + X_fixed['Ti'] / 4) /
                        (X_fixed['C'] + X_fixed['N'] + epsilon)
                    )

                    if 'log10(t_r / h)' in df.columns:
                        y_true = df['log10(t_r / h)'].values
                    else:
                        y_true = None

                    if missing_cols:
                        st.warning(f"자동 생성된 컬럼: {missing_cols}")

                    X_ann_s = ann_scaler_X.transform(X_fixed)
                    ann_mean, ann_std = predict_ann(X_ann_s)

                    X_ngb_s = ngb_scaler_X.transform(X_fixed)
                    ngb_mu, ngb_sig = predict_ngb(X_ngb_s)

                    result_df = df.copy()
                    result_df["Predicted Life (log10 h)"] = ann_mean
                    result_df["ANN Uncertainty"] = ann_std
                    result_df["Predicted Life (hours)"] = 10 ** ann_mean
                    result_df["NGBoost μ"] = ngb_mu
                    result_df["NGBoost σ"] = ngb_sig
                    result_df["95% CI Lower (h)"] = 10 ** (ngb_mu - 1.96 * ngb_sig)
                    result_df["95% CI Upper (h)"] = 10 ** (ngb_mu + 1.96 * ngb_sig)

                    st.session_state.result_df = result_df
                    st.session_state.y_true = y_true
                    st.session_state.y_pred = ann_mean

                st.success("예측 완료! 📊 Result 페이지에서 결과를 확인하세요")

            except Exception as e:
                st.error("❌ 예측 중 오류 발생")
                st.error(str(e))

# ════════════════════════════════════════════════════════════════════
# RESULT
# ════════════════════════════════════════════════════════════════════
elif page == "Result":
    st.header("📊 결과")

    result_df = st.session_state.result_df
    y_true    = st.session_state.y_true
    y_pred    = st.session_state.y_pred

    if result_df is None:
        st.info("아직 예측 결과가 없습니다. ⚙️ Prediction을 먼저 실행하세요.")
    else:
        # ── 성능 지표 ─────────────────────────────────────────────
        if y_true is not None:
            from sklearn.metrics import mean_squared_error, r2_score
            ngb_mu  = result_df["NGBoost μ"].values
            ngb_sig = result_df["NGBoost σ"].values

            ann_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            ann_r2   = r2_score(y_true, y_pred)
            ngb_rmse = np.sqrt(mean_squared_error(y_true, ngb_mu))
            ngb_r2   = r2_score(y_true, ngb_mu)
            cov_95   = np.mean(np.abs(y_true - ngb_mu) <= 1.96 * ngb_sig) * 100

            st.subheader("성능 비교")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ANN RMSE",        f"{ann_rmse:.4f}")
            c2.metric("ANN R²",          f"{ann_r2:.4f}")
            c3.metric("NGBoost RMSE",    f"{ngb_rmse:.4f}")
            c4.metric("NGBoost R²",      f"{ngb_r2:.4f}")
            c5.metric("NGBoost 95% Cov", f"{cov_95:.1f}%")
            st.divider()

        # ── 결과 테이블 ───────────────────────────────────────────
        st.subheader("📋 개별 예측 결과")
        show_cols = ["Predicted Life (log10 h)", "Predicted Life (hours)",
                     "ANN Uncertainty", "NGBoost μ", "NGBoost σ",
                     "95% CI Lower (h)", "95% CI Upper (h)"]
        st.dataframe(result_df[show_cols].round(4))

        csv = result_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 결과 CSV 다운로드", csv, "results.csv", "text/csv")

        st.divider()

        # ── 자동 해석 ─────────────────────────────────────────────
        st.subheader("🤖 결과 해석")
        avg_life = result_df["Predicted Life (log10 h)"].mean()
        if avg_life > 4:
            st.success("👉 전반적으로 수명이 매우 긴 고내열 합금 조건입니다. (10,000시간 이상)")
        elif avg_life > 3:
            st.info("👉 일반적인 고온 환경에서 사용 가능한 수준입니다. (1,000~10,000시간)")
        else:
            st.warning("👉 수명이 짧아 조건이 가혹한 상태입니다. (1,000시간 미만)")

        st.divider()

        # ── 시뮬레이션 ────────────────────────────────────────────
        st.subheader("🧪 합금 성분 시뮬레이션")

        selected_idx = st.selectbox("샘플 선택", result_df.index)
        selected_row = result_df.loc[selected_idx].copy()

        original_pred  = selected_row["Predicted Life (log10 h)"]
        original_hours = 10 ** original_pred

        exclude_cols = ["Predicted Life (log10 h)", "ANN Uncertainty",
                        "Predicted Life (hours)", "NGBoost μ", "NGBoost σ",
                        "95% CI Lower (h)", "95% CI Upper (h)", "log10(t_r / h)"]
        available_cols = [c for c in selected_row.index if c not in exclude_cols]

        selected_features = st.multiselect(
            "조절할 성분 선택", available_cols, default=available_cols[:3]
        )

        modified_row = selected_row.copy()
        for feature in selected_features:
            val     = float(selected_row[feature])
            lo      = float(val * 0.5) if val > 0 else float(val * 1.5) if val < 0 else -1.0
            hi      = float(val * 1.5) if val > 0 else float(val * 0.5) if val < 0 else 1.0
            new_val = st.slider(feature, lo, hi, val)
            modified_row[feature] = new_val

        feature_cols = [c for c in modified_row.index if c not in exclude_cols]
        X_new = pd.DataFrame([modified_row[feature_cols]])
        X_new['Stab_ratio'] = (
            (X_new['Nb'] / 8 + X_new['Ti'] / 4) /
            (X_new['C'] + X_new['N'] + 1e-7)
        )
        X_new_fixed, _ = align_input_dataframe(X_new)
        X_new_ann = ann_scaler_X.transform(X_new_fixed)

        new_pred = float(np.mean([
            ann_scaler_y.inverse_transform(m.predict(X_new_ann, verbose=0)).flatten()[0]
            for m in ann_models
        ]))

        new_hours = 10 ** new_pred
        X_new_ngb = ngb_scaler_X.transform(X_new_fixed).astype(np.float64)
        new_dist = ngb_model.pred_dist(X_new_ngb)
        new_ngb_mu = float(new_dist.loc[0])
        new_ngb_sig = float(new_dist.scale[0])

                
        # 수명 비교
        st.subheader("📊 수명 비교")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**기존**")
            st.metric("수명 (log10 h)", round(original_pred, 3))
            st.metric("수명 (hours)",   f"{original_hours:,.0f}")
        with col2:
            diff = new_pred - original_pred
            st.markdown("**변경 후**")
            st.metric("수명 (log10 h)", round(new_pred, 3), delta=f"{diff:+.3f}")
            st.metric("수명 (hours)",   f"{new_hours:,.0f}")

        st.caption(
            f"NGBoost 변경 후: μ={new_ngb_mu:.4f}, σ={new_ngb_sig:.4f}  |  "
            f"95% CI: {10**(new_ngb_mu-1.96*new_ngb_sig):,.0f} ~ {10**(new_ngb_mu+1.96*new_ngb_sig):,.0f} h"
        )

        if selected_features:
            st.subheader("🧾 성분 변화 비교")
            st.dataframe(pd.DataFrame({
                "Original": selected_row[selected_features],
                "Modified": modified_row[selected_features]
            }))

        # 수명 변화 그래프
        st.subheader("📈 수명 변화 그래프")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [0, 1]
        y = [original_pred, new_pred]
        color = 'steelblue' if new_pred >= original_pred else 'tomato'
        ax.plot(x, y, marker='o', linewidth=3, color=color)
        ax.scatter(x, y, s=120, color=color, zorder=5)
        for i, v in enumerate(y):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Original", "Modified"])
        ax.set_ylim(min(y) - 0.1, max(y) + 0.1)
        ax.set_ylabel("Predicted Life (log10 h)")
        ax.set_title("Life Change (Before → After)")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════
# 단일 샘플 예측
# ════════════════════════════════════════════════════════════════════
elif page == "Single":
    st.header("🔬 단일 샘플 예측")
    st.caption("합금 성분과 시험 조건을 입력하면 두 모델이 동시에 예측합니다")

    with st.form("single_form"):
        st.subheader("합금 성분 (wt%)")
        cols = st.columns(6)
        inp  = {}
        for i, col in enumerate(ALLOY_COLS):
            mn, mx, dv, step, label = META[col]
            with cols[i % 6]:
                inp[col] = st.number_input(
                    label, min_value=float(mn), max_value=float(mx),
                    value=float(dv), step=float(step), format="%.4f"
                )

        st.divider()
        st.subheader("열처리 조건")
        h1, h2, h3 = st.columns(3)
        with h1:
            mn, mx, dv, step, label = META['S_temp / oC']
            inp['S_temp / oC'] = st.number_input(label, float(mn), float(mx), float(dv), float(step))
        with h2:
            mn, mx, dv, step, label = META['log10(S_time / s)']
            inp['log10(S_time / s)'] = st.number_input(label, float(mn), float(mx), float(dv), float(step), format="%.4f")
        with h3:
            inp['WQ'] = st.selectbox("수냉 여부", [1, 0], format_func=lambda x: "수냉 (Yes)" if x else "공냉 (No)")

        st.divider()
        st.subheader("크립 시험 조건")
        t1, t2 = st.columns(2)
        with t1:
            mn, mx, dv, step, label = META['temp / oC']
            inp['temp / oC'] = st.slider(label, int(mn), int(mx), int(dv), int(step))
        with t2:
            mn, mx, dv, step, label = META['stress / Mpa']
            inp['stress / Mpa'] = st.slider(label, float(mn), float(mx), float(dv), float(step))

        submitted = st.form_submit_button("🚀 예측 실행", use_container_width=True, type="primary")

    if submitted:
        try:
            inp['Stab_ratio'] = stab_ratio(inp['Nb'], inp['Ti'], inp['C'], inp['N'])
            X = pd.DataFrame([{c: inp[c] for c in ALL_COLS}])

            X_ann_s            = ann_scaler_X.transform(X)
            ann_mu, ann_std_v  = predict_ann(X_ann_s)
            ann_mu_val         = float(ann_mu[0])
            ann_hours          = 10 ** ann_mu_val

            X_ngb_s            = ngb_scaler_X.transform(X)
            ngb_mu, ngb_sig    = predict_ngb(X_ngb_s)
            ngb_mu_val         = float(ngb_mu[0])
            ngb_sig_val        = float(ngb_sig[0])
            lower_h            = 10 ** (ngb_mu_val - 1.96 * ngb_sig_val)
            upper_h            = 10 ** (ngb_mu_val + 1.96 * ngb_sig_val)

            st.divider()
            left, right = st.columns(2)

            with left:
                st.subheader("🧠 Committee ANN")
                a1, a2 = st.columns(2)
                a1.metric("예측 수명 (log₁₀ h)", f"{ann_mu_val:.4f}")
                a2.metric("예측 수명 (시간)",     f"{ann_hours:,.0f} h")
                st.caption(f"모델 간 표준편차: {float(ann_std_v[0]):.4f}")
                if ann_mu_val > 4:
                    st.success("✅ 수명 매우 김 (10,000h 이상)")
                elif ann_mu_val > 3:
                    st.info("ℹ️ 일반 고온 환경 수준 (1,000~10,000h)")
                else:
                    st.warning("⚠️ 수명 짧음 (1,000h 미만)")

            with right:
                st.subheader("📊 NGBoost 불확실성")
                b1, b2 = st.columns(2)
                b1.metric("불확실성 σ", f"{ngb_sig_val:.4f}")
                b2.metric("95% CI",     f"{lower_h:,.0f} ~ {upper_h:,.0f} h")
                if ngb_sig_val < 0.15:
                    st.success("✅ 불확실성 낮음 — 신뢰도 높은 예측")
                elif ngb_sig_val < 0.30:
                    st.info("ℹ️ 불확실성 보통")
                else:
                    st.warning("⚠️ 불확실성 높음 — 예측이 어려운 조건")

            st.divider()
            st.subheader("95% 신뢰구간")
            fig, ax = plt.subplots(figsize=(9, 2.2))
            ax.barh(["NGBoost 95% CI"], [upper_h - lower_h], left=lower_h,
                    color='steelblue', alpha=0.35, height=0.35)
            ax.axvline(ann_hours, color='tomato', lw=2.0,
                       label=f"ANN: {ann_hours:,.0f}h")
            ax.axvline(10**ngb_mu_val, color='steelblue', lw=2.0, linestyle='--',
                       label=f"NGBoost: {10**ngb_mu_val:,.0f}h")
            ax.set_xlabel("수명 (hours)")
            ax.set_xscale('log')
            ax.legend(fontsize=9)
            ax.grid(True, axis='x', linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)

            st.caption(
                f"Stab_ratio: {inp['Stab_ratio']:.4f}  |  "
                f"temp: {inp['temp / oC']}°C  |  stress: {inp['stress / Mpa']} MPa"
            )

        except Exception as e:
            st.error(f"예측 오류: {e}")

# ════════════════════════════════════════════════════════════════════
# AI ASSISTANT
# ════════════════════════════════════════════════════════════════════
elif page == "Chatbot":
    st.header("🤖 AI Assistant")
    st.caption("크립, 합금, 온도, 응력, 불확실성, 모델 성능 등에 대해 질문해보세요")

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    user_input = st.chat_input("질문을 입력하세요...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        answer = get_answer(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.rerun()
