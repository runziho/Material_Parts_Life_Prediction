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
import plotly.express as px
import plotly.graph_objects as go

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

def predict_ann_batch_df(df_rows):
    df_rows = df_rows.copy()

    if "Stab_ratio" not in df_rows.columns:
        df_rows["Stab_ratio"] = (
            (df_rows["Nb"] / 8 + df_rows["Ti"] / 4) /
            (df_rows["C"] + df_rows["N"] + 1e-7)
        )

    X_fixed, _ = align_input_dataframe(df_rows)
    X_ann_s = ann_scaler_X.transform(X_fixed)

    preds = []
    for m in ann_models:
        p = m.predict(X_ann_s, verbose=0)
        preds.append(ann_scaler_y.inverse_transform(p).flatten())

    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def build_curve_df(base_row, variable_name, variable_values, stress_min, stress_max, n_points=10, fixed_temp=None):
    stress_values = np.linspace(stress_min, stress_max, n_points)
    rows = []
    meta = []

    for val in variable_values:
        for stress in stress_values:
            row = base_row.copy()
            if fixed_temp is not None:
                row["temp / oC"] = fixed_temp
            row[variable_name] = val
            row["stress / Mpa"] = float(stress)

            rows.append(row)
            meta.append({
                "value": val,
                "stress": float(stress)
            })

    batch_df = pd.DataFrame(rows)
    pred_log, pred_std = predict_ann_batch_df(batch_df)

    out = pd.DataFrame(meta)
    out["life_log"] = pred_log
    out["life_hours"] = 10 ** pred_log
    out["model_std"] = pred_std
    return out


def build_point_compare_df(base_row, variable_name, variable_values, fixed_temp, fixed_stress):
    rows = []
    labels = []

    for val in variable_values:
        row = base_row.copy()
        row["temp / oC"] = fixed_temp
        row["stress / Mpa"] = float(fixed_stress)
        row[variable_name] = val

        rows.append(row)
        labels.append(val)

    batch_df = pd.DataFrame(rows)
    pred_log, pred_std = predict_ann_batch_df(batch_df)

    out = pd.DataFrame({
        "value": labels,
        "life_log": pred_log,
        "life_hours": 10 ** pred_log,
        "model_std": pred_std
    })
    return out

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
    "📉 수명 시뮬레이션": "Simulation",
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
    5. **📉 수명 시뮬레이션** — 온도별 응력-수명 곡선 + 시편 파단 애니메이션
    6. **🤖 AI Assistant** — 크립/모델 관련 질문
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
        sim_tab1, sim_tab2 = st.tabs(["기본 시뮬레이션", "고급 시뮬레이션"])
        with sim_tab1:
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
            fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
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
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.pyplot(fig, use_container_width=True)
        # ============================================================
        # 고급 시뮬레이션
        # ============================================================
        with sim_tab2:
            st.subheader("📉 고급 시뮬레이션")
            st.caption("현재 모델은 파손 시간을 예측하는 구조이므로, 실제 시편 변형 애니메이션 대신 조건별 곡선과 파손 예상 시간 비교로 표현함.")

            variable_options = {
                "Cr": "Cr",
                "Ni": "Ni",
                "Nb": "Nb",
                "Ti": "Ti",
                "Al": "Al"
            }

            value_options = {
                "Cr": [10, 12, 14, 16, 18, 20, 22, 24, 26],
                "Ni": [4, 8, 12, 16, 20, 24, 28, 32],
                "Nb": [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0],
                "Ti": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                "Al": [0.0, 0.5, 1.0, 2.0, 3.0]
            }

            with st.form("advanced_sim_form"):
                selected_idx_adv = st.selectbox(
                    "기준 샘플 선택",
                    result_df.index,
                    key="adv_selected_idx"
                )
                base_row_adv = result_df.loc[selected_idx_adv].copy()

                adv_mode = st.radio(
                    "분석 방식",
                    ["온도 영향", "성분 영향", "파손 예상 시간 비교"],
                    horizontal=True,
                    key="adv_mode"
                )

                if adv_mode == "온도 영향":
                    temp_values = st.multiselect(
                        "비교할 온도 선택 (°C)",
                        options=[550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
                        default=[600, 700, 800],
                        key="adv_temp_values"
                    )

                    stress_min, stress_max = st.slider(
                        "응력 범위 (MPa)",
                        min_value=5,
                        max_value=471,
                        value=(50, 300),
                        step=5,
                        key="adv_temp_stress_range"
                    )

                    n_points = st.slider(
                        "곡선 포인트 수",
                        6, 20, 10, 2,
                        key="adv_temp_points"
                    )

                elif adv_mode == "성분 영향":
                    selected_variable_label = st.selectbox(
                        "비교할 성분 선택",
                        list(variable_options.keys()),
                        key="adv_variable_curve"
                    )
                    selected_variable = variable_options[selected_variable_label]

                    fixed_temp = st.slider(
                        "고정 온도 (°C)",
                        500, 1100,
                        int(float(base_row_adv["temp / oC"])),
                        10,
                        key="adv_fixed_temp_curve"
                    )

                    compare_values = st.multiselect(
                        "비교할 값 선택",
                        options=value_options[selected_variable],
                        default=value_options[selected_variable][:3],
                        key="adv_compare_values_curve"
                    )

                    stress_min, stress_max = st.slider(
                        "응력 범위 (MPa)",
                        min_value=5,
                        max_value=471,
                        value=(50, 300),
                        step=5,
                        key="adv_curve_stress_range"
                    )

                    n_points = st.slider(
                        "곡선 포인트 수",
                        6, 20, 10, 2,
                        key="adv_curve_points"
                    )

                else:
                    selected_variable_label = st.selectbox(
                        "비교할 성분 선택",
                        list(variable_options.keys()),
                        key="adv_variable_point"
                    )
                    selected_variable = variable_options[selected_variable_label]

                    fixed_temp = st.slider(
                        "운전 온도 (°C)",
                        500, 1100,
                        int(float(base_row_adv["temp / oC"])),
                        10,
                        key="adv_fixed_temp_point"
                    )

                    fixed_stress = st.slider(
                        "운전 응력 (MPa)",
                        5, 471,
                        int(float(base_row_adv["stress / Mpa"])),
                        5,
                        key="adv_fixed_stress_point"
                    )

                    compare_values = st.multiselect(
                        "비교할 값 선택",
                        options=value_options[selected_variable],
                        default=value_options[selected_variable][:3],
                        key="adv_compare_values_point"
                    )

                run_advanced = st.form_submit_button(
                    "고급 시뮬레이션 실행",
                    use_container_width=True,
                    type="primary"
                )

            if run_advanced:
                if adv_mode == "온도 영향":
                    if not temp_values:
                        st.warning("온도를 하나 이상 선택해야 함")
                    else:
                        curve_df = build_curve_df(
                            base_row=base_row_adv.to_dict(),
                            variable_name="temp / oC",
                            variable_values=temp_values,
                            stress_min=stress_min,
                            stress_max=stress_max,
                            n_points=n_points
                        )

                        fig_temp = px.line(
                            curve_df,
                            x="life_hours",
                            y="stress",
                            color="value",
                            markers=True,
                            labels={
                                "life_hours": "Life (hours)",
                                "stress": "Stress (MPa)",
                                "value": "Temperature (°C)"
                            },
                            title="온도별 Stress–Life Curve"
                        )
                        fig_temp.update_xaxes(type="log")
                        st.plotly_chart(fig_temp, use_container_width=True)

                elif adv_mode == "성분 영향":
                    if not compare_values:
                        st.warning("비교할 값을 하나 이상 선택해야 함")
                    else:
                        curve_df = build_curve_df(
                            base_row=base_row_adv.to_dict(),
                            variable_name=selected_variable,
                            variable_values=compare_values,
                            stress_min=stress_min,
                            stress_max=stress_max,
                            n_points=n_points,
                            fixed_temp=fixed_temp
                        )

                        fig_curve = px.line(
                            curve_df,
                            x="life_hours",
                            y="stress",
                            color="value",
                            markers=True,
                            labels={
                                "life_hours": "Life (hours)",
                                "stress": "Stress (MPa)",
                                "value": selected_variable_label
                            },
                            title=f"{selected_variable_label} 변화에 따른 Stress–Life Curve"
                        )
                        fig_curve.update_xaxes(type="log")
                        st.plotly_chart(fig_curve, use_container_width=True)

                else:
                    if not compare_values:
                        st.warning("비교할 값을 하나 이상 선택해야 함")
                    else:
                        point_df = build_point_compare_df(
                            base_row=base_row_adv.to_dict(),
                            variable_name=selected_variable,
                            variable_values=compare_values,
                            fixed_temp=fixed_temp,
                            fixed_stress=fixed_stress
                        )

                        point_df["label"] = point_df["value"].apply(lambda x: f"{selected_variable_label}={x}")

                        fig_point = px.bar(
                            point_df,
                            x="label",
                            y="life_hours",
                            text="life_hours",
                            labels={
                                "label": "Scenario",
                                "life_hours": "Predicted Rupture Time (hours)"
                            },
                            title="파손 예상 시간 비교"
                        )
                        fig_point.update_yaxes(type="log")
                        fig_point.update_traces(
                            texttemplate="%{text:.2e}",
                            textposition="outside"
                        )
                        st.plotly_chart(fig_point, use_container_width=True)
        

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
# 수명 시뮬레이션
# ════════════════════════════════════════════════════════════════════
elif page == "Simulation":
    st.header("📉 크립 수명 시뮬레이션")
    st.caption("합금 성분과 온도를 바꿔가며 응력-수명 곡선과 시편 파단을 확인하세요")

    tab1, tab2, tab3 = st.tabs(["📈 응력-수명 곡선 (온도별)", "🔬 Cr 농도 영향", "💥 시편 파단 시뮬레이션"])

    def sim_predict(stress, temp, cr, ni=12.0, mo=0.1, s_temp=1150):
        row = {
            'Cr': cr, 'Ni': ni, 'Mo': mo, 'Mn': 1.5, 'Si': 0.5,
            'Nb': 0.0, 'Ti': 0.03, 'V': 0.0, 'W': 0.0, 'Cu': 0.1,
            'N': 0.05, 'C': 0.06, 'B': 0.0, 'P': 0.03, 'S': 0.008,
            'Co': 0.0, 'Al': 0.005, 'Sn': 0.0,
            'S_temp / oC': s_temp, 'log10(S_time / s)': 2.78, 'WQ': 1,
            'temp / oC': temp, 'stress / Mpa': float(stress),
            'Stab_ratio': stab_ratio(0.0, 0.03, 0.06, 0.05),
        }
        X = pd.DataFrame([{c: row[c] for c in ALL_COLS}])
        X_s = ngb_scaler_X.transform(X).astype(np.float64)
        dist = ngb_model.pred_dist(X_s)
        return float(dist.loc[0]), float(dist.scale[0])

    TEMPS_SIM  = [550, 600, 650, 700, 750, 800]
    COLORS_SIM = ['#60a5fa','#34d399','#fbbf24','#f97316','#ef4444','#a855f7']
    CR_VALS    = [12, 14, 16, 18, 20, 22, 24]
    CR_COLS    = ['#60a5fa','#34d399','#a3e635','#fbbf24','#f97316','#ef4444','#a855f7']
    stresses   = list(range(30, 420, 20))

    # ── TAB 1: 온도별 응력-수명 곡선 ─────────────────────────────
    with tab1:
        st.markdown("온도별 응력-수명 곡선 | **곡선이 아래로 갈수록 해당 온도에서 더 빨리 파단**")
        c1, c2, c3, c4 = st.columns(4)
        cr1 = c1.slider("Cr (wt%)",        10.0, 26.0, 18.0, 0.5, key="t1_cr")
        ni1 = c2.slider("Ni (wt%)",         4.0, 35.0, 12.0, 0.5, key="t1_ni")
        mo1 = c3.slider("Mo (wt%)",         0.0,  3.1,  0.1, 0.1, key="t1_mo")
        st1 = c4.slider("고용화 온도 (°C)",1000, 1350, 1150,  10, key="t1_st")
        show_ci = st.checkbox("95% 신뢰구간 표시", value=True, key="ci1")

        with st.spinner("NGBoost로 곡선 계산 중..."):
            fig1, ax1 = plt.subplots(figsize=(11, 6))
            ax1.set_facecolor('#0d1117'); fig1.patch.set_facecolor('#0d1117')
            for temp, color in zip(TEMPS_SIM, COLORS_SIM):
                mus, sigs = [], []
                for s in stresses:
                    try:
                        mu, sig = sim_predict(s, temp, cr1, ni1, mo1, st1)
                        mus.append(mu); sigs.append(sig)
                    except:
                        mus.append(None); sigs.append(None)
                valid = [(s,m,sg) for s,m,sg in zip(stresses,mus,sigs) if m is not None and 0<=m<=6]
                if len(valid) < 2: continue
                xs = [np.power(10, v[1]) for v in valid]
                ys = [v[0] for v in valid]
                ax1.plot(xs, ys, color=color, lw=2.5, label=f"{temp}°C")
                if show_ci:
                    cu = [np.power(10, v[1]+1.96*v[2]) for v in valid]
                    cl = [np.power(10, max(0, v[1]-1.96*v[2])) for v in valid]
                    ax1.fill_betweenx(ys, cl, cu, color=color, alpha=0.12)
            ax1.axvline(x=100000, color='#475569', linestyle='--', lw=1.5, alpha=0.7)
            ax1.text(110000, 380, '100,000h ref.', color='#64748b', fontsize=9)
            ax1.set_xscale('log')
            ax1.set_xlabel('Creep Rupture Life (hours)', color='#94a3b8', fontsize=12)
            ax1.set_ylabel('Stress (MPa)', color='#94a3b8', fontsize=12)
            ax1.set_title(f'Stress-Life Curve by Temperature  |  Cr={cr1}%, Ni={ni1}%, Mo={mo1}%',
                          color='#c7d2fe', fontsize=13, fontweight='bold')
            ax1.tick_params(colors='#64748b')
            ax1.spines[['bottom','left']].set_color('#334155')
            ax1.spines[['top','right']].set_visible(False)
            ax1.grid(True, linestyle='--', alpha=0.2, color='#334155')
            ax1.legend(loc='upper right', framealpha=0.2, labelcolor='white', facecolor='#1e293b', fontsize=10)
            fig1.tight_layout()
            st.pyplot(fig1)
        st.info("💡 x축=수명(시간, 로그), y축=응력(MPa). 온도가 높을수록(붉은 계열) 수명이 짧아집니다.")

    # ── TAB 2: Cr 농도 영향 ──────────────────────────────────────
    with tab2:
        st.markdown("Cr 농도별 응력-수명 곡선 | **Cr이 높을수록 곡선이 위로 이동**")
        c1, c2, c3 = st.columns(3)
        temp2 = c1.slider("온도 (°C)",        500,  900,  650, 25, key="t2_temp")
        ni2   = c2.slider("Ni (wt%)",          4.0, 35.0, 12.0, 0.5, key="t2_ni")
        st2   = c3.slider("고용화 온도 (°C)", 1000, 1350, 1150,  10, key="t2_st")

        with st.spinner("NGBoost로 곡선 계산 중..."):
            fig2, ax2 = plt.subplots(figsize=(11, 6))
            ax2.set_facecolor('#0d1117'); fig2.patch.set_facecolor('#0d1117')
            for cr_v, color in zip(CR_VALS, CR_COLS):
                mus, sigs = [], []
                for s in stresses:
                    try:
                        mu, sig = sim_predict(s, temp2, cr_v, ni2, 0.1, st2)
                        mus.append(mu); sigs.append(sig)
                    except:
                        mus.append(None); sigs.append(None)
                valid = [(s,m,sg) for s,m,sg in zip(stresses,mus,sigs) if m is not None and 0<=m<=6]
                if len(valid) < 2: continue
                xs = [np.power(10, v[1]) for v in valid]
                ys = [v[0] for v in valid]
                ax2.plot(xs, ys, color=color, lw=3.0 if cr_v==18 else 1.8,
                         label=f"Cr={cr_v}%"+(" (ref)" if cr_v==18 else ""))
            ax2.axvline(x=100000, color='#475569', linestyle='--', lw=1.5, alpha=0.7)
            ax2.text(110000, 380, '100,000h ref.', color='#64748b', fontsize=9)
            ax2.set_xscale('log')
            ax2.set_xlabel('Creep Rupture Life (hours)', color='#94a3b8', fontsize=12)
            ax2.set_ylabel('Stress (MPa)', color='#94a3b8', fontsize=12)
            ax2.set_title(f'Stress-Life Curve by Cr Content  |  {temp2}°C, Ni={ni2}%',
                          color='#c7d2fe', fontsize=13, fontweight='bold')
            ax2.tick_params(colors='#64748b')
            ax2.spines[['bottom','left']].set_color('#334155')
            ax2.spines[['top','right']].set_visible(False)
            ax2.grid(True, linestyle='--', alpha=0.2, color='#334155')
            ax2.legend(loc='upper right', framealpha=0.2, labelcolor='white', facecolor='#1e293b', fontsize=10)
            fig2.tight_layout()
            st.pyplot(fig2)
        st.info("💡 Cr은 산화막(Cr₂O₃)을 형성해 고온 내식성을 높입니다. Cr 높을수록 곡선이 위로 이동합니다.")

    # ── TAB 3: 시편 파단 시뮬레이션 ─────────────────────────────
    with tab3:
        st.markdown("온도별 시편이 시간에 따라 변형되다 파단되는 과정을 시뮬레이션합니다")
        import streamlit.components.v1 as components

        sc1, sc2, sc3 = st.columns(3)
        sim_cr3     = sc1.slider("Cr (wt%)",   10.0, 26.0, 18.0, 0.5, key="t3_cr")
        sim_stress3 = sc2.slider("응력 (MPa)",   20,  300,  130,  10,  key="t3_st")
        sim_speed3  = sc3.selectbox("배속", [100, 1000, 10000, 100000], index=1,
                                    format_func=lambda x: f"{x:,}배속", key="t3_sp")

        SIM_TEMPS4 = [600, 650, 700, 750]
        sim_lives  = {}
        for t in SIM_TEMPS4:
            try:
                mu, _ = sim_predict(sim_stress3, t, sim_cr3)
                sim_lives[t] = mu
            except:
                sim_lives[t] = 3.5 - (t - 600) * 0.01

        lives_js = "{" + ", ".join([f"{t}: {v:.4f}" for t, v in sim_lives.items()]) + "}"

        html_code = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body{{background:#0d1117;color:#e2e8f0;font-family:'Segoe UI',sans-serif;margin:0;padding:12px;}}
  .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
  .card{{background:#1a1a2e;border:1px solid #312e81;border-radius:10px;padding:12px;text-align:center;}}
  .card h4{{color:#a5b4fc;font-size:13px;margin:0 0 2px;}}
  .sub{{color:#64748b;font-size:11px;margin-bottom:8px;}}
  canvas{{display:block;margin:0 auto;}}
  .bar-wrap{{background:#0f0f1a;border-radius:4px;height:7px;margin-top:8px;overflow:hidden;}}
  .bar{{height:100%;border-radius:4px;}}
  .lbl{{font-size:10px;color:#64748b;margin-top:4px;}}
  .badge{{display:none;margin-top:6px;background:#7f1d1d;color:#fca5a5;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;}}
  .btns{{display:flex;gap:8px;margin-bottom:12px;}}
  button{{background:#4f46e5;color:white;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:13px;font-weight:600;}}
  button:hover{{background:#6366f1;}}
  button.sec{{background:#1e293b;border:1px solid #4338ca;color:#a5b4fc;}}
</style></head><body>
<div class="btns">
  <button onclick="startSim()">▶ 시뮬레이션 시작</button>
  <button class="sec" onclick="resetSim()">↺ 초기화</button>
</div>
<div class="grid" id="grid"></div>
<script>
const TEMPS=[600,650,700,750];
const COLORS=['#60a5fa','#fbbf24','#f97316','#ef4444'];
const LIVES={lives_js};
const SPEED={sim_speed3};
const ROWS=10,COLS=4;
let animId=null,states=[];
function buildGrid(){{
  const g=document.getElementById('grid');g.innerHTML='';states=[];
  TEMPS.forEach((temp,idx)=>{{
    const lifeH=Math.pow(10,LIVES[temp]);
    const tot=lifeH>=10000?(lifeH/10000).toFixed(1)+'만h':Math.round(lifeH)+'h';
    const card=document.createElement('div');card.className='card';
    card.innerHTML=`<h4>${{temp}}°C</h4><div class="sub">예측 수명: ${{tot}}</div>
      <canvas id="c${{idx}}" width="140" height="200"></canvas>
      <div class="bar-wrap"><div class="bar" id="b${{idx}}" style="background:${{COLORS[idx]}};width:0%"></div></div>
      <div class="lbl" id="l${{idx}}">0h / ${{tot}}</div>
      <div class="badge" id="bd${{idx}}">💥 파단!</div>`;
    g.appendChild(card);
    states.push({{temp,lifeH,elapsed:0,broken:false,color:COLORS[idx]}});
    draw(idx,0,false);
  }});
}}
function draw(idx,strain,broken){{
  const cv=document.getElementById('c'+idx);if(!cv)return;
  const ctx=cv.getContext('2d');const W=cv.width,H=cv.height;
  ctx.clearRect(0,0,W,H);ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const sw=72,sh=160,ox=(W-sw)/2,oy=(H-sh)/2;
  const cw=sw/COLS,ch=sh/ROWS,brow=broken?Math.floor(ROWS/2):-1;
  const col=states[idx].color;
  for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++){{
    const sf=strain*(r/ROWS),ew=sf*cw*0.5;
    let x=ox+c*cw-ew/2,y=oy+r*ch+(broken&&r>brow?14:0);
    const w=cw+ew,h=ch-sf*ch*0.06;
    const heat=Math.min(1,strain*0.8+r/ROWS*0.3);
    const pr=parseInt(col.slice(1,3),16),pg=parseInt(col.slice(3,5),16),pb=parseInt(col.slice(5,7),16);
    ctx.fillStyle=`rgba(${{Math.round(pr*(1-heat)+239*heat)}},${{Math.round(pg*(1-heat)+68*heat)}},${{Math.round(pb*(1-heat)+68*heat)}},0.85)`;
    ctx.fillRect(x+1,y+1,w-2,h-2);
    ctx.strokeStyle=(broken&&r===brow)?'#ff0000':'#1e293b';ctx.lineWidth=(broken&&r===brow)?2:0.5;
    ctx.strokeRect(x+1,y+1,w-2,h-2);
  }}
  if(broken){{
    ctx.strokeStyle='#ff4444';ctx.lineWidth=3;ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(ox-4,oy+(brow+1)*ch);ctx.lineTo(ox+sw+4,oy+(brow+1)*ch);ctx.stroke();ctx.setLineDash([]);
  }}else{{
    ctx.fillStyle='#94a3b8';ctx.font='9px Segoe UI';ctx.textAlign='center';
    ctx.fillText('▼ 하중',W/2,oy-5);ctx.fillText('▲ 고정',W/2,oy+sh+12);
  }}
}}
function startSim(){{
  if(animId)cancelAnimationFrame(animId);buildGrid();let t0=null;
  function animate(ts){{
    if(!t0)t0=ts;const simH=(ts-t0)/1000*SPEED;
    states.forEach((s,idx)=>{{
      if(s.broken)return;
      s.elapsed=Math.min(simH,s.lifeH);const ratio=s.elapsed/s.lifeH;
      draw(idx,Math.pow(ratio,1.5)*0.85,false);
      document.getElementById('b'+idx).style.width=(ratio*100)+'%';
      const cur=s.elapsed>=10000?(s.elapsed/10000).toFixed(1)+'만h':Math.round(s.elapsed)+'h';
      const tot=s.lifeH>=10000?(s.lifeH/10000).toFixed(1)+'만h':Math.round(s.lifeH)+'h';
      document.getElementById('l'+idx).textContent=cur+' / '+tot;
      if(ratio>=1){{s.broken=true;draw(idx,1,true);document.getElementById('bd'+idx).style.display='inline-block';document.getElementById('b'+idx).style.background='#ef4444';}}
    }});
    if(!states.every(s=>s.broken))animId=requestAnimationFrame(animate);
  }}
  animId=requestAnimationFrame(animate);
}}
function resetSim(){{if(animId){{cancelAnimationFrame(animId);animId=null;}}buildGrid();}}
buildGrid();
</script></body></html>"""
        components.html(html_code, height=420, scrolling=False)

        st.subheader("Predicted Life Comparison")
        life_df = pd.DataFrame([
            {
                "Temp (°C)": t,
                "Life (log10 h)": round(sim_lives[t], 4),
                "Life (hours)": f"{round(10**sim_lives[t]):,} h",
                "Life (years)": f"{10**sim_lives[t]/8760:.1f} yr",
            }
            for t in SIM_TEMPS4
        ])
        st.dataframe(life_df, use_container_width=True, hide_index=True)
        st.caption(f"Cr={sim_cr3}%, Stress={sim_stress3} MPa | NGBoost prediction")


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
