import os

#  tf_keras / tensorflow import 보다 먼저 설정해야 적용
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import numpy as np
import pandas as pd
import tf_keras as tfk
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# BNN이 TensorFlow Probability 레이어(DenseFlipout 등)를 쓰는 경우 필요
try:
    import tensorflow_probability as tfp  # noqa: F401
except ImportError:
    tfp = None

# .env 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 0. 파일명 설정 (본인 저장 파일명에 맞게 여기만 수정하세요) ─────────────────
# v2 파일(bnn_v2_*)이 폴더에 있으면 v2를, 없으면 기존 v1(bnn_*)을 자동으로 사용합니다.
_V2 = os.path.exists("bnn_v2_model.keras")
_P = "bnn_v2_" if _V2 else "bnn_"
BNN_MODEL_PATH     = f"{_P}model.keras"
BNN_SCALER_X_PATH  = f"{_P}scaler_X.pkl"
BNN_SCALER_Y_PATH  = f"{_P}scaler_y.pkl"   # 타깃을 스케일링하지 않았다면 파일이 없어도 동작
BNN_FEATURES_PATH  = f"{_P}features.pkl"
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0         # v2 학습 때와 동일한 값 (분산 폭주 방지)
print("📦 사용 모델:", "BNN v2 (평균+노이즈)" if _V2 else "BNN v1 (MC Dropout)")
DEFAULT_N_SAMPLES = 100                  # 몬테카를로 샘플링 횟수

# ── 1. 모델 및 스케일러 로드 ──────────────────────────────────────────────────
import keras as keras3

@keras3.saving.register_keras_serializable(package="Custom", name="MCDropout")
class MCDropout(keras3.layers.Dropout):
    """추론 시에도 항상 드롭아웃을 켜는 레이어 (Monte Carlo Dropout).
    학습 코드의 MCDropout과 같은 이름이어야 모델이 로드됩니다."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def load_model_any():
    """Keras 3로 저장된 .keras 파일은 Keras 3로, 레거시는 tf_keras로 순서대로 시도."""
    errors = []

    # 1) Keras 3 (batch_shape 설정이 들어간 최신 저장 포맷)
    try:
        model = keras3.models.load_model(
            BNN_MODEL_PATH, compile=False,
            custom_objects={"MCDropout": MCDropout},
        )
        print(f"✅ Keras {keras3.__version__} 로 BNN 로드 성공")
        return model
    except Exception as e:
        errors.append(f"[Keras 3] {type(e).__name__}: {e}")

    # 2) 레거시 tf_keras (TFP 레이어를 tf_keras로 저장한 경우)
    try:
        model = tfk.models.load_model(BNN_MODEL_PATH, compile=False)
        print("✅ tf_keras 로 BNN 로드 성공")
        return model
    except Exception as e:
        errors.append(f"[tf_keras] {type(e).__name__}: {e}")

    raise RuntimeError("BNN 모델 로드 실패:\n" + "\n\n".join(errors))

def load_bnn():
    model = load_model_any()
    scaler_X = joblib.load(BNN_SCALER_X_PATH)
    scaler_y = joblib.load(BNN_SCALER_Y_PATH) if os.path.exists(BNN_SCALER_Y_PATH) else None
    return model, scaler_X, scaler_y

bnn_model, bnn_scaler_X, bnn_scaler_y = load_bnn()

def print_scaler_debug():
    print("── [진단] 스케일러 정보 ──")
    print("scaler_X:", type(bnn_scaler_X).__name__, "| n_features_in_ =", getattr(bnn_scaler_X, "n_features_in_", "?"))
    if hasattr(bnn_scaler_X, "mean_"):
        print("scaler_X mean_[:5] =", np.round(bnn_scaler_X.mean_[:5], 3))
    if bnn_scaler_y is None:
        print("scaler_y: 없음 (타깃 역변환 없이 모델 출력을 그대로 사용)")
    else:
        print("scaler_y:", type(bnn_scaler_y).__name__,
              "| mean_ =", getattr(bnn_scaler_y, "mean_", "?"),
              "| scale_ =", getattr(bnn_scaler_y, "scale_", "?"))
    print("──────────────────────────")



# ── 2. 컬럼 및 메타데이터 정의 ────────────────────────────────────────────────
ALLOY_COLS = ['Cr','Ni','Mo','Mn','Si','Nb','Ti','V','W','Cu','N','C','B','P','S','Co','Al','Sn']
COND_COLS  = ['S_temp / oC', 'log10(S_time / s)', 'WQ', 'temp / oC', 'stress / Mpa']
ALL_COLS   = ALLOY_COLS + COND_COLS + ['Stab_ratio']

META = {
    'Cr': (10.0, 26.0, 18.0, 0.1, "Cr (wt%)"), 'Ni': (4.0, 35.0, 12.0, 0.1, "Ni (wt%)"),
    'Mo': (0.0, 3.1, 0.1, 0.01, "Mo (wt%)"), 'Mn': (0.0, 15.0, 1.5, 0.1, "Mn (wt%)"),
    'Si': (0.0, 1.3, 0.5, 0.01, "Si (wt%)"), 'Nb': (0.0, 5.0, 0.0, 0.01, "Nb (wt%)"),
    'Ti': (0.0, 0.56, 0.03, 0.01, "Ti (wt%)"), 'V': (0.0, 0.5, 0.0, 0.01, "V (wt%)"),
    'W': (0.0, 1.03, 0.0, 0.01, "W (wt%)"), 'Cu': (0.0, 3.1, 0.1, 0.01, "Cu (wt%)"),
    'N': (0.0, 0.3, 0.05, 0.001, "N (wt%)"), 'C': (0.0, 0.15, 0.06, 0.001, "C (wt%)"),
    'B': (0.0, 0.01, 0.0, 0.0001, "B (wt%)"), 'P': (0.0, 0.05, 0.03, 0.001, "P (wt%)"),
    'S': (0.0, 0.03, 0.008, 0.001, "S (wt%)"), 'Co': (0.0, 0.54, 0.0, 0.01, "Co (wt%)"),
    'Al': (0.0, 3.93, 0.005, 0.001, "Al (wt%)"), 'Sn': (0.0, 0.02, 0.0, 0.001, "Sn (wt%)"),
    'S_temp / oC': (1000, 1350, 1150, 10, "고용화 온도 (°C)"),
    'log10(S_time / s)': (2.08, 4.26, 2.78, 0.01, "고용화 시간 log10(s)"),
    'WQ': (0, 1, 1, 1, "수냉 여부"), 'temp / oC': (500, 1100, 650, 10, "크립 시험 온도 (°C)"),
    'stress / Mpa': (5.0, 471.0, 130.0, 1.0, "응력 (MPa)")
}

# BNN은 입력 23개(합금 18 + 공정 5)로 학습됨 → Stab_ratio는 모델 입력에서 제외
BNN_COLS = ALLOY_COLS + COND_COLS

def scale_X(df):
    """DataFrame에서 BNN 입력 23개 컬럼만 순서대로 골라 스케일링."""
    return bnn_scaler_X.transform(df[BNN_COLS].values)

# 학습 때 저장한 feature 순서가 있으면 그 순서를 그대로 사용 (없으면 기본 순서)
if os.path.exists(BNN_FEATURES_PATH):
    _feats = [str(c).strip() for c in joblib.load(BNN_FEATURES_PATH)]
    if sorted(_feats) == sorted(BNN_COLS):
        BNN_COLS = _feats
        print("✅ 학습 때 feature 순서 적용:", BNN_COLS)
    else:
        print("⚠️ bnn_features.pkl 컬럼이 앱 컬럼과 달라 기본 순서를 사용합니다.")
        print("   학습에만 있는 컬럼:", sorted(set(_feats) - set(BNN_COLS)))
        print("   앱에만 있는 컬럼  :", sorted(set(BNN_COLS) - set(_feats)))
else:
    print("ℹ️ bnn_features.pkl 없음 → 기본 순서 사용:", BNN_COLS)

print_scaler_debug()

def stab_ratio(nb, ti, c, n):
    return (nb / 8 + ti / 4) / (c + n + 1e-7)

def align_input_dataframe(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    aligned = pd.DataFrame(index=df.index)
    missing_cols = []
    for target in ALL_COLS:
        found = None
        # 1) 정확히 일치 (대소문자 무시) 우선
        for c in df.columns:
            if c.lower() == target.lower():
                found = c
                break
        # 2) 없으면 부분 일치 (원소기호처럼 짧은 이름은 오매칭 위험이 커서 제외)
        if found is None and len(target) > 3:
            for c in df.columns:
                if target.lower() in c.lower() or c.lower() in target.lower():
                    found = c
                    break
        if found is not None:
            aligned[target] = pd.to_numeric(df[found], errors="coerce").fillna(0)
        else:
            aligned[target] = 0
            missing_cols.append(target)
    return aligned, missing_cols

# ── 3. BNN 예측 및 분석 함수 ──────────────────────────────────────────────────
def aggregate_mc(mus, vars_, rng=None):
    """
    MC 샘플링 결과를 합침.
    mus  : (T, n) 각 샘플링의 예측 평균 (원래 단위)
    vars_: (T, n) 각 샘플링의 예측 분산 (v1은 0)
    return: 평균, 총 Std, 모델 불확실성 Std, 데이터 노이즈 Std, 예측 샘플
    """
    rng = rng or np.random.default_rng()
    mean = mus.mean(axis=0)
    epi_var = mus.var(axis=0)            # 모델 불확실성 (epistemic)
    ale_var = vars_.mean(axis=0)         # 데이터 노이즈 (aleatoric, v1은 0)
    samples = mus + np.sqrt(vars_) * rng.standard_normal(mus.shape)
    return mean, np.sqrt(epi_var + ale_var), np.sqrt(epi_var), np.sqrt(ale_var), samples

def predict_bnn_full(X_scaled, n_samples=DEFAULT_N_SAMPLES):
    """
    같은 입력으로 n_samples번 순전파 (MC Dropout).
    - v1(출력 1개): 평균 = 예측 수명(log10 h), Std = 모델 불확실성
    - v2(출력 2개): [평균, 로그분산] → 모델 불확실성 + 데이터 노이즈를 합친 총 Std
    """
    x = np.asarray(X_scaled, dtype=np.float32)
    y_scale = float(bnn_scaler_y.scale_[0]) if bnn_scaler_y is not None else 1.0
    mus, vars_ = [], []
    for _ in range(int(n_samples)):
        out = bnn_model(x, training=False)
        out = np.asarray(out.numpy() if hasattr(out, "numpy") else out, dtype=np.float64)
        out = out.reshape(len(x), -1)
        mu = out[:, :1]
        if bnn_scaler_y is not None:
            mu = bnn_scaler_y.inverse_transform(mu)
        if out.shape[1] >= 2:
            lv = np.clip(out[:, 1], LOGVAR_MIN, LOGVAR_MAX)
            var = np.exp(lv) * y_scale ** 2
        else:
            var = np.zeros(len(x))
        mus.append(mu.flatten()); vars_.append(var)
    mean, std, epi, ale, samples = aggregate_mc(np.array(mus), np.array(vars_))
    return {"mean": mean, "std": std, "epi": epi, "ale": ale, "samples": samples}

def predict_bnn(X_scaled, n_samples=DEFAULT_N_SAMPLES):
    r = predict_bnn_full(X_scaled, n_samples)
    return r["mean"], r["std"], r["samples"]

def predict_bnn_batch_df(df_rows, n_samples=30):
    df_rows = df_rows.copy()
    if "Stab_ratio" not in df_rows.columns:
        df_rows["Stab_ratio"] = stab_ratio(df_rows["Nb"], df_rows["Ti"], df_rows["C"], df_rows["N"])
    X_fixed, _ = align_input_dataframe(df_rows)
    X_s = scale_X(X_fixed)
    mean, std, _ = predict_bnn(X_s, n_samples)
    return mean, std

def sim_predict(stress, temp, cr, ni=12.0, mo=0.1, s_temp=1150, n_samples=30):
    row = {
        'Cr': cr, 'Ni': ni, 'Mo': mo, 'Mn': 1.5, 'Si': 0.5, 'Nb': 0.0, 'Ti': 0.03, 'V': 0.0,
        'W': 0.0, 'Cu': 0.1, 'N': 0.05, 'C': 0.06, 'B': 0.0, 'P': 0.03, 'S': 0.008, 'Co': 0.0,
        'Al': 0.005, 'Sn': 0.0, 'S_temp / oC': s_temp, 'log10(S_time / s)': 2.78, 'WQ': 1,
        'temp / oC': temp, 'stress / Mpa': float(stress), 'Stab_ratio': stab_ratio(0.0, 0.03, 0.06, 0.05)
    }
    X = pd.DataFrame([{c: row[c] for c in ALL_COLS}])
    X_s = scale_X(X)
    mean, std, _ = predict_bnn(X_s, n_samples)
    return float(mean[0]), float(std[0])

# ── 4. LLM 챗봇 로직 ─────────────────────────────────────────────────────────
LLM_SYSTEM_PROMPT = """
당신은 '소재부품 크리프(Creep) 수명 예측 시스템'의 AI 도메인 전문가 챗봇입니다.
사용자의 질문에 대해 친절하고 전문적으로 한국어로 답변해 주세요.

[시스템 및 도메인 지식]
1. 시스템 개요:
   - 합금 성분(Cr, Ni, Mo, Nb, Ti 등)과 공정 조건(온도, 응력)을 입력받아 크리프 수명을 예측하고 불확실성을 분석하는 플랫폼입니다.
   - 엑셀 업로드 시 컬럼 순서 및 누락 항목을 자동 정렬 및 기본값(0) 보정하여 처리합니다.
2. 수명 예측 모델:
   - 베이지안 신경망(BNN, Monte Carlo Dropout 방식): 구조는 입력 23개 → Dense(64) → MCDropout(0.1) → Dense(32) → MCDropout(0.1) → 출력 1개.
     추론 시에도 드롭아웃을 켠 채 같은 입력을 여러 번(예: 100회) 예측하면 예측값 분포가 생기며,
     평균을 예측 수명(log10 시간), 표준편차(Std)를 불확실성(모델 불확실성, epistemic)으로 사용합니다.
     MC Dropout은 베이지안 추론의 근사 방법입니다.\n   - BNN v2는 출력이 2개(예측 평균, 로그분산)로, 모델 불확실성(epistemic)과 데이터 노이즈(aleatoric)를 분리해 추정합니다.\n     총 불확실성 = sqrt(모델 불확실성² + 데이터 노이즈²).
   - 95% 구간은 샘플링된 예측 분포의 2.5~97.5 백분위수로 계산합니다.
   - (참고) 이전 버전: Committee ANN(R²=0.9236, RMSE=0.2570), NGBoost(R²=0.9155, RMSE=0.2694, 95% Coverage=88.1%).
   - BNN 성능 수치(R², RMSE, Coverage)는 여기에 본인 결과로 채워 넣으세요.
3. 재료공학 지식:
   - 크리프(Creep): 고온·고응력 환경에서 재료가 시간에 따라 서서히 변형되는 현상.
   - 주요 변수: 응력(Stress / MPa)과 온도(Temp / °C)가 핵심 영향 변수.
   - 주요 성분: Cr(내산화성), Ni(고온 안정성), Nb/Ti(석출 강화).
   - 파생변수: Stab_ratio = (Nb/8 + Ti/4) / (C + N + ε).
"""

def predict_chat_llm(user_message, history):
    if not user_message.strip():
        return "", history

    api_messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    for msg in history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": user_message})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=api_messages, temperature=0.3, max_tokens=600
        )
        bot_response = response.choices[0].message.content
    except Exception as e:
        bot_response = f"⚠️ LLM 오류: {str(e)}"

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_response})
    return "", history

# ── 5. Gradio 이벤트 처리 함수들 ──────────────────────────────────────────────
def _find_actual_col(df):
    for c in df.columns:
        if "t_r" in str(c).lower():
            return c
    return None

def run_batch_prediction(file, n_samples):
    if file is None:
        return None, "파일을 업로드해주세요.", "", None, None
    try:
        path = file if isinstance(file, str) else file.name
        df = pd.read_excel(path)
        X_fixed, missing_cols = align_input_dataframe(df)
        X_fixed['Stab_ratio'] = stab_ratio(X_fixed['Nb'], X_fixed['Ti'], X_fixed['C'], X_fixed['N'])

        X_s = scale_X(X_fixed)
        mean, std, samples = predict_bnn(X_s, n_samples)
        lo = np.percentile(samples, 2.5, axis=0)
        hi = np.percentile(samples, 97.5, axis=0)

        def to_hours(v):
            return 10.0 ** np.clip(v, -20, 300)

        res = df.copy()
        res["BNN Predicted Life (log10 h)"] = mean
        res["BNN Predicted Life (hours)"] = to_hours(mean)
        res["BNN Uncertainty (Std, log10)"] = std
        res["95% CI Lower (h)"] = to_hours(lo)
        res["95% CI Upper (h)"] = to_hours(hi)

        # 실제값(log10(t_r / h)) 열이 있으면 비교
        actual_col = _find_actual_col(df)
        actual = None
        metrics_md = ""
        if actual_col is not None:
            actual = pd.to_numeric(df[actual_col], errors="coerce").values.astype(float)
            ok = ~np.isnan(actual)
            if ok.sum() > 1:
                err = mean[ok] - actual[ok]
                rmse = float(np.sqrt(np.mean(err ** 2)))
                ss_tot = float(np.sum((actual[ok] - actual[ok].mean()) ** 2))
                r2 = 1 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
                cover = float(np.mean((actual[ok] >= lo[ok]) & (actual[ok] <= hi[ok]))) * 100
                metrics_md = (f"\n\n**실제값 비교** (`{actual_col}`, {int(ok.sum())}개): "
                              f"R² = `{r2:.4f}`, RMSE = `{rmse:.4f}`, 95% 구간 Coverage = `{cover:.1f}%`")

        avg_life = float(np.mean(mean))
        warn = ""
        if np.nanmax(np.abs(mean)) > 20:
            warn = ("\n\n⚠️ **예측값이 비정상적으로 큽니다** (log10 수명은 보통 0~6). "
                    "엑셀 컬럼 매칭, 입력 순서, `bnn_scaler_y.pkl`(타깃 역변환) 설정을 확인하세요.")
        if avg_life > 4:
            comment = "👉 **해석**: 전반적으로 수명이 매우 긴 고내열 합금 조건입니다. (10,000시간 이상)"
        elif avg_life > 3:
            comment = "👉 **해석**: 일반적인 고온 환경에서 사용 가능한 수준입니다. (1,000~10,000시간)"
        else:
            comment = "👉 **해석**: 수명이 짧아 조건이 가혹한 상태입니다. (1,000시간 미만)"
        comment += f"\n\n평균 불확실성(Std): `{std.mean():.4f}` (log10 h)" + metrics_md + warn

        # ── 그래프 1: 샘플별 예측값 + 95% 구간 (+ 실제값) ──
        idx = np.arange(1, len(mean) + 1)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=idx, y=mean, mode="markers", name="BNN 예측 (평균)",
            marker=dict(size=8, color="#4f46e5"),
            error_y=dict(type="data", symmetric=False, array=hi - mean, arrayminus=mean - lo,
                         color="rgba(79,70,229,0.45)", thickness=1.5, width=3),
        ))
        if actual is not None:
            fig1.add_trace(go.Scatter(x=idx, y=actual, mode="markers", name="실제값",
                                      marker=dict(size=8, color="tomato", symbol="diamond")))
        fig1.update_layout(title="샘플별 BNN 예측 수명과 95% 구간",
                           xaxis_title="Sample #", yaxis_title="log10 Life (h)",
                           template="plotly_white", legend=dict(orientation="h", y=-0.2))

        # ── 그래프 2: 실제값 vs 예측값 (없으면 불확실성 분포) ──
        if actual is not None and np.sum(~np.isnan(actual)) > 1:
            ok = ~np.isnan(actual)
            lim_lo = float(min(actual[ok].min(), mean[ok].min()))
            lim_hi = float(max(actual[ok].max(), mean[ok].max()))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[lim_lo, lim_hi], y=[lim_lo, lim_hi], mode="lines",
                                      name="y = x (완벽한 예측)", line=dict(color="gray", dash="dash")))
            fig2.add_trace(go.Scatter(
                x=actual[ok], y=mean[ok], mode="markers", name="BNN 예측",
                marker=dict(size=8, color="#4f46e5"),
                error_y=dict(type="data", symmetric=False, array=(hi - mean)[ok], arrayminus=(mean - lo)[ok],
                             color="rgba(79,70,229,0.35)", thickness=1.2, width=2),
            ))
            fig2.update_layout(title="실제값 vs BNN 예측값 (오차막대: 95% 구간)",
                               xaxis_title="Actual log10 Life (h)", yaxis_title="Predicted log10 Life (h)",
                               template="plotly_white", legend=dict(orientation="h", y=-0.2))
        else:
            fig2 = px.histogram(x=std, nbins=20, labels={"x": "Std (log10 h)"},
                                title="BNN 불확실성(Std) 분포")
            fig2.update_layout(template="plotly_white")

        status = f"✅ 예측 완료 (MC 샘플 {int(n_samples)}회, 자동 보정 컬럼: {missing_cols if missing_cols else '없음'})"
        return res.round(4), status, comment, fig1, fig2
    except Exception as e:
        return None, f"❌ 오류 발생: {str(e)}", "", None, None

def run_single_prediction(*args):
    n_samples = int(args[-1])
    vals = args[:-1]
    inp = {col: vals[i] for i, col in enumerate(ALL_COLS[:-1])}
    inp['Stab_ratio'] = stab_ratio(inp['Nb'], inp['Ti'], inp['C'], inp['N'])
    X = pd.DataFrame([{c: inp[c] for c in ALL_COLS}])

    X_s = scale_X(X)
    r = predict_bnn_full(X_s, n_samples)
    mean, std, samples = r["mean"], r["std"], r["samples"]
    mu, sd = float(mean[0]), float(std[0])
    epi_sd, ale_sd = float(r["epi"][0]), float(r["ale"][0])
    s = samples[:, 0]
    lo_log, hi_log = np.percentile(s, [2.5, 97.5])

    life_h = 10 ** mu
    lower_h, upper_h = 10 ** lo_log, 10 ** hi_log
    split_txt = (f"\n    * ├ 모델 불확실성: `{epi_sd:.4f}` / 데이터 노이즈: `{ale_sd:.4f}`"
                 if ale_sd > 0 else "")

    res_md = f"""
### 🧠 BNN 예측 결과
* **예측 수명**: **{life_h:,.0f} 시간** (log10: `{mu:.4f}`)
* **불확실성 (Std)**: `{sd:.4f}` (log10 h) → 예측 수명 **{mu:.3f} ± {sd:.3f}**{split_txt}
* **95% 예측 구간**: **{lower_h:,.0f} ~ {upper_h:,.0f} 시간**
* MC 샘플링 횟수: {n_samples}회
"""

    fig, ax = plt.subplots(figsize=(8, 3.2))
    hours = 10 ** s
    bins = np.logspace(np.log10(hours.min() * 0.98), np.log10(hours.max() * 1.02), 25)
    ax.hist(hours, bins=bins, color='steelblue', alpha=0.55, label="BNN sampled predictions")
    ax.axvspan(lower_h, upper_h, color='steelblue', alpha=0.15, label="95% interval")
    ax.axvline(life_h, color='tomato', lw=2.0, label=f"Mean: {life_h:,.0f}h")
    ax.set_xscale('log')
    ax.set_xlabel("Life (hours)"); ax.set_ylabel("Count")
    ax.legend(fontsize=9); ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return res_md, fig

def run_advanced_simulation(mode, temp_vals, var_choice, compare_vals, fixed_temp, fixed_stress, stress_min, stress_max, n_pts):
    base_row = {c: META[c][2] if c in META else 0.0 for c in ALL_COLS}

    if mode == "온도 영향":
        if not temp_vals: return go.Figure()
        rows, meta = [], []
        stress_vals = np.linspace(stress_min, stress_max, n_pts)
        for t in temp_vals:
            for s in stress_vals:
                r = base_row.copy(); r['temp / oC'] = t; r['stress / Mpa'] = float(s)
                rows.append(r); meta.append({"value": t, "stress": float(s)})
        df_res = pd.DataFrame(meta)
        pred_log, _ = predict_bnn_batch_df(pd.DataFrame(rows))
        df_res["life_hours"] = 10 ** pred_log

        fig = px.line(df_res, x="life_hours", y="stress", color="value", markers=True,
                      labels={"life_hours": "Life (hours)", "stress": "Stress (MPa)", "value": "Temperature (°C)"},
                      title="온도별 Stress–Life Curve (BNN 평균 예측)")
        fig.update_xaxes(type="log")
        return fig

    elif mode == "성분 영향":
        if not compare_vals: return go.Figure()
        rows, meta = [], []
        stress_vals = np.linspace(stress_min, stress_max, n_pts)
        for v in compare_vals:
            for s in stress_vals:
                r = base_row.copy(); r['temp / oC'] = fixed_temp; r[var_choice] = float(v); r['stress / Mpa'] = float(s)
                rows.append(r); meta.append({"value": v, "stress": float(s)})
        df_res = pd.DataFrame(meta)
        pred_log, _ = predict_bnn_batch_df(pd.DataFrame(rows))
        df_res["life_hours"] = 10 ** pred_log

        fig = px.line(df_res, x="life_hours", y="stress", color="value", markers=True,
                      labels={"life_hours": "Life (hours)", "stress": "Stress (MPa)", "value": var_choice},
                      title=f"{var_choice} 변화에 따른 Stress–Life Curve (BNN 평균 예측)")
        fig.update_xaxes(type="log")
        return fig

    else:
        if not compare_vals: return go.Figure()
        rows, labels = [], []
        for v in compare_vals:
            r = base_row.copy(); r['temp / oC'] = fixed_temp; r['stress / Mpa'] = float(fixed_stress); r[var_choice] = float(v)
            rows.append(r); labels.append(f"{var_choice}={v}")
        pred_log, pred_std = predict_bnn_batch_df(pd.DataFrame(rows))
        df_res = pd.DataFrame({"label": labels, "life_hours": 10 ** pred_log})
        # 불확실성(Std)을 log 공간에서 오차막대로 표시
        upper = 10 ** (pred_log + pred_std) - 10 ** pred_log
        lower = 10 ** pred_log - 10 ** (pred_log - pred_std)

        fig = px.bar(df_res, x="label", y="life_hours", text="life_hours",
                     labels={"label": "Scenario", "life_hours": "Predicted Rupture Time (hours)"},
                     title="파손 예상 시간 비교 (오차막대: ±1 Std)")
        fig.update_traces(texttemplate="%{text:.2e}", textposition="outside",
                          error_y=dict(type="data", symmetric=False, array=upper, arrayminus=lower))
        fig.update_yaxes(type="log")
        return fig

def update_rupture_html(cr, stress, speed):
    temps = [600, 650, 700, 750]
    sim_lives = {}
    for t in temps:
        try:
            mu, _ = sim_predict(stress, t, cr)
            sim_lives[t] = mu
        except Exception:
            sim_lives[t] = 3.5 - (t - 600) * 0.01
    lives_js = "{" + ", ".join([f"{t}: {v:.4f}" for t, v in sim_lives.items()]) + "}"

    html_code = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
      body{{background:#0d1117;color:#e2e8f0;font-family:'Segoe UI',sans-serif;margin:0;padding:12px;}}
      .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
      .card{{background:#1a1a2e;border:1px solid #312e81;border-radius:10px;padding:12px;text-align:center;}}
      .card h4{{color:#a5b4fc;font-size:13px;margin:0 0 2px;}} .sub{{color:#64748b;font-size:11px;margin-bottom:8px;}}
      canvas{{display:block;margin:0 auto;}} .bar-wrap{{background:#0f0f1a;border-radius:4px;height:7px;margin-top:8px;overflow:hidden;}}
      .bar{{height:100%;border-radius:4px;}} .lbl{{font-size:10px;color:#64748b;margin-top:4px;}}
      .badge{{display:none;margin-top:6px;background:#7f1d1d;color:#fca5a5;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;}}
      .btns{{display:flex;gap:8px;margin-bottom:12px;}}
      button{{background:#4f46e5;color:white;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:13px;font-weight:600;}}
      button.sec{{background:#1e293b;border:1px solid #4338ca;color:#a5b4fc;}}
    </style></head><body>
    <div class="btns"><button onclick="startSim()">▶ 시뮬레이션 시작</button><button class="sec" onclick="resetSim()">↺ 초기화</button></div>
    <div class="grid" id="grid"></div>
    <script>
    const TEMPS=[600,650,700,750], COLORS=['#60a5fa','#fbbf24','#f97316','#ef4444'], LIVES={lives_js}, SPEED={speed}, ROWS=10, COLS=4;
    let animId=null, states=[];
    function buildGrid(){{
      const g=document.getElementById('grid');g.innerHTML='';states=[];
      TEMPS.forEach((temp,idx)=>{{
        const lifeH=Math.pow(10,LIVES[temp]), tot=lifeH>=10000?(lifeH/10000).toFixed(1)+'만h':Math.round(lifeH)+'h';
        const card=document.createElement('div');card.className='card';
        card.innerHTML=`<h4>${{temp}}°C</h4><div class="sub">예측 수명: ${{tot}}</div><canvas id="c${{idx}}" width="140" height="200"></canvas>
          <div class="bar-wrap"><div class="bar" id="b${{idx}}" style="background:${{COLORS[idx]}};width:0%"></div></div>
          <div class="lbl" id="l${{idx}}">0h / ${{tot}}</div><div class="badge" id="bd${{idx}}">💥 파단!</div>`;
        g.appendChild(card); states.push({{temp,lifeH,elapsed:0,broken:false,color:COLORS[idx]}}); draw(idx,0,false);
      }});
    }}
    function draw(idx,strain,broken){{
      const cv=document.getElementById('c'+idx);if(!cv)return; const ctx=cv.getContext('2d'), W=cv.width, H=cv.height;
      ctx.clearRect(0,0,W,H);ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
      const sw=72,sh=160,ox=(W-sw)/2,oy=(H-sh)/2, cw=sw/COLS,ch=sh/ROWS, brow=broken?Math.floor(ROWS/2):-1, col=states[idx].color;
      for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++){{
        const sf=strain*(r/ROWS),ew=sf*cw*0.5; let x=ox+c*cw-ew/2,y=oy+r*ch+(broken&&r>brow?14:0), w=cw+ew,h=ch-sf*ch*0.06;
        const heat=Math.min(1,strain*0.8+r/ROWS*0.3), pr=parseInt(col.slice(1,3),16),pg=parseInt(col.slice(3,5),16),pb=parseInt(col.slice(5,7),16);
        ctx.fillStyle=`rgba(${{Math.round(pr*(1-heat)+239*heat)}},${{Math.round(pg*(1-heat)+68*heat)}},${{Math.round(pb*(1-heat)+68*heat)}},0.85)`;
        ctx.fillRect(x+1,y+1,w-2,h-2); ctx.strokeStyle=(broken&&r===brow)?'#ff0000':'#1e293b';ctx.lineWidth=(broken&&r===brow)?2:0.5; ctx.strokeRect(x+1,y+1,w-2,h-2);
      }}
      if(broken){{ ctx.strokeStyle='#ff4444';ctx.lineWidth=3;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(ox-4,oy+(brow+1)*ch);ctx.lineTo(ox+sw+4,oy+(brow+1)*ch);ctx.stroke();ctx.setLineDash([]); }}
      else{{ ctx.fillStyle='#94a3b8';ctx.font='9px Segoe UI';ctx.textAlign='center';ctx.fillText('▼ 하중',W/2,oy-5);ctx.fillText('▲ 고정',W/2,oy+sh+12); }}
    }}
    function startSim(){{
      if(animId)cancelAnimationFrame(animId);buildGrid();let t0=null;
      function animate(ts){{
        if(!t0)t0=ts;const simH=(ts-t0)/1000*SPEED;
        states.forEach((s,idx)=>{{
          if(s.broken)return; s.elapsed=Math.min(simH,s.lifeH);const ratio=s.elapsed/s.lifeH;
          draw(idx,Math.pow(ratio,1.5)*0.85,false); document.getElementById('b'+idx).style.width=(ratio*100)+'%';
          const cur=s.elapsed>=10000?(s.elapsed/10000).toFixed(1)+'만h':Math.round(s.elapsed)+'h', tot=s.lifeH>=10000?(s.lifeH/10000).toFixed(1)+'만h':Math.round(s.lifeH)+'h';
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
    escaped_html = html_code.replace('"', '&quot;')
    return f'<iframe srcdoc="{escaped_html}" style="width:100%; height:450px; border:none;"></iframe>'


# ── 6. Gradio 전체 화면 구성 ──────────────────────────────────────────────────
with gr.Blocks(title="소재부품 크립 수명 예측 시스템 (BNN)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚙️ 소재부품 크리프 수명 예측 및 불확실성 분석 플랫폼 (BNN)")

    with gr.Tabs():
        # TAB 1: 📂 배치 데이터 업로드 및 예측
        with gr.Tab("📂 Data Upload & Prediction"):
            gr.Markdown("### 엑셀 데이터를 업로드하고 BNN 배치 수명 예측을 실행합니다.")
            excel_file = gr.File(label="엑셀 파일 업로드 (.xlsx)", file_types=[".xlsx"])
            batch_samples = gr.Slider(10, 300, value=50, step=10, label="MC 샘플링 횟수 (많을수록 정밀, 느려짐)")
            batch_btn = gr.Button("🚀 배치 예측 실행", variant="primary")
            status_txt = gr.Textbox(label="상태 메시지")
            res_df_table = gr.Dataframe(label="예측 결과 테이블")
            comment_txt = gr.Markdown()
            with gr.Row():
                batch_plot1 = gr.Plot(label="샘플별 예측 및 95% 구간")
                batch_plot2 = gr.Plot(label="실제값 vs 예측값 / 불확실성 분포")
            batch_btn.click(run_batch_prediction, inputs=[excel_file, batch_samples],
                            outputs=[res_df_table, status_txt, comment_txt, batch_plot1, batch_plot2])

        # TAB 2: 🔬 단일 샘플 예측
        with gr.Tab("🔬 단일 샘플 예측"):
            gr.Markdown("### 합금 성분 및 열처리/시험 조건을 직접 입력하여 BNN 예측과 불확실성을 확인합니다.")
            inputs_single = []
            with gr.Accordion("합금 성분 (wt%) 입력", open=True):
                with gr.Row():
                    for col in ALLOY_COLS:
                        mn, mx, dv, step, label = META[col]
                        inputs_single.append(gr.Number(label=label, value=dv, minimum=mn, maximum=mx, step=step))
            with gr.Accordion("공정 및 시험 조건 입력", open=True):
                with gr.Row():
                    for col in COND_COLS:
                        mn, mx, dv, step, label = META[col]
                        if col == 'WQ':
                            inputs_single.append(gr.Dropdown(choices=[1, 0], value=1, label=label))
                        else:
                            inputs_single.append(gr.Number(label=label, value=dv, minimum=mn, maximum=mx, step=step))

            single_samples = gr.Slider(10, 500, value=DEFAULT_N_SAMPLES, step=10, label="MC 샘플링 횟수")
            single_btn = gr.Button("🚀 단일 예측 실행", variant="primary")
            single_md = gr.Markdown()
            single_plot = gr.Plot(label="BNN 예측 분포 및 95% 구간")
            single_btn.click(run_single_prediction, inputs=inputs_single + [single_samples],
                             outputs=[single_md, single_plot])

        # TAB 3: 📉 고급 수명 시뮬레이션
        with gr.Tab("📉 고급 수명 시뮬레이션"):
            gr.Markdown("### 온도 및 성분 영향성을 조건별 인터랙티브 차트로 확인합니다.")
            with gr.Row():
                sim_mode = gr.Radio(["온도 영향", "성분 영향", "파손 예상 시간 비교"], label="분석 방식", value="온도 영향")
                var_choice = gr.Dropdown(["Cr", "Ni", "Nb", "Ti", "Al"], value="Cr", label="비교 성분")
            with gr.Row():
                temp_vals = gr.CheckboxGroup([550, 600, 650, 700, 750, 800], value=[600, 700, 800], label="비교 온도 (°C)")
                compare_vals = gr.CheckboxGroup([10, 12, 14, 16, 18, 20, 22, 24, 26], value=[12, 18, 24], label="비교 성분값")
            with gr.Row():
                fixed_temp = gr.Slider(500, 1100, value=650, step=10, label="고정 온도 (°C)")
                fixed_stress = gr.Slider(5, 471, value=130, step=5, label="고정 응력 (MPa)")
                stress_min = gr.Slider(5, 200, value=50, step=5, label="최소 응력 (MPa)")
                stress_max = gr.Slider(200, 471, value=300, step=5, label="최대 응력 (MPa)")
                n_pts = gr.Slider(6, 20, value=10, step=2, label="곡선 포인트 수")

            adv_btn = gr.Button("📊 고급 시뮬레이션 실행", variant="primary")
            adv_plot = gr.Plot()

            adv_inputs = [sim_mode, temp_vals, var_choice, compare_vals, fixed_temp, fixed_stress, stress_min, stress_max, n_pts]
            adv_btn.click(run_advanced_simulation, inputs=adv_inputs, outputs=[adv_plot])

        # TAB 4: 💥 시편 파단 시뮬레이션
        with gr.Tab("💥 시편 파단 시뮬레이션"):
            gr.Markdown("### 시간 경과에 따른 온도별 시편 파단 과정을 애니메이션으로 시뮬레이션합니다.")
            with gr.Row():
                cr_anim = gr.Slider(10.0, 26.0, value=18.0, step=0.5, label="Cr (wt%)")
                stress_anim = gr.Slider(20, 300, value=130, step=10, label="응력 (MPa)")
                speed_anim = gr.Dropdown(choices=[100, 1000, 10000, 100000], value=1000, label="시뮬레이션 배속")

            html_out = gr.HTML()
            anim_inputs = [cr_anim, stress_anim, speed_anim]

            demo.load(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            cr_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            stress_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            speed_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])

        # TAB 5: 🤖 LLM AI 도메인 챗봇
        with gr.Tab("🤖 LLM AI 도메인 챗봇"):
            gr.Markdown("### 💬 크리프 현상, 모델 해석, 합금 성분 영향에 대해 질문해보세요!")
            chatbot_ui = gr.Chatbot(height=450, placeholder="크리프 현상, 모델 정확도, 신뢰구간에 대해 무엇이든 물어보세요.")
            msg_input = gr.Textbox(placeholder="질문을 입력하고 Enter를 누르세요...", show_label=False)
            clear_btn = gr.ClearButton([msg_input, chatbot_ui], value="대화 초기화")

            msg_input.submit(predict_chat_llm, inputs=[msg_input, chatbot_ui], outputs=[msg_input, chatbot_ui])

if __name__ == "__main__":
    demo.launch(share=True)