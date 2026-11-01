import os

# tf_keras / tensorflow import 보다 먼저 설정해야 적용
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

# ── 0. 파일명 설정 ─────────────────────────────────────────────────
_V2 = os.path.exists("bnn_v2_model.keras")
_P = "bnn_v2_" if _V2 else "bnn_"
BNN_MODEL_PATH     = f"{_P}model.keras"
BNN_SCALER_X_PATH  = f"{_P}scaler_X.pkl"
BNN_SCALER_Y_PATH  = f"{_P}scaler_y.pkl"
BNN_FEATURES_PATH  = f"{_P}features.pkl"
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0
print("📦 사용 모델:", "BNN v2 (평균+노이즈)" if _V2 else "BNN v1 (MC Dropout)")
DEFAULT_N_SAMPLES = 100

# ── 1. 모델 및 스케일러 로드 ──────────────────────────────────────────
import keras as keras3

@keras3.saving.register_keras_serializable(package="Custom", name="MCDropout")
class MCDropout(keras3.layers.Dropout):
    """추론 시에도 항상 드롭아웃을 켜는 레이어 (Monte Carlo Dropout)."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def load_model_any():
    errors = []
    try:
        model = keras3.models.load_model(
            BNN_MODEL_PATH, compile=False,
            custom_objects={"MCDropout": MCDropout},
        )
        print(f"✅ Keras {keras3.__version__} 로 BNN 로드 성공")
        return model
    except Exception as e:
        errors.append(f"[Keras 3] {type(e).__name__}: {e}")

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
        print("scaler_y: 없음")
    else:
        print("scaler_y:", type(bnn_scaler_y).__name__,
              "| mean_ =", getattr(bnn_scaler_y, "mean_", "?"),
              "| scale_ =", getattr(bnn_scaler_y, "scale_", "?"))
    print("──────────────────────────")

# ── 2. 컬럼 및 메타데이터 정의 ───────────────────────────────────────
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

# 학습된 피처 순서(Stab_ratio 포함 24개)를 파일에서 안전하게 로드
if os.path.exists(BNN_FEATURES_PATH):
    BNN_COLS = [str(c).strip() for c in joblib.load(BNN_FEATURES_PATH)]
    print("✅ 학습 때 feature 순서 적용:", BNN_COLS)
else:
    BNN_COLS = ALLOY_COLS + COND_COLS + ['Stab_ratio']
    print("ℹ️ bnn_features.pkl 없음 → 기본 24개 순서 사용")

print_scaler_debug()

def stab_ratio(nb, ti, c, n):
    return (nb / 8.0 + ti / 4.0) / (c + n + 1e-7)

def align_input_dataframe(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    
    # Stab_ratio가 없으면 자동 계산해서 채워줌
    if "Stab_ratio" not in df.columns:
        nb = pd.to_numeric(df.get('Nb', 0), errors="coerce").fillna(0)
        ti = pd.to_numeric(df.get('Ti', 0), errors="coerce").fillna(0)
        c = pd.to_numeric(df.get('C', 0), errors="coerce").fillna(0)
        n = pd.to_numeric(df.get('N', 0), errors="coerce").fillna(0)
        df['Stab_ratio'] = stab_ratio(nb, ti, c, n)

    aligned = pd.DataFrame(index=df.index)
    missing_cols = []
    for target in BNN_COLS:
        found = None
        for c in df.columns:
            if c.lower() == target.lower():
                found = c
                break
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

def scale_X(df):
    return bnn_scaler_X.transform(df[BNN_COLS].values)

# ── 3. BNN 예측 및 분석 함수 ────────────────────────────────────────
def aggregate_mc(mus, vars_, rng=None):
    rng = rng or np.random.default_rng()
    mean = mus.mean(axis=0)
    epi_var = mus.var(axis=0)
    ale_var = vars_.mean(axis=0)
    samples = mus + np.sqrt(vars_) * rng.standard_normal(mus.shape)
    return mean, np.sqrt(epi_var + ale_var), np.sqrt(epi_var), np.sqrt(ale_var), samples

def predict_bnn_full(X_scaled, n_samples=DEFAULT_N_SAMPLES):
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
    X_fixed, _ = align_input_dataframe(df_rows)
    X_s = scale_X(X_fixed)
    mean, std, _ = predict_bnn(X_s, n_samples)
    return mean, std

def sim_predict(stress, temp, cr, ni=12.0, mo=0.1, s_temp=1150, n_samples=30):
    row = {
        'Cr': cr, 'Ni': ni, 'Mo': mo, 'Mn': 1.5, 'Si': 0.5, 'Nb': 0.0, 'Ti': 0.03, 'V': 0.0,
        'W': 0.0, 'Cu': 0.1, 'N': 0.05, 'C': 0.06, 'B': 0.0, 'P': 0.03, 'S': 0.008, 'Co': 0.0,
        'Al': 0.005, 'Sn': 0.0, 'S_temp / oC': s_temp, 'log10(S_time / s)': 2.78, 'WQ': 1,
        'temp / oC': temp, 'stress / Mpa': float(stress), 
        'Stab_ratio': stab_ratio(0.0, 0.03, 0.06, 0.05)
    }
    X = pd.DataFrame([{c: row.get(c, 0.0) for c in BNN_COLS}])
    X_s = scale_X(X)
    mean, std, _ = predict_bnn(X_s, n_samples)
    return float(mean[0]), float(std[0])

# ── 4. LLM 챗봇 로직 ────────────────────────────────────────────────
LLM_SYSTEM_PROMPT = """
당신은 '소재부품 크리프(Creep) 수명 예측 시스템'의 AI 도메인 전문가 챗봇입니다.
사용자의 질문에 대해 친절하고 전문적으로 한국어로 답변해 주세요.
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

# ── 5. Gradio 이벤트 처리 함수들 ──────────────────────────────────────
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

        X_s = scale_X(X_fixed)
        mean, std, samples = predict_bnn(X_s, n_samples)
        lo = np.percentile(samples, 2.5, axis=0)
        hi = np.percentile(samples, 97.5, axis=0)

        def to_hours(v):
            return 10.0 ** np.clip(v, -20, 300)

        actual_col = _find_actual_col(df)
        actual = None
        if actual_col is not None:
            actual = pd.to_numeric(df[actual_col], errors="coerce").values.astype(float)

        # ⭐️ [수정 핵심] 예측 수명(mean) 기준 오름차순 정렬 인덱스 생성
        sort_idx = np.argsort(mean)
        mean_s = mean[sort_idx]
        std_s = std[sort_idx]
        lo_s = lo[sort_idx]
        hi_s = hi[sort_idx]
        actual_s = actual[sort_idx] if actual is not None else None

        # 결과 데이터프레임도 정렬된 순서로 재구성
        res = df.iloc[sort_idx].copy().reset_index(drop=True)
        res["BNN Predicted Life (log10 h)"] = mean_s
        res["BNN Predicted Life (hours)"] = to_hours(mean_s)
        res["BNN Uncertainty (Std, log10)"] = std_s
        res["95% CI Lower (h)"] = to_hours(lo_s)
        res["95% CI Upper (h)"] = to_hours(hi_s)

        metrics_md = ""
        if actual is not None:
            ok = ~np.isnan(actual_s)
            if ok.sum() > 1:
                err = mean_s[ok] - actual_s[ok]
                rmse = float(np.sqrt(np.mean(err ** 2)))
                ss_tot = float(np.sum((actual_s[ok] - actual_s[ok].mean()) ** 2))
                r2 = 1 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
                cover = float(np.mean((actual_s[ok] >= lo_s[ok]) & (actual_s[ok] <= hi_s[ok]))) * 100
                metrics_md = (f"\n\n**실제값 비교** (`{actual_col}`, {int(ok.sum())}개): "
                            f"R² = `{r2:.4f}`, RMSE = `{rmse:.4f}`, 95% 구간 Coverage = `{cover:.1f}%`")

        avg_life = float(np.mean(mean_s))
        warn = ""
        if avg_life > 4:
            comment = "👉 **해석**: 전반적으로 수명이 매우 긴 고내열 합금 조건입니다. (10,000시간 이상)"
        elif avg_life > 3:
            comment = "👉 **해석**: 일반적인 고온 환경에서 사용 가능한 수준입니다. (1,000~10,000시간)"
        else:
            comment = "👉 **해석**: 수명이 짧아 조건이 가혹한 상태입니다. (1,000시간 미만)"
        comment += f"\n\n평균 불확실성(Std): `{std_s.mean():.4f}` (log10 h)" + metrics_md + warn

        idx = np.arange(1, len(mean_s) + 1)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=idx, y=mean_s, mode="markers", name="BNN 예측 (평균)",
            marker=dict(size=8, color="#4f46e5"),
            error_y=dict(type="data", symmetric=False, array=hi_s - mean_s, arrayminus=mean_s - lo_s,
                         color="rgba(79,70,229,0.45)", thickness=1.5, width=3),
        ))
        if actual_s is not None:
            fig1.add_trace(go.Scatter(x=idx, y=actual_s, mode="markers", name="실제값",
                                      marker=dict(size=8, color="tomato", symbol="diamond")))
        fig1.update_layout(title="샘플별 BNN 예측 수명 오름차순 정렬 및 95% 구간",
                           xaxis_title="Sorted Sample Index", yaxis_title="log10 Life (h)",
                           template="plotly_white", legend=dict(orientation="h", y=-0.2))

        # 오른쪽 그래프는 기존 원본 데이터 기준 산점도 유지 (회귀 평가용)
        if actual is not None and np.sum(~np.isnan(actual)) > 1:
            ok = ~np.isnan(actual)
            lim_lo = float(min(actual[ok].min(), mean[ok].min()))
            lim_hi = float(max(actual[ok].max(), mean[ok].max()))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[lim_lo, lim_hi], y=[lim_lo, lim_hi], mode="lines",
                                      name="y = x (완벽한 예측)", line=dict(color="gray", dash="dash")))
            fig2.add_trace(go.Scatter(
                x=actual, y=mean, mode="markers", name="BNN 예측",
                marker=dict(size=8, color="#4f46e5"),
                error_y=dict(type="data", symmetric=False, array=(hi - mean), arrayminus=(mean - lo),
                           color="rgba(79,70,229,0.35)", thickness=1.2, width=2),
            ))
            fig2.update_layout(title="실제값 vs BNN 예측값 (오차막대: 95% 구간)",
                               xaxis_title="Actual log10 Life (h)", yaxis_title="Predicted log10 Life (h)",
                               template="plotly_white", legend=dict(orientation="h", y=-0.2))
        else:
            fig2 = px.histogram(x=std, nbins=20, labels={"x": "Std (log10 h)"},
                                title="BNN 불확실성(Std) 분포")
            fig2.update_layout(template="plotly_white")

        status = f"✅ 예측 완료 (MC 샘플 {int(n_samples)}회, 예측 수명 오름차순 정렬 완료)"
        return res.round(4), status, comment, fig1, fig2
    except Exception as e:
        return None, f"❌ 오류 발생: {str(e)}", "", None, None

def run_single_prediction(*args):
    n_samples = int(args[-1])
    vals = args[:-1]
    inp = {col: vals[i] for i, col in enumerate(ALLOY_COLS + COND_COLS)}
    inp['Stab_ratio'] = stab_ratio(inp.get('Nb', 0), inp.get('Ti', 0), inp.get('C', 0), inp.get('N', 0))
    X = pd.DataFrame([{c: inp.get(c, 0.0) for c in BNN_COLS}])

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
- **예측 수명**: **{life_h:,.0f} 시간** (log10: `{mu:.4f}`)
- **불확실성 (Std)**: `{sd:.4f}` (log10 h) → 예측 수명 **{mu:.3f} ± {sd:.3f}**{split_txt}
- **95% 예측 구간**: **{lower_h:,.0f} ~ {upper_h:,.0f} 시간**
- MC 샘플링 횟수: {n_samples}회
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
    base_row = {c: META[c][2] if c in META else 0.0 for c in ALLOY_COLS + COND_COLS}
    base_row['Stab_ratio'] = stab_ratio(base_row.get('Nb', 0), base_row.get('Ti', 0), base_row.get('C', 0), base_row.get('N', 0))

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

# ── 6. Gradio 전체 화면 구성 ──────────────────────────────────────────
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
