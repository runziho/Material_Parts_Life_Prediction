import os

# tf_keras / tensorflow import 보다 먼저 설정해야 적용
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import re
from html import escape as _esc
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
print("📦 사용 모델:", "BNN v3 (평균+노이즈)" if _V2 else "BNN v1 (MC Dropout)")
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

def _norm(s):
    """컬럼명 비교용 정규화: 소문자, 공백/_/괄호 제거, (wt%) 제거, ℃ → oc"""
    s = str(s).strip().lower().replace('℃', 'oc').replace('°c', 'oc')
    s = re.sub(r'\(?wt\.?\s*%\)?', '', s)
    return re.sub(r'[\s_\-\.\(\)\[\]]', '', s)

# 자주 쓰는 다른 표기들 (정규화 후 비교)
COL_ALIASES = {
    'S_temp / oC':       ['solution temp', 'solution temperature', 'S temp', 'S_temp/C'],
    'log10(S_time / s)': ['log S_time', 'log10 S_time', 'S_time', 'solution time'],
    'WQ':                ['water quench', 'water quenching'],
    'temp / oC':         ['temp', 'temperature', 'test temp', 'test temperature', 'creep temp', 'temp/C'],
    'stress / Mpa':      ['stress', 'stress (MPa)', 'stress/MPa'],
    'Stab_ratio':        ['stability ratio'],
}

def align_input_dataframe(df, return_map=False):
    """셔플된 엑셀 컬럼을 학습 때 순서(BNN_COLS)로 정렬.
    1) 정규화 후 정확히 일치 → 2) 별칭 → 3) 남은 컬럼만 부분 일치. 없으면 0으로 채움."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    norm_map = {}
    for c in df.columns:
        norm_map.setdefault(_norm(c), c)

    found_map = {}
    for t in BNN_COLS:                      # 1단계
        if _norm(t) in norm_map:
            found_map[t] = norm_map[_norm(t)]
    used = set(found_map.values())

    for t in BNN_COLS:                      # 2단계
        if t in found_map:
            continue
        for a in COL_ALIASES.get(t, []):
            c = norm_map.get(_norm(a))
            if c is not None and c not in used:
                found_map[t] = c; used.add(c)
                break

    for t in BNN_COLS:                      # 3단계 (C/N/S 같은 짧은 이름은 제외)
        tn = _norm(t)
        if t in found_map or len(tn) <= 3:
            continue
        for c in df.columns:
            cn = _norm(c)
            if c in used or len(cn) <= 3:
                continue
            if tn in cn or cn in tn:
                found_map[t] = c; used.add(c)
                break

    aligned = pd.DataFrame(index=df.index)
    missing_cols = []
    for t in BNN_COLS:
        if t in found_map:
            aligned[t] = df[found_map[t]]
        else:
            aligned[t] = 0
            missing_cols.append(t)
    return (aligned, missing_cols, found_map) if return_map else (aligned, missing_cols)

def _recalc_stab(X):
    """Nb/Ti/C/N이 바뀌면 Stab_ratio도 바뀌므로 항상 다시 계산."""
    if 'Stab_ratio' in X.columns and all(c in X.columns for c in ('Nb', 'Ti', 'C', 'N')):
        X['Stab_ratio'] = stab_ratio(X['Nb'], X['Ti'], X['C'], X['N'])
    return X

def prepare_uploaded_df(df):
    """업로드 데이터를 정렬 + 숫자 변환 + Stab_ratio 계산. (X, 누락컬럼, 매핑) 반환"""
    X, missing, mapping = align_input_dataframe(df, return_map=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    alloy = [c for c in ALLOY_COLS if c in X.columns]
    X[alloy] = X[alloy].fillna(0.0)                     # 성분 빈칸은 0(미첨가)으로 처리
    bad = [c for c in COND_COLS if c in X.columns and X[c].isna().any()]
    if bad:
        raise ValueError(f"공정/시험 조건 컬럼에 빈 값이나 숫자가 아닌 값이 있습니다: {bad}")
    X = _recalc_stab(X.fillna(0.0))
    return X, [c for c in missing if c != 'Stab_ratio'], mapping

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
    X_fixed, _ = align_input_dataframe(df_rows)
    X_fixed = _recalc_stab(X_fixed)
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
        bot_response = f"LLM 오류: {str(e)}"

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_response})
    return "", history

# ── 5. 대시보드 공통 요소 ───────────────────────────────────────────
MODEL_LABEL = "BNN v2 · 평균+노이즈 모델" if _V2 else "BNN v1 · MC Dropout"

NAVY = gr.themes.Color(c50="#f1f6fb", c100="#e3edf7", c200="#c9dcef", c300="#a8c4e2", c400="#7ea4d0",
                       c500="#5b86bb", c600="#46699a", c700="#38557d", c800="#2d4466", c900="#233553", c950="#1a2840",
                       name="navy")
THEME = gr.themes.Soft(primary_hue=NAVY, secondary_hue="sky", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "sans-serif"]).set(
    body_background_fill="#f3f8fc",
    button_primary_background_fill="#46699a",
    button_primary_background_fill_hover="#38557d",
    button_primary_text_color="#ffffff",
)

CSS = """
.hero{background:#e4eff9;border:1px solid #c9dcef;border-radius:12px;padding:16px 22px;margin-bottom:8px}
.hero h1{margin:0;font-size:1.3rem;font-weight:700;color:#2d4466}
.hero p{margin:4px 0 0;font-size:.85rem;color:#46699a}
.dark .hero{background:#1f2f45;border-color:#2d4466}
.dark .hero h1{color:#dbe8f5}
.dark .hero p{color:#a8c4e2}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:2px 0 10px}
.kpi{background:var(--background-fill-secondary);border:1px solid #d6e4f2;border-left:4px solid #5b86bb;
     border-radius:10px;padding:12px 14px}
.kpi .lbl{font-size:.75rem;color:var(--body-text-color-subdued)}
.kpi .val{font-size:1.35rem;font-weight:700;line-height:1.3;color:#2d4466}
.dark .kpi .val{color:#dbe8f5}
.kpi .sub{font-size:.73rem;color:var(--body-text-color-subdued);margin-top:2px}
.kpi.info{border-left-color:#7fb8dc}.kpi.good{border-left-color:#6aa6a0}
.kpi.warn{border-left-color:#d0a85c}.kpi.bad{border-left-color:#c98585}
.kpi-empty{border:1px dashed #c9dcef;border-radius:10px;padding:14px;margin:2px 0 10px;
           color:var(--body-text-color-subdued);text-align:center;font-size:.9rem}
.panel{background:var(--block-background-fill);border:1px solid #d6e4f2;border-radius:12px;padding:14px !important}
.panel h4{margin:0 0 4px 0}
"""

HERO_HTML = f"""
<div class="hero">
  <h1>소재부품 크리프 수명 예측 대시보드</h1>
  <p>Bayesian Neural Network 예측 · 모델/데이터 불확실성 분석 · 고급 수명 시뮬레이션 &nbsp;|&nbsp; {MODEL_LABEL}</p>
</div>
"""

EMPTY_KPI = '<div class="kpi-empty">실행하면 핵심 지표가 여기에 표시됩니다.</div>'

def _fmt_h(h):
    h = float(h)
    return f"{h:,.0f}" if h < 1e6 else f"{h:.2e}"

def _span(h):
    h = float(h)
    return f"≈ {h/24:,.0f}일 · {h/8760:,.1f}년"

def kpi_cards(items):
    """items: [(라벨, 값, 보조설명, tone)] → KPI 카드 HTML. tone: info/good/warn/bad/''"""
    cards = "".join(
        f'<div class="kpi {tone}"><div class="lbl">{_esc(str(lbl))}</div>'
        f'<div class="val">{_esc(str(val))}</div><div class="sub">{_esc(str(sub))}</div></div>'
        for lbl, val, sub, tone in items)
    return f'<div class="kpi-grid">{cards}</div>'

def _style_fig(fig):
    fig.update_layout(template="plotly_white", height=480, margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=-0.18), font=dict(color="#2d4466"),
                      plot_bgcolor="#fbfdff",
                      colorway=["#2d4466", "#5b86bb", "#7fb8dc", "#a8c4e2", "#46699a", "#6aa6a0", "#d0a85c", "#c98585", "#8ea3b8"])
    return fig

def _file_path(file):
    # Gradio 버전에 따라 str 경로 또는 tempfile 객체로 들어옴
    return file if isinstance(file, str) else file.name

# ── 6. Gradio 이벤트 처리 함수들 ──────────────────────────────────────
def _find_actual_col(df):
    for c in df.columns:
        if "t_r" in str(c).lower():
            return c
    return None

def run_batch_prediction(file, n_samples):
    if file is None:
        return None, "파일을 업로드해주세요.", "", EMPTY_KPI, None, None
    try:
        df = pd.read_excel(_file_path(file))
        df.columns = df.columns.astype(str).str.strip()
        X_fixed, missing_cols, mapping = prepare_uploaded_df(df)

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

        # 예측 수명(mean) 기준 오름차순 정렬
        sort_idx = np.argsort(mean)
        mean_s, std_s, lo_s, hi_s = mean[sort_idx], std[sort_idx], lo[sort_idx], hi[sort_idx]
        actual_s = actual[sort_idx] if actual is not None else None

        # 결과표: 학습 순서로 정렬된 입력 컬럼 → 사용하지 않은 원본 컬럼 → 예측 결과
        extras = [c for c in df.columns if c not in set(mapping.values())]
        res = pd.concat([X_fixed, df[extras]], axis=1).iloc[sort_idx].reset_index(drop=True)
        res["BNN Predicted Life (log10 h)"] = mean_s
        res["BNN Predicted Life (hours)"] = to_hours(mean_s)
        res["BNN Uncertainty (Std, log10)"] = std_s
        res["95% CI Lower (h)"] = to_hours(lo_s)
        res["95% CI Upper (h)"] = to_hours(hi_s)

        metrics = None
        if actual is not None:
            ok = ~np.isnan(actual_s)
            if ok.sum() > 1:
                err = mean_s[ok] - actual_s[ok]
                rmse = float(np.sqrt(np.mean(err ** 2)))
                ss_tot = float(np.sum((actual_s[ok] - actual_s[ok].mean()) ** 2))
                r2 = 1 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
                cover = float(np.mean((actual_s[ok] >= lo_s[ok]) & (actual_s[ok] <= hi_s[ok]))) * 100
                metrics = (r2, rmse, cover, int(ok.sum()), actual_col)

        avg_life = float(np.mean(mean_s))
        if avg_life > 4:
            comment = "**해석**: 전반적으로 수명이 매우 긴 고내열 합금 조건입니다. (10,000시간 이상)"
        elif avg_life > 3:
            comment = "**해석**: 일반적인 고온 환경에서 사용 가능한 수준입니다. (1,000~10,000시간)"
        else:
            comment = "**해석**: 수명이 짧아 조건이 가혹한 상태입니다. (1,000시간 미만)"

        cards = [
            ("분석 샘플 수", f"{len(df):,}개", "업로드한 행 수", "info"),
            ("평균 예측 수명", f"{_fmt_h(10 ** avg_life)} h", f"log10 평균 {avg_life:.3f} · {_span(10 ** avg_life)}", ""),
            ("최단 수명", f"{_fmt_h(to_hours(mean_s.min()))} h", _span(to_hours(mean_s.min())), "bad"),
            ("최장 수명", f"{_fmt_h(to_hours(mean_s.max()))} h", _span(to_hours(mean_s.max())), "good"),
            ("평균 불확실성 σ", f"{float(std_s.mean()):.3f}", "log10 h · 모델+데이터", "warn"),
        ]
        if metrics is not None:
            r2, rmse, cover, n_ok, a_col = metrics
            cards += [
                ("R²", f"{r2:.4f}", f"실제값 `{a_col}` {n_ok}개 기준".replace("`", ""), "info"),
                ("RMSE", f"{rmse:.4f}", "log10 h", ""),
                ("95% 구간 Coverage", f"{cover:.1f}%", "실제값이 예측 구간 안에 들어온 비율", "good"),
            ]
        kpi = kpi_cards(cards)

        idx = np.arange(1, len(mean_s) + 1)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=idx, y=mean_s, mode="markers", name="BNN 예측 (평균)",
            marker=dict(size=8, color="#46699a"),
            error_y=dict(type="data", symmetric=False, array=hi_s - mean_s, arrayminus=mean_s - lo_s,
                         color="rgba(70,105,154,0.40)", thickness=1.5, width=3),
        ))
        if actual_s is not None:
            fig1.add_trace(go.Scatter(x=idx, y=actual_s, mode="markers", name="실제값",
                                      marker=dict(size=8, color="#d0a85c", symbol="diamond")))
        fig1.update_layout(title="샘플별 BNN 예측 수명 (오름차순) 및 95% 구간",
                           xaxis_title="Sorted Sample Index", yaxis_title="log10 Life (h)")
        _style_fig(fig1).update_layout(height=400)

        # 오른쪽 그래프는 원본 데이터 기준 산점도 (회귀 평가용)
        if actual is not None and np.sum(~np.isnan(actual)) > 1:
            ok = ~np.isnan(actual)
            lim_lo = float(min(actual[ok].min(), mean[ok].min()))
            lim_hi = float(max(actual[ok].max(), mean[ok].max()))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=[lim_lo, lim_hi], y=[lim_lo, lim_hi], mode="lines",
                                      name="y = x (완벽한 예측)", line=dict(color="#8ea3b8", dash="dash")))
            fig2.add_trace(go.Scatter(
                x=actual, y=mean, mode="markers", name="BNN 예측",
                marker=dict(size=8, color="#46699a"),
                error_y=dict(type="data", symmetric=False, array=(hi - mean), arrayminus=(mean - lo),
                             color="rgba(70,105,154,0.35)", thickness=1.2, width=2),
            ))
            fig2.update_layout(title="실제값 vs BNN 예측값 (오차막대: 95% 구간)",
                               xaxis_title="Actual log10 Life (h)", yaxis_title="Predicted log10 Life (h)")
        else:
            fig2 = px.histogram(x=std, nbins=20, labels={"x": "Std (log10 h)"},
                                title="BNN 불확실성(Std) 분포", color_discrete_sequence=["#5b86bb"])
        _style_fig(fig2).update_layout(height=400)

        status = f"예측 완료 ({len(df)}행, MC 샘플 {int(n_samples)}회, 예측 수명 오름차순 정렬)"
        renamed = [f"{src}→{tgt}" for tgt, src in mapping.items() if src != tgt]
        if renamed:
            status += f"\n[보정] 이름이 달라서 매칭한 컬럼: {renamed}"
        if missing_cols:
            status += f"\n[주의] 파일에 없어서 0으로 채운 컬럼: {missing_cols}"
        if extras:
            status += f"\n[참고] 예측에 쓰지 않은 컬럼: {extras}"
        return res.round(4), status, comment, kpi, fig1, fig2
    except Exception as e:
        return None, f"오류 발생: {str(e)}", "", EMPTY_KPI, None, None

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

    cards = [
        ("BNN 예측 수명", f"{_fmt_h(life_h)} h", f"log10 {mu:.3f} ± {sd:.3f}", ""),
        ("95% 예측 구간 하한", f"{_fmt_h(lower_h)} h", _span(lower_h), "warn"),
        ("95% 예측 구간 상한", f"{_fmt_h(upper_h)} h", _span(upper_h), "good"),
        ("불확실성 σ", f"{sd:.3f}", f"log10 h · MC 샘플 {n_samples}회", "info"),
    ]
    if ale_sd > 0:
        cards += [("모델 불확실성", f"{epi_sd:.3f}", "epistemic (log10 h)", ""),
                  ("데이터 노이즈", f"{ale_sd:.3f}", "aleatoric (log10 h)", "")]
    kpi = kpi_cards(cards)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    hours = 10 ** s
    bins = np.logspace(np.log10(hours.min() * 0.98), np.log10(hours.max() * 1.02 + 1e-9), 25)
    ax.hist(hours, bins=bins, color='#9bbbe0', alpha=0.85, label="BNN sampled predictions")
    ax.axvspan(lower_h, upper_h, color='#c9dcef', alpha=0.45, label="95% interval")
    ax.axvline(life_h, color='#2d4466', lw=2.0, label=f"Mean: {life_h:,.0f}h")
    ax.set_xscale('log')
    ax.set_xlabel("Life (hours)"); ax.set_ylabel("Count")
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, frameon=False); ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return kpi, fig

# ── 7. 고급 수명 시뮬레이션 ───────────────────────────────────────────
SIM_VAR_CHOICES = ALLOY_COLS + ['S_temp / oC', 'log10(S_time / s)']

def load_uploaded_file(file):
    """업로드한 엑셀을 정렬해서 State에 저장하고, 기준 샘플 드롭다운을 채운다."""
    default_choice = [("기본값 (슬라이더 기준)", -1)]
    if file is None:
        return None, gr.update(choices=default_choice, value=-1)
    df = pd.read_excel(_file_path(file))
    try:
        X, _, _ = prepare_uploaded_df(df)
    except ValueError as e:
        raise gr.Error(str(e))
    X = X.reset_index(drop=True)
    choices = list(default_choice)
    for i in range(min(len(X), 500)):  # 드롭다운이 너무 길어지지 않게 500행까지만 표시
        r = X.iloc[i]
        choices.append((f"샘플 {i} | Cr={r.get('Cr', 0):g}, Ni={r.get('Ni', 0):g}, "
                        f"{r.get('temp / oC', 0):g}°C, {r.get('stress / Mpa', 0):g}MPa", i))
    return X, gr.update(choices=choices, value=0 if len(X) else -1)

def make_base_row(df_state, sample_idx):
    """기준 조건: 업로드 파일의 선택된 행, 없으면 META 기본값."""
    row = {c: META[c][2] for c in ALLOY_COLS + COND_COLS}
    if df_state is not None and sample_idx is not None and 0 <= int(sample_idx) < len(df_state):
        src = df_state.iloc[int(sample_idx)]
        for c in df_state.columns:
            row[c] = float(src[c])
    return row

def _fmt(v):
    return f"{float(v):g}"

def _snap(v, var):
    """META의 step에 맞춰 값을 반올림 (예: Cr 0.1, S_temp 10)."""
    step = META[var][3]
    decimals = max(0, int(round(-np.log10(step)))) if step < 1 else 0
    return round(round(v / step) * step, decimals)

def _preset_choices(base_row, var, n=9):
    """선택 목록: META 범위를 n등분한 값 + 현재 기준값."""
    lo, hi = META[var][0], META[var][1]
    vals = {_snap(v, var) for v in np.linspace(lo, hi, n)}
    vals.add(round(float(base_row[var]), 6))
    return [_fmt(v) for v in sorted(vals)]

def _default_selection(base_row, var):
    """기본 선택: 최소 / 중간 / 최대 + 기준값."""
    lo, hi = META[var][0], META[var][1]
    vals = {_snap(lo, var), _snap((lo + hi) / 2, var), _snap(hi, var), round(float(base_row[var]), 6)}
    return [_fmt(v) for v in sorted(vals)]

def refresh_controls(df_state, sample_idx, var):
    """기준 샘플/비교 변수가 바뀌면 고정 온도·응력, 비교 값 선택 목록을 자동으로 맞춘다."""
    base = make_base_row(df_state, sample_idx)
    t = int(round(min(max(base['temp / oC'], 500), 1100) / 10) * 10)
    s = int(round(min(max(base['stress / Mpa'], 5), 471) / 5) * 5)
    return (gr.update(value=t), gr.update(value=s),
            gr.update(choices=_preset_choices(base, var), value=_default_selection(base, var)))

def parse_values(selected):
    """드롭다운에서 고른 값 + 직접 입력한 값(문자열)을 숫자 리스트로 변환."""
    if isinstance(selected, (str, int, float)):
        selected = [selected]
    vals = []
    try:
        for item in (selected or []):
            for tok in str(item).replace("，", ",").split(","):   # "12, 18"처럼 입력해도 허용
                if tok.strip():
                    vals.append(float(tok))
    except ValueError:
        raise gr.Error("비교 값은 숫자만 입력할 수 있어요. 예: 12 또는 18.5")
    if not vals:
        raise gr.Error("비교 값을 하나 이상 선택하거나 입력하세요.")
    return sorted(set(vals))

def _sim_kpis(df_res, label_fn, n_scn):
    top = df_res.loc[df_res['life_hours'].idxmax()]
    low = df_res.loc[df_res['life_hours'].idxmin()]
    ratio = top['life_hours'] / max(low['life_hours'], 1e-12)
    return kpi_cards([
        ("비교 시나리오", f"{n_scn}개", f"예측 포인트 {len(df_res)}개", "info"),
        ("최장 수명", f"{_fmt_h(top['life_hours'])} h", f"{label_fn(top)} · {_span(top['life_hours'])}", "good"),
        ("최단 수명", f"{_fmt_h(low['life_hours'])} h", f"{label_fn(low)} · {_span(low['life_hours'])}", "bad"),
        ("최장/최단 비", f"{ratio:,.1f}배", "조건 변화에 따른 수명 차이", "warn"),
    ])

def toggle_sim_controls(mode):
    """분석 방식에 맞는 컨트롤만 보여준다. (temp_vals, var_choice, compare_vals, fixed_temp, fixed_stress, stress_min, stress_max, n_pts)"""
    temp_mode, comp_mode = mode == "온도 영향", mode == "파손 예상 시간 비교"
    return (gr.update(visible=temp_mode), gr.update(visible=not temp_mode), gr.update(visible=not temp_mode),
            gr.update(visible=not temp_mode), gr.update(visible=comp_mode),
            gr.update(visible=not comp_mode), gr.update(visible=not comp_mode), gr.update(visible=not comp_mode))

def run_advanced_simulation(mode, temp_vals, var_choice, compare_vals, fixed_temp, fixed_stress,
                            stress_min, stress_max, n_pts, df_state, sample_idx):
    base_row = make_base_row(df_state, sample_idx)
    n_pts = int(n_pts)
    if stress_min >= stress_max:
        raise gr.Error("최소 응력은 최대 응력보다 작아야 합니다.")
    stress_vals = np.linspace(stress_min, stress_max, n_pts)
    src = f"샘플 {int(sample_idx)} 기준" if (df_state is not None and sample_idx is not None and int(sample_idx) >= 0) else "기본값 기준"

    if mode == "온도 영향":
        if not temp_vals:
            raise gr.Error("비교 온도를 하나 이상 선택하세요.")
        rows, meta = [], []
        for t in temp_vals:
            for s in stress_vals:
                r = base_row.copy(); r['temp / oC'] = float(t); r['stress / Mpa'] = float(s)
                rows.append(r); meta.append({"value": f"{float(t):g}", "stress": float(s)})
        df_res = pd.DataFrame(meta)
        pred_log, _ = predict_bnn_batch_df(pd.DataFrame(rows))
        df_res["life_hours"] = 10 ** pred_log

        fig = px.line(df_res, x="life_hours", y="stress", color="value", markers=True,
                      labels={"life_hours": "Life (hours)", "stress": "Stress (MPa)", "value": "Temperature (°C)"},
                      title=f"온도별 Stress–Life Curve (BNN 평균 예측, {src})")
        fig.update_xaxes(type="log")
        kpi = _sim_kpis(df_res, lambda r: f"{r['value']}°C @ {r['stress']:.0f}MPa", len(temp_vals))
        return _style_fig(fig), kpi

    vals = parse_values(compare_vals)

    if mode == "성분 영향":
        rows, meta = [], []
        for v in vals:
            for s in stress_vals:
                r = base_row.copy(); r['temp / oC'] = float(fixed_temp); r[var_choice] = float(v); r['stress / Mpa'] = float(s)
                rows.append(r); meta.append({"value": f"{v:g}", "stress": float(s)})
        df_res = pd.DataFrame(meta)
        pred_log, _ = predict_bnn_batch_df(pd.DataFrame(rows))
        df_res["life_hours"] = 10 ** pred_log

        fig = px.line(df_res, x="life_hours", y="stress", color="value", markers=True,
                      labels={"life_hours": "Life (hours)", "stress": "Stress (MPa)", "value": var_choice},
                      title=f"{var_choice} 변화에 따른 Stress–Life Curve (BNN 평균 예측, {src}, {fixed_temp:g}°C)")
        fig.update_xaxes(type="log")
        kpi = _sim_kpis(df_res, lambda r: f"{var_choice}={r['value']} @ {r['stress']:.0f}MPa", len(vals))
        return _style_fig(fig), kpi

    # 파손 예상 시간 비교
    rows, labels = [], []
    for v in vals:
        r = base_row.copy(); r['temp / oC'] = float(fixed_temp); r['stress / Mpa'] = float(fixed_stress); r[var_choice] = float(v)
        rows.append(r); labels.append(f"{var_choice}={v:g}")
    pred_log, pred_std = predict_bnn_batch_df(pd.DataFrame(rows))
    df_res = pd.DataFrame({"label": labels, "life_hours": 10 ** pred_log})
    upper = 10 ** (pred_log + pred_std) - 10 ** pred_log
    lower = 10 ** pred_log - 10 ** (pred_log - pred_std)

    fig = px.bar(df_res, x="label", y="life_hours", text="life_hours",
                 labels={"label": "Scenario", "life_hours": "Predicted Rupture Time (hours)"},
                 title=f"파손 예상 시간 비교 ({src}, {fixed_temp:g}°C / {fixed_stress:g}MPa, 오차막대 ±1 Std)")
    fig.update_traces(texttemplate="%{text:.2e}", textposition="outside",
                      error_y=dict(type="data", symmetric=False, array=upper, arrayminus=lower))
    fig.update_yaxes(type="log")
    kpi = _sim_kpis(df_res, lambda r: r['label'], len(vals))
    return _style_fig(fig), kpi

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
      body{{background:#f3f8fc;color:#2d4466;font-family:'Segoe UI',sans-serif;margin:0;padding:12px;}}
      .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}}
      .card{{background:#ffffff;border:1px solid #d6e4f2;border-radius:10px;padding:12px;text-align:center;}}
      .card h4{{color:#2d4466;font-size:13px;margin:0 0 2px;}} .sub{{color:#6b7f99;font-size:11px;margin-bottom:8px;}}
      canvas{{display:block;margin:0 auto;}} .bar-wrap{{background:#e3edf7;border-radius:4px;height:7px;margin-top:8px;overflow:hidden;}}
      .bar{{height:100%;border-radius:4px;}} .lbl{{font-size:10px;color:#6b7f99;margin-top:4px;}}
      .badge{{display:none;margin-top:6px;background:#f6dede;color:#a34a4a;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;}}
      .btns{{display:flex;gap:8px;margin-bottom:12px;}}
      button{{background:#46699a;color:white;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:13px;font-weight:600;}}
      button.sec{{background:#e4eff9;border:1px solid #c9dcef;color:#2d4466;}}
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
          <div class="lbl" id="l${{idx}}">0h / ${{tot}}</div><div class="badge" id="bd${{idx}}">파단</div>`;
        g.appendChild(card); states.push({{temp,lifeH,elapsed:0,broken:false,color:COLORS[idx]}}); draw(idx,0,false);
      }});
    }}
    function draw(idx,strain,broken){{
      const cv=document.getElementById('c'+idx);if(!cv)return; const ctx=cv.getContext('2d'), W=cv.width, H=cv.height;
      ctx.clearRect(0,0,W,H);ctx.fillStyle='#f7fafd';ctx.fillRect(0,0,W,H);
      const sw=72,sh=160,ox=(W-sw)/2,oy=(H-sh)/2, cw=sw/COLS,ch=sh/ROWS, brow=broken?Math.floor(ROWS/2):-1, col=states[idx].color;
      for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++){{
        const sf=strain*(r/ROWS),ew=sf*cw*0.5; let x=ox+c*cw-ew/2,y=oy+r*ch+(broken&&r>brow?14:0), w=cw+ew,h=ch-sf*ch*0.06;
        const heat=Math.min(1,strain*0.8+r/ROWS*0.3), pr=parseInt(col.slice(1,3),16),pg=parseInt(col.slice(3,5),16),pb=parseInt(col.slice(5,7),16);
        ctx.fillStyle=`rgba(${{Math.round(pr*(1-heat)+239*heat)}},${{Math.round(pg*(1-heat)+68*heat)}},${{Math.round(pb*(1-heat)+68*heat)}},0.85)`;
        ctx.fillRect(x+1,y+1,w-2,h-2); ctx.strokeStyle=(broken&&r===brow)?'#c0504d':'#c9dcef';ctx.lineWidth=(broken&&r===brow)?2:0.5; ctx.strokeRect(x+1,y+1,w-2,h-2);
      }}
      if(broken){{ ctx.strokeStyle='#ff4444';ctx.lineWidth=3;ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(ox-4,oy+(brow+1)*ch);ctx.lineTo(ox+sw+4,oy+(brow+1)*ch);ctx.stroke();ctx.setLineDash([]); }}
      else{{ ctx.fillStyle='#6b7f99';ctx.font='9px Segoe UI';ctx.textAlign='center';ctx.fillText('▼ 하중',W/2,oy-5);ctx.fillText('▲ 고정',W/2,oy+sh+12); }}
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
with gr.Blocks(title="소재부품 크리프 수명 예측 대시보드 (BNN)", theme=THEME, css=CSS) as demo:
    gr.HTML(HERO_HTML)

    with gr.Tabs():
        # TAB 1: 배치 데이터 업로드 및 예측
        with gr.Tab("Data Upload & Prediction"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes=["panel"]):
                    gr.Markdown("#### 데이터 업로드")
                    excel_file = gr.File(label="엑셀 파일 업로드 (.xlsx)", file_types=[".xlsx"])
                    batch_samples = gr.Slider(10, 300, value=50, step=10, label="MC 샘플링 횟수 (많을수록 정밀, 느려짐)")
                    batch_btn = gr.Button("배치 예측 실행", variant="primary")
                    status_txt = gr.Textbox(label="상태 메시지", lines=4, interactive=False)
                with gr.Column(scale=3):
                    batch_kpi = gr.HTML(EMPTY_KPI)
                    comment_txt = gr.Markdown()
                    with gr.Row():
                        batch_plot1 = gr.Plot(label="샘플별 예측 및 95% 구간")
                        batch_plot2 = gr.Plot(label="실제값 vs 예측값 / 불확실성 분포")
            res_df_table = gr.Dataframe(label="예측 결과 테이블")
            batch_btn.click(run_batch_prediction, inputs=[excel_file, batch_samples],
                            outputs=[res_df_table, status_txt, comment_txt, batch_kpi, batch_plot1, batch_plot2])

        # TAB 2: 단일 샘플 예측
        with gr.Tab("단일 샘플 예측"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_classes=["panel"]):
                    inputs_single = []
                    gr.Markdown("#### 합금 성분 (wt%)")
                    for i in range(0, len(ALLOY_COLS), 6):
                        with gr.Row():
                            for col in ALLOY_COLS[i:i + 6]:
                                mn, mx, dv, step, label = META[col]
                                inputs_single.append(gr.Number(label=label, value=dv, minimum=mn, maximum=mx, step=step, min_width=90))
                    gr.Markdown("#### 공정 및 시험 조건")
                    with gr.Row():
                        for col in COND_COLS:
                            mn, mx, dv, step, label = META[col]
                            if col == 'WQ':
                                inputs_single.append(gr.Dropdown(choices=[1, 0], value=1, label=label, min_width=90))
                            else:
                                inputs_single.append(gr.Number(label=label, value=dv, minimum=mn, maximum=mx, step=step, min_width=110))
                    single_samples = gr.Slider(10, 500, value=DEFAULT_N_SAMPLES, step=10, label="MC 샘플링 횟수")
                    single_btn = gr.Button("단일 예측 실행", variant="primary")
                with gr.Column(scale=2):
                    single_md = gr.HTML(EMPTY_KPI)
                    single_plot = gr.Plot(label="BNN 예측 분포 및 95% 구간")
            single_btn.click(run_single_prediction, inputs=inputs_single + [single_samples],
                             outputs=[single_md, single_plot])

        # TAB 3: 고급 수명 시뮬레이션
        with gr.Tab("고급 수명 시뮬레이션"):
            uploaded_state = gr.State(None)
            _b0 = make_base_row(None, -1)
            with gr.Row(equal_height=False):
                # 왼쪽: 컨트롤 패널
                with gr.Column(scale=1, min_width=340, elem_classes=["panel"]):
                    gr.Markdown("#### 시뮬레이션 설정")
                    sample_dd = gr.Dropdown(choices=[("기본값 (슬라이더 기준)", -1)], value=-1,
                                            label="기준 샘플 (Data Upload 탭에서 올린 파일의 행)")
                    sim_mode = gr.Radio(["온도 영향", "성분 영향", "파손 예상 시간 비교"], label="분석 방식", value="온도 영향")
                    var_choice = gr.Dropdown(SIM_VAR_CHOICES, value="Cr", label="비교 변수 (컬럼)", visible=False)
                    temp_vals = gr.CheckboxGroup(list(range(500, 1101, 50)), value=[600, 700, 800], label="비교 온도 (°C)")
                    compare_vals = gr.Dropdown(choices=_preset_choices(_b0, "Cr"), value=_default_selection(_b0, "Cr"),
                                               multiselect=True, allow_custom_value=True, visible=False,
                                               label="비교 값 - 목록에서 선택하거나, 직접 입력 후 Enter")
                    fixed_temp = gr.Slider(500, 1100, value=650, step=10, label="고정 온도 (°C)", visible=False)
                    fixed_stress = gr.Slider(5, 471, value=130, step=5, label="고정 응력 (MPa)", visible=False)
                    with gr.Row():
                        stress_min = gr.Slider(5, 200, value=50, step=5, label="최소 응력 (MPa)", min_width=120)
                        stress_max = gr.Slider(200, 471, value=300, step=5, label="최대 응력 (MPa)", min_width=120)
                    n_pts = gr.Slider(6, 20, value=10, step=2, label="곡선 포인트 수")
                    adv_btn = gr.Button("시뮬레이션 실행", variant="primary")
                # 오른쪽: 결과 대시보드
                with gr.Column(scale=3):
                    adv_kpi = gr.HTML(EMPTY_KPI)
                    adv_plot = gr.Plot()

            sim_mode.change(toggle_sim_controls, inputs=[sim_mode],
                            outputs=[temp_vals, var_choice, compare_vals, fixed_temp, fixed_stress, stress_min, stress_max, n_pts])

            # 파일 업로드 → 기준 샘플 목록 갱신 → 고정 온도/응력/비교 값 자동 세팅
            ctrl_inputs = [uploaded_state, sample_dd, var_choice]
            ctrl_outputs = [fixed_temp, fixed_stress, compare_vals]
            excel_file.upload(load_uploaded_file, inputs=[excel_file], outputs=[uploaded_state, sample_dd]) \
                      .then(refresh_controls, inputs=ctrl_inputs, outputs=ctrl_outputs)
            sample_dd.change(refresh_controls, inputs=ctrl_inputs, outputs=ctrl_outputs)
            var_choice.change(refresh_controls, inputs=ctrl_inputs, outputs=ctrl_outputs)

            adv_inputs = [sim_mode, temp_vals, var_choice, compare_vals, fixed_temp, fixed_stress,
                          stress_min, stress_max, n_pts, uploaded_state, sample_dd]
            adv_btn.click(run_advanced_simulation, inputs=adv_inputs, outputs=[adv_plot, adv_kpi])

        # TAB 4: 시편 파단 시뮬레이션
        with gr.Tab("시편 파단 시뮬레이션"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=260, elem_classes=["panel"]):
                    gr.Markdown("#### 조건 설정\n시간 경과에 따른 온도별 시편 파단 과정을 애니메이션으로 보여줍니다.")
                    cr_anim = gr.Slider(10.0, 26.0, value=18.0, step=0.5, label="Cr (wt%)")
                    stress_anim = gr.Slider(20, 300, value=130, step=10, label="응력 (MPa)")
                    speed_anim = gr.Dropdown(choices=[100, 1000, 10000, 100000], value=1000, label="시뮬레이션 배속")
                with gr.Column(scale=3):
                    html_out = gr.HTML()
            anim_inputs = [cr_anim, stress_anim, speed_anim]

            demo.load(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            cr_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            stress_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            speed_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])

        # TAB 5: LLM AI 도메인 챗봇
        with gr.Tab("LLM AI 도메인 챗봇"):
            gr.Markdown("### 크리프 현상, 모델 해석, 합금 성분 영향에 대해 질문해보세요")
            chatbot_ui = gr.Chatbot(height=450, placeholder="크리프 현상, 모델 정확도, 신뢰구간에 대해 무엇이든 물어보세요.")
            msg_input = gr.Textbox(placeholder="질문을 입력하고 Enter를 누르세요...", show_label=False)
            clear_btn = gr.ClearButton([msg_input, chatbot_ui], value="대화 초기화")

            msg_input.submit(predict_chat_llm, inputs=[msg_input, chatbot_ui], outputs=[msg_input, chatbot_ui])

if __name__ == "__main__":
    demo.launch(server_port=7861, share=True)