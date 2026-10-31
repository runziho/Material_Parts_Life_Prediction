import os

# tf_keras / tensorflow import 보다 먼저 설정해야 적용
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import joblib
import re
import numpy as np
import pandas as pd
import tf_keras as tfk
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

try:
    import tensorflow_probability as tfp  # noqa: F401
except ImportError:
    tfp = None

# .env 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 0. 파일명 설정 ─────────────────────────────────────────────────
PREFIX = "health_bnn"
MODEL_PATH     = f"{PREFIX}_model.keras"
SCALER_X_PATH  = f"{PREFIX}_scaler_X.pkl"
SCALER_Y_PATH  = f"{PREFIX}_scaler_y.pkl"
FEATURES_PATH  = f"{PREFIX}_features.pkl"
LOGVAR_MIN, LOGVAR_MAX = -6.0, 4.0

print("📦 사용 모델: 건강·생활습관 BNN 예측 모델")
DEFAULT_N_SAMPLES = 100

# ── 1. 모델 및 스케일러 로드 ──────────────────────────────────────────
import keras as keras3

@keras3.saving.register_keras_serializable(package="Custom", name="MCDropout")
class MCDropout(keras3.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def load_model_any():
    errors = []
    try:
        model = keras3.models.load_model(
            MODEL_PATH, compile=False,
            custom_objects={"MCDropout": MCDropout},
        )
        print(f"✅ Keras {keras3.__version__} 로 BNN 로드 성공")
        return model
    except Exception as e:
        errors.append(f"[Keras 3] {type(e).__name__}: {e}")

    try:
        model = tfk.models.load_model(MODEL_PATH, compile=False)
        print("✅ tf_keras 로 BNN 로드 성공")
        return model
    except Exception as e:
        errors.append(f"[tf_keras] {type(e).__name__}: {e}")

    raise RuntimeError("건강 BNN 모델 로드 실패:\n" + "\n\n".join(errors))

bnn_model = load_model_any()
bnn_scaler_X = joblib.load(SCALER_X_PATH)
bnn_scaler_y = joblib.load(SCALER_Y_PATH) if os.path.exists(SCALER_Y_PATH) else None

# 학습된 피처 순서 로드
if os.path.exists(FEATURES_PATH):
    raw_cols = joblib.load(FEATURES_PATH)
    EXCLUDE_TARGETS = ['health_risk', 'bmi']
    HEALTH_COLS = [str(c).strip() for c in raw_cols if str(c).strip() not in EXCLUDE_TARGETS]
    print("✅ 단일/배치 예측용 폼 피처 적용:", HEALTH_COLS)
else:
    HEALTH_COLS = ['age', 'weight', 'height', 'exercise', 'sleep', 'sugar_intake', 'smoking', 'alcohol', 'married', 'profession']
    print("ℹ️ features.pkl 없음 → 기본 컬럼 사용")

def align_input_dataframe(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    norm_map = {str(c).strip().lower(): c for c in df.columns}

    aligned = pd.DataFrame(index=df.index)
    missing_cols = []
    for t in HEALTH_COLS:
        t_lower = t.lower()
        if t_lower in norm_map:
            col_name = norm_map[t_lower]
            val = df[col_name]
            if val.dtype == 'object':
                aligned[t] = pd.factorize(val.astype(str))[0].astype(float)
            else:
                aligned[t] = pd.to_numeric(val, errors="coerce").fillna(0.0)
        else:
            aligned[t] = 0.0
            missing_cols.append(t)
    return aligned, missing_cols

def scale_X(df):
    return bnn_scaler_X.transform(df[HEALTH_COLS].values)

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

# ── 4. LLM 챗봇 로직 ────────────────────────────────────────────────
LLM_SYSTEM_PROMPT = """
당신은 '건강·생활습관 및 질환 위험도 예측 시스템'의 AI 헬스케어 도메인 전문가 챗봇입니다.
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

# ── 5. 대시보드 공통 요소 및 디자인 ──────────────────────────────────
CSS = """
.hero{background:#e4eff9;border:1px solid #c9dcef;border-radius:12px;padding:16px 22px;margin-bottom:8px}
.hero h1{margin:0;font-size:1.3rem;font-weight:700;color:#2d4466}
.hero p{margin:4px 0 0;font-size:.85rem;color:#46699a}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:2px 0 10px}
.kpi{background:var(--background-fill-secondary);border:1px solid #d6e4f2;border-left:4px solid #5b86bb;
     border-radius:10px;padding:12px 14px}
.kpi .lbl{font-size:.75rem;color:var(--body-text-color-subdued)}
.kpi .val{font-size:1.35rem;font-weight:700;line-height:1.3;color:#2d4466}
.kpi .sub{font-size:.73rem;color:var(--body-text-color-subdued);margin-top:2px}
.kpi.info{border-left-color:#7fb8dc}.kpi.good{border-left-color:#6aa6a0}
.kpi.warn{border-left-color:#d0a85c}.kpi.bad{border-left-color:#c98585}
.kpi-empty{border:1px dashed #c9dcef;border-radius:10px;padding:14px;margin:2px 0 10px;
           color:var(--body-text-color-subdued);text-align:center;font-size:.9rem}
.panel{background:var(--block-background-fill);border:1px solid #d6e4f2;border-radius:12px;padding:14px !important}
"""

HERO_HTML = """
<div class="hero">
  <h1>건강·생활습관 및 위험도 예측 시스템 (BNN)</h1>
  <p>Bayesian Neural Network 기반 개인별 건강 지표 예측 및 95% 신뢰구간 불확실성 분석 플랫폼</p>
</div>
"""

EMPTY_KPI = '<div class="kpi-empty">실행하면 핵심 지표가 여기에 표시됩니다.</div>'

def kpi_cards(items):
    cards = "".join(
        f'<div class="kpi {tone}"><div class="lbl">{str(lbl)}</div>'
        f'<div class="val">{str(val)}</div><div class="sub">{str(sub)}</div></div>'
        for lbl, val, sub, tone in items)
    return f'<div class="kpi-grid">{cards}</div>'

def _style_fig(fig):
    fig.update_layout(template="plotly_white", height=450, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.18), font=dict(color="#2d4466"),
                      plot_bgcolor="#fbfdff",
                      colorway=["#2d4466", "#5b86bb", "#7fb8dc", "#a8c4e2", "#46699a", "#6aa6a0", "#d0a85c", "#c98585"])
    return fig

# ── 6. Gradio 이벤트 처리 함수들 ──────────────────────────────────────
def run_batch_prediction(file, n_samples):
    if file is None:
        return None, "파일을 업로드해주세요.", "", EMPTY_KPI, None, None
    try:
        path = file if isinstance(file, str) else file.name
        df = pd.read_excel(path)
        X_fixed, missing_cols = align_input_dataframe(df)

        X_s = scale_X(X_fixed)
        mean, std, samples = predict_bnn(X_s, int(n_samples))
        lo = np.percentile(samples, 2.5, axis=0)
        hi = np.percentile(samples, 97.5, axis=0)

        sort_idx = np.argsort(mean)
        mean_s, std_s, lo_s, hi_s = mean[sort_idx], std[sort_idx], lo[sort_idx], hi[sort_idx]

        res = df.iloc[sort_idx].copy().reset_index(drop=True)
        res["BNN Predicted Value"] = mean_s
        res["BNN Uncertainty (Std)"] = std_s
        res["95% CI Lower"] = lo_s
        res["95% CI Upper"] = hi_s

        avg_val = float(mean_s.mean())
        if avg_val > 25:
            comment = "**👉 해석**: 분석된 집단의 평균 예측 수치가 다소 높은 편입니다. 생활습관 개선 및 지속적인 모니터링이 권장됩니다."
        else:
            comment = "**👉 해석**: 분석된 집단의 건강 지표가 안정적인 범위 내에 분포하고 있습니다."

        cards = [
            ("분석 샘플 수", f"{len(df):,}개", "업로드한 행 수", "info"),
            ("평균 예측값", f"{avg_val:.3f}", f"표준편차 σ: {std_s.mean():.3f}", ""),
            ("최소 예측값", f"{mean_s.min():.3f}", "하한 지표", "bad"),
            ("최대 예측값", f"{mean_s.max():.3f}", "상한 지표", "good"),
        ]
        kpi = kpi_cards(cards)

        idx = np.arange(1, len(mean_s) + 1)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=idx, y=mean_s, mode="markers", name="BNN 예측 (평균)",
            marker=dict(size=8, color="#46699a"),
            error_y=dict(type="data", symmetric=False, array=hi_s - mean_s, arrayminus=mean_s - lo_s,
                         color="rgba(70,105,154,0.4)", thickness=1.5, width=3),
        ))
        fig1.update_layout(title="샘플별 BNN 예측값 오름차순 정렬 및 95% 구간",
                           xaxis_title="Sorted Sample Index", yaxis_title="Predicted Value")
        _style_fig(fig1)

        fig2 = px.histogram(x=std_s, nbins=20, labels={"x": "Std"},
                            title="BNN 불확실성(Std) 분포", color_discrete_sequence=["#5b86bb"])
        _style_fig(fig2)

        status = f"✅ 예측 완료 (행 수: {len(df)}, MC 샘플: {int(n_samples)}회)"
        if missing_cols:
            status += f"\n[주의] 누락되어 0으로 처리된 컬럼: {missing_cols}"
        return res.round(4), status, comment, kpi, fig1, fig2
    except Exception as e:
        return None, f"❌ 오류 발생: {str(e)}", "", EMPTY_KPI, None, None

def run_single_prediction(*args):
    n_samples = int(args[-1])
    vals = args[:-1]
    
    inp = {}
    for i, col in enumerate(HEALTH_COLS):
        val = vals[i] if i < len(vals) else 0.0
        if isinstance(val, str):
            inp[col] = float(abs(hash(val)) % 100) / 10.0
        else:
            inp[col] = float(val)

    X = pd.DataFrame([inp])
    X_s = scale_X(X)
    r = predict_bnn_full(X_s, n_samples)
    mean, std, samples = r["mean"], r["std"], r["samples"]
    mu, sd = float(mean[0]), float(std[0])
    s = samples[:, 0]
    lo_val, hi_val = np.percentile(s, [2.5, 97.5])

    # 🎯 예측 수치 기반 건강 상태 자동 진단
    if mu < 18.5:
        risk_level = "🟢 저체중 주의군"
        risk_desc = "예측된 수치가 다소 낮습니다. 균형 잡힌 영양 섭취와 체중 관리가 필요합니다."
        tone = "warn"
    elif 18.5 <= mu < 23:
        risk_level = "🟢 정상 범위 (안정)"
        risk_desc = "현재 생활습관과 신체 조건이 매우 안정적입니다. 이 상태를 유지하세요!"
        tone = "good"
    elif 23 <= mu < 25:
        risk_level = "🟡 과체중 관심군"
        risk_desc = "수치가 주의 단계입니다. 유산소 운동과 가벼운 식단 조절을 권장합니다."
        tone = "warn"
    else:
        risk_level = "🔴 비만 및 건강 고위험군"
        risk_desc = "예측 수치가 높게 나타났습니다. 대사 질환 예방을 위한 전문적 관리가 필요합니다."
        tone = "bad"

    cards = [
        ("BNN 예측 결과 (BMI)", f"{mu:.3f}", f"표준편차 σ: {sd:.3f}", ""),
        ("상태 진단", f"{risk_level}", risk_desc, tone),
        ("95% 신뢰구간", f"{lo_val:.3f} ~ {hi_val:.3f}", "예측 범위 (하한~상한)", "info"),
        ("모델 불확실성 σ", f"{sd:.3f}", f"MC 샘플 {n_samples}회 기반", "info"),
    ]
    kpi = kpi_cards(cards)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(s.min() * 0.98, s.max() * 1.02 + 1e-9, 25)
    ax.hist(s, bins=bins, color='#9bbbe0', alpha=0.85, label="BNN sampled predictions")
    ax.axvspan(lo_val, hi_val, color='#c9dcef', alpha=0.45, label="95% interval")
    ax.axvline(mu, color='#2d4466', lw=2.0, label=f"Mean: {mu:.3f}")
    ax.set_xlabel("Predicted Value (BMI)"); ax.set_ylabel("Count")
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, frameon=False); ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return kpi, fig

# ── 7. Gradio 전체 화면 구성 ──────────────────────────────────────────
NAVY = gr.themes.Color(c50="#f1f6fb", c100="#e3edf7", c200="#c9dcef", c300="#a8c4e2", c400="#7ea4d0",
                     c500="#5b86bb", c600="#46699a", c700="#38557d", c800="#2d4466", c900="#233553", c950="#1a2840", name="navy")
THEME = gr.themes.Soft(primary_hue=NAVY, secondary_hue="sky", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "sans-serif"]).set(
    body_background_fill="#f3f8fc",
    button_primary_background_fill="#46699a",
    button_primary_background_fill_hover="#38557d",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(title="건강·생활습관 BNN 예측 시스템", theme=THEME, css=CSS) as demo:
    gr.HTML(HERO_HTML)

    with gr.Tabs():
        with gr.Tab("📂 Data Upload & Prediction"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes=["panel"]):
                    gr.Markdown("#### 건강 데이터 업로드")
                    excel_file = gr.File(label="엑셀 파일 업로드 (.xlsx)", file_types=[".xlsx"])
                    batch_samples = gr.Slider(10, 300, value=50, step=10, label="MC 샘플링 횟수")
                    batch_btn = gr.Button("배치 예측 실행", variant="primary")
                    status_txt = gr.Textbox(label="상태 메시지", lines=4, interactive=False)
                with gr.Column(scale=3):
                    batch_kpi = gr.HTML(EMPTY_KPI)
                    comment_md = gr.Markdown()
                    with gr.Row():
                        batch_plot1 = gr.Plot(label="샘플별 예측 및 95% 구간")
                        batch_plot2 = gr.Plot(label="불확실성 분포")
            res_df_table = gr.Dataframe(label="예측 결과 테이블")
            batch_btn.click(run_batch_prediction, inputs=[excel_file, batch_samples],
                            outputs=[res_df_table, status_txt, comment_md, batch_kpi, batch_plot1, batch_plot2])

        with gr.Tab("🔬 단일 샘플 예측"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_classes=["panel"]):
                    gr.Markdown("#### 개인 건강 및 생활습관 조건 입력")
                    with gr.Row():
                        in_age = gr.Number(label="나이 (Age)", value=40, minimum=10, maximum=100, step=1)
                        in_weight = gr.Number(label="체중 (Weight, kg)", value=65, minimum=30, maximum=150, step=0.5)
                        in_height = gr.Number(label="키 (Height, cm)", value=170, minimum=130, maximum=200, step=0.5)
                    with gr.Row():
                        in_exercise = gr.Dropdown(choices=["none", "low", "medium", "high"], value="medium", label="운동 빈도/강도 (Exercise)")
                        in_sleep = gr.Number(label="수면 시간 (Sleep, h)", value=7, minimum=1, maximum=15, step=0.5)
                        in_sugar = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label="당 섭취 수준 (Sugar Intake)")
                    with gr.Row():
                        in_smoking = gr.Dropdown(choices=["no", "yes"], value="no", label="흡연 여부 (Smoking)")
                        in_alcohol = gr.Dropdown(choices=["no", "yes"], value="no", label="음주 여부 (Alcohol)")
                        in_married = gr.Dropdown(choices=["no", "yes"], value="yes", label="결혼 여부 (Married)")
                    with gr.Row():
                        in_profession = gr.Dropdown(choices=["office_worker", "teacher", "artist", "farmer", "driver", "engineer", "student", "doctor"], value="office_worker", label="직업군 (Profession)")
                    
                    single_samples = gr.Slider(10, 500, value=DEFAULT_N_SAMPLES, step=10, label="MC 샘플링 횟수")
                    single_btn = gr.Button("단일 예측 실행", variant="primary")
                with gr.Column(scale=2):
                    single_md = gr.HTML(EMPTY_KPI)
                    single_plot = gr.Plot(label="BNN 예측 분포 및 95% 구간")

            single_inputs = [in_age, in_weight, in_height, in_exercise, in_sleep, in_sugar, in_smoking, in_alcohol, in_married, in_profession, single_samples]
            single_btn.click(run_single_prediction, inputs=single_inputs, outputs=[single_md, single_plot])

        with gr.Tab("🤖 LLM AI 헬스케어 챗봇"):
            gr.Markdown("### 💬 건강 지표 예측, 생활습관 개선, 모델 불확실성에 대해 무엇이든 물어보세요!")
            chatbot_ui = gr.Chatbot(height=450, placeholder="생활습관 데이터가 건강 지표에 미치는 영향 등에 대해 질문해보세요.")
            msg_input = gr.Textbox(placeholder="질문을 입력하고 Enter를 누르세요...", show_label=False)
            clear_btn = gr.ClearButton([msg_input, chatbot_ui], value="대화 초기화")

            msg_input.submit(predict_chat_llm, inputs=[msg_input, chatbot_ui], outputs=[msg_input, chatbot_ui])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)