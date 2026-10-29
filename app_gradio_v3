import os
import re
import joblib
import numpy as np
import pandas as pd
import tf_keras as tfk
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import gradio as gr
from html import escape as _esc
from dotenv import load_dotenv
from openai import OpenAI

# TensorFlow Keras 레거시 설정
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# .env 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 1. 모델 및 스케일러 로드 ──────────────────────────────────────────────────
def load_ann():
    models = [tfk.models.load_model(f"model_{i}.keras") for i in range(10)]
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return models, scaler_X, scaler_y

def load_ngb():
    ngb = joblib.load("ngboost_model.pkl")
    scaler_X = joblib.load("scaler_X_ngb.pkl")
    return ngb, scaler_X

ann_models, ann_scaler_X, ann_scaler_y = load_ann()
ngb_model, ngb_scaler_X = load_ngb()

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

def stab_ratio(nb, ti, c, n):
    return (nb / 8 + ti / 4) / (c + n + 1e-7)

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
    """셔플된 엑셀 컬럼을 ALL_COLS(개발 당시 순서)로 정렬.
    1) 정규화 후 정확히 일치 → 2) 별칭 → 3) 남은 컬럼만 부분 일치. 없으면 0으로 채움."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    norm_map = {}
    for c in df.columns:
        norm_map.setdefault(_norm(c), c)

    found_map = {}
    for t in ALL_COLS:                      # 1단계
        if _norm(t) in norm_map:
            found_map[t] = norm_map[_norm(t)]
    used = set(found_map.values())

    for t in ALL_COLS:                      # 2단계
        if t in found_map:
            continue
        for a in COL_ALIASES.get(t, []):
            c = norm_map.get(_norm(a))
            if c is not None and c not in used:
                found_map[t] = c; used.add(c)
                break

    for t in ALL_COLS:                      # 3단계 (C/N/S 같은 짧은 이름은 제외)
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
    for t in ALL_COLS:
        if t in found_map:
            aligned[t] = df[found_map[t]]
        else:
            aligned[t] = 0
            missing_cols.append(t)
    return (aligned, missing_cols, found_map) if return_map else (aligned, missing_cols)

def prepare_uploaded_df(df):
    """업로드 데이터를 정렬 + 숫자 변환 + Stab_ratio 계산. (X, 누락컬럼, 매핑) 반환"""
    X, missing, mapping = align_input_dataframe(df, return_map=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    X[ALLOY_COLS] = X[ALLOY_COLS].fillna(0.0)          # 성분 빈칸은 0(미첨가)으로 처리
    bad = [c for c in COND_COLS if X[c].isna().any()]
    if bad:
        raise ValueError(f"공정/시험 조건 컬럼에 빈 값이나 숫자가 아닌 값이 있습니다: {bad}")
    X['Stab_ratio'] = stab_ratio(X['Nb'], X['Ti'], X['C'], X['N'])
    return X, [c for c in missing if c != 'Stab_ratio'], mapping

# ── 3. 예측 및 분석 함수 ──────────────────────────────────────────────────────
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
    X_fixed, _ = align_input_dataframe(df_rows)
    # Stab_ratio는 Nb/Ti/C/N이 바뀌면 같이 바뀌므로 항상 다시 계산
    X_fixed['Stab_ratio'] = stab_ratio(X_fixed['Nb'], X_fixed['Ti'], X_fixed['C'], X_fixed['N'])
    X_ann_s = ann_scaler_X.transform(X_fixed)
    return predict_ann(X_ann_s)

def sim_predict(stress, temp, cr, ni=12.0, mo=0.1, s_temp=1150):
    row = {
        'Cr': cr, 'Ni': ni, 'Mo': mo, 'Mn': 1.5, 'Si': 0.5, 'Nb': 0.0, 'Ti': 0.03, 'V': 0.0,
        'W': 0.0, 'Cu': 0.1, 'N': 0.05, 'C': 0.06, 'B': 0.0, 'P': 0.03, 'S': 0.008, 'Co': 0.0,
        'Al': 0.005, 'Sn': 0.0, 'S_temp / oC': s_temp, 'log10(S_time / s)': 2.78, 'WQ': 1,
        'temp / oC': temp, 'stress / Mpa': float(stress), 'Stab_ratio': stab_ratio(0.0, 0.03, 0.06, 0.05)
    }
    X = pd.DataFrame([{c: row[c] for c in ALL_COLS}])
    X_s = ngb_scaler_X.transform(X).astype(np.float64)
    dist = ngb_model.pred_dist(X_s)
    return float(dist.loc[0]), float(dist.scale[0])

# ── 4. LLM 챗봇 로직 ─────────────────────────────────────────────────────────
LLM_SYSTEM_PROMPT = """
당신은 '소재부품 크리프(Creep) 수명 예측 시스템'의 AI 도메인 전문가 챗봇입니다.
사용자의 질문에 대해 친절하고 전문적으로 한국어로 답변해 주세요.

[시스템 및 도메인 지식]
1. 시스템 개요:
   - 합금 성분(Cr, Ni, Mo, Nb, Ti 등)과 공정 조건(온도, 응력)을 입력받아 크리프 수명을 예측하고 불확실성을 분석하는 플랫폼입니다.
   - 엑셀 업로드 시 컬럼 순서 및 누락 항목을 자동 정렬 및 기본값(0) 보정하여 처리합니다.
2. 수명 예측 모델 및 성능:
   - Committee ANN: 동일 구조 ANN 10개 앙상블. R²=0.9236, RMSE=0.2570. 정확한 점 추정 수명 도출.
   - NGBoost: Natural Gradient Boosting 기반. R²=0.9155, RMSE=0.2694, 95% Coverage=88.1%. 95% 신뢰구간(μ ± 1.96σ) 출력.
3. 재료공학 지식:
   - 크리프(Creep): 고온·고응력 환경에서 재료가 시간에 따라 서서히 변형되는 현상.
   - 주요 변수: 응력(Stress / MPa, 중요도 약 28%)과 온도(Temp / °C)가 핵심 영향 변수.
   - 주요 성분: Cr(내산화성), Ni(고온 안정성), Nb/Ti(석출 강화).
   - 파생변수: Stab_ratio = (Nb/8 + Ti/4) / (C + N + ε).
"""

def predict_chat_llm(user_message, history):
    if not user_message.strip():
        return "", history
    
    # OpenAI 전달용 메시지 리스트
    api_messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    
    # 이전 히스토리 추가 (dict 형식)
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
        
    # Gradio history 업데이트 (dict 형식)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_response})
    
    return "", history

# ── 5. Gradio 이벤트 처리 함수들 ──────────────────────────────────────────────
# ── 대시보드 공통 요소 ────────────────────────────────────────────
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

HERO_HTML = """
<div class="hero">
  <h1>소재부품 크리프 수명 예측 대시보드</h1>
  <p>Committee ANN 예측 · NGBoost 불확실성 분석 · 고급 수명 시뮬레이션</p>
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

def run_batch_prediction(file):
    if file is None:
        return None, "파일을 업로드해주세요.", "", EMPTY_KPI
    try:
        df = pd.read_excel(_file_path(file))
        df.columns = df.columns.astype(str).str.strip()
        X_fixed, missing_cols, mapping = prepare_uploaded_df(df)

        X_ann_s = ann_scaler_X.transform(X_fixed)
        ann_mean, ann_std = predict_ann(X_ann_s)

        X_ngb_s = ngb_scaler_X.transform(X_fixed)
        ngb_mu, ngb_sig = predict_ngb(X_ngb_s)

        # 결과표: 정렬된 표준 컬럼 순서 → 사용하지 않은 원본 컬럼 → 예측 결과
        extras = [c for c in df.columns if c not in set(mapping.values())]
        res = pd.concat([X_fixed, df[extras]], axis=1)
        res["Predicted Life (log10 h)"] = ann_mean
        res["Predicted Life (hours)"] = 10 ** ann_mean
        res["ANN Uncertainty"] = ann_std
        res["NGBoost μ"] = ngb_mu
        res["NGBoost σ"] = ngb_sig
        res["95% CI Lower (h)"] = 10 ** (ngb_mu - 1.96 * ngb_sig)
        res["95% CI Upper (h)"] = 10 ** (ngb_mu + 1.96 * ngb_sig)

        avg_life = ann_mean.mean()
        if avg_life > 4:
            comment = "**해석**: 전반적으로 수명이 매우 긴 고내열 합금 조건입니다. (10,000시간 이상)"
        elif avg_life > 3:
            comment = "**해석**: 일반적인 고온 환경에서 사용 가능한 수준입니다. (1,000~10,000시간)"
        else:
            comment = "**해석**: 수명이 짧아 조건이 가혹한 상태입니다. (1,000시간 미만)"

        renamed = [f"{src}→{tgt}" for tgt, src in mapping.items() if src != tgt]
        status = f"예측 완료 ({len(df)}행, 컬럼을 표준 순서로 정렬함)"
        if renamed:
            status += f"\n[보정] 이름이 달라서 매칭한 컬럼: {renamed}"
        if missing_cols:
            status += f"\n[주의] 파일에 없어서 0으로 채운 컬럼: {missing_cols}"
        if extras:
            status += f"\n[참고] 예측에 쓰지 않은 컬럼: {extras}"
        kpi = kpi_cards([
            ("분석 샘플 수", f"{len(df):,}개", "업로드한 행 수", "info"),
            ("평균 예측 수명", f"{_fmt_h(10 ** avg_life)} h", f"log10 평균 기준 · {_span(10 ** avg_life)}", ""),
            ("최단 수명", f"{_fmt_h(10 ** ann_mean.min())} h", _span(10 ** ann_mean.min()), "bad"),
            ("최장 수명", f"{_fmt_h(10 ** ann_mean.max())} h", _span(10 ** ann_mean.max()), "good"),
            ("평균 불확실성 σ", f"{float(np.mean(ngb_sig)):.3f}", "NGBoost (log10 h)", "warn"),
        ])
        return res.round(4), status, comment, kpi
    except Exception as e:
        return None, f"오류 발생: {str(e)}", "", EMPTY_KPI

def run_single_prediction(*args):
    inp = {col: args[i] for i, col in enumerate(ALL_COLS[:-1])}
    inp['Stab_ratio'] = stab_ratio(inp['Nb'], inp['Ti'], inp['C'], inp['N'])
    X = pd.DataFrame([{c: inp[c] for c in ALL_COLS}])
    
    X_ann_s = ann_scaler_X.transform(X)
    ann_mu, ann_std_v = predict_ann(X_ann_s)
    ann_mu_val = float(ann_mu[0])
    ann_hours = 10 ** ann_mu_val
    
    X_ngb_s = ngb_scaler_X.transform(X)
    ngb_mu, ngb_sig = predict_ngb(X_ngb_s)
    ngb_mu_val, ngb_sig_val = float(ngb_mu[0]), float(ngb_sig[0])
    lower_h = 10 ** (ngb_mu_val - 1.96 * ngb_sig_val)
    upper_h = 10 ** (ngb_mu_val + 1.96 * ngb_sig_val)
    
    res_md = kpi_cards([
        ("Committee ANN 수명", f"{_fmt_h(ann_hours)} h", f"log10 {ann_mu_val:.3f} · 모델간 σ {float(ann_std_v[0]):.3f}", ""),
        ("NGBoost 중앙값", f"{_fmt_h(10 ** ngb_mu_val)} h", f"불확실성 σ {ngb_sig_val:.3f}", "info"),
        ("95% 신뢰구간 하한", f"{_fmt_h(lower_h)} h", _span(lower_h), "warn"),
        ("95% 신뢰구간 상한", f"{_fmt_h(upper_h)} h", _span(upper_h), "good"),
    ])

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(["NGBoost 95% CI"], [upper_h - lower_h], left=lower_h, color='#9bbbe0', alpha=0.45, height=0.35)
    ax.axvline(ann_hours, color='#2d4466', lw=2.0, label=f"ANN: {ann_hours:,.0f}h")
    ax.axvline(10**ngb_mu_val, color='#5b86bb', lw=2.0, linestyle='--', label=f"NGBoost: {10**ngb_mu_val:,.0f}h")
    ax.set_xlabel("Life (hours)"); ax.set_xscale('log'); ax.legend(fontsize=9); ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    return res_md, fig

SIM_VAR_CHOICES = ALLOY_COLS + ['S_temp / oC', 'log10(S_time / s)']

def _file_path(file):
    # Gradio 버전에 따라 str 경로 또는 tempfile 객체로 들어옴
    return file if isinstance(file, str) else file.name

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
        choices.append((f"샘플 {i} | Cr={r['Cr']:g}, Ni={r['Ni']:g}, {r['temp / oC']:g}°C, {r['stress / Mpa']:g}MPa", i))
    return X, gr.update(choices=choices, value=0 if len(X) else -1)

def make_base_row(df_state, sample_idx):
    """기준 조건: 업로드 파일의 선택된 행, 없으면 META 기본값."""
    row = {c: META[c][2] for c in ALL_COLS if c in META}
    if df_state is not None and sample_idx is not None and 0 <= int(sample_idx) < len(df_state):
        src = df_state.iloc[int(sample_idx)]
        for c in ALL_COLS[:-1]:
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
            for tok in str(item).replace("，", ",").split(","):   # 직접 "12, 18"처럼 입력해도 허용
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
        pred_log, _ = predict_ann_batch_df(pd.DataFrame(rows))
        df_res["life_hours"] = 10 ** pred_log

        fig = px.line(df_res, x="life_hours", y="stress", color="value", markers=True,
                      labels={"life_hours": "Life (hours)", "stress": "Stress (MPa)", "value": "Temperature (°C)"},
                      title=f"온도별 Stress–Life Curve ({src})")
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
        pred_log, _ = predict_ann_batch_df(pd.DataFrame(rows))
        df_res["life_hours"] = 10 ** pred_log

        fig = px.line(df_res, x="life_hours", y="stress", color="value", markers=True,
                      labels={"life_hours": "Life (hours)", "stress": "Stress (MPa)", "value": var_choice},
                      title=f"{var_choice} 변화에 따른 Stress–Life Curve ({src}, {fixed_temp:g}°C)")
        fig.update_xaxes(type="log")
        kpi = _sim_kpis(df_res, lambda r: f"{var_choice}={r['value']} @ {r['stress']:.0f}MPa", len(vals))
        return _style_fig(fig), kpi

    # 파손 예상 시간 비교
    rows, labels = [], []
    for v in vals:
        r = base_row.copy(); r['temp / oC'] = float(fixed_temp); r['stress / Mpa'] = float(fixed_stress); r[var_choice] = float(v)
        rows.append(r); labels.append(f"{var_choice}={v:g}")
    pred_log, _ = predict_ann_batch_df(pd.DataFrame(rows))
    df_res = pd.DataFrame({"label": labels, "life_hours": 10 ** pred_log})

    fig = px.bar(df_res, x="label", y="life_hours", text="life_hours",
                 labels={"label": "Scenario", "life_hours": "Predicted Rupture Time (hours)"},
                 title=f"파손 예상 시간 비교 ({src}, {fixed_temp:g}°C / {fixed_stress:g}MPa)")
    fig.update_yaxes(type="log"); fig.update_traces(texttemplate="%{text:.2e}", textposition="outside")
    kpi = _sim_kpis(df_res, lambda r: r['label'], len(vals))
    return _style_fig(fig), kpi

def update_rupture_html(cr, stress, speed):
    temps = [600, 650, 700, 750]
    sim_lives = {}
    for t in temps:
        try:
            mu, _ = sim_predict(stress, t, cr)
            sim_lives[t] = mu
        except:
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
with gr.Blocks(title="소재부품 크립 수명 예측 시스템", theme=THEME, css=CSS) as demo:
    gr.HTML(HERO_HTML)

    with gr.Tabs():
        # TAB 1: 배치 데이터 업로드 및 예측
        with gr.Tab("Data Upload & Prediction"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes=["panel"]):
                    gr.Markdown("#### 데이터 업로드")
                    excel_file = gr.File(label="엑셀 파일 업로드 (.xlsx)", file_types=[".xlsx"])
                    batch_btn = gr.Button("배치 예측 실행", variant="primary")
                    status_txt = gr.Textbox(label="상태 메시지", lines=4, interactive=False)
                with gr.Column(scale=3):
                    batch_kpi = gr.HTML(EMPTY_KPI)
                    comment_txt = gr.Markdown()
                    res_df_table = gr.Dataframe(label="예측 결과 테이블")
            batch_btn.click(run_batch_prediction, inputs=[excel_file],
                            outputs=[res_df_table, status_txt, comment_txt, batch_kpi])

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
                    single_btn = gr.Button("단일 예측 실행", variant="primary")
                with gr.Column(scale=2):
                    single_md = gr.HTML(EMPTY_KPI)
                    single_plot = gr.Plot(label="95% 신뢰구간")
            single_btn.click(run_single_prediction, inputs=inputs_single, outputs=[single_md, single_plot])

        # TAB 3: 고급 수명 시뮬레이션
        with gr.Tab("고급 수명 시뮬레이션"):
            uploaded_state = gr.State(None)
            _b0 = make_base_row(None, -1)
            with gr.Row(equal_height=False):
                # ── 왼쪽: 컨트롤 패널 ──
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
                # ── 오른쪽: 결과 대시보드 ──
                with gr.Column(scale=3):
                    adv_kpi = gr.HTML(EMPTY_KPI)
                    adv_plot = gr.Plot()

            # 분석 방식에 맞는 컨트롤만 표시
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

            # 초기 로드 및 값 변경 시 자동 업데이트
            demo.load(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            cr_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            stress_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])
            speed_anim.change(update_rupture_html, inputs=anim_inputs, outputs=[html_out])

        # TAB 5: LLM AI 도메인 챗봇
        with gr.Tab("LLM AI 도메인 챗봇"):
            gr.Markdown("### 크리프 현상, 모델 해석, 합금 성분 영향에 대해 질문해보세요!")
            chatbot_ui = gr.Chatbot(height=450, placeholder="크리프 현상, 모델 정확도, 신뢰구간에 대해 무엇이든 물어보세요.")
            msg_input = gr.Textbox(placeholder="질문을 입력하고 Enter를 누르세요...", show_label=False)
            clear_btn = gr.ClearButton([msg_input, chatbot_ui], value="대화 초기화")

            msg_input.submit(predict_chat_llm, inputs=[msg_input, chatbot_ui], outputs=[msg_input, chatbot_ui])

if __name__ == "__main__":
    demo.launch(share=True)
