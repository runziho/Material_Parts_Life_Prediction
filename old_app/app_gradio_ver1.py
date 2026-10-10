import os
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

def align_input_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    aligned = pd.DataFrame()
    missing_cols = []
    for target in ALL_COLS:
        found = None
        for c in df.columns:
            if c == target or c.lower() == target.lower() or target.lower() in c.lower() or c.lower() in target.lower():
                found = c
                break
        if found:
            aligned[target] = df[found]
        else:
            aligned[target] = 0
            missing_cols.append(target)
    return aligned, missing_cols

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
    df_rows = df_rows.copy()
    if "Stab_ratio" not in df_rows.columns:
        df_rows["Stab_ratio"] = stab_ratio(df_rows["Nb"], df_rows["Ti"], df_rows["C"], df_rows["N"])
    X_fixed, _ = align_input_dataframe(df_rows)
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
def run_batch_prediction(file):
    if file is None:
        return None, "파일을 업로드해주세요.", ""
    try:
        df = pd.read_excel(file.name)
        X_fixed, missing_cols = align_input_dataframe(df)
        X_fixed['Stab_ratio'] = stab_ratio(X_fixed['Nb'], X_fixed['Ti'], X_fixed['C'], X_fixed['N'])
        
        X_ann_s = ann_scaler_X.transform(X_fixed)
        ann_mean, ann_std = predict_ann(X_ann_s)
        
        X_ngb_s = ngb_scaler_X.transform(X_fixed)
        ngb_mu, ngb_sig = predict_ngb(X_ngb_s)
        
        res = df.copy()
        res["Predicted Life (log10 h)"] = ann_mean
        res["Predicted Life (hours)"] = 10 ** ann_mean
        res["ANN Uncertainty"] = ann_std
        res["NGBoost μ"] = ngb_mu
        res["NGBoost σ"] = ngb_sig
        res["95% CI Lower (h)"] = 10 ** (ngb_mu - 1.96 * ngb_sig)
        res["95% CI Upper (h)"] = 10 ** (ngb_mu + 1.96 * ngb_sig)
        
        avg_life = ann_mean.mean()
        if avg_life > 4:
            comment = "👉 **해석**: 전반적으로 수명이 매우 긴 고내열 합금 조건입니다. (10,000시간 이상)"
        elif avg_life > 3:
            comment = "👉 **해석**: 일반적인 고온 환경에서 사용 가능한 수준입니다. (1,000~10,000시간)"
        else:
            comment = "👉 **해석**: 수명이 짧아 조건이 가혹한 상태입니다. (1,000시간 미만)"
            
        status = f"✅ 예측 완료 (자동 보정 컬럼: {missing_cols if missing_cols else '없음'})"
        return res.round(4), status, comment
    except Exception as e:
        return None, f"❌ 오류 발생: {str(e)}", ""

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
    
    res_md = f"""
    ### 🧠 예측 결과
    * **Committee ANN 수명**: **{ann_hours:,.0f} 시간** (log10: `{ann_mu_val:.4f}`, 모델간 σ: `{float(ann_std_v[0]):.4f}`)
    * **NGBoost 95% 신뢰구간**: **{lower_h:,.0f} ~ {upper_h:,.0f} 시간** (불확실성 σ: `{ngb_sig_val:.4f}`)
    """
    
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(["NGBoost 95% CI"], [upper_h - lower_h], left=lower_h, color='steelblue', alpha=0.35, height=0.35)
    ax.axvline(ann_hours, color='tomato', lw=2.0, label=f"ANN: {ann_hours:,.0f}h")
    ax.axvline(10**ngb_mu_val, color='steelblue', lw=2.0, linestyle='--', label=f"NGBoost: {10**ngb_mu_val:,.0f}h")
    ax.set_xlabel("Life (hours)"); ax.set_xscale('log'); ax.legend(fontsize=9); ax.grid(True, axis='x', linestyle='--', alpha=0.4)
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
        pred_log, _ = predict_ann_batch_df(pd.DataFrame(rows))
        df_res["life_hours"] = 10 ** pred_log
        
        fig = px.line(df_res, x="life_hours", y="stress", color="value", markers=True,
                      labels={"life_hours": "Life (hours)", "stress": "Stress (MPa)", "value": "Temperature (°C)"},
                      title="온도별 Stress–Life Curve")
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
        pred_log, _ = predict_ann_batch_df(pd.DataFrame(rows))
        df_res["life_hours"] = 10 ** pred_log
        
        fig = px.line(df_res, x="life_hours", y="stress", color="value", markers=True,
                      labels={"life_hours": "Life (hours)", "stress": "Stress (MPa)", "value": var_choice},
                      title=f"{var_choice} 변화에 따른 Stress–Life Curve")
        fig.update_xaxes(type="log")
        return fig

    else:
        if not compare_vals: return go.Figure()
        rows, labels = [], []
        for v in compare_vals:
            r = base_row.copy(); r['temp / oC'] = fixed_temp; r['stress / Mpa'] = float(fixed_stress); r[var_choice] = float(v)
            rows.append(r); labels.append(f"{var_choice}={v}")
        pred_log, _ = predict_ann_batch_df(pd.DataFrame(rows))
        df_res = pd.DataFrame({"label": labels, "life_hours": 10 ** pred_log})
        
        fig = px.bar(df_res, x="label", y="life_hours", text="life_hours",
                     labels={"label": "Scenario", "life_hours": "Predicted Rupture Time (hours)"}, title="파손 예상 시간 비교")
        fig.update_yaxes(type="log"); fig.update_traces(texttemplate="%{text:.2e}", textposition="outside")
        return fig

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
with gr.Blocks(title="소재부품 크립 수명 예측 시스템", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚙️ 소재부품 크리프 수명 예측 및 불확실성 분석 플랫폼")

    with gr.Tabs():
        # TAB 1: 📂 배치 데이터 업로드 및 예측
        with gr.Tab("📂 Data Upload & Prediction"):
            gr.Markdown("### 엑셀 데이터를 업로드하고 배치 수명 예측을 실행합니다.")
            excel_file = gr.File(label="엑셀 파일 업로드 (.xlsx)", file_types=[".xlsx"])
            batch_btn = gr.Button("🚀 배치 예측 실행", variant="primary")
            status_txt = gr.Textbox(label="상태 메시지")
            res_df_table = gr.Dataframe(label="예측 결과 테이블")
            comment_txt = gr.Markdown()
            batch_btn.click(run_batch_prediction, inputs=[excel_file], outputs=[res_df_table, status_txt, comment_txt])

        # TAB 2: 🔬 단일 샘플 예측
        with gr.Tab("🔬 단일 샘플 예측"):
            gr.Markdown("### 합금 성분 및 열처리/시험 조건을 직접 입력하여 정밀 예측을 수행합니다.")
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
            
            single_btn = gr.Button("🚀 단일 예측 실행", variant="primary")
            single_md = gr.Markdown()
            single_plot = gr.Plot(label="95% 신뢰구간")
            single_btn.click(run_single_prediction, inputs=inputs_single, outputs=[single_md, single_plot])

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
            
            # 초기 로드 및 값 변경 시 자동 업데이트
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
