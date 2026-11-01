import os
import subprocess
import sys
import time
import gradio as gr

# ==============================================================================
# 1. 각 도메인별 Gradio 서버 백그라운드 자동 실행 (포트 분리)
# ==============================================================================
PYTHON_CMD = sys.executable  # 현재 파이썬 가상환경/실행 파일 경로 자동 사용

print("🚀 [자동 런처] 소재부품 크리프 수명 예측 서버(포트 7861) 구동 중...")
creep_process = subprocess.Popen([PYTHON_CMD, "app_gradio_creep.py"])

print("🚀 [자동 런처] 건강·생활습관 리스크 예측 서버(포트 7860) 구동 중...")
health_process = subprocess.Popen([PYTHON_CMD, "app_gradio_health.py"])

print("🚀 [자동 런처] 교통 통행속도 & 정체 불확실성 예측 서버(포트 7862) 구동 중...")
traffic_process = subprocess.Popen([PYTHON_CMD, "app_gradio_traffic.py"])

# 각 서버들이 안정적으로 구동될 때까지 대기
time.sleep(4)

# ==============================================================================
# 2. UI 테마 및 커스텀 CSS 스타일 정의
# ==============================================================================
NAVY = gr.themes.Color(
    c50="#f1f6fb", c100="#e3edf7", c200="#c9dcef", c300="#a8c4e2", c400="#7ea4d0",
    c500="#5b86bb", c600="#46699a", c700="#38557d", c800="#2d4466", c900="#233553", c950="#1a2840", name="navy"
)
THEME = gr.themes.Soft(
    primary_hue=NAVY, secondary_hue="sky", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "sans-serif"]
)

CSS = """
.main-hero { 
    background: linear-gradient(135deg, #2d4466 0%, #46699a 100%); 
    color: white !important; 
    border-radius: 16px; 
    padding: 30px; 
    text-align: center; 
    margin-bottom: 20px; 
}
.main-hero h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; color: white !important; }
.main-hero p { font-size: 1rem; opacity: 0.95; color: white !important; }

.card-box { 
    background: #ffffff !important; 
    border: 1px solid #d6e4f2 !important; 
    border-radius: 16px !important; 
    padding: 25px !important; 
    text-align: center; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
}
.card-box h3 { color: #2d4466 !important; font-size: 1.3rem; font-weight: 700; margin-bottom: 10px; }
.card-box p { color: #46699a !important; font-size: 0.95rem; margin-bottom: 20px; line-height: 1.5; }
"""

# ==============================================================================
# 3. 메인 허브 웹 화면 구성
# ==============================================================================
with gr.Blocks(title="BNN 멀티 도메인 AI 예측 플랫폼", css=CSS) as demo:
    
    gr.HTML("""
    <div class="main-hero">
        <h1>Bayesian Neural Network (BNN) 통합 AI 예측 플랫폼</h1>
        <p>불확실성 정량화(95% 신뢰구간) 기반 소재부품, 헬스케어, 교통 정체 AI 정밀 예측 시스템</p>
    </div>
    """)

    gr.Markdown("### 원하시는 분석 도메인을 선택하세요")
    gr.Markdown("아래 버튼을 클릭하시면 새 탭으로 해당 예측 시스템이 연결됩니다.")
    
    with gr.Row(equal_height=True):
        # 1. 크립 수명 예측 카드 (포트 7861)
        with gr.Column(elem_classes=["card-box"]):
            gr.Markdown("### ⚙️ 소재부품 크립 수명 예측")
            gr.Markdown("고내열 합금 소재의 고온 크립 수명 예측, 95% 신뢰구간 불확실성 분석, 고급 시뮬레이션 및 파단 애니메이션 제공")
            gr.HTML("""
                <div style="margin-top: 15px;">
                    <a href="http://127.0.0.1:7861" target="_blank" style="background-color: #46699a; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block; box-shadow: 0 3px 6px rgba(0,0,0,0.1);">
                        ⚙️ 크리프 수명 예측으로 이동 ➔
                    </a>
                </div>
            """)
        
        # 2. 건강 예측 카드 (포트 7860)
        with gr.Column(elem_classes=["card-box"]):
            gr.Markdown("### 🩺 건강·생활습관 위험도 예측")
            gr.Markdown("개인 신체 조건 및 생활습관 기반 건강 지표(BMI 등) 예측, 95% 신뢰구간 정량화 및 맞춤형 상태 진단 가이드 제공")
            gr.HTML("""
                <div style="margin-top: 15px;">
                    <a href="http://127.0.0.1:7860" target="_blank" style="background-color: #5b86bb; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block; box-shadow: 0 3px 6px rgba(0,0,0,0.1);">
                        🩺 건강 예측으로 이동 ➔
                    </a>
                </div>
            """)

        # 3. 교통 통행속도 & 정체 예측 카드 (포트 7862)
        with gr.Column(elem_classes=["card-box"]):
            gr.Markdown("### 🚗 도로 통행 & 정체 불확실성 예측")
            gr.Markdown("DSRC 센서 공공데이터 기반 도로 구간별 예상 통행속도 및 출퇴근/우천 시 정체 불확실성(Std) 예측 시스템 제공")
            gr.HTML("""
                <div style="margin-top: 15px;">
                    <a href="http://127.0.0.1:7862" target="_blank" style="background-color: #38557d; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block; box-shadow: 0 3px 6px rgba(0,0,0,0.1);">
                        🚗 교통 정체 예측으로 이동 ➔
                    </a>
                </div>
            """)

# ==============================================================================
# 4. 서버 실행 및 종료 처리
# ==============================================================================
if __name__ == "__main__":
    try:
        # 메인 허브는 7880 포트로 구동 (theme 매개변수는 launch에 전달)
        demo.launch(server_port=7880, share=True, theme=THEME)
    finally:
        # 메인 창 종료 시 백그라운드 서버 프로세스 깔끔하게 정돈
        creep_process.terminate()
        health_process.terminate()
        traffic_process.terminate()
