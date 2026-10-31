import os
import subprocess
import time
import gradio as gr

# 1. 포트 충돌이 나지 않도록 각각 다른 포트로 백그라운드 자동 실행
print("🚀 [자동 런처] 크립 수명 예측 서버(포트 7861)를 구동합니다...")
creep_process = subprocess.Popen(["py", "-3.11", "app_gradio_creep.py"])

print("🚀 [자동 런처] 건강 예측 서버(포트 7860)를 구동합니다...")
health_process = subprocess.Popen(["py", "-3.11", "app_gradio_health.py"])

# 서버들이 안정적으로 켜질 때까지 4초 대기
time.sleep(4)

# 2. 예쁜 UI 카드 홈 화면 구성
NAVY = gr.themes.Color(c50="#f1f6fb", c100="#e3edf7", c200="#c9dcef", c300="#a8c4e2", c400="#7ea4d0",
                     c500="#5b86bb", c600="#46699a", c700="#38557d", c800="#2d4466", c900="#233553", c950="#1a2840", name="navy")
THEME = gr.themes.Soft(primary_hue=NAVY, secondary_hue="sky", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "sans-serif"])

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

with gr.Blocks(title="BNN 멀티 도메인 AI 예측 플랫폼") as demo:
    
    gr.HTML("""
    <div class="main-hero">
        <h1>Bayesian Neural Network (BNN) 통합 AI 예측 플랫폼</h1>
        <p>불확실성 정량화(95% 신뢰구간) 기반 소재부품 수명 및 헬스케어 리스크 정밀 예측 시스템</p>
    </div>
    """)

    gr.Markdown("### 🚀 원하시는 분석 도메인을 선택하세요")
    gr.Markdown("아래 버튼을 클릭하시면 새 탭으로 안전하게 연결됩니다.")
    
    with gr.Row(equal_height=True):
        # 크립 수명 예측 링크 카드 (포트 7861)
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
        
        # 건강 예측 링크 카드 (포트 7860)
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

if __name__ == "__main__":
    try:
        # 메인 허브는 7880 포트로 실행
        demo.launch(server_port=7880, share=True, theme=THEME, css=CSS)
    finally:
        # 메인 창을 닫으면 백그라운드 서버들도 함께 깔끔하게 종료됨
        creep_process.terminate()
        health_process.terminate()