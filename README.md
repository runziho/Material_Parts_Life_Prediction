# 🧪 Material Parts Life Prediction

### 소재부품 수명 예측 및 설계 지원 AI 시스템
👉 합금 성분을 바꿔보면서 수명 변화를 직접 확인할 수 있는 크립 수명 예측 시스템

## 📸 Demo
### 🔹 메인 화면

![Home](Home.png)

### 🔹 데이터 업로드 & 예측

![Result](Result.png)

### 🔹 시뮬레이션

*GIF추가하기

### 💡 Quick Preview
데이터 넣으면 수명 바로 예측됨
성분 바꾸면 결과 실시간으로 바뀜
데이터 형식 안 맞아도 자동으로 처리됨

### 🧠 핵심 포인트
단순 예측 모델 ❌
직접 값 바꿔보면서 확인 가능한 구조 ⭕


## 👥 Team 올클

강다영 · 이혜원 · 최지호

---

## 📌 Project Overview

고온·고압 환경에서 사용되는 내열강 및 초내열합금의 크립 수명 예측 시스템 설계.

합금 성분, 온도, 응력 조건을 입력으로 활용하여 수명 예측 수행.
단순 예측을 넘어 성분 변경에 따른 수명 변화를 확인할 수 있는 시뮬레이션 기능 포함.

엑셀 데이터의 컬럼 순서 및 일부 누락 문제를 고려하여 자동 정렬 및 보정 기능 구현.

---

## 🔄 Workflow

![workflow](workflow.png)

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras (ANN 앙상블 모델)
* NGBoost (불확실성 분석)
* Pandas / NumPy / Scikit-learn
* Matplotlib
* Streamlit

---

## 🚀 Key Features

* **수명 예측**
  → 합금 성분 및 조건 기반 크립 수명 예측 수행

* **불확실성 분석**
  → NGBoost 활용, 예측값과 함께 신뢰구간 산출

* **데이터 자동 처리**
  → 컬럼 순서 정렬 및 누락값 보정 기능 구현

* **시간 단위 변환**
  → log10 기반 결과를 실제 시간(hour)으로 변환

* **합금 시뮬레이션**
  → 성분 변화에 따른 수명 변화 실시간 반영

* **결과 비교 및 시각화**
  → 기존 조건과 변경 조건 비교 및 그래프 제공

* **AI Assistant**
  → 주요 개념 설명을 위한 키워드 기반 응답 기능 구현

---

## ▶️ How to Run

### 1. 데이터 준비

`Data/creep_austenite.xlsx` 파일 저장

### 2. 필요 패키지 설치 및 모델 학습

```bash
py -3.11 -m pip install numpy pandas openpyxl scikit-learn joblib tensorflow tf_keras tensorflow-probability ngboost matplotlib plotly streamlit
python train_ANN.py
python train_NGBoost.py
```

학습 완료 후 모델 및 스케일러 파일 생성됨

### 3. 웹 서비스 실행

```bash
streamlit run app_ver4.py
```

웹 UI를 통해 데이터 업로드, 예측, 결과 확인, 시뮬레이션 기능 사용 가능

---

## 📊 Result

* ANN 앙상블 기반 수명 예측 성능 확보
* NGBoost 적용을 통한 불확실성 정량화
* 다양한 입력 데이터 조건에서도 안정적 동작 확인

---

## 🧠 Summary

수명 예측, 불확실성 분석, 시뮬레이션 기능을 통합한 AI 기반 설계 지원 시스템 구현
