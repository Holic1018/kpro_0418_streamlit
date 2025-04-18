import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Malgun Gothic'

# 사이드바 옵션은 항상 보이도록 유지
st.sidebar.title("주요 변수")

# 드롭다운 선택
target = st.sidebar.selectbox( "■ 주요 target :",
    ("로그 응집제 주입률"))

# 체크 박스를 이용한 입력 변수 넣는 방식
st.sidebar.write( "■ Input_options 선택하기 : ")
st.sidebar.write( "로그 원수 탁도는 기본으로 선택되어 있습니다.")
input_options = {"원수 탁도", "원수 pH", "원수 알칼리도", "원수 전기전도도",
                 "원수 수온", "3단계 원수 유입 유량", "3단계 1계열 응집제 주입률",
                 "3단계 침전지 탁도", "3단계 침전지 체류시간", "3단계 여과지 탁도"}

selected_input = []
for col in input_options:
    if st.sidebar.checkbox(col):
        selected_input.append(col)

st.sidebar.write( "■ XGboost 변수를 설정하세요")

# 숫자 슬라이더
max_depth_1 = st.sidebar.slider("max_depth:", 0, 20, 1)
n_estimator1 = st.sidebar.slider("n_estimator:", 0, 500, 25)
learnig_rate1 = st.sidebar.slider("learning_rate:", 0.00,1.00 , 0.01)
subsample1 = st.sidebar.slider("subsample:", 0.00, 1.00, 0.01)

# 실행 버튼
run_button = st.sidebar.button("🚀 실행하기")

# 버튼 클릭 시에만 본문 내용이 표시되도록 처리
if run_button:
    st.title(":blue[스마트 정수장 XGboost: 약품 공정]")
    
    df = pd.read_csv("SN_total.csv", encoding='utf-8-sig')

    st.header("선택한 DataFrame:")
    
    if selected_input:
        st.dataframe(df[selected_input])
    else:
        st.warning("⛔ 열을 하나 이상 선택해주세요.")
    
    X = df[list(set(selected_input + ["로그 원수 탁도"]))]  # "로그 원수 탁도"는 항상 포함
    y = df[target]

    Xt, Xts, yt, yts = train_test_split(X, y, test_size=0.2, shuffle=False)

    # XGBoost 모델 훈련
    xg_reg = XGBRegressor(
        max_depth=max_depth_1,
        n_estimators=n_estimator1,
        eta=learnig_rate1,
        subsample=subsample1,
        random_state=2
    )

    # 훈련 및 예측
    xg_reg.fit(Xt, yt)
    yt_pred = xg_reg.predict(Xt)
    yts_pred = xg_reg.predict(Xts)

    # 모델 평가
    st.header("XGboost 평가 항목:")
    mse_train = mean_squared_error(10**yt, 10**yt_pred)
    mse_test = mean_squared_error(10**yts, 10**yts_pred)
    st.write(f"학습 데이터 MSE: {mse_train}")
    st.write(f"테스트 데이터 MSE: {mse_test}")

    r2_train = r2_score(10**yt, 10**yt_pred)
    r2_test = r2_score(10**yts, 10**yts_pred)
    st.write(f"학습 데이터 R2: {r2_train}")
    st.write(f"테스트 데이터 R2: {r2_test}")

    # 그래프 그리기
    st.header("그래프:")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(Xt["로그 원수 탁도"], yt, s=3, label="학습 데이터 (실제)")
    ax.scatter(Xt["로그 원수 탁도"], yt_pred, s=3, label="학습 데이터 (예측)", c="r")
    ax.grid()
    ax.legend(fontsize=13)
    ax.set_xlabel("로그 원수 탁도")
    ax.set_ylabel("로그 응집제 주입률")
    ax.set_title(
        rf"학습 데이터  MSE: {round(mse_train, 4)}, $R^2$: {round(r2_train, 2)}",
        fontsize=18,
    )

    ax = axes[1]
    ax.scatter(Xts["로그 원수 탁도"], yts, s=3, label="테스트 데이터 (실제)")
    ax.scatter(Xts["로그 원수 탁도"], yts_pred, s=3, label="테스트 데이터 (예측)", c="r")
    ax.grid()
    ax.legend(fontsize=13)
    ax.set_xlabel("로그 원수 탁도")
    ax.set_ylabel("로그 응집제 주입률")
    ax.set_title(
        rf"테스트 데이터  MSE: {round(mse_test, 4)}, $R^2$: {round(r2_test, 2)}",
        fontsize=18,
    )

    st.pyplot(fig)

else:
    st.title(":blue[스마트 정수장 XGboost: 약품 공정]")
    st.write("사이드바에서 변수를 선택하고 '실행하기' 버튼을 눌러주세요.")
