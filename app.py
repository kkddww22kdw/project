import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 깨짐 방지 설정
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# 페이지 설정
st.set_page_config(
    page_title="슬기로운 의사 생활 - 초미세 먼지 프로젝트",
    page_icon="🏥",
    layout="wide"
)

# 제목
st.title("🏥 슬기로운 의사 생활: 초미세 먼지와 환자 수 분석")
st.write(
    "초미세 먼지 농도에 따른 호흡기 환자 수 변화를 분석하는 "
    "데이터 과학 웹 애플리케이션입니다."
)


# 데이터 로드 함수
@st.cache_data
def load_data():
    df = pd.read_csv("ultrafine_dust_data.csv")
    return df


try:
    # 데이터 불러오기
    df = load_data()

    # 1. 데이터셋 확인
    st.subheader("1. 📊 수집된 데이터셋 확인")
    st.dataframe(df.head())

    # 2. 기술 통계
    st.subheader("2. 📈 데이터 기술 통계")
    st.dataframe(df.describe())

    # 3. 산점도
    st.subheader("3. 📉 초미세 먼지 농도와 환자 수의 관계")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df["ultrafine_dust"],
        df["patients"],
        color="steelblue",
        alpha=0.8,
        edgecolors="black"
    )

    ax.set_xlabel("초미세 먼지 농도")
    ax.set_ylabel("환자 수")
    ax.set_title("초미세 먼지 농도에 따른 환자 수 변화")

    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)

except FileNotFoundError:
    st.error(
        "데이터 파일 'ultrafine_dust_data.csv'를 찾을 수 없습니다.\n\n"
        "파이썬 파일과 CSV 파일이 같은 폴더에 있는지 확인해주세요."
    )

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다.\n\n에러 내용: {e}")
