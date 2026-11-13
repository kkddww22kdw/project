import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import glob

# 한글 폰트 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="태양광 발전량 예측 프로젝트", page_icon="☀️", layout="wide")

st.title("☀️ 전국 태양광 발전량 예측 및 분석")
st.write("태양광 발전량 데이터와 요인들을 분석하는 데이터 과학 웹 어플리케이션입니다.")

@st.cache_data
def load_data():
    # 깃허브 폴더 안의 CSV 파일을 자동으로 찾습니다
    csv_files = glob.glob('*.csv')
    if not csv_files:
        return None, None
    filename = csv_files[0]
    
    try:
        df = pd.read_csv(filename, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filename, encoding='euc-kr')
    return df, filename

try:
    df, filename = load_data()
    
    if df is not None:
        st.success(f"📂 연결된 데이터 파일: `{filename}`")
        
        st.subheader("1. 📊 수집된 데이터셋 확인")
        st.dataframe(df.head())
        
        st.subheader("2. 📈 데이터 기술 통계 (describe)")
        st.write(df.describe())
        
        st.subheader("3. 📉 데이터 시각화 및 분석")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("X축 데이터 선택", numeric_cols, index=0)
            col2 = st.selectbox("Y축 데이터 선택", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(df[col1], df[col2], color='darkorange', alpha=0.8, edgecolors='k')
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.set_title(f'{col1}에 따른 {col2} 분석')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
        else:
            st.line_chart(df)
            
    else:
        st.error("깃허브 저장소에 CSV 데이터 파일이 없습니다. 태양광 데이터 파일을 업로드해 주세요!")

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다.\n\n에러 내용: {e}")
