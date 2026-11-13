import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="Solar Insight | 태양광 발전량 분석",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 그래프 기본 설정
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 디자인 CSS
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2.5rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #fff7e6, #fffdf7);
        border: 1px solid #f2d49b;
        margin-bottom: 2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
    }

    .hero p {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background-color: #fafafa;
        border: 1px solid #eeeeee;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/kkddww22kdw/project/main/sun.csv"

    try:
        return pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(url, encoding="euc-kr")


# =========================================================
# 선형회귀 계산
# =========================================================
def linear_regression(x, y):
    x_mean = x.mean()
    y_mean = y.mean()

    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        return None, None

    slope = ((x - x_mean) * (y - y_mean)).sum() / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept


# =========================================================
# 로지스틱 회귀 계산
# =========================================================
def sigmoid(z):
    z = max(min(z, 500), -500)
    return 1 / (1 + math.exp(-z))


def logistic_regression(x, y, learning_rate=0.01, epochs=1000):

    x_mean = x.mean()
    x_std = x.std()

    if x_std == 0:
        return None

    x_scaled = (x - x_mean) / x_std

    weight = 0.0
    bias = 0.0
    n = len(x_scaled)

    for _ in range(epochs):

        predictions = [
            sigmoid(weight * value + bias)
            for value in x_scaled
        ]

        errors = [
            predictions[i] - y.iloc[i]
            for i in range(n)
        ]

        dw = sum(
            errors[i] * x_scaled.iloc[i]
            for i in range(n)
        ) / n

        db = sum(errors) / n

        weight -= learning_rate * dw
        bias -= learning_rate * db

    return weight, bias, x_mean, x_std


# =========================================================
# 메인 프로그램
# =========================================================
try:

    df = load_data()
    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # -----------------------------------------------------
    # HERO 영역
    # -----------------------------------------------------
    st.markdown("""
    <div class="hero">
        <h1>☀️ Solar Insight</h1>
        <p>
            전국 태양광 발전 데이터를 분석하고,
            머신러닝을 활용하여 데이터의 패턴을 발견합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # 데이터 상태
    # -----------------------------------------------------
    st.markdown(
        '<div class="section-title">📊 데이터 현황</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("전체 데이터", f"{len(df):,}")
    c2.metric("전체 변수", f"{len(df.columns)}")
    c3.metric("숫자형 변수", f"{len(numeric_cols)}")
    c4.metric("결측값", f"{df.isnull().sum().sum():,}")

    st.divider()

    # =====================================================
    # 탭
    # =====================================================
    tab1, tab2, tab3 = st.tabs([
        "🗂️ 데이터 탐색",
        "📈 데이터 분석",
        "🤖 AI 모델"
    ])

    # =====================================================
    # 데이터 탐색
    # =====================================================
    with tab1:

        st.subheader("데이터 미리보기")
        st.caption("수집된 태양광 발전 데이터를 확인할 수 있습니다.")

        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=400
        )

        st.subheader("변수 정보")

        info_df = pd.DataFrame({
            "데이터 유형": df.dtypes.astype(str),
            "결측값": df.isnull().sum(),
            "고유값 수": df.nunique()
        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

        if numeric_cols:
            st.subheader("기술 통계")

            st.dataframe(
                df[numeric_cols].describe().T,
                use_container_width=True
            )

    # =====================================================
    # 데이터 분석
    # =====================================================
    with tab2:

        st.subheader("변수 관계 분석")
        st.caption(
            "두 개의 변수를 선택하여 데이터 사이의 관계를 확인하세요."
        )

        if len(numeric_cols) >= 2:

            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox(
                    "X축 변수",
                    numeric_cols,
                    key="x_axis"
                )

            with col2:
                y_col = st.selectbox(
                    "Y축 변수",
                    numeric_cols,
                    index=1,
                    key="y_axis"
                )

            graph_data = df[
                [x_col, y_col]
            ].dropna()

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.scatter(
                graph_data[x_col],
                graph_data[y_col],
                alpha=0.7
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col}와 {y_col}의 관계")
            ax.grid(alpha=0.2)

            fig.tight_layout()

            st.pyplot(fig)

            # 상관계수
            correlation = graph_data[
                [x_col, y_col]
            ].corr().iloc[0, 1]

            st.info(
                f"📌 두 변수의 상관계수는 "
                f"**{correlation:.3f}**입니다."
            )

    # =====================================================
    # AI 모델
    # =====================================================
    with tab3:

        st.subheader("🤖 머신러닝 분석")
        st.caption(
            "데이터를 선택하고 선형회귀 또는 로지스틱 회귀를 실행하세요."
        )

        if len(numeric_cols) >= 2:

            model_type = st.selectbox(
                "분석 모델 선택",
                [
                    "📈 선형회귀",
                    "🎯 로지스틱 회귀"
                ]
            )

            st.divider()

            setting1, setting2 = st.columns(2)

            with setting1:
                input_col = st.selectbox(
                    "입력 변수",
                    numeric_cols,
                    key="input_model"
                )

            with setting2:
                target_cols = [
                    col for col in numeric_cols
                    if col != input_col
                ]

                target_col = st.selectbox(
                    "예측 변수",
                    target_cols,
                    key="target_model"
                )

            model_data = df[
                [input_col, target_col]
            ].dropna()

            # -------------------------------------------------
            # 선형회귀
            # -------------------------------------------------
            if model_type == "📈 선형회귀":

                st.markdown("""
                <div class="info-box">
                    <b>📈 선형회귀란?</b><br>
                    입력 변수의 변화를 이용하여 숫자 데이터를 예측하는 방법입니다.
                </div>
                """, unsafe_allow_html=True)

                st.write("")

                if st.button(
                    "🚀 선형회귀 분석 시작",
                    type="primary",
                    use_container_width=True
                ):

                    x = model_data[input_col]
                    y = model_data[target_col]

                    slope, intercept = linear_regression(x, y)

                    if slope is None:
                        st.error("입력 데이터의 값이 모두 같아 분석할 수 없습니다.")

                    else:
                        prediction = slope * x + intercept

                        mae = abs(y - prediction).mean()

                        ss_total = ((y - y.mean()) ** 2).sum()
                        ss_error = ((y - prediction) ** 2).sum()

                        r2 = (
                            1 - ss_error / ss_total
                            if ss_total != 0 else 0
                        )

                        m1, m2, m3 = st.columns(3)

                        m1.metric("기울기", f"{slope:.4f}")
                        m2.metric("평균 오차", f"{mae:.4f}")
                        m3.metric("결정계수 R²", f"{r2:.4f}")

                        st.subheader("분석 결과")

                        fig, ax = plt.subplots(figsize=(10, 5))

                        ax.scatter(
                            x,
                            y,
                            alpha=0.6,
                            label="실제 데이터"
                        )

                        line_data = pd.DataFrame({
                            "x": x,
                            "y": prediction
                        }).sort_values("x")

                        ax.plot(
                            line_data["x"],
                            line_data["y"],
                            linewidth=2,
                            label="예측선"
                        )

                        ax.set_xlabel(input_col)
                        ax.set_ylabel(target_col)
                        ax.set_title("선형회귀 분석 결과")
                        ax.legend()
                        ax.grid(alpha=0.2)

                        fig.tight_layout()

                        st.pyplot(fig)

            # -------------------------------------------------
            # 로지스틱 회귀
            # -------------------------------------------------
            else:

                st.markdown("""
                <div class="info-box">
                    <b>🎯 로지스틱 회귀란?</b><br>
                    결과를 두 가지 범주로 나누고 각 결과가 발생할 확률을 계산하는 방법입니다.
                </div>
                """, unsafe_allow_html=True)

                unique_values = sorted(
                    model_data[target_col].unique()
                )

                if len(unique_values) == 2:

                    st.success(
                        f"분류 가능한 두 가지 값이 확인되었습니다: "
                        f"{unique_values[0]}, {unique_values[1]}"
                    )

                    if st.button(
                        "🚀 로지스틱 회귀 분석 시작",
                        type="primary",
                        use_container_width=True
                    ):

                        x = model_data[input_col].reset_index(drop=True)

                        original_y = model_data[
                            target_col
                        ].reset_index(drop=True)

                        value_map = {
                            unique_values[0]: 0,
                            unique_values[1]: 1
                        }

                        y = original_y.map(value_map)

                        result = logistic_regression(x, y)

                        if result is not None:

                            weight, bias, x_mean, x_std = result

                            x_scaled = (
                                x - x_mean
                            ) / x_std

                            probabilities = [
                                sigmoid(
                                    weight * value + bias
                                )
                                for value in x_scaled
                            ]

                            predictions = [
                                1 if value >= 0.5 else 0
                                for value in probabilities
                            ]

                            accuracy = sum(
                                predictions[i] == y.iloc[i]
                                for i in range(len(y))
                            ) / len(y)

                            m1, m2, m3 = st.columns(3)

                            m1.metric(
                                "분류 정확도",
                                f"{accuracy * 100:.1f}%"
                            )

                            m2.metric(
                                "가중치",
                                f"{weight:.3f}"
                            )

                            m3.metric(
                                "분석 데이터",
                                f"{len(y):,}개"
                            )

                            result_df = pd.DataFrame({
                                input_col: x,
                                "실제값": original_y,
                                "예측 확률": probabilities,
                                "예측 결과": [
                                    unique_values[value]
                                    for value in predictions
                                ]
                            })

                            st.subheader("예측 결과")

                            st.dataframe(
                                result_df.head(30),
                                use_container_width=True
                            )

                else:
                    st.warning(
                        "로지스틱 회귀를 사용하려면 예측 변수의 "
                        "값이 정확히 두 종류여야 합니다."
                    )

        else:
            st.warning(
                "분석을 위해 숫자형 변수가 2개 이상 필요합니다."
            )

except Exception as e:
    st.error("⚠️ 데이터를 불러오는 중 문제가 발생했습니다.")
    st.exception(e)
