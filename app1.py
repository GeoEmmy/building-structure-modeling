import streamlit as st
import geopandas as gpd
import plotly.express as px

# 📂 Parquet 파일 경로
PARQUET_PATH = "D:/python/lca/merged_address_only.parquet"  # ← 경로 수정 가능

@st.cache_data
def load_data():
    return gpd.read_parquet(PARQUET_PATH)

df = load_data()

st.title("🏡 주소 기반 지번 Polygon 시각화")

# 🎯 시도 → 시군구 → 법정동 선택
sido = st.selectbox("시도 선택", sorted(df["SIDO_NM"].dropna().unique()))
sgg_list = sorted(df[df["SIDO_NM"] == sido]["SGG_NM"].dropna().unique())
sgg = st.selectbox("시군구 선택", sgg_list)

emd_list = sorted(df[(df["SIDO_NM"] == sido) & (df["SGG_NM"] == sgg)]["EMD_NM"].dropna().unique())
emd = st.selectbox("법정동 선택", emd_list)

# 🔢 본번 / 부번 입력
mnnm = st.number_input("본번 (예: 9)", min_value=1, step=1)
slno = st.number_input("부번 (예: 0)", min_value=0, step=1)

# ▶️ 버튼 누를 때만 실행
if st.button("지번 조회"):
    filtered = df[
        (df["SIDO_NM"] == sido) &
        (df["SGG_NM"] == sgg) &
        (df["EMD_NM"] == emd) &
        (df["MNNM"] == mnnm) &
        (df["SLNO"] == slno)
    ]

    if not filtered.empty:
        st.success(f"✅ {len(filtered)}개 지번이 일치합니다.")
        fig = px.choropleth_mapbox(
            filtered,
            geojson=filtered.geometry.__geo_interface__,
            locations=filtered.index,
            color_discrete_sequence=["blue"],
            mapbox_style="carto-positron",
            center={"lat": filtered.geometry.centroid.y.mean(), "lon": filtered.geometry.centroid.x.mean()},
            zoom=17,
            opacity=0.5
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("❌ 해당 조건과 일치하는 지번이 없습니다.")
