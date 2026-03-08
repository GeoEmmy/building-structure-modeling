import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px

# 📂 데이터 로딩
@st.cache_data
def load_data():
    return gpd.read_parquet("merged_address_with_area.parquet")

df = load_data()

st.title("📍 주소 기반 지번 폴리곤 조회")

# ✅ 디폴트값
기본_시도 = "서울특별시"
기본_시군구 = "마포구"
기본_읍면동 = "서교동"
기본_본번 = 375
기본_부번 = 26

# ✅ 입력
시도 = st.selectbox("시도 선택", sorted(df["SIDO_NM"].dropna().unique()))
시군구 = st.selectbox("시군구 선택", sorted(df[df["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique()))
읍면동 = st.selectbox("읍면동 선택", sorted(df[(df["SIDO_NM"] == 시도) & (df["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique()))
본번 = st.number_input("본번 (MNNM)", min_value=1, step=1)
부번 = st.number_input("부번 (SLNO)", min_value=0, step=1)

if st.button("🔍 지번 조회"):
    filtered = df[
        (df["SIDO_NM"] == 시도) &
        (df["SGG_NM"] == 시군구) &
        (df["EMD_NM"] == 읍면동) &
        (df["MNNM"] == 본번) &
        (df["SLNO"] == 부번)
    ]

    if filtered.empty:
        st.warning("❌ 해당 지번의 데이터를 찾을 수 없습니다.")
    else:
        gdf = gpd.GeoDataFrame(filtered, geometry="geometry", crs="EPSG:4326")
        geojson = gdf.__geo_interface__

        # 🌐 지도 시각화
        fig = px.choropleth_mapbox(
            gdf,
            geojson=geojson,
            locations=gdf.index,
            color_discrete_sequence=["blue"],
            center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
            mapbox_style="open-street-map",
            zoom=17
        )
        fig.update_traces(marker_line_width=2, marker_line_color="white")
        st.plotly_chart(fig)

        # 📏 면적 표시
        면적 = filtered.iloc[0]["PAREA"]
        st.subheader("📐 면적 정보")
        st.metric(label="면적 (㎡)", value=f"{면적:,.2f}")
