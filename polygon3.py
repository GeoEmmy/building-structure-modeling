import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
import numpy as np

@st.cache_data
def load_data():
    return gpd.read_parquet("merged_address_with_area.parquet")

df = load_data()

st.title("📍 대지 3D 시각화 (Z=30m + 동서남북 표시)")

# 디폴트 주소
default_시도 = "서울특별시"
default_시군구 = "마포구"
default_읍면동 = "서교동"
default_본번 = 375
default_부번 = 26

# ✅ 주소 입력 (디폴트 적용)
시도_options = sorted(df["SIDO_NM"].dropna().unique())
시도 = st.selectbox("시도 선택", 시도_options, index=시도_options.index(default_시도))

시군구_options = sorted(df[df["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique())
시군구 = st.selectbox("시군구 선택", 시군구_options, index=시군구_options.index(default_시군구))

읍면동_options = sorted(df[(df["SIDO_NM"] == 시도) & (df["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique())
읍면동 = st.selectbox("읍면동 선택", 읍면동_options, index=읍면동_options.index(default_읍면동))

본번 = st.number_input("본번 (MNNM)", min_value=1, step=1, value=default_본번)
부번 = st.number_input("부번 (SLNO)", min_value=0, step=1, value=default_부번)

if st.button("🔍 대지 시각화"):
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
        # EPSG:5179로 변환 (미터 기반)
        gdf = gpd.GeoDataFrame(filtered, geometry="geometry", crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=5179)

        polygon = gdf.geometry.iloc[0]
        if polygon.geom_type == "MultiPolygon":
            polygon = list(polygon.geoms)[0]

        coords = polygon.exterior.coords.xy
        x = list(coords[0])
        y = list(coords[1])
        z = [0] * len(x)

        # 중심 정규화
        min_x, min_y, max_x, max_y = polygon.bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        x_centered = [xi - center_x for xi in x]
        y_centered = [yi - center_y for yi in y]

        # 바닥 그리드 생성
        padding = 10  # meter
        grid_x = np.linspace(min_x - padding, max_x + padding, 30)
        grid_y = np.linspace(min_y - padding, max_y + padding, 30)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        grid_zz = np.zeros_like(grid_xx)
        grid_xx_centered = grid_xx - center_x
        grid_yy_centered = grid_yy - center_y

        # 동서남북 텍스트 라벨
        direction_labels = go.Scatter3d(
            x=[20, -20, 0, 0],
            y=[0, 0, -20, 20],
            z=[0, 0, 0, 0],
            mode='text',
            text=["E", "W", "S", "N"],
            textposition="middle center",
            textfont=dict(size=16, color="black"),
            name="방위"
        )

        # 📊 3D 시각화
        fig = go.Figure()

        # 대지 경계선
        fig.add_trace(go.Scatter3d(
            x=x_centered + [x_centered[0]],
            y=y_centered + [y_centered[0]],
            z=z + [0],
            mode='lines',
            line=dict(color='blue', width=6),
            name="대지경계"
        ))

        # 바닥면
        fig.add_trace(go.Surface(
            x=grid_xx_centered,
            y=grid_yy_centered,
            z=grid_zz,
            showscale=False,
            opacity=0.2,
            colorscale="Greys",
            name="그리드"
        ))

        # 방향 텍스트
        fig.add_trace(direction_labels)

        # 카메라 및 축 설정
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Height (m)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                zaxis=dict(range=[0, 30])
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            title="3D 대지경계 + 동서남북 방향 표시"
        )

        st.plotly_chart(fig, use_container_width=True)
