import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from shapely.geometry import Polygon, LineString
# -----------------------------
# ✅ Parquet 데이터 불러오기
# -----------------------------
def load_parquet():
    return gpd.read_parquet("merged_address_with_area.parquet")

# -----------------------------
# ✅ Streamlit UI
# -----------------------------
st.title("🏗️ 대지 기반 건물 3D 모델링: 이격거리 → 건폐율 → 최고높이 → 정북일조 → 용적률")

gdf = load_parquet()

시도 = st.selectbox("시도 선택", sorted(gdf["SIDO_NM"].dropna().unique()), index=sorted(gdf["SIDO_NM"].dropna().unique()).index("서울특별시"))
시군구 = st.selectbox("시군구 선택", sorted(gdf[gdf["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique()), index=sorted(gdf[gdf["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique()).index("마포구"))
읍면동 = st.selectbox("읍면동 선택", sorted(gdf[(gdf["SIDO_NM"] == 시도) & (gdf["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique()), index=sorted(gdf[(gdf["SIDO_NM"] == 시도) & (gdf["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique()).index("서교동"))
본번 = st.number_input("본번 (MNNM)", min_value=1, step=1, value=375)
부번 = st.number_input("부번 (SLNO)", min_value=0, step=1, value=26)
층고 = st.number_input("층고 (m)", min_value=2.5, value=3.0)
건폐율 = st.number_input("건폐율 (%)", min_value=1.0, max_value=100.0, value=60.0)
용적률 = st.number_input("용적률 (%)", min_value=1.0, max_value=1000.0, value=200.0)
이격거리 = st.number_input("이격거리 (m)", min_value=0.0, value=0.0)

col1, col2 = st.columns(2)
with col1:
    최고높이_제한_적용 = st.checkbox("가로구역별 최고높이 적용")
    if 최고높이_제한_적용:
        최고높이 = st.number_input("가로구역별 최고높이 (m)", min_value=1.0, value=20.0)
    else:
        최고높이 = float('inf')

with col2:
    정북일조_적용 = st.checkbox("정북일조 적용")

if st.button("🔍 건물 모델링 계산"):
    filtered = gdf[
        (gdf["SIDO_NM"] == 시도) &
        (gdf["SGG_NM"] == 시군구) &
        (gdf["EMD_NM"] == 읍면동) &
        (gdf["MNNM"] == 본번) &
        (gdf["SLNO"] == 부번)
    ]

    if filtered.empty:
        st.warning("❌ 해당 지번의 데이터를 찾을 수 없습니다.")
        st.stop()

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    polygon = Polygon([transformer.transform(x, y) for x, y in filtered.geometry.iloc[0].exterior.coords])
    

    x_site = list(polygon.exterior.xy[0])
    y_site = list(polygon.exterior.xy[1])

    # 중심 좌표 기준 상대 좌표 변환
    x_center = np.mean(x_site)
    y_center = np.mean(y_site)
    x_site = [x - x_center for x in x_site]
    y_site = [y - y_center for y in y_site]
    z_base = [0] * len(x_site)
    대지면적 = filtered.iloc[0]["PAREA"]

    # -----------------------------
    # ✅ 정북일조 사선 계산 (10m까지 1.5m 이격, 이후 1:2 사선)
    # -----------------------------
    정북일조_제한높이 = float('inf')
    if 정북일조_적용:
        def get_north_edge(polygon):
            coords = list(polygon.exterior.coords)
            # 북쪽은 Y값이 가장 큰 쪽
            north_edges = []
            max_y = max(c[1] for c in coords)
            for c1, c2 in zip(coords[:-1], coords[1:]):
                if abs((c1[1] + c2[1])/2 - max_y) < 1e-6:
                    north_edges.append((c1, c2))
            # 가장 긴 북측 변 선택
            if north_edges:
                return max(north_edges, key=lambda e: np.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]))
            return (coords[0], coords[1])

        (p0, p1) = get_north_edge(polygon)
        x_north = np.linspace(p0[0], p1[0], 10)
        y_north = np.linspace(p0[1], p1[1], 10)

        x_sun, y_sun, z_sun = [], [], []
        for i in range(len(x_north)):
            for dx in np.linspace(0, 20, 20):  # 북쪽에서 남쪽으로 20m 범위
                setback = 1.5 if dx <= 10 else 1.5 + (dx - 10) * 0.5
                x_sun.append(x_north[i])
                y_sun.append(y_north[i] - dx)  # 남쪽 방향 (Y 감소)
                z_sun.append(setback)

        정북일조_제한높이 = min(z_sun)
    else:
        x_sun, y_sun, z_sun = [], [], []

    # -----------------------------
    # ✅ 최종 허용 높이 및 층수 계산
    # -----------------------------
    최종_허용높이 = min(정북일조_제한높이, 최고높이)
    if 층고 > 0 and np.isfinite(최종_허용높이):
        이론_최대_층수 = int(최종_허용높이 // 층고)
    else:
        이론_최대_층수 = 0
    이론_연면적 = 이론_최대_층수 * 대지면적
    용적률_제한_연면적 = 대지면적 * (용적률 / 100)
    최종_연면적 = min(이론_연면적, 용적률_제한_연면적)
    층수 = int(최종_연면적 // 대지면적)
    건물높이 = 층수 * 층고

    # -----------------------------
    # ✅ 건폐율 반영 여부 판단
    # -----------------------------
    건폐율_최대면적 = 대지면적 * (건폐율 / 100)
    실제_건축면적 = min(대지면적, 건폐율_최대면적)
    면적_비율 = 실제_건축면적 / 대지면적

    

    

    

    x_center = np.mean(x_site)
    y_center = np.mean(y_site)

    # 건물 평면 좌표 (건폐율 + 이격거리 반영)
    building_footprint = [(x * np.sqrt(면적_비율), y * np.sqrt(면적_비율)) for x, y in zip(x_site, y_site)]

    # 이격거리 반영 (단순 폴리곤 내축소)
    from shapely.affinity import scale
    bldg_polygon = Polygon(building_footprint).buffer(-이격거리)
    if not bldg_polygon.is_empty and isinstance(bldg_polygon, Polygon):
        building_footprint = list(bldg_polygon.exterior.coords)
    else:
        building_footprint = building_footprint  # 그대로 유지

    # ✅ 항상 z_top 정의
    z_top = [건물높이] * len(building_footprint)

    

    

    침범_여부 = f"✅ 최종 {층수}층 ({건물높이}m), 건폐율/용적률 만족" if 층수 > 0 else "⚠️ 제한 조건으로 건물 불가"

    # -----------------------------
    # ✅ Plotly 시각화 (3D 모델링)
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=x_site + [x_site[0]],
        y=y_site + [y_site[0]],
        z=z_base + [z_base[0]],
        color='gray',
        opacity=0.5,
        name='대지'
    ))

    fig.add_trace(go.Mesh3d(
        x=[x for x, y in building_footprint],
        y=[y for x, y in building_footprint],
        z=z_top,
        color='royalblue',
        opacity=0.9,
        name='건물'
    ))

    if 정북일조_적용:
        fig.add_trace(go.Scatter3d(
            x=x_sun,
            y=y_sun,
            z=z_sun,
            mode='markers',
            marker=dict(size=1, color='orange'),
            name='정북일조 사선'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='높이(m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric("📏 대지면적 (㎡)", f"{대지면적:.2f}")
    st.metric("🏢 허용 층수", f"{층수}층")
    st.metric("📐 연면적 (㎡)", f"{최종_연면적:.2f}")
