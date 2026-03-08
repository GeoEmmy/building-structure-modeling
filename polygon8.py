import streamlit as st
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Polygon
import trimesh

@st.cache_data
def load_data():
    return gpd.read_parquet("merged_address_with_area.parquet")

def create_box(origin, size):
    return trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(origin))

def auto_structure_model(total_area, num_floors, span, floor_height, offset=(0, 0)):
    floor_area = total_area / num_floors
    side_len = int(np.sqrt(floor_area))
    num_grids = int(side_len // span) + 1

    column_size = (0.6, 0.6, floor_height)
    beam_size = (span, 0.3, 0.6)
    beam_y_size = (0.3, span, 0.6)
    slab_thickness = 0.2
    slab_size = (side_len, side_len, slab_thickness)

    boxes = []
    ox, oy = offset

    for floor in range(num_floors):
        z_base = floor * floor_height

        for i in range(num_grids):
            for j in range(num_grids):
                x = ox + i * span
                y = oy + j * span
                boxes.append(create_box((x, y, z_base + column_size[2]/2), column_size))

        for i in range(num_grids - 1):
            for j in range(num_grids):
                x = ox + (i + 0.5) * span
                y = oy + j * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_size[2]/2), beam_size))

        for i in range(num_grids):
            for j in range(num_grids - 1):
                x = ox + i * span
                y = oy + (j + 0.5) * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_y_size[2]/2), beam_y_size))

        boxes.append(create_box((ox + side_len / 2, oy + side_len / 2, z_base + floor_height + slab_thickness / 2), slab_size))

    return boxes

def visualize_trimesh_boxes_plotly(boxes, polygon=None, setback=None):
    combined = trimesh.util.concatenate(boxes)
    vertices = combined.vertices
    faces = combined.faces

    fig = go.Figure()

    # 구조체 Mesh
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.5,
        name='Structure'
    ))

    def add_boundary_trace(geom, name, color, z=0):
        if geom and isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            fig.add_trace(go.Scatter3d(
                x=list(x) + [x[0]],
                y=list(y) + [y[0]],
                z=[z] * (len(x) + 1),
                mode='lines',
                line=dict(color=color, width=5),
                name=name
            ))

    add_boundary_trace(polygon, "대지경계선", "green")
    add_boundary_trace(setback, "유효경계선", "red")

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        title="🏗️ 대지 및 구조체 3D 시각화",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

# Streamlit 앱 시작
df = load_data()
st.title("📐 지번 기반 구조 자동 모델링")

# 주소 입력 UI
def_시도 = "서울특별시"
def_시군구 = "마포구"
def_읍면동 = "서교동"
def_본번 = 375
def_부번 = 26

시도 = st.selectbox("시도", sorted(df["SIDO_NM"].dropna().unique()), index=0)
시군구 = st.selectbox("시군구", sorted(df[df["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique()), index=0)
읍면동 = st.selectbox("읍면동", sorted(df[(df["SIDO_NM"] == 시도) & (df["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique()), index=0)
본번 = st.number_input("본번", min_value=1, step=1, value=def_본번)
부번 = st.number_input("부번", min_value=0, step=1, value=def_부번)

# 건축 조건
이격거리 = st.number_input("이격거리 (m)", value=2.0, step=0.5)
건폐율 = st.number_input("건폐율 (%)", value=60.0, step=5.0) / 100
용적률 = st.number_input("용적률 (%)", value=300.0, step=50.0) / 100
최대높이 = st.number_input("최대높이 (m)", value=15.0, step=1.0)
층고 = st.number_input("층당 층고 (m)", value=3.3, step=0.1)
스팬 = st.number_input("기둥 스팬 거리 (m)", value=6.0, step=0.5)

if st.button("🔍 모델 생성"):
    filtered = df[
        (df["SIDO_NM"] == 시도) &
        (df["SGG_NM"] == 시군구) &
        (df["EMD_NM"] == 읍면동) &
        (df["MNNM"] == 본번) &
        (df["SLNO"] == 부번)
    ]

    if filtered.empty:
        st.error("❌ 해당 지번의 데이터를 찾을 수 없습니다.")
        st.stop()

    gdf = gpd.GeoDataFrame(filtered, geometry="geometry", crs="EPSG:4326").to_crs(epsg=5179)
    polygon = gdf.geometry.iloc[0]
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    setback = polygon.buffer(-이격거리).buffer(0)
    if setback.is_empty or not isinstance(setback, Polygon):
        st.error("⚠️ 이격거리 적용 후 유효 대지가 없습니다.")
        st.stop()

    대지면적 = polygon.area
    유효면적 = setback.area
    건축면적 = 유효면적 * 건폐율
    연면적 = 유효면적 * 용적률
    층수 = int(min(연면적 / 건축면적, 최대높이 // 층고))

    if 층수 < 1:
        st.error("⚠️ 유효 층수가 1층 미만입니다.")
        st.stop()

    st.markdown(f"✅ **대지면적**: {대지면적:.1f}㎡ | **유효면적**: {유효면적:.1f}㎡")
    st.markdown(f"🏢 **건축면적**: {건축면적:.1f}㎡ | **연면적**: {연면적:.1f}㎡ | **층수**: {층수}층")

    minx, miny, *_ = setback.bounds
    boxes = auto_structure_model(total_area=연면적, num_floors=층수, span=스팬, floor_height=층고, offset=(minx, miny))
    visualize_trimesh_boxes_plotly(boxes, polygon=polygon, setback=setback)
