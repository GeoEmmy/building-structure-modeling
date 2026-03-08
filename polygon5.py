import streamlit as st
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

@st.cache_data
def load_data():
    return gpd.read_parquet("merged_address_with_area.parquet")

def create_box(origin, size):
    return trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(origin))

def auto_structure_model(total_area, num_floors, span, floor_height):
    floor_area = total_area / num_floors
    side_len = int(np.sqrt(floor_area))
    num_grids = int(side_len // span) + 1

    column_size = (0.6, 0.6, floor_height)
    beam_size = (span, 0.3, 0.6)
    slab_thickness = 0.2
    slab_size = (side_len, side_len, slab_thickness)

    boxes = []

    for floor in range(num_floors):
        z_base = floor * floor_height

        # 기둥
        for i in range(num_grids):
            for j in range(num_grids):
                x = i * span
                y = j * span
                boxes.append(create_box((x, y, z_base + column_size[2]/2), column_size))

        # 보 X방향
        for i in range(num_grids - 1):
            for j in range(num_grids):
                x = (i + 0.5) * span
                y = j * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_size[2]/2), beam_size))

        # 보 Y방향
        for i in range(num_grids):
            for j in range(num_grids - 1):
                x = i * span
                y = (j + 0.5) * span
                beam_y = (0.3, span, 0.6)
                boxes.append(create_box((x, y, z_base + floor_height - beam_y[2]/2), beam_y))

        # 슬래브
        boxes.append(create_box((side_len / 2, side_len / 2, z_base + floor_height + slab_thickness / 2), slab_size))

    return boxes

def visualize_trimesh_boxes(boxes):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for mesh in boxes:
        for face in mesh.faces:
            tri = mesh.vertices[face]
            poly = Poly3DCollection([tri], alpha=0.5, edgecolor='k')
            ax.add_collection3d(poly)

    all_vertices = np.vstack([m.vertices for m in boxes])
    ax.set_xlim([np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])])
    ax.set_ylim([np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])])
    ax.set_zlim([0, np.max(all_vertices[:, 2])])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit 앱 시작
df = load_data()
st.title("🏗️ 대지 기반 구조 모델링 (정밀 구조 + 자동 계산)")

# 디폴트 주소
default_시도 = "서울특별시"
default_시군구 = "마포구"
default_읍면동 = "서교동"
default_본번 = 375
default_부번 = 26

시도 = st.selectbox("시도", sorted(df["SIDO_NM"].dropna().unique()), index=0)
시군구 = st.selectbox("시군구", sorted(df[df["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique()), index=0)
읍면동 = st.selectbox("읍면동", sorted(df[(df["SIDO_NM"] == 시도) & (df["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique()), index=0)
본번 = st.number_input("본번", min_value=1, step=1, value=default_본번)
부번 = st.number_input("부번", min_value=0, step=1, value=default_부번)

# 추가 조건 입력
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
        st.error("⚠️ 유효 대지가 없습니다.")
        st.stop()

    대지면적 = polygon.area
    유효면적 = setback.area
    건축면적 = 유효면적 * 건폐율
    연면적 = 유효면적 * 용적률
    층수 = int(min(연면적 / 건축면적, 최대높이 // 층고))
    if 층수 < 1:
        st.error("⚠️ 유효 층수가 1층 미만입니다.")
        st.stop()

    st.markdown(f"✅ **대지면적**: {대지면적:.1f}㎡ | **이격 후 면적**: {유효면적:.1f}㎡")
    st.markdown(f"🏢 **건축면적**: {건축면적:.1f}㎡ | **연면적**: {연면적:.1f}㎡ | **자동 층수**: {층수}층")

    boxes = auto_structure_model(total_area=연면적, num_floors=층수, span=스팬, floor_height=층고)
    visualize_trimesh_boxes(boxes)
