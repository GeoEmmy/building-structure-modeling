import streamlit as st
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import math
import trimesh
import shapely.affinity
from shapely.geometry import Polygon
from stable_baselines3 import PPO
from mass_placement_env import MassPlacementEnv

# ------------------- 유틸리티 함수 -------------------

def create_box(origin, size):
    return trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(origin))

def auto_structure_model(total_area, num_floors, span, floor_height, offset=(0, 0), basement_floors=0):
    floor_area = total_area / num_floors
    side_len = int(np.sqrt(floor_area))
    num_grids = int(side_len // span) + 1

    column_size = (0.6, 0.6, floor_height)
    beam_size = (span, 0.3, 0.6)
    beam_y_size = (0.3, span, 0.6)
    slab_thickness = 0.2
    foundation_thickness = 0.6
    boxes = []
    ox, oy = offset

    for b in range(basement_floors):
        z_base = - (b + 1) * floor_height
        for i in range(num_grids):
            for j in range(num_grids):
                x = ox + i * span
                y = oy + j * span
                boxes.append(create_box((x, y, z_base + column_size[2] / 2), column_size))
        for i in range(num_grids - 1):
            for j in range(num_grids):
                boxes.append(create_box((ox + (i + 0.5) * span, oy + j * span, z_base + floor_height - beam_size[2]/2), beam_size))
        for i in range(num_grids):
            for j in range(num_grids - 1):
                boxes.append(create_box((ox + i * span, oy + (j + 0.5) * span, z_base + floor_height - beam_y_size[2]/2), beam_y_size))
        slab_thick = foundation_thickness if b == basement_floors - 1 else slab_thickness
        z_slab = z_base - slab_thick / 2 if b == basement_floors - 1 else z_base + slab_thick / 2
        boxes.append(create_box((ox + side_len / 2, oy + side_len / 2, z_slab), (side_len, side_len, slab_thick)))

    for floor in range(num_floors):
        z_base = floor * floor_height
        for i in range(num_grids):
            for j in range(num_grids):
                boxes.append(create_box((ox + i * span, oy + j * span, z_base + column_size[2] / 2), column_size))
        for i in range(num_grids - 1):
            for j in range(num_grids):
                boxes.append(create_box((ox + (i + 0.5) * span, oy + j * span, z_base + floor_height - beam_size[2]/2), beam_size))
        for i in range(num_grids):
            for j in range(num_grids - 1):
                boxes.append(create_box((ox + i * span, oy + (j + 0.5) * span, z_base + floor_height - beam_y_size[2]/2), beam_y_size))
        boxes.append(create_box((ox + side_len / 2, oy + side_len / 2, z_base + 0.2 / 2), (side_len, side_len, 0.2)))
        boxes.append(create_box((ox + side_len / 2, oy + side_len / 2, z_base + floor_height + 0.2 / 2), (side_len, side_len, 0.2)))

    return boxes, side_len

def rotate_boxes(boxes, center, angle_rad):
    rot = trimesh.transformations.rotation_matrix(angle_rad, direction=[0, 0, 1], point=[center[0], center[1], 0])
    return [box.copy().apply_transform(rot) for box in boxes]

def visualize_trimesh_boxes_plotly(boxes, polygon=None, setback=None):
    combined = trimesh.util.concatenate(boxes)
    vertices, faces = combined.vertices, combined.faces
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color='lightblue', opacity=1.0, name='Structure'))
    def add_boundary_trace(geom, name, color, z=0):
        if geom and isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            fig.add_trace(go.Scatter3d(x=list(x) + [x[0]], y=list(y) + [y[0]], z=[z] * (len(x) + 1), mode='lines', line=dict(color=color, width=5), name=name))
    add_boundary_trace(polygon, "대지경계선", "green")
    add_boundary_trace(setback, "유효경계선", "red")
    fig.update_layout(scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'), title="🏗️ 대지 및 구조체 3D 시각화", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ------------------- Streamlit 앱 -------------------

df = gpd.read_parquet("merged_address_with_area.parquet")
st.title("🏗️ RL 기반 최적 매스 + 구조모델 자동 생성")

시도 = st.selectbox("시도 선택", sorted(df["SIDO_NM"].dropna().unique()))
시군구 = st.selectbox("시군구 선택", sorted(df[df["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique()))
읍면동 = st.selectbox("읍면동 선택", sorted(df[(df["SIDO_NM"] == 시도) & (df["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique()))
본번 = st.number_input("본번 (MNNM)", min_value=1, step=1)
부번 = st.number_input("부번 (SLNO)", min_value=0, step=1)

이격거리 = st.number_input("이격거리 (m)", value=2.0, step=0.5)
건폐율 = st.number_input("건폐율 (%)", value=60.0, step=5.0) / 100
용적률 = st.number_input("용적률 (%)", value=300.0, step=50.0) / 100
최대높이 = st.number_input("최대높이 (m)", value=15.0, step=1.0)
층고 = st.number_input("층당 층고 (m)", value=3.3, step=0.1)
스팬 = st.number_input("기둥 스팬 거리 (m)", value=6.0, step=0.5)
지하층수 = st.number_input("지하층 수", min_value=0, step=1, value=1)

if st.button("🏁 최적 매스 + 구조모델 생성"):
    filtered = df[(df["SIDO_NM"] == 시도) & (df["SGG_NM"] == 시군구) & (df["EMD_NM"] == 읍면동) & (df["MNNM"] == 본번) & (df["SLNO"] == 부번)]
    if filtered.empty:
        st.error("❌ 해당 지번의 데이터를 찾을 수 없습니다.")
        st.stop()

    gdf = gpd.GeoDataFrame(filtered, geometry="geometry", crs="EPSG:4326").to_crs(epsg=5179)
    polygon = gdf.geometry.iloc[0]
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)
    setback = polygon.buffer(-이격거리).buffer(0)

    대지면적 = polygon.area
    유효면적 = setback.area
    건축면적 = 유효면적 * 건폐율
    연면적 = 유효면적 * 용적률
    층수 = max(1, int(min(연면적 / 건축면적, 최대높이 // 층고)))

    try:
        model = PPO.load("ppo_mass.zip")
        env = MassPlacementEnv(setback, 대지면적, 건폐율, 용적률, 층수)
        observation, info = env.reset()
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
    except Exception as e:
        st.error(f"⚠️ 강화학습 예측 오류: {e}")
        action = [0, 0, 10, 10, 0]
        env = MassPlacementEnv(setback, 대지면적, 건폐율, 용적률, 층수)
        observation, reward, terminated, truncated, info = env.step(action)

    x, y, w, h, angle = action
    origin_x, origin_y = x, y
    mass = info.get("mass", None)

    if mass and isinstance(mass, Polygon) and mass.area > 0:
        st.success("✅ 유효한 mass 생성 완료")
    else:
        st.warning("❌ 유효한 mass를 찾지 못했습니다. 정책이 setback 또는 건폐율 조건을 반복적으로 위반했을 수 있습니다.")

    boxes, _ = auto_structure_model(
        total_area=연면적,
        num_floors=층수,
        span=스팬,
        floor_height=층고,
        offset=(origin_x, origin_y),
        basement_floors=지하층수
    )

    center = (origin_x + w / 2, origin_y + h / 2)
    rotated_boxes = rotate_boxes(boxes, center=center, angle_rad=np.radians(angle))

    polygon_local = shapely.affinity.translate(polygon, xoff=-origin_x, yoff=-origin_y)
    setback_local = shapely.affinity.translate(setback, xoff=-origin_x, yoff=-origin_y)

    st.markdown(f"✅ **대지면적**: {대지면적:.1f}㎡ | **유효면적**: {유효면적:.1f}㎡")
    st.markdown(f"🏢 **건축면적**: {건축면적:.1f}㎡ | **연면적**: {연면적:.1f}㎡ | **지상층수**: {층수}층 | **지하층수**: {지하층수}층")

    visualize_trimesh_boxes_plotly(rotated_boxes, polygon=polygon_local, setback=setback_local)
