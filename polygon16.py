import streamlit as st
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Polygon, LineString, Point
import shapely.affinity
import trimesh
import math

@st.cache_data
def load_data():
    return gpd.read_parquet("merged_address_with_area.parquet")

def create_beam(start, end, z, beam_width, beam_depth, part_type="beam"):
    start = np.array(start)
    end = np.array(end)
    center = (start + end) / 2
    vec = end - start
    length = np.linalg.norm(vec)
    if length == 0:
        return None
    direction = vec / length
    angle = math.atan2(direction[1], direction[0])
    box = trimesh.creation.box(extents=(length, beam_width, beam_depth))
    box.apply_translation([length / 2, 0, 0])  # move to start
    tf = trimesh.transformations.rotation_matrix(angle, [0, 0, 1], point=[0,0,0])
    box.apply_transform(tf)
    box.apply_translation([center[0], center[1], z + beam_depth/2])
    box.metadata = {"type": part_type}
    return box

def create_column(pt, z, col_width, col_depth, col_height):
    box = trimesh.creation.box(extents=(col_width, col_depth, col_height))
    box.apply_translation([pt[0], pt[1], z + col_height/2])
    box.metadata = {"type": "column"}
    return box

def create_slab(polygon, z, thickness):
    minx, miny, maxx, maxy = polygon.bounds
    w, h = maxx - minx, maxy - miny
    center = [(minx + maxx)/2, (miny + maxy)/2, z + thickness/2]
    slab = trimesh.creation.box(extents=(w, h, thickness))
    slab.apply_translation(center)
    slab.metadata = {"type": "slab"}
    return slab

def generate_beams_and_columns(slab_polygon, span, floor_z, beam_size, col_size):
    coords = list(slab_polygon.exterior.coords)
    beams = []
    beam_points = []
    # 1. 외곽선 따라 보 배치 (segment별)
    for i in range(len(coords) - 1):
        p1, p2 = np.array(coords[i]), np.array(coords[i + 1])
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        direction = seg_vec / seg_len if seg_len != 0 else np.zeros(2)
        num = int(seg_len // span)
        for j in range(num):
            s = p1 + direction * span * j
            e = p1 + direction * span * (j + 1)
            beams.append((tuple(s), tuple(e)))
            beam_points.extend([tuple(s), tuple(e)])
        # 남는 부분(끝까지)
        if (seg_len - (span * num)) > 0.1:
            s = p1 + direction * span * num
            e = p2
            beams.append((tuple(s), tuple(e)))
            beam_points.extend([tuple(s), tuple(e)])

    # 2. 보 교차점에만 기둥 생성 (중복제거)
    unique_points = {pt for pt in beam_points if slab_polygon.contains(Point(pt)) or slab_polygon.touches(Point(pt))}
    # 보끼리 교차점도 찾아 추가(선택)
    # 단순화: 모든 보 시작점/끝점이 교차점
    columns = list(unique_points)
    return beams, columns

def visualize_trimesh_boxes_plotly(boxes, polygon=None, setback=None):
    fig = go.Figure()
    COLOR_MAP = {
        "column": "orange",
        "beam": "blue",
        "slab": "lightgray",
        "foundation": "brown"
    }
    categorized = {k: [] for k in COLOR_MAP}
    for box in boxes:
        t = box.metadata.get("type", "slab")
        categorized.setdefault(t, []).append(box)
    for part_type, part_boxes in categorized.items():
        if not part_boxes:
            continue
        mesh = trimesh.util.concatenate(part_boxes)
        vertices = mesh.vertices
        faces = mesh.faces
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=COLOR_MAP.get(part_type, 'gray'), name=part_type, opacity=1.0
        ))
    def add_boundary_trace(geom, name, color, z=0):
        if geom and isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            fig.add_trace(go.Scatter3d(
                x=list(x) + [x[0]], y=list(y) + [y[0]], z=[z] * (len(x) + 1),
                mode='lines', line=dict(color=color, width=5), name=name
            ))
    add_boundary_trace(polygon, "대지경계선", "green")
    add_boundary_trace(setback, "유효경계선", "red")
    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        title="\U0001f3d7️ 부재별 구조 색상 시각화",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================= Streamlit UI =======================
df = load_data()
st.title("📐 슬라브 외곽선 기반 보/기둥 자동 구조 모델링")

# 디폴트 주소
def_시도 = "서울특별시"
def_시군구 = "영등포구"
def_읍면동 = "양평동1가"
def_본번 = 270
def_부번 = 0

시도_options = sorted(df["SIDO_NM"].dropna().unique())
시도 = st.selectbox("시도", 시도_options, index=시도_options.index(def_시도))
시군구_options = sorted(df[df["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique())
시군구 = st.selectbox("시군구", 시군구_options, index=시군구_options.index(def_시군구))
읍면동_options = sorted(df[(df["SIDO_NM"] == 시도) & (df["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique())
읍면동 = st.selectbox("읍면동", 읍면동_options, index=읍면동_options.index(def_읍면동))
본번 = st.number_input("본번", min_value=1, step=1, value=def_본번)
부번 = st.number_input("부번", min_value=0, step=1, value=def_부번)

이격거리 = st.number_input("이격거리 (m)", value=2.0, step=0.5)
건폐율 = st.number_input("건폐율 (%)", value=60.0, step=5.0) / 100
슬라브두께 = st.number_input("슬라브 두께(m)", value=0.2, step=0.05)
보폭 = st.number_input("보 간격(span, m)", value=6.0, step=0.5)
보폭_너비 = st.number_input("보 너비(m)", value=0.3, step=0.05)
보폭_깊이 = st.number_input("보 깊이(m)", value=0.6, step=0.05)
기둥폭 = st.number_input("기둥 너비(m)", value=0.6, step=0.05)
기둥깊이 = st.number_input("기둥 깊이(m)", value=0.6, step=0.05)
기둥높이 = st.number_input("기둥 높이(m)", value=3.3, step=0.1)

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

    # 슬라브는 setback 영역 전체(=유효경계) 기준
    slab_polygon = setback

    # 1. 슬라브 생성
    boxes = [create_slab(slab_polygon, z=0, thickness=슬라브두께)]

    # 2. 슬라브 외곽선 따라 보 배치 및 교차점 기둥 생성
    beams, columns = generate_beams_and_columns(
        slab_polygon, span=보폭, floor_z=0,
        beam_size=(보폭, 보폭_너비, 보폭_깊이),
        col_size=(기둥폭, 기둥깊이, 기둥높이)
    )

    # 3. 보 생성
    for s, e in beams:
        beam_box = create_beam(s, e, z=슬라브두께, beam_width=보폭_너비, beam_depth=보폭_깊이)
        if beam_box is not None:
            boxes.append(beam_box)
    # 4. 기둥 생성
    for pt in columns:
        col_box = create_column(pt, z=0, col_width=기둥폭, col_depth=기둥깊이, col_height=기둥높이)
        boxes.append(col_box)

    st.markdown(f"✅ **대지면적**: {대지면적:.1f}㎡ | **유효면적**: {유효면적:.1f}㎡ | **건축면적(슬라브기준)**: {건축면적:.1f}㎡")
    visualize_trimesh_boxes_plotly(boxes, polygon=polygon, setback=setback)
