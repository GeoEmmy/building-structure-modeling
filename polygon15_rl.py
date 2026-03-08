import streamlit as st
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Polygon
from shapely.affinity import scale, translate
import shapely.affinity
import trimesh
import math
import gymnasium as gym
from gymnasium import spaces

# ==================== 최적화 환경 ====================

class MassPlacementEnv(gym.Env):
    """휴리스틱 최적화 기반 매스 배치 환경"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_steps = 50
        self.current_step = 0
        self.best_area = 0
        self.best_mass = None
        self.set_parameters(config)
        self.define_spaces()

    def set_parameters(self, config):
        self.setback = config["setback"]
        self.site_polygon = config["site_polygon"]
        self.대지면적 = config["대지면적"]
        self.건폐율 = config["건폐율"]
        self.용적률 = config["용적률"]
        self.최대높이 = config["최대높이"]
        self.층고 = config.get("층고", 3.3)

        self.max_total_area = self.대지면적 * self.용적률
        self.max_building_area = self.대지면적 * self.건폐율
        self.max_floors = int(self.최대높이 // self.층고)

    def define_spaces(self):
        self.action_space = spaces.Box(
            low=np.array([0.3, -10.0, -10.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.best_area = 0
        self.best_mass = None
        self.current_state = np.random.uniform(0, 1, size=7).astype(np.float32)
        return self.current_state, {"mass": None}

    def step(self, action):
        self.current_step += 1
        scale_ratio, dx, dy = action

        centroid = self.setback.centroid
        scaled = scale(self.setback, xfact=scale_ratio, yfact=scale_ratio, origin=centroid)
        moved = translate(scaled, xoff=dx, yoff=dy)
        mass = moved

        total_area = 0
        floors = 0
        reasons = []

        if not self.setback.contains(mass):
            reasons.append("setback 위반")
        if mass.area > self.max_building_area:
            reasons.append("건폐율 초과")
        if mass.area > 0:
            floors = min(self.max_floors, int(self.max_total_area // mass.area))
            total_area = mass.area * floors
            if total_area > self.max_total_area:
                reasons.append("용적률 초과")
        else:
            reasons.append("면적 0")

        penalty = 0
        if "setback 위반" in reasons: penalty += 1.0
        if "건폐율 초과" in reasons: penalty += 1.0
        if "용적률 초과" in reasons: penalty += 1.0
        if "면적 0" in reasons: penalty += 2.0

        area_ratio = total_area / self.max_total_area if self.max_total_area > 0 else 0
        floor_bonus = floors / self.max_floors if self.max_floors > 0 else 0
        reward = (area_ratio ** 1.2) * 10 + (floor_bonus ** 1.2) * 5 - (penalty ** 1.5)

        if not reasons and total_area > self.best_area:
            self.best_area = total_area
            self.best_mass = mass

        obs = np.array([
            mass.area / self.max_building_area if self.max_building_area > 0 else 0,
            floor_bonus,
            area_ratio,
            dx / 10.0,
            dy / 10.0,
            self.건폐율,
            self.용적률 / 5.0
        ], dtype=np.float32)

        info = {
            "mass": self.best_mass if self.best_mass else mass,
            "층수": floors,
            "연면적": total_area,
            "층고": self.층고,
            "violation_reason": reasons,
            "scale_ratio": scale_ratio,
            "offset": (dx, dy)
        }

        terminated = self.current_step >= self.max_steps
        return obs, reward, terminated, False, info


# ==================== 구조 모델링 함수 ====================

@st.cache_data
def load_data():
    return gpd.read_parquet("merged_address_with_area.parquet")

def create_box(origin, size, part_type="slab"):
    box = trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(origin))
    box.metadata["type"] = part_type
    return box

def auto_structure_model(floor_area, num_floors, span, floor_height, offset=(0, 0), basement_floors=0):
    side_len = np.sqrt(floor_area)
    num_grids = int(side_len // span) + 1

    column_size = (0.6, 0.6, floor_height)
    beam_size = (span, 0.3, 0.6)
    beam_y_size = (0.3, span, 0.6)
    slab_thickness = 0.2
    foundation_thickness = 0.6
    boxes = []
    ox, oy = offset

    # Basement
    for b in range(basement_floors):
        z_base = - (b + 1) * floor_height
        for i in range(num_grids):
            for j in range(num_grids):
                x = ox + i * span
                y = oy + j * span
                boxes.append(create_box((x, y, z_base + column_size[2] / 2), column_size, part_type="column"))
        for i in range(num_grids - 1):
            for j in range(num_grids):
                x = ox + (i + 0.5) * span
                y = oy + j * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_size[2] / 2), beam_size, part_type="beam_x"))
        for i in range(num_grids):
            for j in range(num_grids - 1):
                x = ox + i * span
                y = oy + (j + 0.5) * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_y_size[2] / 2), beam_y_size, part_type="beam_y"))
        slab_thick = foundation_thickness if (b == basement_floors - 1) else slab_thickness
        z_slab_center = z_base - slab_thick / 2 if (b == basement_floors - 1) else z_base + slab_thick / 2
        part = "foundation" if (b == basement_floors - 1) else "slab"
        boxes.append(create_box((ox + side_len / 2, oy + side_len / 2, z_slab_center), (side_len, side_len, slab_thick), part_type=part))

    # Ground and above-ground floors
    for floor in range(num_floors):
        z_base = floor * floor_height
        for i in range(num_grids):
            for j in range(num_grids):
                x = ox + i * span
                y = oy + j * span
                boxes.append(create_box((x, y, z_base + column_size[2] / 2), column_size, part_type="column"))
        for i in range(num_grids - 1):
            for j in range(num_grids):
                x = ox + (i + 0.5) * span
                y = oy + j * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_size[2] / 2), beam_size, part_type="beam_x"))
        for i in range(num_grids):
            for j in range(num_grids - 1):
                x = ox + i * span
                y = oy + (j + 0.5) * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_y_size[2] / 2), beam_y_size, part_type="beam_y"))
        boxes.append(create_box((ox + side_len / 2, oy + side_len / 2, z_base + slab_thickness / 2), (side_len, side_len, slab_thickness), part_type="slab"))
        boxes.append(create_box((ox + side_len / 2, oy + side_len / 2, z_base + floor_height + slab_thickness / 2), (side_len, side_len, slab_thickness), part_type="slab"))

    return boxes, side_len

def create_polygon_slab(polygon, thickness, z_position, part_type="slab"):
    """polygon 형태 그대로 슬라브 메쉬 생성 (삼각화 사용)"""
    try:
        # triangle 엔진 사용하여 extrude
        slab_mesh = trimesh.creation.extrude_polygon(polygon, height=thickness, engine="triangle")
        slab_mesh.apply_translation([0, 0, z_position])
        slab_mesh.metadata["type"] = part_type
        return slab_mesh
    except Exception as e1:
        try:
            # mapbox_earcut 엔진으로 시도
            slab_mesh = trimesh.creation.extrude_polygon(polygon, height=thickness, engine="earcut")
            slab_mesh.apply_translation([0, 0, z_position])
            slab_mesh.metadata["type"] = part_type
            return slab_mesh
        except Exception as e2:
            # 수동으로 메쉬 생성
            return create_polygon_mesh_manual(polygon, thickness, z_position, part_type)


def create_polygon_mesh_manual(polygon, thickness, z_position, part_type="slab"):
    """수동으로 polygon 메쉬 생성 (fallback)"""
    import numpy as np
    from shapely.geometry import Polygon

    # polygon 좌표 추출
    coords = np.array(polygon.exterior.coords[:-1])  # 마지막 점 제외 (시작점과 동일)
    n = len(coords)

    if n < 3:
        # 너무 적은 점이면 bounding box로 대체
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        box = create_box((center_x, center_y, z_position + thickness/2), (width, height, thickness), part_type=part_type)
        return box

    # 상부/하부 면 vertices
    bottom_verts = np.column_stack([coords, np.full(n, z_position)])
    top_verts = np.column_stack([coords, np.full(n, z_position + thickness)])
    vertices = np.vstack([bottom_verts, top_verts])

    # triangulate 상부/하부 면 (간단한 fan triangulation)
    bottom_faces = []
    top_faces = []
    for i in range(1, n - 1):
        bottom_faces.append([0, i + 1, i])  # CCW for bottom (reversed)
        top_faces.append([n, n + i, n + i + 1])  # CCW for top

    # 측면 faces
    side_faces = []
    for i in range(n):
        next_i = (i + 1) % n
        # 두 개의 삼각형으로 사각형 측면 구성
        side_faces.append([i, next_i, n + next_i])
        side_faces.append([i, n + next_i, n + i])

    faces = np.array(bottom_faces + top_faces + side_faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    mesh.metadata["type"] = part_type
    return mesh


def create_polygon_structure(mass_polygon, num_floors, span, floor_height, basement_floors=0):
    """비정형 polygon 형태 구조물 생성 (매스라인 방향에 맞춘 축열)"""
    from shapely.geometry import Point

    # 원본 mass polygon 바운딩 박스 (슬라브 위치용)
    orig_bounds = mass_polygon.bounds
    orig_minx, orig_miny, orig_maxx, orig_maxy = orig_bounds

    # 1. 최소 회전 사각형으로 주축 방향 찾기
    min_rect = mass_polygon.minimum_rotated_rectangle
    angle = get_longest_edge_angle(min_rect)
    centroid = mass_polygon.centroid

    # 2. polygon을 주축 방향으로 회전 (world 축에 정렬) - 기둥/보 배치용
    aligned_poly = shapely.affinity.rotate(mass_polygon, -math.degrees(angle), origin=centroid)

    # 3. 정렬된 polygon의 바운딩 박스 기준으로 그리드 생성
    bounds = aligned_poly.bounds
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny

    num_grids_x = int(width // span) + 2
    num_grids_y = int(height // span) + 2

    column_size = (0.6, 0.6, floor_height)
    beam_size_x = (span, 0.3, 0.6)
    beam_size_y = (0.3, span, 0.6)
    slab_thickness = 0.2
    foundation_thickness = 0.6

    structural_boxes = []  # 기둥, 보 (회전 필요)
    slab_boxes = []        # 슬라브 (회전 불필요, 원본 형태 유지)

    # 정렬된 polygon 내부 그리드 포인트 찾기
    grid_points = []
    for i in range(num_grids_x):
        for j in range(num_grids_y):
            x = minx + i * span
            y = miny + j * span
            if aligned_poly.contains(Point(x, y)) or aligned_poly.buffer(0.5).contains(Point(x, y)):
                grid_points.append((i, j, x, y))

    grid_set = {(p[0], p[1]) for p in grid_points}

    # 원본 polygon을 로컬 좌표로 변환 (슬라브용 - 회전 없이 원본 형태 그대로)
    local_mass_poly = shapely.affinity.translate(mass_polygon, xoff=-orig_minx, yoff=-orig_miny)

    def add_floor_elements(z_base, is_foundation=False):
        # 기둥 (정렬된 좌표계에서)
        for i, j, x, y in grid_points:
            structural_boxes.append(create_box((x - minx, y - miny, z_base + column_size[2] / 2), column_size, part_type="column"))

        # X방향 보 (주축 방향)
        for i, j, x, y in grid_points:
            if (i + 1, j) in grid_set:
                bx = x - minx + span / 2
                by = y - miny
                structural_boxes.append(create_box((bx, by, z_base + floor_height - beam_size_x[2] / 2), beam_size_x, part_type="beam_x"))

        # Y방향 보 (부축 방향)
        for i, j, x, y in grid_points:
            if (i, j + 1) in grid_set:
                bx = x - minx
                by = y - miny + span / 2
                structural_boxes.append(create_box((bx, by, z_base + floor_height - beam_size_y[2] / 2), beam_size_y, part_type="beam_y"))

        # 슬라브 (원본 polygon 형태 그대로 - 회전 없음)
        slab_thick = foundation_thickness if is_foundation else slab_thickness
        z_slab = z_base
        part_type = "foundation" if is_foundation else "slab"

        slab_mesh = create_polygon_slab(local_mass_poly, slab_thick, z_slab, part_type)
        slab_boxes.append(slab_mesh)

    # Basement
    for b in range(basement_floors):
        z_base = - (b + 1) * floor_height
        is_foundation = (b == basement_floors - 1)
        add_floor_elements(z_base, is_foundation=is_foundation)

    # Ground and above-ground floors
    for floor in range(num_floors):
        z_base = floor * floor_height
        add_floor_elements(z_base)

        # 상부 슬라브 (천장) - 원본 polygon 형태
        top_slab = create_polygon_slab(local_mass_poly, slab_thickness, z_base + floor_height, "slab")
        slab_boxes.append(top_slab)

    # 4. 기둥/보만 원래 방향으로 회전
    local_center = ((maxx - minx) / 2, (maxy - miny) / 2)
    rotated_structural = rotate_boxes(structural_boxes, center=local_center, angle_rad=angle)

    # 5. 회전된 기둥/보를 원본 polygon 좌표계로 이동
    # 정렬된 중심에서 원본 중심으로의 오프셋 계산
    aligned_center_world = (minx + local_center[0], miny + local_center[1])
    offset_to_origin = (orig_minx - aligned_center_world[0] + local_center[0],
                        orig_miny - aligned_center_world[1] + local_center[1])

    # 기둥/보 이동
    translation_matrix = trimesh.transformations.translation_matrix([offset_to_origin[0], offset_to_origin[1], 0])
    moved_structural = [box.copy().apply_transform(translation_matrix) for box in rotated_structural]

    # 슬라브는 이미 원본 좌표계에 있음 (이동 불필요)
    all_boxes = moved_structural + slab_boxes

    return all_boxes, (orig_minx, orig_miny)


def create_rectangular_structure(width, height, num_floors, span, floor_height, basement_floors=0):
    """직사각형 형태의 구조물 생성"""
    num_grids_x = int(width // span) + 1
    num_grids_y = int(height // span) + 1

    column_size = (0.6, 0.6, floor_height)
    beam_x_size = (span, 0.3, 0.6)
    beam_y_size = (0.3, span, 0.6)
    slab_thickness = 0.2
    foundation_thickness = 0.6
    boxes = []

    # Basement
    for b in range(basement_floors):
        z_base = - (b + 1) * floor_height
        for i in range(num_grids_x):
            for j in range(num_grids_y):
                x = i * span
                y = j * span
                boxes.append(create_box((x, y, z_base + column_size[2] / 2), column_size, part_type="column"))
        for i in range(num_grids_x - 1):
            for j in range(num_grids_y):
                x = (i + 0.5) * span
                y = j * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_x_size[2] / 2), beam_x_size, part_type="beam_x"))
        for i in range(num_grids_x):
            for j in range(num_grids_y - 1):
                x = i * span
                y = (j + 0.5) * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_y_size[2] / 2), beam_y_size, part_type="beam_y"))
        slab_thick = foundation_thickness if (b == basement_floors - 1) else slab_thickness
        z_slab_center = z_base - slab_thick / 2 if (b == basement_floors - 1) else z_base + slab_thick / 2
        part = "foundation" if (b == basement_floors - 1) else "slab"
        boxes.append(create_box((width / 2, height / 2, z_slab_center), (width, height, slab_thick), part_type=part))

    # Ground and above-ground floors
    for floor in range(num_floors):
        z_base = floor * floor_height
        for i in range(num_grids_x):
            for j in range(num_grids_y):
                x = i * span
                y = j * span
                boxes.append(create_box((x, y, z_base + column_size[2] / 2), column_size, part_type="column"))
        for i in range(num_grids_x - 1):
            for j in range(num_grids_y):
                x = (i + 0.5) * span
                y = j * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_x_size[2] / 2), beam_x_size, part_type="beam_x"))
        for i in range(num_grids_x):
            for j in range(num_grids_y - 1):
                x = i * span
                y = (j + 0.5) * span
                boxes.append(create_box((x, y, z_base + floor_height - beam_y_size[2] / 2), beam_y_size, part_type="beam_y"))
        boxes.append(create_box((width / 2, height / 2, z_base + slab_thickness / 2), (width, height, slab_thickness), part_type="slab"))
        boxes.append(create_box((width / 2, height / 2, z_base + floor_height + slab_thickness / 2), (width, height, slab_thickness), part_type="slab"))

    return boxes

def get_longest_edge_angle(rect: Polygon):
    coords = list(rect.exterior.coords)
    max_len = 0
    angle = 0
    for i in range(len(coords) - 1):
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]
        length = (dx**2 + dy**2) ** 0.5
        if length > max_len:
            max_len = length
            angle = math.atan2(dy, dx)
    return angle

def rotate_boxes(boxes, center, angle_rad):
    rot = trimesh.transformations.rotation_matrix(angle_rad, direction=[0, 0, 1], point=[center[0], center[1], 0])
    return [box.copy().apply_transform(rot) for box in boxes]

def visualize_trimesh_boxes_plotly(boxes, polygon=None, setback=None, mass=None):
    fig = go.Figure()
    COLOR_MAP = {
        "column": "orange",
        "beam_x": "blue",
        "beam_y": "deepskyblue",
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
    if mass:
        add_boundary_trace(mass, "최적화 매스", "purple")

    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        title="🏗️ 부재별 구조 색상 시각화",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


def run_rl_optimization(env, num_episodes=100):
    """휴리스틱 탐색으로 최적 매스 찾기"""
    best_reward = -float('inf')
    best_action = None
    best_info = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    for episode in range(num_episodes):
        obs, info = env.reset()

        # 랜덤 탐색 + 그리디 개선
        for step in range(env.max_steps):
            if episode < num_episodes // 2:
                # 전반부: 랜덤 탐색
                action = env.action_space.sample()
            else:
                # 후반부: 이전 best 주변 탐색
                if best_action is not None:
                    noise = np.random.normal(0, 0.1, size=3).astype(np.float32)
                    action = np.clip(best_action + noise, env.action_space.low, env.action_space.high)
                else:
                    action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            if reward > best_reward and info.get("mass") is not None:
                if not info.get("violation_reason"):
                    best_reward = reward
                    best_action = action.copy()
                    best_info = info.copy()

            if terminated:
                break

        progress_bar.progress((episode + 1) / num_episodes)
        status_text.text(f"탐색 중... {episode + 1}/{num_episodes} (최고 보상: {best_reward:.2f})")

    progress_bar.empty()
    status_text.empty()

    return best_action, best_info, best_reward


def try_load_trained_model():
    """학습된 SAC 모델 로드 시도"""
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_names = ["sac_mass_stable_v1", "sac_mass_general_v1_stable"]

    for name in model_names:
        path = os.path.join(script_dir, name + ".zip")
        if os.path.exists(path):
            try:
                from stable_baselines3 import SAC
                model = SAC.load(path.replace(".zip", ""))
                return model, name
            except Exception as e:
                return None, str(e)
    return None, None


# ==================== Streamlit UI ====================

df = load_data()
st.title("📐 지번 기반 구조 자동 모델링")

# 모드 선택
st.sidebar.header("🎯 모델링 모드")
optimization_mode = st.sidebar.radio(
    "최적화 방식 선택",
    ["📐 규칙 기반", "🔍 휴리스틱 최적화", "🤖 강화학습 (SAC)"],
    index=0
)

if optimization_mode == "🔍 휴리스틱 최적화":
    st.sidebar.info("랜덤 탐색 + 로컬 서치로 현재 대지에 최적화된 매스를 찾습니다.")
    rl_episodes = st.sidebar.slider("탐색 횟수", 50, 500, 100, step=50)
elif optimization_mode == "🤖 강화학습 (SAC)":
    st.sidebar.info("학습된 SAC 신경망으로 빠르게 매스를 배치합니다.")
    st.sidebar.warning("⚠️ 일반화 성능이 낮을 수 있음")
else:
    st.sidebar.info("정사각형 그리드로 구조물을 배치합니다.")

# 디폴트 주소
def_시도 = "서울특별시"
def_시군구 = "영등포구"
def_읍면동 = "양평동1가"
def_본번 = 270
def_부번 = 0

시도_options = sorted(df["SIDO_NM"].dropna().unique())
시도 = st.selectbox("시도", 시도_options, index=시도_options.index(def_시도) if def_시도 in 시도_options else 0)
시군구_options = sorted(df[df["SIDO_NM"] == 시도]["SGG_NM"].dropna().unique())
시군구 = st.selectbox("시군구", 시군구_options, index=시군구_options.index(def_시군구) if def_시군구 in 시군구_options else 0)
읍면동_options = sorted(df[(df["SIDO_NM"] == 시도) & (df["SGG_NM"] == 시군구)]["EMD_NM"].dropna().unique())
읍면동 = st.selectbox("읍면동", 읍면동_options, index=읍면동_options.index(def_읍면동) if def_읍면동 in 읍면동_options else 0)
본번 = st.number_input("본번", min_value=1, step=1, value=def_본번)
부번 = st.number_input("부번", min_value=0, step=1, value=def_부번)

이격거리 = st.number_input("이격거리 (m)", value=2.0, step=0.5)
건폐율 = st.number_input("건폐율 (%)", value=60.0, step=5.0) / 100
용적률 = st.number_input("용적률 (%)", value=300.0, step=50.0) / 100
최대높이 = st.number_input("최대높이 (m)", value=15.0, step=1.0)
층고 = st.number_input("층당 층고 (m)", value=3.3, step=0.1)
스팬 = st.number_input("기둥 스팬 거리 (m)", value=6.0, step=0.5)
지하층수 = st.number_input("지하층 수", min_value=0, step=1, value=1)

if optimization_mode == "📐 규칙 기반":
    button_label = "📐 모델 생성"
elif optimization_mode == "🔍 휴리스틱 최적화":
    button_label = "🔍 휴리스틱 최적화 + 모델 생성"
else:
    button_label = "🤖 강화학습 + 모델 생성"

if st.button(button_label):
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

    mass_polygon = None
    rl_info = None
    use_optimization = optimization_mode != "📐 규칙 기반"

    if optimization_mode == "🔍 휴리스틱 최적화":
        # 휴리스틱 최적화 모드
        st.subheader("🔍 휴리스틱 최적화 진행 중...")

        config = {
            "setback": setback,
            "site_polygon": polygon,
            "대지면적": 대지면적,
            "건폐율": 건폐율,
            "용적률": 용적률,
            "최대높이": 최대높이,
            "층고": 층고
        }
        env = MassPlacementEnv(config)

        best_action, rl_info, best_reward = run_rl_optimization(env, num_episodes=rl_episodes)

        if rl_info:
            mass_polygon = rl_info.get("mass")
            st.success(f"✅ 최적화 완료! 최고 보상: {best_reward:.2f}")

        if mass_polygon and isinstance(mass_polygon, Polygon) and mass_polygon.area > 0:
            건축면적 = mass_polygon.area
            연면적 = rl_info.get("연면적", 건축면적 * (용적률 / 건폐율))
            층수 = rl_info.get("층수", max(1, int(min(연면적 / 건축면적, 최대높이 // 층고))))

            st.markdown("### 📊 휴리스틱 최적화 결과")
            col1, col2, col3 = st.columns(3)
            col1.metric("건축면적", f"{건축면적:.1f}㎡")
            col2.metric("연면적", f"{연면적:.1f}㎡")
            col3.metric("층수", f"{층수}층")
        else:
            st.warning("⚠️ 휴리스틱 최적화 실패. 규칙 기반 모드로 전환합니다.")
            use_optimization = False

    elif optimization_mode == "🤖 강화학습 (SAC)":
        # 강화학습 모드
        st.subheader("🤖 강화학습 (SAC) 추론 중...")

        config = {
            "setback": setback,
            "site_polygon": polygon,
            "대지면적": 대지면적,
            "건폐율": 건폐율,
            "용적률": 용적률,
            "최대높이": 최대높이,
            "층고": 층고
        }
        env = MassPlacementEnv(config)

        model, model_info = try_load_trained_model()

        if model is not None:
            st.success(f"✅ 학습된 모델 로드: {model_info}")
            obs, info = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _, rl_info = env.step(action)
            mass_polygon = rl_info.get("mass")

            if mass_polygon and isinstance(mass_polygon, Polygon) and mass_polygon.area > 0:
                건축면적 = mass_polygon.area
                연면적 = rl_info.get("연면적", 건축면적 * (용적률 / 건폐율))
                층수 = rl_info.get("층수", max(1, int(min(연면적 / 건축면적, 최대높이 // 층고))))

                st.markdown("### 📊 강화학습 추론 결과")
                col1, col2, col3 = st.columns(3)
                col1.metric("건축면적", f"{건축면적:.1f}㎡")
                col2.metric("연면적", f"{연면적:.1f}㎡")
                col3.metric("층수", f"{층수}층")
            else:
                st.warning("⚠️ 강화학습 최적화 실패. 규칙 기반 모드로 전환합니다.")
                use_optimization = False
        else:
            st.error(f"❌ 모델 로드 실패: {model_info}")
            st.warning("⚠️ 규칙 기반 모드로 전환합니다.")
            use_optimization = False

    if not use_optimization or mass_polygon is None:
        # 규칙 기반 모드
        건축면적 = 유효면적 * 건폐율
        연면적 = 건축면적 * (용적률 / 건폐율)
        층수 = max(1, int(min(연면적 / 건축면적, 최대높이 // 층고)))

    # mass_polygon이 있으면 그 형상 그대로 구조물 배치 (비정형)
    if mass_polygon and isinstance(mass_polygon, Polygon) and mass_polygon.area > 0:
        # 비정형 polygon 구조물 생성
        boxes, (minx, miny) = create_polygon_structure(
            mass_polygon=mass_polygon,
            num_floors=층수,
            span=스팬,
            floor_height=층고,
            basement_floors=지하층수
        )

        # 비정형은 회전 없음 (이미 polygon 형태 그대로)
        rotated_boxes = boxes
        origin_x = minx
        origin_y = miny

    else:
        # 규칙 기반: 정사각형
        side_len = np.sqrt(건축면적)
        width = height = side_len
        center = setback.centroid
        origin_x = center.x - side_len / 2
        origin_y = center.y - side_len / 2

        boxes, _ = auto_structure_model(
            floor_area=건축면적,
            num_floors=층수,
            span=스팬,
            floor_height=층고,
            offset=(0, 0),
            basement_floors=지하층수
        )
        angle = get_longest_edge_angle(setback.minimum_rotated_rectangle)
        rotated_boxes = rotate_boxes(boxes, center=(side_len / 2, side_len / 2), angle_rad=angle)

    polygon_local = shapely.affinity.translate(polygon, xoff=-origin_x, yoff=-origin_y)
    setback_local = shapely.affinity.translate(setback, xoff=-origin_x, yoff=-origin_y)
    mass_local = None
    if mass_polygon:
        mass_local = shapely.affinity.translate(mass_polygon, xoff=-origin_x, yoff=-origin_y)

    st.markdown(f"✅ **대지면적**: {대지면적:.1f}㎡ | **유효면적**: {유효면적:.1f}㎡")
    st.markdown(f"🏢 **건축면적**: {건축면적:.1f}㎡ | **연면적**: {연면적:.1f}㎡ | **지상층수**: {층수}층 | **지하층수**: {지하층수}층")

    if optimization_mode == "🔍 휴리스틱 최적화" and mass_polygon:
        st.markdown("🔍 **모드**: 휴리스틱 최적화")
    elif optimization_mode == "🤖 강화학습 (SAC)" and mass_polygon:
        st.markdown("🤖 **모드**: 강화학습 (SAC)")
    else:
        st.markdown("📐 **모드**: 규칙 기반")

    visualize_trimesh_boxes_plotly(rotated_boxes, polygon=polygon_local, setback=setback_local, mass=mass_local)
