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

# ==================== RL 환경 ====================

class MassPlacementEnv(gym.Env):
    """강화학습 기반 매스 배치 환경"""
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
        add_boundary_trace(mass, "RL 매스", "purple")

    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        title="🏗️ 부재별 구조 색상 시각화",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


def run_rl_optimization(env, num_episodes=100):
    """RL 탐색으로 최적 매스 찾기 (학습된 모델 없이 탐색)"""
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
    try:
        from stable_baselines3 import SAC
        import os

        model_paths = [
            "sac_mass_stable_v1.zip",
            "sac_mass_stable_v1",
            "sac_mass_general_v1_stable.zip",
            "sac_mass_general_v1_stable"
        ]

        for path in model_paths:
            if os.path.exists(path) or os.path.exists(path + ".zip"):
                model = SAC.load(path.replace(".zip", ""))
                return model, path
        return None, None
    except ImportError:
        return None, None
    except Exception as e:
        st.warning(f"모델 로드 실패: {e}")
        return None, None


# ==================== Streamlit UI ====================

df = load_data()
st.title("📐 지번 기반 구조 자동 모델링")

# 모드 선택
st.sidebar.header("🎯 모델링 모드")
use_rl = st.sidebar.checkbox("🤖 강화학습(RL) 모드 사용", value=False)

if use_rl:
    st.sidebar.info("RL 모드: 대지 형상에 최적화된 매스 배치를 탐색합니다.")
    rl_episodes = st.sidebar.slider("탐색 횟수", 50, 500, 100, step=50)
else:
    st.sidebar.info("규칙 기반 모드: 정사각형 그리드로 구조물을 배치합니다.")

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

button_label = "🤖 RL 최적화 + 모델 생성" if use_rl else "🔍 모델 생성"

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

    if use_rl:
        # RL 모드
        st.subheader("🤖 강화학습 최적화 진행 중...")

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

        # 학습된 모델 시도
        model, model_path = try_load_trained_model()

        if model is not None:
            st.success(f"✅ 학습된 모델 로드: {model_path}")
            obs, info = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _, rl_info = env.step(action)
            mass_polygon = rl_info.get("mass")
        else:
            st.info("📊 학습된 모델이 없어 탐색 기반 최적화를 진행합니다...")
            best_action, rl_info, best_reward = run_rl_optimization(env, num_episodes=rl_episodes)

            if rl_info:
                mass_polygon = rl_info.get("mass")
                st.success(f"✅ 최적화 완료! 최고 보상: {best_reward:.2f}")

        if mass_polygon and isinstance(mass_polygon, Polygon) and mass_polygon.area > 0:
            건축면적 = mass_polygon.area
            연면적 = rl_info.get("연면적", 건축면적 * (용적률 / 건폐율))
            층수 = rl_info.get("층수", max(1, int(min(연면적 / 건축면적, 최대높이 // 층고))))

            st.markdown("### 📊 RL 최적화 결과")
            col1, col2, col3 = st.columns(3)
            col1.metric("건축면적", f"{건축면적:.1f}㎡")
            col2.metric("연면적", f"{연면적:.1f}㎡")
            col3.metric("층수", f"{층수}층")
        else:
            st.warning("⚠️ RL 최적화 실패. 규칙 기반 모드로 전환합니다.")
            use_rl = False

    if not use_rl or mass_polygon is None:
        # 규칙 기반 모드
        건축면적 = 유효면적 * 건폐율
        연면적 = 건축면적 * (용적률 / 건폐율)
        층수 = max(1, int(min(연면적 / 건축면적, 최대높이 // 층고)))

    side_len = np.sqrt(건축면적)
    center = setback.centroid
    origin_x = center.x - side_len / 2
    origin_y = center.y - side_len / 2

    # 구조 모델 생성
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

    if use_rl and mass_polygon:
        st.markdown("🤖 **모드**: 강화학습 최적화")
    else:
        st.markdown("📐 **모드**: 규칙 기반")

    visualize_trimesh_boxes_plotly(rotated_boxes, polygon=polygon_local, setback=setback_local, mass=mass_local)
