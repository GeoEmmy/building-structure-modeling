# 목표: 조건을 만족하며 용적률(연면적/대지면적)을 최대화하는 매스 찾기

import gymnasium as gym
import numpy as np
from shapely.geometry import box, Polygon
import shapely.affinity
from gymnasium import spaces

class MassPlacementEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_steps = 30
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

        minx, miny, maxx, maxy = self.setback.bounds
        self.minx, self.miny, self.maxx, self.maxy = minx, miny, maxx, maxy
        self.width = maxx - minx
        self.height = maxy - miny

    def define_spaces(self):
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([self.minx, self.miny, 3, 3, -90], dtype=np.float32),
            high=np.array([self.maxx, self.maxy, 20, 20, 90], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.best_area = 0
        self.best_mass = None
        self.current_state = np.random.uniform(0, 1, size=9).astype(np.float32)
        return self.current_state, {}

    def step(self, action):
        self.current_step += 1
        x, y, w, h, angle = action
        mass = box(x, y, x + w, y + h)
        rotated = shapely.affinity.rotate(mass, angle, origin='center')

        total_area = 0
        floors = 0
        reward = 0
        reasons = []

        # ✅ 조건 평가
        if not self.setback.contains(rotated):
            reasons.append("setback 위반")
        if rotated.area > self.max_building_area:
            reasons.append("건폐율 초과")
        if rotated.area > 0:
            floors = min(self.max_floors, int(self.max_total_area // rotated.area))
            total_area = rotated.area * floors
            if total_area > self.max_total_area:
                reasons.append("용적률 초과")
        else:
            reasons.append("면적 0")

        # ✅ 보상 계산 개선
        penalty = 0
        if "setback 위반" in reasons:
            penalty += 1.0
        if "건폐율 초과" in reasons:
            penalty += 1.0
        if "용적률 초과" in reasons:
            penalty += 1.0
        if "면적 0" in reasons:
            penalty += 2.0

        area_ratio = total_area / self.max_total_area if self.max_total_area > 0 else 0
        floor_bonus = floors / self.max_floors if self.max_floors > 0 else 0
        reward = (area_ratio * 10) + (floor_bonus * 5) - penalty * 1.0

        # ✅ 최적 매스 저장
        if not reasons and total_area > self.best_area:
            self.best_area = total_area
            self.best_mass = rotated

        # ✅ 관측값 구성
        norm_x = (rotated.bounds[0] - self.minx) / self.width
        norm_y = (rotated.bounds[1] - self.miny) / self.height
        norm_area = rotated.area / self.max_building_area
        norm_length = rotated.length / (2 * (self.width + self.height))

        obs = np.array([
            norm_x,
            norm_y,
            norm_area,
            norm_length,
            self.건폐율,
            self.용적률 / 5.0,
            self.최대높이 / 50.0,
            self.층고 / 5.0,
            total_area / self.max_total_area
        ], dtype=np.float32)

        info = {
            "mass": self.best_mass,
            "층수": floors,
            "연면적": total_area,
            "층고": self.층고,
            "violation_reason": reasons
        }

        terminated = self.current_step >= self.max_steps
        return obs, reward, terminated, False, info
