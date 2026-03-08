import gymnasium as gym
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import scale, translate
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

    def define_spaces(self):
        self.action_space = spaces.Box(
            low=np.array([0.5, -5.0, -5.0], dtype=np.float32),
            high=np.array([1.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.best_area = 0
        self.best_mass = None
        self.current_state = np.random.uniform(0, 1, size=7).astype(np.float32)
        return self.current_state, {}

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
            "mass": self.best_mass,
            "층수": floors,
            "연면적": total_area,
            "층고": self.층고,
            "violation_reason": reasons
        }

        terminated = self.current_step >= self.max_steps
        return obs, reward, terminated, False, info
