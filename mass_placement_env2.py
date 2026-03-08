import gymnasium as gym
import numpy as np
from shapely.geometry import box, Polygon
import shapely.affinity
from gymnasium import spaces
from stable_baselines3 import PPO

class MassPlacementEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.set_parameters(config)
        self.define_spaces()

    def set_parameters(self, config):
        self.setback = config["setback"]
        self.site_polygon = config["site_polygon"]
        self.대지면적 = config["대지면적"]
        self.건폐율 = config["건폐율"]
        self.용적률 = config["용적률"]
        self.층수 = config["층수"]

        self.target_building_area = self.대지면적 * self.건폐율
        self.target_total_area = self.대지면적 * self.용적률

        minx, miny, maxx, maxy = self.setback.bounds
        self.minx, self.miny, self.maxx, self.maxy = minx, miny, maxx, maxy
        self.width = maxx - minx
        self.height = maxy - miny

    def define_spaces(self):
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([self.minx, self.miny, 5, 5, -180], dtype=np.float32),
            high=np.array([self.maxx, self.maxy, self.width, self.height, 180], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = np.zeros(4, dtype=np.float32)
        return self.current_state, {}

    def step(self, action):
        x, y, w, h, angle = action
        mass = box(x, y, x + w, y + h)
        rotated = shapely.affinity.rotate(mass, angle, origin='center')

        valid = True
        reasons = []

        if not self.setback.contains(rotated):
            valid = False
            reasons.append("setback 위반")
        if rotated.area > self.target_building_area:
            valid = False
            reasons.append("건폐율 초과")

        reward = 1.0 if valid else -1.0
        terminated = True
        truncated = False

        norm_x = (rotated.bounds[0] - self.minx) / self.width
        norm_y = (rotated.bounds[1] - self.miny) / self.height
        norm_area = rotated.area / self.target_building_area
        norm_length = rotated.length / (2 * (self.width + self.height))

        obs = np.array([norm_x, norm_y, norm_area, norm_length], dtype=np.float32)

        info = {"mass": rotated if valid else None, "violation_reason": reasons}
        return obs, reward, terminated, truncated, info

def generate_random_config():
    site = box(0, 0, 30, 30)
    setback = site.buffer(-2).buffer(0)
    return {
        "setback": setback,
        "site_polygon": site,
        "대지면적": site.area,
        "건폐율": np.random.uniform(0.4, 0.7),
        "용적률": np.random.uniform(1.5, 3.0),
        "층수": np.random.randint(1, 6)
    }

def train_general_model():
    env = MassPlacementEnv(generate_random_config())
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("ppo_mass_general")

if __name__ == '__main__':
    train_general_model()
