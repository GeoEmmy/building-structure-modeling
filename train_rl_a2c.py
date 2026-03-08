import numpy as np
from shapely.geometry import box
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# ✅ GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ✅ 랜덤 환경 구성
def generate_random_config():
    w, h = np.random.uniform(20, 50), np.random.uniform(20, 50)
    site = box(0, 0, w, h)
    setback_distance = np.random.uniform(1.0, 5.0)
    setback = site.buffer(-setback_distance).buffer(0)
    area = site.area

    return {
        "setback": setback,
        "site_polygon": site,
        "대지면적": area,
        "건폐율": np.random.uniform(0.4, 0.7),
        "용적률": np.random.uniform(1.5, 3.5),
        "최대높이": np.random.uniform(9.0, 30.0),
        "층고": np.random.choice([3.0, 3.3, 3.6])
    }

# ✅ VecEnv를 위한 wrapper
class RandomizedEnvWrapper:
    def __call__(self):
        return MassPlacementEnv(generate_random_config())

# ✅ 학습 실행
if __name__ == "__main__":
    env = DummyVecEnv([RandomizedEnvWrapper()])

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log="./a2c_logs"
    )

    model.learn(total_timesteps=1_000_000)
    model.save("a2c_mass_general_v1")
