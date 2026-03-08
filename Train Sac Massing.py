import numpy as np
from shapely.geometry import box
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from MaximizeMassing import MassPlacementEnv  # 환경 파일

# ✅ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ 랜덤 환경 구성 함수
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

# ✅ 환경 래퍼
class RandomizedEnvWrapper:
    def __call__(self):
        return MassPlacementEnv(generate_random_config())

if __name__ == "__main__":
    # ✅ 벡터화 환경 구성
    env = DummyVecEnv([RandomizedEnvWrapper()])

    # ✅ SAC 모델 정의 및 안정화 파라미터
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.02,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto_0.1',
        target_update_interval=1,
        tensorboard_log="./sac_logs"
    )

    # ✅ 학습 시작
    model.learn(total_timesteps=1_000_000)

    # ✅ 모델 저장
    model.save("sac_mass_general_v2")
