import numpy as np
from shapely.geometry import box
from MaximizeMassing import MassPlacementEnv
from stable_baselines3 import SAC
import matplotlib.pyplot as plt

# ✅ 학습된 모델 불러오기
model = SAC.load("sac_mass_stable_v1")

# ✅ 검증용 환경 목록 생성
def generate_test_configs():
    configs = []
    sizes = [(30, 30), (40, 40), (50, 50)]
    setbacks = [2.0, 3.0, 5.0]
    건폐율s = [0.4, 0.5, 0.6]
    용적률s = [1.5, 2.0, 3.0]
    최대높이s = [9.0, 15.0, 24.0]
    층고s = [3.0, 3.3, 3.6]

    for (w, h) in sizes:
        for setback_d in setbacks:
            for 건폐율 in 건폐율s:
                for 용적률 in 용적률s:
                    for 최대높이 in 최대높이s:
                        for 층고 in 층고s:
                            site = box(0, 0, w, h)
                            setback = site.buffer(-setback_d).buffer(0)
                            area = site.area
                            config = {
                                "setback": setback,
                                "site_polygon": site,
                                "대지면적": area,
                                "건폐율": 건폐율,
                                "용적률": 용적률,
                                "최대높이": 최대높이,
                                "층고": 층고
                            }
                            configs.append(config)
    return configs

# ✅ 단일 검증 함수
def evaluate_config(config):
    env = MassPlacementEnv(config)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    return total_reward, info

# ✅ 실행
if __name__ == "__main__":
    configs = generate_test_configs()
    print(f"🔍 총 검증 케이스 수: {len(configs)}")

    good = 0
    for i, config in enumerate(configs):
        reward, info = evaluate_config(config)
        is_valid = (len(info["violation_reason"]) == 0)
        if is_valid:
            good += 1
        print(f"[{i+1}/{len(configs)}] ✅ 보상: {reward:.2f} | 유효: {is_valid} | 연면적: {info['연면적']:.2f} | 층수: {info['층수']} | 위반: {info['violation_reason']}")

    print(f"\n🎯 조건 충족률: {good}/{len(configs)} ({(good/len(configs))*100:.1f}%)")
