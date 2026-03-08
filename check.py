from stable_baselines3 import PPO

model = PPO.load("ppo_mass.zip")

print("Policy:", model.policy)
print("Observation space:", model.observation_space)
print("Action space:", model.action_space)
print("Number of parameters:", sum(p.numel() for p in model.policy.parameters()))
