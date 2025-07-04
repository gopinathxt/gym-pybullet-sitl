import os
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.CFAviary import CFAviary

def main():
    output_folder = 'results'
    filename = os.path.join(output_folder, 'cf_save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(filename, exist_ok=True)

    env = make_vec_env(CFAviary, n_envs=1)
    print('[INFO] Action space:', env.action_space)
    print('[INFO] Observation space:', env.observation_space)

    model = PPO('MlpPolicy', env, verbose=1,
                n_epochs=2,
                batch_size=64,
                learning_rate=1e-3,
                n_steps=512,
                policy_kwargs=dict(net_arch=[32, 32])
                )
    model.learn(total_timesteps=100_000)
    model.save(filename + '/final_model.zip')
    print("Model saved to", filename)

    # Test the trained model
    test_env = CFAviary(gui=True)
    obs, info = test_env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"Step {i}: reward={reward}, terminated={terminated}, truncated={truncated}")
        test_env.render()
        if terminated or truncated:
            obs, info = test_env.reset()
    test_env.close()

if __name__ == '__main__':
    main()