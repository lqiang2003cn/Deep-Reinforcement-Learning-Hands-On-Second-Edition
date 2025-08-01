import sys
sys.path.append("/home/lq/lqtech/Deep-Reinforcement-Learning-Hands-On-Second-Edition/ptan")


import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from lib import model

import numpy as np
import torch

# ENV_ID = "MinitaurBulletEnv-v0"
ENV_ID = "Hopper-v5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="/home/lq/lqtech/Deep-Reinforcement-Learning-Hands-On-Second-Edition/saves/d4pg-d4pg/best_+432.499_541000.dat", help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", default=True, help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="rgb_array")
    if args.record:
        env = RecordVideo(
            env,
            video_folder=ENV_ID,    # Folder to save videos
            name_prefix="eval",               # Prefix for video filenames
            episode_trigger=lambda x: True    # Record every episode
        )

    # d4pg uses the same actor as ddpg
    net = model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))

    obs = env.reset()[0]
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, _,_ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
    env.close()
