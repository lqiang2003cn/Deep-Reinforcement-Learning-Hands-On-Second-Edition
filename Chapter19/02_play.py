import sys
sys.path.append("/home/lq/lqtech/Deep-Reinforcement-Learning-Hands-On-Second-Edition/ptan")

import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from lib import model, kfac
from PIL import Image

import numpy as np
import torch


# ENV_ID = "HalfCheetahBulletEnv-v0"
ENV_ID = "Ant-v4"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="/home/lq/lqtech/Deep-Reinforcement-Learning-Hands-On-Second-Edition/saves/a2c-a2c_baseline_ant/best_+960.559_800000.dat", help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", default=True, help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("-s", "--save", type=int, help="If specified, save every N-th step as an image")
    parser.add_argument("--acktr", default=False, action='store_true', help="Enable Acktr-specific tweaks")
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="rgb_array")
    if args.record:
        env = RecordVideo(
            env,
            video_folder=ENV_ID,    # Folder to save videos
            name_prefix="eval",               # Prefix for video filenames
            episode_trigger=lambda x: True    # Record every episode
        )

    net = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0])
    if args.acktr:
        opt = kfac.KFACOptimizer(net)
    net.load_state_dict(torch.load(args.model))

    obs = env.reset()[0]
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor(obs)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        if np.isscalar(action): 
            action = [action]
        obs, reward, done, is_pruned, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done or is_pruned:
            break
        if args.save is not None and total_steps % args.save == 0:
            o = env.render('rgb_array')
            img = Image.fromarray(o)
            img.save("img_%05d.png" % total_steps)
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
    env.close()
