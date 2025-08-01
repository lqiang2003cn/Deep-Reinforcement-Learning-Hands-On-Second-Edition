import ptan
import numpy as np
import torch
import math

def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()[0]
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            
            # no noise added, so test rewards are more stable
            action = np.clip(action, -1, 1)
            if np.isscalar(action): 
                action = [action]
            obs, reward, done, is_pruned, b = env.step(action)
            rewards += reward
            steps += 1
            if done or is_pruned:
                break
    return rewards / count, steps / count

def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2



