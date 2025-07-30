import gymnasium as gym
# import pybullet_envs

# ENV_ID = "MinitaurBulletEnv-v0"
# ENV_ID = "HalfCheetah-v4"
ENV_ID = "InvertedPendulum-v4"
RENDER = True


if __name__ == "__main__":
    spec = gym.envs.registry[ENV_ID]
    # spec.kwargs['render'] = RENDER
    env = gym.make(ENV_ID, render_mode="human")

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(env)
    print(env.reset())
    input("Press any key to exit\n")
    env.close()
