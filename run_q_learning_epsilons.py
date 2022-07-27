import numpy as np
import matplotlib.pyplot as plt

from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from q_learn_mountain_car import Solver, run_episode, moving_average


if __name__ == "__main__":
    env = MountainCarWithResetEnv()
    seed = 123
    np.random.seed(seed)
    env.seed(seed)

    gamma = 0.999
    learning_rate = 0.05
    epsilon = 0.01

    max_episodes = 100000
    solver = Solver(
        # learning parameters
        gamma=gamma, learning_rate=learning_rate,
        # feature extraction parameters
        number_of_kernels_per_dim=[7, 5],
        # env dependencies (DO NOT CHANGE):
        number_of_actions=env.action_space.n,
    )
    reward_for_plot = []
    for episode_index in range(1, max_episodes + 1):
    # for episode_index in range(1, 300):
        episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon)
        print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon}, average error {mean_delta}')

        # saving data
        reward_for_plot.append(episode_gain)

        # termination condition:
        if episode_index % 10 == 9:
            test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
            mean_test_gain = np.mean(test_gains)
            print(f'tested 10 episodes: mean gain is {mean_test_gain}')
            if mean_test_gain >= -75.:
                print(f'solved in {episode_index} episodes')
                break
    Reward = np.array(reward_for_plot)
    # Reward = moving_average(Reward, 100)
    X1 = range(0, len(Reward))
    np.savez('epsilon5', Reward=Reward, X1=X1)
    run_episode(env, solver, is_train=False, render=True)
