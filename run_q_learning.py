import numpy as np
import matplotlib.pyplot as plt

from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from q_learn_mountain_car import Solver, run_episode


def evaluate_criterion(env, solver):
    num_of_states = 10
    test_gains = [run_episode(env, solver, is_train=False, epsilon=None)[0] for _ in range(num_of_states)]
    mean_test_gain = np.mean(test_gains)
    return mean_test_gain


if __name__ == "__main__":
    env = MountainCarWithResetEnv()
    gamma = 0.999
    learning_rate = 0.05
    epsilon_current = 0.1
    epsilon_decrease = 1.
    epsilon_min = 0.05

    max_episodes = 10000
    seeds = [123, 321, 234]
    for i in range(len(seeds)):
        solver = Solver(
            # learning parameters
            gamma=gamma, learning_rate=learning_rate,
            # feature extraction parameters
            number_of_kernels_per_dim=[7, 5],
            # env dependencies (DO NOT CHANGE):
            number_of_actions=env.action_space.n,
        )
        seed = seeds[i]
        env.seed(seed)
        reward_for_plot = []
        SR_for_plot = []
        initial_state_value_for_plot = []
        avg_bellman_err_for_plot = []
        for episode_index in range(1, max_episodes + 1):
            episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon_current)
            # saving data
            reward_for_plot.append(episode_gain)
            SR_for_plot.append(evaluate_criterion(env, solver))

            s0 = env.reset_specific(-0.5, 0)
            phi_s0 = solver.get_features(s0)
            s0_greedy_action = solver.get_max_action(s0)
            initial_state_value_for_plot.append(solver.get_q_val(phi_s0, s0_greedy_action))

            avg_bellman_err_for_plot.append(mean_delta)

            # reduce epsilon if required
            # epsilon_current *= epsilon_decrease
            # epsilon_current = max(epsilon_current, epsilon_min)

            print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}, average error {mean_delta}')

            # termination condition:
            if episode_index % 10 == 9:
                print(f'tested 10 episodes: mean gain is {SR_for_plot[-1]}')
                if SR_for_plot[-1] >= -75.:
                    print(f'solved in {episode_index} episodes')
                    break
        if i == 1:
            Reward1 = np.array(reward_for_plot)
            SR1 = np.array(SR_for_plot)
            InitStateV1 = np.array(initial_state_value_for_plot)
            BellmanErr1 = np.array(avg_bellman_err_for_plot)
            X1 = range(0, len(reward_for_plot))
        if i == 2:
            Reward2 = np.array(reward_for_plot)
            SR2 = np.array(SR_for_plot)
            InitStateV2 = np.array(initial_state_value_for_plot)
            BellmanErr2 = np.array(avg_bellman_err_for_plot)
            X2 = range(0, len(reward_for_plot))
        if i == 3:
            Reward3 = np.array(reward_for_plot)
            SR3 = np.array(SR_for_plot)
            InitStateV3 = np.array(initial_state_value_for_plot)
            BellmanErr3 = np.array(avg_bellman_err_for_plot)
            X3 = range(0, len(reward_for_plot))

    # plotting the saved data
    # seed 1
    fig1, axs1 = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
    axs1[0, 0].plot(X1, Reward1)
    axs1[0, 0].set_xlabel('Episode')
    axs1[0, 0].set_ylabel('Reward')
    axs1[0, 0].set_title('Total reward gained as function of episode number')

    axs1[0, 1].plot(X1, SR1)
    axs1[0, 1].set_xlabel('Episode')
    axs1[0, 1].set_ylabel('Success Rate')
    axs1[0, 1].set_title('Success rate of the greedy policy as function of episode number')

    axs1[1, 0].plot(X1, InitStateV1)
    axs1[1, 0].set_xlabel('Episode')
    axs1[1, 0].set_ylabel('V(S0)')
    axs1[1, 0].set_title('The initial state value as function of episode number')

    axs1[1, 1].plot(X1, BellmanErr1)
    axs1[1, 1].set_xlabel('Episode')
    axs1[1, 1].set_ylabel('Bellman Error')
    axs1[1, 1].set_title('Average Bellman error as function of episode number')


    # run_episode(env, solver, is_train=False, render=True)