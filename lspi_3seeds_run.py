import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer
from lspi import compute_lspi_iteration
import matplotlib.pyplot as plt
import time


def evaluation_criterion(evaluator):
    samples_to_collect = 50
    max_steps_per_game = 1000
    sum = 0
    for j in range(samples_to_collect):
        done = float(evaluator.play_game(max_steps_per_game, render=False))
        sum += done
    success_rate = sum / samples_to_collect
    return success_rate


if __name__ == '__main__':
    samples_to_collect = 100000
    number_of_kernels_per_dim = [12, 10]
    success_rate = np.zeros(3)
    gamma = 0.99
    w_updates = 1
    evaluation_number_of_games = 10
    evaluation_max_steps_per_game = 1000
    seeds = [123, 321, 234]
    this_iter_success_rate = np.zeros((len(seeds), w_updates + 1))
    env = MountainCarWithResetEnv()
    for i in range(len(seeds)):
        seed = seeds[i]
        print(f'begin lspi for seed "{seed}"')
        env.seed(seed)
        # collect data
        states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
        # get data success rate
        data_success_rate = np.sum(rewards) / len(rewards)
        print(f'data success rate {data_success_rate}')
        # standardize data
        data_transformer = DataTransformer()
        data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
        states = data_transformer.transform_states(states)
        next_states = data_transformer.transform_states(next_states)
        # process with radial basis functions
        feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        # encode all states:
        encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
        encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
        # set a new linear policy
        linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
        # but set the weights as random
        linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
        # start an object that evaluates the success rate over time
        evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
        this_iter_success_rate[i, 0] = evaluation_criterion(evaluator)
        for lspi_iteration in range(1, w_updates + 1):
            print(f'starting lspi iteration {lspi_iteration}')

            new_w = compute_lspi_iteration(
                encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
            )
            norm_diff = linear_policy.set_w(new_w)
            # this_iter_success_rate[lspi_iteration] = np.mean(success_rate_array_per_seed)
            this_iter_success_rate[i, lspi_iteration] = evaluation_criterion(evaluator)
            if norm_diff < 0.00001:
                break
        print(f'done lspi for seed {seed}')

    fig, ax = plt.subplots()
    ax.plot(range(w_updates + 1), np.mean(this_iter_success_rate, axis=1))
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('Iteration #')
    ax.title('Success rate as function of iterations number')
    plt.show()
    # evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
    # evaluator.play_game(evaluation_max_steps_per_game, render=True)