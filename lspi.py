import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer


def features_tile(encoded_state, action):
    d = encoded_state.shape[0]
    A = 3
    encoded_state_action = np.zeros((d*A))
    encoded_state_with_bias = np.zeros((d))
    # encoded_state_with_bias[-1] = 1
    # encoded_state_with_bias[:-1] = encoded_state
    encoded_state_action[action * d:(action + 1) * d] = encoded_state
    return encoded_state_action


def compute_lspi_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma):
    next_states_actions = linear_policy.get_max_action(encoded_next_states)
    if linear_policy.include_bias:
        number_of_states = len(encoded_states)
        encoded_states = np.concatenate((encoded_states, np.ones((number_of_states, 1), np.float64)), axis=1)
        encoded_next_states = np.concatenate((encoded_next_states, np.ones((number_of_states, 1), np.float64)), axis=1)
    samples_num = encoded_states.shape[0]
    d = encoded_states.shape[1]
    A1 = np.zeros((3*d, 3*d))
    A2 = np.zeros((3*d, 3*d))
    b = np.zeros((3*d))
    for i in range(samples_num):
        phi_s = encoded_states[i, :]
        phi_sPrime = encoded_next_states[i, :]
        a = actions[i]
        r = rewards[i]
        phi_s_a = features_tile(phi_s, a)  # size of 3*d
        aPrime = next_states_actions[i]
        if not done_flags[i]:
            aPrime = next_states_actions[i]
        else:
            aPrime = 1
        phi_sPrime_a = features_tile(phi_sPrime, aPrime)
        # A1 += phi_s_a @ phi_sPrime_a.T
        # A2 += phi_s_a @ phi_s_a.T
        A1 += np.outer(phi_s_a,phi_sPrime_a)
        A2 += np.outer(phi_s_a,phi_s_a)
        b += r * phi_s_a
    A = gamma * A1 - A2
    next_w = np.linalg.inv(-A) @ b
    next_w = np.expand_dims(next_w, axis=1)
    return next_w


if __name__ == '__main__':
    samples_to_collect = 90000
    # samples_to_collect = 150000
    # samples_to_collect = 10000
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.999
    w_updates = 100
    evaluation_number_of_games = 10
    evaluation_max_steps_per_game = 1000

    np.random.seed(123)
    # np.random.seed(234)

    env = MountainCarWithResetEnv()
    # collect data
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    # get data success rate
    data_success_rate = np.sum(rewards) / len(rewards)
    print(f'success rate {data_success_rate}')
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
    for lspi_iteration in range(w_updates):
        print(f'starting lspi iteration {lspi_iteration}')

        new_w = compute_lspi_iteration(
            encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
        )
        norm_diff = linear_policy.set_w(new_w)
        if norm_diff < 0.00001:
            break
    print('done lspi')
    evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
    evaluator.play_game(evaluation_max_steps_per_game, render=True)



