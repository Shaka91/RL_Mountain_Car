from radial_basis_function_extractor import RadialBasisFunctionExtractor
from mountain_car_with_data_collection import MountainCarWithResetEnv
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt


if __name__ == "__main__":
    number_of_kernels_per_dim = [12, 10]
    RBF = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    model = MountainCarWithResetEnv()
    state_discrete = []
    # minPos = model.min_position
    # maxPos = model.max_position
    # maxSpeed = model.max_speed
    # minSpeed = -model.max_speed
    minPos = -2.5
    maxPos = -1
    maxSpeed = -1
    minSpeed = -2.5
    Res = 1000
    posDiscrete = np.linspace(minPos, maxPos, Res)
    speedDiscrete = np.linspace(minSpeed, maxSpeed, Res)
    Posv, Speedv = np.meshgrid(posDiscrete, speedDiscrete, indexing='ij')
    features = []
    states = []
    for i in range(len(Posv)):
        for j in range(len(Speedv)):
            states.append([Posv[i, j], Speedv[i, j]])
    states = np.array(states)
    # states = states.T
    features = RBF.encode_states_with_radial_basis_functions(states)
    k = 0
    feature1_plot = np.reshape(features[:, 0], (Res, Res))
    feature2_plot= np.reshape(features[:, 1], (Res, Res))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(Posv, Speedv, feature1_plot, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(Posv, Speedv, feature2_plot, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

