from radial_basis_function_extractor import RadialBasisFunctionExtractor
from mountain_car_with_data_collection import MountainCarWithResetEnv
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import matplotlib as plt


if __name__ == "__main__":
    RBF = RadialBasisFunctionExtractor([2, 2])
    model = MountainCarWithResetEnv()
    state_discrete = []
    minPos = model.min_position
    maxPos = model.max_position
    maxSpeed = model.max_speed
    minSpeed = -model.max_speed
    posDiscrete = np.linspace(minPos, maxPos, 300)
    speedDiscrete = np.linspace(minSpeed, maxSpeed, 300)
    Posv, Speedv = np.meshgrid(posDiscrete, speedDiscrete, indexing='ij')
    features = []
    for i in range(len(Posv)):
        for j in range(len(Speedv)):
            state = np.array([Posv[i,j], Speedv[i,j]])
            features.append(RBF.encode_states_with_radial_basis_functions(state))
    features = np.array(features)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(Posv, Speedv, features, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

