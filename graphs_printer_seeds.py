import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = np.load('seed234.npz')
    X1 = dataset['X1']
    Reward = dataset['Reward']
    SR = dataset['SR']
    InitStateV = dataset['InitStateV']
    BellmanErr = dataset['BellmanErr']
    fig1, axs1 = plt.subplots(2, 2)
    axs1[0, 0].plot(X1, Reward)
    axs1[0, 0].set_xlabel('Episode')
    axs1[0, 0].set_ylabel('Reward')
    axs1[0, 0].set_title('Total reward gained as function of episode number')

    SRaxis = np.arange(len(SR)) * 10
    axs1[0, 1].plot(SRaxis, SR)
    axs1[0, 1].set_xlabel('Episode')
    axs1[0, 1].set_ylabel('Success Rate')
    axs1[0, 1].set_title('Success rate of the greedy policy as function of episode number')

    axs1[1, 0].plot(X1, InitStateV)
    axs1[1, 0].set_xlabel('Episode')
    axs1[1, 0].set_ylabel('V(S0)')
    axs1[1, 0].set_title('The initial state value as function of episode number')
    X1 = np.arange(len(BellmanErr))
    axs1[1, 1].plot(X1, BellmanErr)
    axs1[1, 1].set_xlabel('Episode')
    axs1[1, 1].set_ylabel('Bellman Error')
    axs1[1, 1].set_title('Average Bellman error as function of episode number')
    plt.show()

    # run_episode(env, solver, is_train=False, render=True)