import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset1 = np.load('epsilon1.npz')
    X1 = dataset1['X1']
    Reward1 = dataset1['Reward']
    dataset2 = np.load('epsilon2.npz')
    X2 = dataset2['X1']
    Reward2 = dataset2['Reward']
    dataset3 = np.load('epsilon3.npz')
    X3 = dataset3['X1']
    Reward3 = dataset3['Reward']
    dataset4 = np.load('epsilon4.npz')
    X4 = dataset4['X1']
    Reward4 = dataset4['Reward']
    dataset5 = np.load('epsilon5.npz')
    X5 = dataset5['X1']
    Reward5 = dataset5['Reward']
    axis_list = [X1, X2, X3, X4, X5]
    idx = np.argmin([len(X1), len(X2), len(X3), len(X4), len(X5)])
    final_axis = axis_list[idx]
    fig, ax = plt.subplots()
    ax.plot(final_axis, Reward1[:len(axis_list[idx])])
    ax.plot(final_axis, Reward2[:len(axis_list[idx])])
    ax.plot(final_axis, Reward3[:len(axis_list[idx])])
    ax.plot(final_axis, Reward4[:len(axis_list[idx])])
    ax.plot(final_axis, Reward5[:len(axis_list[idx])])
    ax.legend(['epsilon = 1', 'epsilon = 0.75', 'epsilon = 0.5', 'epsilon = 0.3', 'epsilon = 0.01'])
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episode')
    ax.set_title('Total reward gained as function of episode number for different exploration parameters')
    plt.show()