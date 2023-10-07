import numpy as np 
import matplotlib.pyplot as plt

def plot_event(X, y, en, ax=None): 
    """
    Plot a single event from the dataset. 
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # prepare data for plotting
    X[X == 0] = np.nan
    yi, xi = np.mgrid[slice(0, X.shape[0] + 1, 1), slice(0, X.shape[1] + 1, 1)]
    eb = ax.pcolormesh(xi, yi, X, cmap='viridis')

    # plotting
    plt.colorbar(eb, ax=ax)
    ax.scatter(y[:,1], y[:, 0], c='red', marker='*', s=10, label='true')

    # set labels and style
    ax.legend()