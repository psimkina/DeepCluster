import pandas as pd 
import numpy as np

def create_dataframe(pred, true):
    """
    Create a dataframe with the predictions and the true values.
    Args:
        - pred: tuple, predictions from the model
        - true: tuple, true values
    Returns:
        - df: pd.DataFrame, dataframe with the predictions and the true values
    """
    # get the predictions and the true values
    yc, en, ys, seed_pr = pred
    X, y, en = true

    #en_centr = X[:,:,3,3]

    # dataset with true info
    df_true = pd.DataFrame()
    df_true['true_en'] = en.flatten()
    df_true['deposited_en'] = np.sum(X, axis=(1,2)).flatten()
    df_true['true_x'] = y[:,:,0].flatten()
    df_true['true_y'] = y[:,:,1].flatten()

    return df_true