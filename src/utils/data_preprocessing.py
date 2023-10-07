import numpy as np
from tqdm import tqdm


def apply_noise(X, n=0.167, s=0.03, c=0.7 * 0.005, cut=0.05):
    """
    Apply noise on the initial sample.

    Args:
        - X: np.array, shape=(n_events, 51, 51)
        - n: float, noise term (GeV)
        - s: float, stochastic term (GeV)
        - c: float, constant term (GeV)
    Returns:
        - X: np.array, shape=(n_events, 51, 51) with noise applied
    """

    # apply noise using Gaussian distribution
    sigma = np.sqrt(X * s**2 + n**2 + X**2 * c**2)
    X = np.random.normal(X, sigma)

    # apply cut on 0.05 GeV
    X[X < cut] = 0.0
    return X


def get_window(X, y, en, crop_size=7, threshold=0.5):
    """
    Function to crop the windows of the size crop_size x crop_size around crystals with deposited energy > threshold.
    Args:
        - X: np.array, one event, shape=(51, 51)
        - y: np.array, true coordinates, shape=(n_pcl, 2)
        - en: np.array, true energies, shape=(n_pcl, 1)
        - crop_size: int, size of the crop window
        - threshold: float, energy threshold for the crystals
    Returns:
        - X_crop: np.array, cropped windows, shape=(n_windows, crop_size, crop_size)
        - indices: np.array, indices of the crystals that pass the threshold, shape=(n_windows, 2)
        - is_seed: np.array, 1 if the crystal is a seed, 0 otherwise, shape=(n_windows)
        - y_crop: np.array, true coordinates of the crystals that are 1 crystal away from the seed, shape=(n_windows, 2)
        - en_crop: np.array, true energies of the crystals that are the same as the seed, shape=(n_windows, 1)
    """
    # create lists to store the data
    X_crop, is_seed, indices, y_crop, en_crop = [], [], [], [], []

    # find all the indices of the crystals that pass energy threshold (0.5 GeV)
    ind = np.argwhere(X > threshold)

    # prepare the cropped windows
    for j in ind:
        # crop the window around the energetic crystal
        Xi = X[
            j[0] - crop_size // 2 : j[0] + crop_size // 2 + 1,
            j[1] - crop_size // 2 : j[1] + crop_size // 2 + 1,
        ]

        if Xi.shape == (7, 7):  # don't include the windows on the borders
            indices.append(j)
            X_crop.append(Xi)

            # find the true coordinates that fall inside the processed crystal (if any)
            arg_seed = np.where(
                (j[0] == np.floor(y[:, 0])) & (j[1] == np.floor(y[:, 1]))
            )

            # find the true coordinates that are less than 1 crystal away from the processed crystal (if any)
            dr = np.sqrt((j[0] + 0.5 - y[:, 0]) ** 2 + (j[1] + 0.5 - y[:, 1]) ** 2)
            arg_coord = np.where(dr <= 1.0)

            is_seed.append(
                1 if len(arg_seed[0]) else 0
            )  # add 1 if the crystal is a seed, 0 otherwise

            # add the true coordinates (1 crystal away) and energies (same crystal) to the lists, if none - add 0 instead
            y_crop.append(y[arg_coord][0] if len(arg_coord[0]) else y[0] * 0.0)
            en_crop.append(en[arg_seed][0] if len(arg_seed[0]) else en[0] * 0.0)

    return X_crop, indices, is_seed, y_crop, en_crop


def get_model_samples(X, y, en, crop_size=7, threshold=0.5):
    """
    Function to format the input data for the model.
    Args:
        - X: np.array, input events, shape=(n_events, 51, 51)
        - y: np.array, true coordinates, shape=(n_events, n_pcl, 2)
        - en: np.array, true energies, shape=(n_events, n_pcl, 1)
    Returns:
        - X_crop: np.array, cropped windows, shape=(n_windows, crop_size, crop_size)
        - indices: np.array, indices of the crystals that pass the threshold, shape=(n_windows, 2)
        - is_seed: np.array, 1 if the crystal is a seed, 0 otherwise, shape=(n_windows)
        - y_crop: np.array, true coordinates of the crystals that are 1 crystal away from the seed, shape=(n_windows, 2)
        - en_crop: np.array, true energies of the crystals that are the same as the seed, shape=(n_windows, 1)
    """
    X_crop, indices, is_seed, y_crop, en_crop = [], [], [], [], []
    final_var = [X_crop, indices, is_seed, y_crop, en_crop]

    # number of the particles in the samples
    n_pcl = y.shape[1]

    for i, (X, y, en) in tqdm(enumerate(zip(X, y, en)), total=X.shape[0]):
        Xi, indi, is_seedi, yi, eni = get_window(X, y, en)

        # don't include the samples where two seeds fall into one crystal
        if np.sum(is_seedi) != n_pcl:
            continue

        # format the variables
        var = [Xi, indi, is_seedi, yi, eni]
        for i, _ in enumerate(var):
            var[i] = format_variable(var[i], pad=35)

        is_seedi = np.array(is_seedi).astype(int)

        # append the variables to the final lists
        for v, fv in zip(var, final_var):
            fv.append(v)
    # change the final lists to np.arrays
    final_var = [np.asarray(fv) for fv in final_var]

    return final_var


def format_variable(var, pad=35):
    """
    Function to format the variables for the model.
    """
    var = np.array(var)
    var = np.pad(
        var,
        [(0, pad - var.shape[0])] + [(0, 0)] * (var.ndim - 1),
        constant_values=-1,
    )

    return var
