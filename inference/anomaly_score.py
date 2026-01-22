import numpy as np

def reconstruction_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes reconstruction error per window
    y_true shape: (n_windows, window_size, n_features)
    y_pred shape: (n_windows, window_size, n_features)
    returns shape: (n_windows,)
    """
    errors = np.mean(np.square(y_pred - y_true), axis=(1, 2))
    return errors
