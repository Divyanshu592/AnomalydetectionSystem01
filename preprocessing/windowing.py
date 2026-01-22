import numpy as np

def create_windows(data: np.ndarray, window_size: int):
    """
    data shape: (n_samples, n_features)
    return shape: (n_windows, window_size, n_features)
    """
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return np.array(windows)
