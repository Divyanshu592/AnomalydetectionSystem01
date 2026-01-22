import numpy as np

def static_threshold(value: float = 0.05) -> float:
    """
    Basic fixed threshold (for quick testing)
    """
    return value


def percentile_threshold(errors: np.ndarray, percentile: float = 95) -> float:
    """
    Dynamic threshold based on training reconstruction errors.
    Example: 95th percentile
    """
    return float(np.percentile(errors, percentile))
