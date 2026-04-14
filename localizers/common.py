import numpy as np


def get_default_K(shape: tuple[int, int]) -> np.ndarray:
    max_size = max(shape)
    FOCAL_PRIOR = 1.2
    f = FOCAL_PRIOR * max_size
    height, width = shape
    cx = width / 2
    cy = height / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    return K
