import numpy as np


def map(X, YMIN, YMAX, XMIN, XMAX):
    X = np.array(X)  # Ensure X is a NumPy array
    Y = (YMAX - YMIN) * (X - XMIN) / (XMAX - XMIN) + YMIN
    return Y