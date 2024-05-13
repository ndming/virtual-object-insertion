import numpy as np


def adapt_basis(vectors: np.ndarray) -> np.ndarray:
    i = np.array([ 0.,  0.,  1.])
    j = np.array([-1.,  0.,  0.])
    k = np.array([ 0.,  1.,  0.])

    # Since the transform is orthonormal, the inverse of the column matrix
    # is equivalent to its transpose (the row matrix itself)
    adapted_vectors = (np.vstack([i, j, k]) @ vectors.T).T
    return adapted_vectors.astype(np.float32)