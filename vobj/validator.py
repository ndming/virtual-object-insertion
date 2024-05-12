import numpy as np


def is_ccw(coords: np.ndarray) -> bool:
    """
    Check if the sequence of 2D coordinates is specified in CCW order. Note that 
    coordinates obtained from 2D images usually have the `y`-aixs pointing 
    downward, thus the result of `is_ccw` for such coordinates is flipped.

    Parameters:
    ---
    - `coords`: an `n-by-2` array of 2D coordinates

    Returns:
    ---
    `True` if coordinates are specified in CCW order, `False` otherwise.
    """

    # Construct an array of adjacent coordinates
    adj_coords = np.zeros(coords.shape)
    adj_coords[0:-1] = coords[1:]
    adj_coords[-1] = coords[0]
    
    # Compute the sign area using the shoelace algorithm
    s_area = coords * adj_coords[:, [1, 0]]
    s_area = np.sum(s_area[:, 0] - s_area[:, 1])
    
    if s_area == 0:
        raise ValueError(s_area)
    
    return s_area > 0


def is_convex(coords: np.ndarray) -> bool:
    """
    Check if the sequence of 2D coordinates forms a convex polygon..

    Parameters:
    ---
    - `coords`: an `n-by-2` array of 2D coordinates

    Returns:
    ---
    `True` if coordinates form a convex polygon, `False` otherwise.
    """

    if len(coords) < 3:
        return False
    
    # Calculate vectors between consecutive points
    vectors = np.diff(coords, axis=0)

    # Calculate cross product of consecutive vectors
    cross_product = np.cross(vectors[:-1], vectors[1:])

    # Check if all cross products have the same sign
    return np.all(cross_product >= 0) or np.all(cross_product <= 0)