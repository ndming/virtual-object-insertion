import numpy as np


def polygon_mask(
    img_coords: np.ndarray, 
    shape: np.ndarray | tuple[int, int]
) -> np.ndarray[bool]:
    """
    Generates a boolean mask of size `shape` indicating whether each point in a 
    grid lies within the polygon defined by coordinates sequence in `img_coords`.

    Parameters:
    ---
    - `img_coords`: an `n-by-2` array containing the image coordinates defining 
    the polygon vertices
    - `shape`: the shape of the output mask

    Returns:
    ---
    A boolean mask where `True` indicates points within the polygon and `False` 
    indicates points outside the polygon.

    Raises:
    - `ValueError`: if the number of points is less than 3 or the input array is 
    not two-dimensional.

    Example:
    ---
    >>> # Define polygon vertices
    >>> polygon_vertices = np.array([[100, 50], [200, 100], [150, 200]])

    >>> # Define the shape of the output mask
    >>> output_shape = (300, 300)

    >>> # Generate boolean mask for the polygon
    >>> mask = polygon_mask(polygon_vertices, output_shape)

    """

    n_points = img_coords.shape[0]
    if n_points < 3 or np.ndim(img_coords) < 2:
        raise ValueError(f"could not build a polygon with less than 3 points")
    
    # Avoid modifying the passed array
    coords = np.copy(img_coords).astype(float)

    # Turn y-coordinates into standard direction and check for winding order
    cts_coords = np.copy(img_coords)
    cts_coords[:, 1] = 255 - cts_coords[:, 1]
    if not is_ccw(cts_coords):
        coords = np.flip(coords, axis=0)

    # Define the next adjacent points
    adj_coords = np.zeros(coords.shape)
    adj_coords[0:-1] = coords[1:]
    adj_coords[-1] = coords[0]

    # The normal vectors for each line segmentS
    directions = adj_coords - coords
    normals = directions[:, [1, 0]] * np.array([1, -1])

    rows, cols = shape[0:2]
    m_x, m_y = np.meshgrid(np.arange(cols), np.arange(rows))
    grid = np.vstack([m_x.flatten(), m_y.flatten()]).T
    grid = np.tile(np.expand_dims(grid, axis=2), [1, 1, n_points])
    normals = np.swapaxes(np.expand_dims(normals, axis=2), 0, 2)
    coords = np.swapaxes(np.expand_dims(coords, axis=2), 0, 2)

    # Check if points in the mesh grid lie on the left or on the right to each
    # line segment, then "and" all results
    mask = np.sum((grid - coords) * normals, axis=1) >= 0
    mask = np.logical_and.reduce(mask, axis=1).reshape(rows, cols)

    return mask