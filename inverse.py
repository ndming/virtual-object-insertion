import numpy as np
import cv2, random
import tqdm as tqdm


def prompt_image_points(
    image: np.ndarray, 
    n_points: int | tuple[int, int], 
    window_name=""
) -> np.ndarray:
    """
    Shows a window containing the `image` and listens for `n_points` clicks. If
    `n_points` is given as a tuple (`min`, `max`), waits for point click event 
    until an ENTER or an interrupt event is signaled. The number of received 
    image points will be validated with the provided `n_points`.

    Parameters:
    ---
    - `image`: the image 2D array, guaranteed to be intact throughout.
    - `n_points`: an integer or a tuple specifying the number of points.
    - `window_name`: the string to be shown on the window title bar.

    Returns:
    ---
    An `n-by-2` array containing the received image coordinates.
    """
    
    img = np.copy(image)  # prevent overwriting on the original
    if not window_name:
        window_name = f"window-{hex(random.randint(0, 2**32))}"
    
    points = []
    callback_param = (img, points, window_name)
    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, _select_point_callback, callback_param)
    
    if isinstance(n_points, int):
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.waitKey(1)
            if len(points) >= n_points:
                cv2.destroyAllWindows()

        if len(points) < n_points:
            raise ValueError(f"received less than {n_points} points")
    else:
        min, max = n_points
        cv2.waitKey()
        cv2.destroyAllWindows()
        if len(points) > max or len(points) < min:
            raise ValueError(f"received outside ({min}, {max}) number of points")

    return np.asarray(points)


def _select_point_callback(event, x, y, _, param: tuple[np.ndarray, list[tuple], str]):
    if event == cv2.EVENT_LBUTTONDOWN:
        img, points, window_name = param
        points.append((x, y))
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        cv2.imshow(window_name, img)


def polygon_mask(
    img_coords: np.ndarray, 
    shape: np.ndarray | tuple[int, int]
) -> np.ndarray[bool]:
    """
    Generates a boolean mask of size `shape` indicating whether each point in a 
    grid lies within the polygon defined by image coordinates `img_coords`.

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
    if not _is_ccw(cts_coords):
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


def _is_ccw(coords: np.ndarray) -> bool:
    # Construct an array of adjacent coordinates
    adj_coords = np.zeros(coords.shape)
    adj_coords[0:-1] = coords[1:]
    adj_coords[-1] = coords[0]
    
    # Compute the sign area using the shoelace algorithm
    s_area = coords * adj_coords[:, [1, 0]]
    s_area = np.sum(s_area[:, 0] - s_area[:, 1])
    
    if s_area == 0:
        raise ValueError(f"the provided coordinate sequence is not valid")
    
    return s_area > 0
    

def erode_mask(mask: np.ndarray, radius) -> np.ndarray[bool]:
    """
    Erode a binary mask using an elliptical structuring element.

    Parameters:
    ---
    - `mask`: the input binary mask to be eroded.
    - `radius`: radius of the elliptical kernel.

    Returns:
    ---
    The eroded binary mask.

    Example:
    ---
    >>> # Erode a mask with a radius of 3
    >>> eroded_mask = erode_mask(mask, 3.0)
    """

    diameter = radius * 2
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.erode(mask.astype(float), es) > 0


def world_from_image(
    coords: np.ndarray, 
    fov: float, 
    shape: np.ndarray | tuple[int, int],
    up=np.array([0, 1, 0]),
    front=np.array([0, 0, 1]),
    depth_ref=None,
) -> np.ndarray[np.float32]:
    """
    Converts image coordinates in `coords` to world coordinates, given the
    camera's `fov` in the x-axis and the image'`shape`. The camera is assumed
    to always locate at the origin, looking down the `front` direction with
    the associated `up` vector. The optional `depth_ref` can be specified as 
    the depth reference point in world space to the camera. If `None` is 
    provided, the first image coordinate will be used as `depth_ref` after 
    transforming to 3D local coordinates. The final world coordinates will be 
    transformed relative to this reference depth and the `up` vector.

    Parameters:
    ---
    - `coords`: an `n-by-2` array specifying the image coordinates.
    - `fov`: the camera fiew-of-view in degree, along the x-axis.
    - `shape`: the image's shape, given as an array or a tuple of `[rows, cols]`
    - `up`: the camera's up vector, default to +Y.
    - `front`: the direction along which the camera is looking down, default to +Z.
    - `depth_ref`: the depth reference point in world coordinates.

    Returns:
    ---
    An `n-by-3` array containing the converted world coordinates.
    """

    rows, cols = shape[0:2]
    aspect = float(cols) / rows  # in case using Python 2.x
    tan_x = np.tan(np.radians(fov / 2.0))
    tan_y = tan_x / aspect

    n_points = 1 if np.ndim(coords) == 1 else coords.shape[0]
    half_img = np.array([(cols - 1) / 2.0, (rows - 1) / 2.0])
    half_mat = np.tile(half_img, (n_points, 1))

    # Convert to world coordinates with depth=1, up=+Y, and front=+Z
    local_coords = np.array([tan_x, tan_y]) * (half_mat - coords) / half_mat
    local_coords = np.hstack([local_coords, np.ones((n_points, 1))])
    world_coords = change_basis(local_coords, up, front)

    # Compute depth, or exit early if necessary
    if depth_ref is None and n_points == 1:
        return world_coords.astype(np.float32)
    elif depth_ref is None:
        depth_ref = world_coords[0]

    depth_scales = np.dot(depth_ref, up) / np.sum(world_coords * up, axis=1)
    if not np.all((depth_scales > 0) & np.isfinite(depth_scales)):
        raise ValueError(f"depth scales are not valid")
    
    world_coords = world_coords * depth_scales[:, np.newaxis]
    return world_coords.astype(np.float32)


def change_basis(coords, up, front):
    """
    Change the basis of a set of `coords` to a new coordinate system defined by 
    the given `up` and `front` vectors.

    Parameters:
    ---
    - `coords`: the coordinates to be transformed, as an array of shape `(n, 3)`.
    - `up`: the up vector defining the new coordinate system's y-axis.
    - `front`: the front vector defining the new coordinate system's z-axis.

    Returns:
    ---
    The transformed coordinates in the new basis, as an array of shape (n, 3).

    Raises:
    ---
    - `ValueError`: if the `up` and `front` vectors are not orthogonal.

    Examples:
    ---
    >>> # Define the original coordinates and new basis vectors
    >>> coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> up = np.array([0, 0, 1])
    >>> front = np.array([0, 1, 0])

    >>> # Transform the coordinates to the new basis
    >>> new_coords = change_basis(coords, up, front)
    """

    # Orthogonal check
    up_n = up / np.linalg.norm(up)
    front_n = front / np.linalg.norm(front)
    if not np.isclose(np.dot(up_n, front_n), 0):
        raise ValueError(f"{up} and {front} are not orthogonal")
    
    # Transform to the basis specified by up and front
    right = np.cross(up_n, front_n)
    right_n = right / np.linalg.norm(right)
    world_mat = np.vstack([right_n, up_n, front_n]).T
    return (world_mat @ coords.T).T  # using row-vectors


def rotate_env_map(map: np.ndarray, normal: np.ndarray) -> np.ndarray[np.float32]:
    """
    Rotates the environment `map` based on the given `normal` vector.

    Parameters:
    ---
    - `map`: a 3D environment map array (rows, cols, colors).
    - `normal`: the vector representing the normal direction relative to the y-axis. 

    Returns:
    ---
    The rotated environment map with the same shape as `map`.

    Example:
    ---
    >>> # Rotate an environment map based on the normal vector (0, 1, 0)
    >>> rotated_map = rotate_env_map(env_map, np.array([0, 1, 0]))
    """

    up = np.array([0, 1, 0])
    right = np.cross(up, normal)
    front = np.cross(normal, right)

    norml = normal / np.linalg.norm(normal)
    right = right / np.linalg.norm(right)
    front = front / np.linalg.norm(front)

    rows, cols, _ = map.shape
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    phi = 2.0 * np.pi * grid_x / (cols - 1) - np.pi
    theta = np.pi * grid_y / (rows - 1)

    z = np.sin(theta) * np.cos(phi)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)

    coords = np.vstack([x.flatten(), y.flatten(), z.flatten()])
    mat_rotate = np.vstack([right, front, norml]).T
    coords_r = mat_rotate @ coords

    x_rotate = coords_r[0, :].reshape(rows, cols)
    y_rotate = coords_r[1, :].reshape(rows, cols)
    z_rotate = coords_r[2, :].reshape(rows, cols)

    x_rotate = x_rotate / (np.sqrt(1 - z_rotate * z_rotate) + 1e-12)
    y_rotate = y_rotate / (np.sqrt(1 - z_rotate * z_rotate) + 1e-12)
    x_rotate = np.clip(x_rotate, -1, 1)
    y_rotate = np.clip(y_rotate, -1, 1)
    z_rotate = np.clip(z_rotate, -1, 1)

    theta_r = np.arccos(z_rotate)
    phi_r = np.arccos(x_rotate)
    sign = np.ones_like(y_rotate)
    sign[y_rotate < 0] = -1
    phi_r = phi_r * sign

    u = (phi_r + np.pi) / (2.0 * np.pi)  # [-pi; pi] to [0; 1]
    v = theta_r / np.pi                  # [0; pi] to [0; 1]

    return _uv_to_color(map, u, v).astype(np.float32)


def _uv_to_color(map: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    rows, cols = map.shape[0], map.shape[1]

    # Find the target and neighbor pixels
    x, y = u * (cols - 1), v * (rows - 1)
    px, py = x.astype(int), y.astype(int)
    qx = np.minimum(cols - 1, px + 1)  # the next neighbor pixel along x
    qy = np.minimum(rows - 1, py + 1)  # the next neighbor pixel along y

    # Fraction to interleave pixels
    fx, fy = x - px, y - py
    fx = np.stack([fx, fx, fx], axis=2)
    fy = np.stack([fy, fy, fy], axis=2)

    # Interleave along x first, then y
    p_color = (1 - fx) * map[py, px, :] + fx * map[py, qx, :]
    q_color = (1 - fx) * map[qy, px, :] + fx * map[qy, qx, :]
    color = (1 - fy) * p_color + fy * q_color

    return color


def get_relative_transform(orientation, up, position, depth_ref):
    axis = np.cross(up, orientation)

    if np.sum(axis * axis) <= 1e-6:
        axis = up
        rotate_angle = 0.0
    else:
        axis = axis / np.linalg.norm(axis)
        rotate_angle = np.arccos(np.sum(orientation * up))
    
    rotate = np.array([rotate_angle, axis[0], axis[1], axis[2]])

    scale = np.linalg.norm(position - depth_ref)
    scale = np.array([scale, scale, scale])

    return (rotate, scale)