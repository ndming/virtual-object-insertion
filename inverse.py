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


def _select_point_callback(event, x, y, _, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img, pts, name = param
        pts.append((x, y))
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        cv2.imshow(name, img)


def build_plane_mask(plane_x, plane_y, rows, cols):
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    mask = np.ones((rows, cols))

    for n in range(len(plane_x)):
        if n == 3:
            sx = plane_x[0] - plane_x[n]
            sy = plane_y[0] - plane_y[n]
        else:
            sx = plane_x[n + 1] - plane_x[n]
            sy = plane_y[n + 1] - plane_y[n]

        mask *= ((grid_x - plane_x[n]) * sy - (grid_y - plane_y[n]) * sx) > 0

    return mask > 0
    

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

    if np.ndim(coords) == 1:
        coords = coords.reshape(1, 2)

    n_points = coords.shape[0]
    half_img = np.array([(cols - 1) / 2.0, (rows - 1) / 2.0])
    half_mat = np.tile(half_img, (n_points, 1))

    # Convert to world coordinates with depth=1, up=+Y, and front=+Z
    local_coords = np.array([tan_x, tan_y]) * (half_mat - coords) / half_mat
    local_coords = np.hstack((local_coords, np.ones((n_points, 1))))
    world_coords = change_basis(local_coords, up, front)

    # Compute depth, or exit early if necessary
    if depth_ref is None and n_points == 1:
        return world_coords.astype(np.float32)
    elif depth_ref is None:
        depth_ref = world_coords[0]

    depth_scales = np.dot(depth_ref, up) / np.sum(world_coords * up, axis=1)
    if not np.all((depth_scales > 0) & np.isfinite(depth_scales)):
        raise ValueError(f"focal values are not valid")
    
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
    - `ValueError`: if the up and front vectors are not orthogonal.

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
    up = up / np.linalg.norm(up)
    front = front / np.linalg.norm(front)
    if not np.isclose(np.dot(up, front), 0):
        raise ValueError(f"{up} and {front} are not orthogonal")
    
    # Transform to the basis specified by up and front
    right = np.cross(up, front)
    right = right / np.linalg.norm(right)
    world_mat = np.vstack((right, up, front)).T
    return (world_mat @ coords.T).T  # using row-vectors


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


def rotate_env_map(
    map: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray[np.float32]:
    up = np.array([0, 1, 0])
    right = np.cross(up, normal)
    front = np.cross(normal, right)

    normal = normal / np.linalg.norm(normal)
    right = right / np.linalg.norm(right)
    front = front / np.linalg.norm(front)

    rows, cols, _ = map.shape
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    phi = 2.0 * np.pi * grid_x / (cols - 1) - np.pi
    theta = np.pi * grid_y / (rows - 1)

    z = np.sin(theta) * np.cos(phi)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)

    coords = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    mat_rotate = np.vstack((right, front, normal)).T
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

    x, y = u * (cols - 1), v * (rows - 1)
    px, py = x.astype(int), y.astype(int)
    qx = np.minimum(cols - 1, px + 1)  # the next neighbor pixel along x
    qy = np.minimum(rows - 1, py + 1)  # the next neighbor pixel along y

    # Fraction to interleave pixels
    fx, fy = x - px, y - py
    fx = np.stack((fx, fx, fx), axis=2)
    fy = np.stack((fy, fy, fy), axis=2)

    p_color = (1 - fx) * map[py, px, :] + fx * map[py, qx, :]
    q_color = (1 - fx) * map[qy, px, :] + fx * map[qy, qx, :]
    color = (1 - fy) * p_color + fy * q_color

    return color