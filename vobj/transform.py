import numpy as np
from scipy.signal import convolve2d


def rotate_env_map(
    envmap: np.ndarray, 
    normal: np.ndarray
) -> np.ndarray[np.float32]:
    """
    Rotates the environment `envmap` based on the given `normal` vector.

    Parameters:
    ---
    - `envmap`: a 3D environment map array (rows, cols, colors).
    - `normal`: a vector representing the normal direction relative to the y-axis. 

    Returns:
    ---
    The rotated environment map with the same shape as `envmap`.

    Example:
    ---
    >>> # Rotate an environment map based on the normal vector (0, 1, 0)
    >>> rotated_map = rotate_env_map(env_map, np.array([0, 1, 0]))
    """

    up = np.array([0., 1., 0.])
    right = np.cross(up, normal)
    front = np.cross(normal, right)

    norml = normal / np.linalg.norm(normal)
    right = right / np.linalg.norm(right)
    front = front / np.linalg.norm(front)

    rows, cols, _ = envmap.shape
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    phi = 2.0 * np.pi * grid_x / (cols - 1) - np.pi
    theta = np.pi * grid_y / (rows - 1)

    z = np.sin(theta) * np.cos(phi)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)

    coords = np.vstack([x.flatten(), y.flatten(), z.flatten()])
    rotate = np.vstack([right, front, norml]).T
    coords = rotate @ coords

    x_rotate = coords[0, :].reshape(rows, cols)
    y_rotate = coords[1, :].reshape(rows, cols)
    z_rotate = coords[2, :].reshape(rows, cols)

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

    hdr = _uv_to_color(envmap, u, v)
    return np.flip(hdr, axis=1).astype(np.float32)


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


def ldr_to_hdr(ldr: np.ndarray, gamma=2.2) -> np.ndarray:
    return (ldr.astype(np.float32) / 255.) ** gamma


def fov_to_focal(fov_x, image_size) -> tuple[float, float]:
    n_cols, n_rows = image_size
    fx = n_cols / (2. * np.tan(np.radians(fov_x / 2.0)))
    aspect = n_cols / n_rows
    fov_y = fov_x / aspect
    fy = n_rows / (2. * np.tan(np.radians(fov_y / 2.0)))
    return (fx, fy)


def depth_to_ref(depth: np.ndarray, min_depth, max_depth) -> np.ndarray:
    min, max = np.min(depth), np.max(depth)
    # Higher depth values should correspond to further pixels
    ref_depth = max - depth + min
    # Rescale to the specified min and max depth
    ratio = (max_depth - min_depth) / (max - min)
    ref_depth = min_depth + (ref_depth - min) * ratio
    return ref_depth.astype(np.float32)


def convolve_ref_depth(ref_depth: np.ndarray, k_size=21) -> np.ndarray:
    kernel = np.ones((k_size, k_size)) / (k_size * k_size)
    return convolve2d(ref_depth, kernel, mode='same', boundary='wrap')


def image_to_world(
    im_coords: np.ndarray,
    ref_depth: np.ndarray, 
    focal: tuple[float, float],
    normal: np.ndarray = None,
    ref_point: np.ndarray = None,
) -> np.ndarray:
    n_rows, n_cols = ref_depth.shape

    # Camera coordinates
    cam_coords = np.copy(im_coords)
    cam_coords[:, 1] *= -1
    cam_coords = cam_coords + np.array([-n_cols / 2., n_rows / 2.])

    # Initial depth of each point based on the reference depth
    z = ref_depth[im_coords[:, 1], im_coords[:, 0]]

    # Set out the world coordinates at depth -1
    n_points = im_coords.shape[0]
    w_coords = np.hstack([cam_coords / np.array(focal), np.ones((n_points, 1))])
    w_coords[:, 2] *= -1  # camera faces down the -Z direction

    # If normality is imposed, tailor the depth so that points form a polygon
    # whose surface normal as specified
    if normal is not None:
        if ref_point is None:
            ref_point = w_coords[np.argmin(z)]
            ref_point = ref_point * np.min(z)

        # Compute and replace with the new depth
        z = np.dot(ref_point, normal) / np.sum(w_coords * normal, axis=1)

    w_coords *= z[:, np.newaxis]
    return w_coords.astype(np.float32), ref_point