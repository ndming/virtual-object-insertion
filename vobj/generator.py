from cv2 import Sobel, CV_32F
import numpy as np
from PIL.Image import Image, fromarray
from transformers import pipeline


MAX_DEPTH = 2.0


def generate_depth(image: Image) -> Image:
    depth_model = "LiheYoung/depth-anything-base-hf"
    pipe = pipeline(task="depth-estimation", model=depth_model)
    depth = pipe(image)["depth"]
    return depth


def generate_normal(depth: Image) -> Image:
    x = Sobel(np.array(depth), CV_32F, 1, 0, ksize=3)
    y = Sobel(np.array(depth), CV_32F, 0, 1, ksize=3)
    z = np.ones_like(x) * np.pi * 2.0

    normal = np.stack([x, y, z], axis=2)
    normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
    normal = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    normal = fromarray(normal)

    return normal


def generate_bundle(
    target: Image, source: Image, depth: Image, 
    fov, env_coords: np.ndarray, ep_fraction
) -> dict:
    ref_image, src_image = _get_ref_src_image(target, source)
    ref_depth = _get_ref_depth(depth)
    intrinsic = _get_intrinsic(ref_depth[0].shape, fov)
    ref_pose, src_pose = _get_ref_src_pose()
    focal = intrinsic[0, 0, 0]
    env_pose = _get_env_pose(env_coords, ep_fraction, focal, ref_depth)

    return {
        'intrinsics': intrinsic,
        'ref_image': ref_image,
        'ref_depth': ref_depth,
        'src_images': src_image,
        'ref_pose': ref_pose,
        'src_poses': src_pose,
        'env_pose': env_pose,
    }


def _get_ref_src_image(target, source):
    ref_image = np.array(target) / 255.
    src_image = np.array(source) / 255.

    ref_image = np.expand_dims(ref_image, axis=0).astype(np.float32)
    src_image = np.expand_dims(src_image, axis=0).astype(np.float32)

    return ref_image, src_image


def _get_ref_depth(depth):
    ref_depth = np.array(depth)

    # Higher depth values correspond to further pixels
    ref_depth = np.max(ref_depth) - ref_depth + np.min(ref_depth)
    ref_depth = ref_depth / 255. * MAX_DEPTH

    ref_depth = np.expand_dims(ref_depth, axis=0).astype(np.float32)
    return ref_depth


def _get_intrinsic(image_size, fov_x):
    n_rows, n_cols = image_size[0:2]
    fx = n_cols / (2. * np.tan(np.radians(fov_x / 2.0)))
    aspect = n_cols / n_rows
    fov_y = fov_x / aspect
    fy = n_rows / (2. * np.tan(np.radians(fov_y / 2.0)))

    intrinsic = np.array([
        [fx, 0, n_cols / 2.], 
        [0, fy, n_rows / 2.], 
        [0, 0, 1]
    ], dtype=np.float32)

    intrinsic = np.expand_dims(intrinsic, axis=0)
    return intrinsic


def _get_ref_src_pose():
    ref_pose = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]
    ], dtype=np.float32)

    ref_pose = np.expand_dims(ref_pose, axis=0)
    src_pose = np.copy(ref_pose)
    src_pose = np.expand_dims(src_pose, axis=3)

    return ref_pose, src_pose


def _get_env_pose(coords, fraction, focal, ref_depth):
    n_rows, n_cols = ref_depth[0].shape

    # Coordinates in camera space
    x_cam, y_cam = coords
    x_cam = fraction * x_cam - n_cols / 2.
    y_cam =  n_rows / 2. - fraction * y_cam

    # Depth of the env location
    env_z = ref_depth[0, int(fraction * coords[1]), int(fraction * coords[0])]
    # Because ref_depth was scaled by max depth
    env_z = env_z / MAX_DEPTH

    # Coordinates in world space
    env_x = x_cam * env_z / focal
    env_y = y_cam * env_z / focal

    env_pose = np.array([
        [1, 0, 0, env_x], 
        [0, 1, 0, env_y], 
        [0, 0, 1, env_z], 
        [0, 0, 0, 1]
    ], dtype=np.float32)
    env_pose = np.expand_dims(env_pose, axis=0)

    return env_pose