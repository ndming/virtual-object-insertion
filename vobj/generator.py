from cv2 import Sobel, CV_32F
import numpy as np
from PIL.Image import Image, fromarray
from transformers import pipeline


def generate_depth(image: Image, estimator: str) -> Image:
    if estimator == 'depthanything':
        return _generate_depth_anything(image)
    else:
        return _generate_depth_midas(image)
    

def _generate_depth_anything(image: Image) -> Image:
    depth_model = "LiheYoung/depth-anything-base-hf"
    pipe = pipeline(task="depth-estimation", model=depth_model)
    depth = pipe(image)["depth"]
    return depth


def _generate_depth_midas(image: Image) -> Image:
    depth_model = "Intel/dpt-hybrid-midas"
    pipe = pipeline(task="depth-estimation", model=depth_model)
    depth = pipe(image)['predicted_depth'][0]
    depth = fromarray(depth.numpy()).resize(image.size)
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
    target: Image, source: Image, rdepth: np.ndarray, 
    focal: tuple[float, float], env_coords: np.ndarray
) -> dict:
    ref_image, src_image = _get_ref_src_image(target, source)
    ref_depth = np.expand_dims(rdepth, axis=0).astype(np.float32)
    intrinsic = _get_intrinsic(rdepth.shape, focal)
    ref_pose, src_pose = _get_ref_src_pose()
    env_pose = _get_env_pose(env_coords, focal, ref_depth[0])

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


def _get_intrinsic(image_size, focal):
    n_rows, n_cols = image_size[0:2]
    fx, fy = focal

    intrinsic = np.array([
        [fx, 0, (n_cols - 1.) / 2.], 
        [0, fy, (n_rows - 1.) / 2.], 
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


def _get_env_pose(coords, focal, ref_depth):
    n_rows, n_cols = ref_depth.shape

    # Coordinates in camera space
    x_cam, y_cam = coords
    x_cam = x_cam - n_cols / 2.
    y_cam = n_rows / 2. - y_cam

    # Depth of the env location
    env_z = ref_depth[int(coords[1]), int(coords[0])]

    # Coordinates in world space
    fx, fy = focal
    env_x = x_cam * env_z / fx
    env_y = y_cam * env_z / fy

    env_pose = np.array([
        [1, 0, 0, env_x], 
        [0, 1, 0, env_y], 
        [0, 0, 1, env_z], 
        [0, 0, 0, 1]
    ], dtype=np.float32)
    env_pose = np.expand_dims(env_pose, axis=0)

    return env_pose