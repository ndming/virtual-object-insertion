import cv2, coloredlogs, logging, os, pyexr
from argparse import ArgumentParser
import inverse as inv
import numpy as np
from pathlib import Path
from pbrw import PBRW
from pbrm import coated_diffuse, coated_conductor_texture, diffuse


def verify_filepath(file: Path, logger: logging.Logger):
    if not file.exists():
        logger.error("could not find the file specified!")
        raise FileNotFoundError(file.resolve().relative_to(Path.cwd()))


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    coloredlogs.install(
        level='DEBUG', logger=logger, 
        fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'
    )

    parser = ArgumentParser()
    parser.add_argument(
        '--working-dir', type=str, required=True, 
        help="path to the directory containing feature maps"
    )
    parser.add_argument(
        '--target', type=str, default="", 
        help="path to the target image, default to im.png in working dir"
    )
    parser.add_argument(
        '-exr', action='store_true', 
        help="run in HDR mode"
    )
    parser.add_argument(
        '--fov', type=float, default=63.4149, 
        help="field of view in the x-axis, default to 63.4149"
    )
    parser.add_argument(
        '--env-scale', type=float, default=1, 
        help="scaling factor applied to the EXR environment map, default to 1"
    )
    parser.add_argument(
        '--ground-fraction', type=float, default=0.5, 
        help="fraction of the env map that will be filled with 0, default to 0.5"
    )
    parser.add_argument(
        '--feature-scale', type=int, default=2, 
        help="a power of 2 to resize (scale) normal, albedo, and rough maps"
    )
    parser.add_argument(
        '--pixel-samples', type=int, default=1024, 
        help="register the number of sample pixels, default to 1024"
    )
    parser.add_argument(
        '--conductor-roughness', type=float, default=0.2, 
        help="rouhgness for the object's conductor layer"
    )
    parser.add_argument(
        '--interface-roughness', type=float, default=0.2, 
        help="rouhgness for the object's interface layer"
    )
    parser.add_argument(
        '--roughness', type=float, default=0.2, 
        help="rouhgness of the diffuse layer for coated diffuse material"
    )
    parser.add_argument(
        '--albedo', type=float, nargs=3, default=[0.8, 0.8, 0.8], 
        help="scattering albedo of the medium between the interface and diffuse layer"
    )
    parser.add_argument(
        '--reflectance', type=float, nargs=3, default=[1.0, 1.0, 1.0], 
        help="average spectral reflectance of the object"
    )
    parser.add_argument(
        '--thickness', type=float, default=0.01, 
        help="thickness of the medium between the two layers"
    )
    parser.add_argument(
        '--scale', type=float, default=1.0, 
        help="custom uniform scale factor to be applied to the rendered object"
    )
    parser.add_argument(
        '--rotate', type=float, nargs=4, default=[0.0, 0.0, 1.0, 0.0], 
        help="custom rotation to be applied to the rendered object"
    )
    args = parser.parse_args()

    # Construct paths to feature maps and the output directory
    # ---
    working_dir = Path(os.path.abspath(args.working_dir))
    output_dir = working_dir/"out"
    
    # Where all generated files will go to
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the target image
    # ---
    target_path = Path(os.path.abspath(args.target))
    target_file = working_dir/"im.png" if not args.target else target_path
    verify_filepath(target_file, logger)
    target_img = cv2.imread(str(target_file))
    logger.debug(f"loaded target image of size: {target_img.shape}")

    # Load the normal map
    # ---
    normal_file = working_dir/"normal.png"
    verify_filepath(normal_file, logger)
    normal_map = cv2.imread(str(normal_file))
    logger.debug(f"feature size: {normal_map.shape}")

    # Resize images
    # ---
    target_rows, target_cols, _ = normal_map.shape
    target_rows = int(target_rows * args.feature_scale)
    target_cols = int(target_cols * args.feature_scale)
    target_size = (target_cols, target_rows)
    normal_map = cv2.resize(normal_map, target_size, interpolation=cv2.INTER_LINEAR)
    target_img = cv2.resize(target_img, target_size, interpolation=cv2.INTER_LINEAR)
    logger.debug(f"target size: {target_img.shape}")

    # Prompt coordinates for the plane receiving shadow
    # ---
    logger.info("select 4 points in counter-clockwise order...")
    plane_coords = None
    try:
        plane_coords = inv.prompt_image_points(target_img, 4, 'target_img')
    except ValueError:
        logger.error(f"could not get the requested number of points")
        exit(1)

    plane_x = plane_coords[:, 0]
    plane_y = plane_coords[:, 1]
    logger.debug(f"plane_x {plane_x}")
    logger.debug(f"plane_y {plane_y}")

    # Construct the plane's mask
    # ---
    plane_mask = inv.polygon_mask(plane_coords, target_img.shape)
    plane_eroded_mask = inv.erode_mask(plane_mask, 1)

    # Average the normals within the mask region
    # ---
    normal_z, normal_y, normal_x = cv2.split(normal_map)  # weird?!
    plane_nx = np.mean(normal_x[plane_eroded_mask].astype(np.float32))
    plane_ny = np.mean(normal_y[plane_eroded_mask].astype(np.float32))
    plane_nz = np.mean(normal_z[plane_eroded_mask].astype(np.float32))

    plane_normal = np.array([plane_nx, plane_ny, plane_nz], dtype=np.float32)
    plane_normal = plane_normal / 127.5 - 1
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Using right-handed coordinate systems, with the camera 
    # placed at (0, 0, 0) facing down the +Z direction
    plane_normal[0] = -plane_normal[0]
    plane_normal[2] = -plane_normal[2]
    logger.debug(f"plane normal: {plane_normal}")

    # Compute the plane's world coordinates; the up and front vector are set
    # to directions preferred by pbrt
    # ---
    plane_p = inv.world_from_image(
        coords=plane_coords, 
        fov=args.fov, 
        shape=target_img.shape,
        up=np.array([0, 0, 1]),
        front=np.array([1, 0, 0])
    )

    # Compute the plane's uv coordinates, with v facing up
    # ---
    pu = (plane_x - 1.0) / (target_cols - 1.0)
    pv = (target_rows - plane_y) / (target_rows - 1.0)
    plane_uv = np.column_stack([pu, pv]).astype(np.float32)

    # Write out the plane shape, transforming the normal to the direction
    # preferred by pbrt
    # ---
    plane_shape_file = output_dir/"plane_shape.pbrt"
    plane_normal_pbrt = inv.change_basis(
        coords=plane_normal,
        up=np.array([0, 0, 1]),
        front=np.array([1, 0, 0]),
    )
    plane_normal_pbrt = plane_normal_pbrt / np.linalg.norm(plane_normal_pbrt)
    plane_n = np.tile(plane_normal_pbrt, (plane_p.shape[0], 1))
    plane_indices = [0, 1, 2, 0, 2, 3]
    PBRW.write_shape(plane_shape_file, plane_p, plane_n, plane_uv, plane_indices)

    # Prompt the object location
    # ---
    logger.info("select object location...")
    hint_img = np.copy(target_img)
    plane_poly = np.array([plane_x, plane_y], dtype=np.int32).T.reshape((-1, 1, 2))
    cv2.polylines(hint_img, [plane_poly], isClosed=True, color=(255, 0, 0), thickness=2)
    obj_coords = None
    try:
        obj_coords = inv.prompt_image_points(hint_img, 1, 'hint_img')
    except ValueError:
        logger.error(f"could not get the requested number of points")
        exit(1)
    logger.debug(f"object location: {obj_coords[0]}")

    # Compute the object's world coordinates
    # ---
    obj_p = inv.world_from_image(
        coords=obj_coords, 
        fov=args.fov, 
        shape=target_img.shape,
        up=np.array([0, 0, 1]),
        front=np.array([1, 0, 0]), 
        depth_ref=plane_p[0],
    )

    # Load the numpy environment map
    # ---
    env_file = working_dir/"env.npz"
    verify_filepath(env_file, logger)
    env_map = np.load(env_file)['env']
    logger.debug(f"loaded env map: {env_map.shape}")

    # Pick the local environment map according to the object's location
    # ---
    obj_x = obj_coords[:, 0]
    obj_y = obj_coords[:, 1]
    obj_frac_x = (obj_x.astype(float)[0] - 1) / (target_cols - 1)
    obj_frac_y = (obj_y.astype(float)[0] - 1) / (target_rows - 1)
    env_rows, env_cols = env_map.shape[0], env_map.shape[1]
    env_x, env_y = (env_cols - 1) * obj_frac_x, (env_rows - 1) * obj_frac_y
    env_y = np.clip(np.round(env_y), 0, env_rows - 1)
    env_x = np.clip(np.round(env_x), 0, env_cols - 1)
    env_x, env_y = int(env_x), int(env_y)
    logger.debug(f"env coordinates: [{env_x}, {env_y}]")
    env_local = env_map[env_y, env_x, :, :, :]
    
    # Fill the ground with zeros and rotate the env map
    # ---
    ground_portion = int(args.ground_fraction * 512)
    intact_portion = 512 - ground_portion
    env_local = cv2.resize(
        env_local, (1024, intact_portion), interpolation = cv2.INTER_LINEAR
    )
    env_ground = np.zeros([ground_portion, 1024, 3], dtype=np.float32)
    env = np.concatenate([env_local, env_ground], axis=0)
    # plane_normal[0] = -plane_normal[0]
    hdr = inv.rotate_env_map(env, plane_normal)
    logger.debug(f"generated env HDR with range: [{np.min(hdr)} - {np.max(hdr)}]")

    # Export the rotated env map as EXR
    # ---
    hdr_flipped = np.flip(hdr, axis=2)  # convert from BGR to RGB
    exr_file = output_dir/"env.exr"
    pyexr.write(str(exr_file), np.maximum(hdr_flipped, 0))
    hdr_file = output_dir/"env.hdr"     # for debug
    cv2.imwrite(str(hdr_file), np.maximum(hdr, 0))
    
    logger.info(f"EXR file written to: {exr_file.resolve().relative_to(Path.cwd())}")
    logger.info(f"HDR file written to: {hdr_file.resolve().relative_to(Path.cwd())}")

    # Load and resize the albedo map to the target size
    # ---
    albedo_file = working_dir/"albedo.png"
    verify_filepath(albedo_file, logger)
    albedo_map = cv2.imread(str(albedo_file))
    albedo_map = cv2.resize(albedo_map, target_size, interpolation = cv2.INTER_LINEAR)
    albedo_resized_file = output_dir/"albedo.png"
    cv2.imwrite(str(albedo_resized_file), albedo_map)

    # Copy the resized target image to output dir
    # ---
    target_out_file = output_dir/"target.png"
    cv2.imwrite(str(target_out_file), target_img)

    # Resize and save rough map to the output directory
    # ---
    rough_file = working_dir/"rough.png"
    verify_filepath(rough_file, logger)
    rough_map = cv2.imread(str(rough_file))
    rough_map = cv2.resize(rough_map, target_size, interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(str(output_dir/"rough.png"), rough_map)

    # pbrt works with vertical fov, in degree
    # ---
    target_aspect = float(target_cols) / float(target_rows)
    fov_rad = 2 * np.arctan(np.tan(np.radians(args.fov / 2.0)) / target_aspect)
    fov_deg = np.degrees(fov_rad)

    # Object's material and transform
    # ---
    obj_mat_textures, obj_mat_type, obj_mat_params = coated_diffuse(
        roughness=args.roughness,
        albedo=args.albedo,
        reflectance=args.reflectance,
        thickness=args.thickness,
    )
    obj_transforms = PBRW.transform_sequence(
        translate=obj_p.flatten() + np.array([0, 0, args.scale]), 
        scale=args.scale
    )

    # Plane's material
    # ---
    albedo_exr_file = output_dir/"albedo.exr"
    pln_mat_textures, pln_mat_type, pln_mat_params = coated_conductor_texture(
        albedo_file=albedo_exr_file, rough_file=rough_file,
    )

    film_type = "spectral" if args.exr else "rgb"

    # Generate I_all scene file description
    # ---
    scene_film_name = "scene.exr" if args.exr else "scene.png"
    scene_writer_builder = PBRW.Builder()
    scene_writer = scene_writer_builder\
        .sensor("perspective", fov_deg)\
        .sampler("zsobol", 1024)\
        .integrator("volpath")\
        .film(film_type, target_cols, target_rows)\
        .build(scene_film_name)
    
    scene_writer.add_light_infinite(exr_file, scale=args.env_scale)
    scene_writer.add_material('obj_mat', obj_mat_textures, obj_mat_type, obj_mat_params)
    scene_writer.add_material('pln_mat', pln_mat_textures, pln_mat_type, pln_mat_params)
    
    obj_shape = f"Shape \"sphere\" \"float radius\" 1"
    scene_writer.add_attribute('obj_mat', obj_transforms, obj_shape)

    pln_shape = f"Import \"{plane_shape_file.name}\""
    scene_writer.add_attribute('pln_mat', [], pln_shape)

    scene_render_file = output_dir/"scene.pbrt"
    scene_writer.write_scene(scene_render_file)

    # Generate I_pln scene file description
    # ---
    plane_film_name = "plane.exr" if args.exr else "plane.png"
    plane_writer_builder = PBRW.Builder()
    plane_writer = plane_writer_builder\
        .sensor("perspective", fov_deg)\
        .sampler("zsobol", 1024)\
        .integrator("volpath")\
        .film(film_type, target_cols, target_rows)\
        .build(plane_film_name)
    
    plane_writer.add_light_infinite(exr_file, scale=args.env_scale)
    plane_writer.add_material('pln_mat', pln_mat_textures, pln_mat_type, pln_mat_params)

    plane_writer.add_attribute('pln_mat', [], pln_shape)

    plane_render_file = output_dir/"plane.pbrt"
    plane_writer.write_scene(plane_render_file)

    # Generate M_all scene file description
    # ---
    mall_film_name = "mask_all.png"
    mall_writer_builder = PBRW.Builder()
    mall_writer = mall_writer_builder\
        .sensor("perspective", fov_deg)\
        .sampler("zsobol", 512)\
        .integrator("volpath")\
        .film("rgb", target_cols, target_rows)\
        .build(mall_film_name)
    
    mall_writer.add_light_distant(
        l_to=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        rgb_L=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        scale=2,
    )
    mall_writer.add_light_distant(
        l_to=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        rgb_L=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        scale=2,
    )
    mall_writer.add_light_distant(
        l_to=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        rgb_L=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        scale=2,
    )

    diff_textures, diff_material, diff_params = diffuse()
    mall_writer.add_material('diff', diff_textures, diff_material, diff_params)

    mall_writer.add_attribute('diff', obj_transforms, obj_shape)
    mall_writer.add_attribute('diff', [], pln_shape)

    mall_render_file = output_dir/"mask_all.pbrt"
    mall_writer.write_scene(mall_render_file)

    # Generate M_obj scene file description
    # ---
    mobj_film_name = "mask_obj.png"
    mobj_writer_builder = PBRW.Builder()
    mobj_writer = mobj_writer_builder\
        .sensor("perspective", fov_deg)\
        .sampler("zsobol", 512)\
        .integrator("volpath")\
        .film("rgb", target_cols, target_rows)\
        .build(mobj_film_name)
    
    mobj_writer.add_light_distant(
        l_to=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        rgb_L=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    mobj_writer.add_light_distant(
        l_to=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        rgb_L=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    mobj_writer.add_light_distant(
        l_to=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        rgb_L=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        scale=2,
    )

    mobj_writer.add_material('diff', diff_textures, diff_material, diff_params)
    mobj_writer.add_attribute('diff', obj_transforms, obj_shape)

    mobj_render_file = output_dir/"mask_obj.pbrt"
    mobj_writer.write_scene(mobj_render_file)