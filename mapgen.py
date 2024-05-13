from argparse import ArgumentParser
import cv2 as cv
import numpy as np
from os import makedirs
from os.path import abspath
from pathlib import Path
from PIL import Image
import shutil, subprocess, sys

from vobj.dimension import resize_efficient, resize_prompting, get_edim, E2P
from vobj.generator import generate_depth, generate_normal, generate_bundle
from vobj.interface import prompt_image_point
from vobj.transform import fov_to_focal, depth_to_ref
from vobj.validator import is_ccw, is_convex

from terminal import get_logger, log_output, verify_path
from warnings import filterwarnings


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generates all resources for virtual object insertion.")

    parser.add_argument('-debug', action='store_true',
        help="run in debug mode")
    parser.add_argument('-cuda', action='store_true',
        help="enalble CUDA when running Li et al.'s inverse rendering")
    parser.add_argument('--img', type=str, required=True, 
        help="path to the target image file")
    parser.add_argument('--py37', type=str, required=True, 
        help="path to the python 3.7 executable, see README.md for more info")
    parser.add_argument('--fov', type=float, default=63.5, 
        help="the camera's horizontal field of view in degree, default to 63.5")
    parser.add_argument('--dpe', type=str, default='midas', 
        help="choose the depth estimator, one of ['midas', 'depthanything']")
    parser.add_argument('--dmax', type=float, default=4.0,
        help="the maximum depth value (meter) in the scene, default to 4.0")
    parser.add_argument('--dmin', type=float, default=1.0,
        help="the minimum depth value (meter) in the scene, default to 1.0")

    args = parser.parse_args()

    verbose = 'DEBUG' if args.debug else 'INFO'
    logger = get_logger(__name__, verbose)

    # Verify depth bounds
    if args.dmin <= 0:
        logger.error(f"min depth of {args.dmin} is invalid, must be positive")
        exit(1)
    if args.dmin >= args.dmax:
        logger.error(
            f"min depth must be less than max depth [{args.dmin}, {args.dmax}]")
        exit(1)

    # Verify the python 3.7 executable
    py37_file = Path(abspath(args.py37))
    try:
        verify_path(logger, py37_file)
    except FileNotFoundError:
        logger.error("could not find the executable!")
        exit(1)

    # Load and verify the target image
    image_file = Path(abspath(args.img))
    try:
        verify_path(logger, image_file)
    except FileNotFoundError:
        logger.error("could not find the file specified!")
        exit(1)

    image = Image.open(image_file)
    logger.info(f"[>] loaded target image of size: {image.size}")

    # Prompt coordinates of the plane
    logger.info("[!] select 4 points enclosing a plane for shadow...")
    prompt_image = resize_prompting(image)
    
    pln_coords = None
    try:
        pln_coords = prompt_image_point(prompt_image, 4, 'plane')
    except ValueError as e:
        logger.error(f"could not get the requested number of points!")
        logger.error(f"expected 4, but received {e}")
        exit(1)
        
    logger.debug(f"plane coordinates[0]: {pln_coords[0]}")
    logger.debug(f"plane coordinates[1]: {pln_coords[1]}")
    logger.debug(f"plane coordinates[2]: {pln_coords[2]}")
    logger.debug(f"plane coordinates[3]: {pln_coords[3]}")

    # Verify the plane's covexity and validity
    if not is_convex(pln_coords):
        logger.error(f"points must form a convex polygon!")
        exit(1)

    try:
        is_ccw(pln_coords)
    except ValueError as e:
        logger.error(f"expected area different to {e}")
        logger.error(f"points are not valid!")
        exit(1)

    # Prompt coordinates of the virtual object
    logger.info("[!] select the location of the virtual object in the plane...")
    prompt_image = np.array(resize_prompting(image))
    pln_poly = pln_coords.astype(np.int32).reshape((-1, 1, 2))
    cv.polylines(
        prompt_image, [pln_poly], isClosed=True, color=(0, 255, 0), thickness=2)
    
    obj_coords = None
    try:
        obj_coords = prompt_image_point(prompt_image, 1, 'object')
    except ValueError as e:
        logger.error(f"could not get the requested number of points!")
        logger.error(f"expected 1, but received {e}")
        exit(1)

    logger.debug(f"object coordinates: {obj_coords[0]}")

    # Generate depth map
    if not args.dpe in ['midas', 'depthanything']:
        logger.error(f"unsupported depth estimator {args.dpe}")
    
    dpe_name = 'DepthAnything' if args.dpe == 'depthanything' else 'MiDaS 3.0'
    logger.info(f"[>] estimating depth map with {dpe_name}...")
    filterwarnings("ignore", category=FutureWarning)
    depth = generate_depth(image, args.dpe)
    filterwarnings("default", category=FutureWarning)

    # Generate normal map, it's important to pass the unmodified depth map
    normal = generate_normal(depth)

    # Where all generated files will go to
    output_dir = image_file.parent/"gen"
    makedirs(output_dir, exist_ok=True)
    
    # Rescale to [0, 255] and save depth map for visualization
    depth_file = output_dir/"depth.png"
    sdepth = np.array(depth)
    sd_min, sd_max = np.min(sdepth), np.max(sdepth)
    sdepth = ((sdepth - sd_min) / (sd_max - sd_min)) * 255.
    sdepth = Image.fromarray(sdepth.astype(np.uint8))
    sdepth.save(depth_file)
    log_output(logger, "depth map generated at", depth_file)

    # Save the raw depth as reference depth
    ref_depth = depth_to_ref(np.array(depth), args.dmin, args.dmax)
    ref_depth_file = output_dir/"ref_depth.npy"
    np.save(ref_depth_file, ref_depth)
    log_output(logger, "reference depth saved to", ref_depth_file)

    # Save normal map
    normal_file = output_dir/"normal.png"
    normal.save(output_dir/"normal.png")
    log_output(logger, "normal map generated at", normal_file)

    # Also save the target image
    target_file = output_dir/"target.png"
    image.save(target_file)
    log_output(logger, "target image saved to", target_file)

    # Save the plane and object coordinates
    coords_file = output_dir/"coords.npz"
    np.savez(coords_file, pln=pln_coords, obj=obj_coords)
    log_output(logger, "selected coordinates saved to", coords_file)

    # Where all outputs of inverse rendering will go to
    irois_dir = output_dir/"irois"
    makedirs(irois_dir, exist_ok=True)

    # The inverse rendering repo requires a text file pointing to the target
    imlist_file = irois_dir/"imlist.txt"
    with open(imlist_file, 'w') as file:
        file.write("target.png")

    # Launch a subprocess to run inverse rendering
    irois_args = [
        sys.executable, 'irois/testReal.py', '--isLight', '--isBS',
        '--experiment0', 'irois/models/check_cascade0_w320_h240', 
        '--experimentLight0', 'irois/models/check_cascadeLight0_sg12_offset1.0',
        '--experimentBS0', 'irois/models/checkBs_cascade0_w320_h240',
        '--experiment1', 'irois/models/check_cascade1_w320_h240', 
        '--experimentLight1', 'irois/models/check_cascadeLight1_sg12_offset1.0', 
        '--experimentBS1', 'irois/models/checkBs_cascade1_w320_h240',
        '--dataRoot', f'{output_dir}', '--imList', f'{imlist_file}', 
        '--testRoot', f'{irois_dir}'
    ]
    if args.cuda:
        irois_args.append('--cuda')

    logger.info("[>] running Li et al.'s inverse rendering...")
    result = subprocess.run(irois_args, stderr=subprocess.DEVNULL)
    rc = result.returncode
    if rc != 0:
        logger.error(f"subprocess execution failed with return code {rc}")
        exit(rc)

    # Copy out the albedo and rough maps from irois
    cascade_level = 1

    albedo_src_file = irois_dir/f"target_albedoBS{cascade_level}.png"
    albedo_file = output_dir/"albedo.png"
    shutil.copy(albedo_src_file, albedo_file)
    log_output(logger, "albedo map saved to", albedo_file)

    rough_src_file = irois_dir/f"target_roughBS{cascade_level}.png"
    rough_file = output_dir/"rough.png"
    shutil.copy(rough_src_file, rough_file)
    log_output(logger, "rough map saved to", rough_file)

    # Load and save the spatially varying environment map
    irois_src_file = irois_dir/f"target_envmap{cascade_level}.png.npz"
    irois_env = np.load(irois_src_file)['env']
    logger.debug(f"loaded irois env map of shape: {irois_env.shape}")
    irois_file = output_dir/"irois.npy"
    np.save(irois_file, irois_env)
    log_output(logger, "Li et al.'s env map saved to", irois_file)

    # Generate lighthouse's input bundle to a separate folder
    lighthouse_dir = output_dir/"lighthouse"
    makedirs(lighthouse_dir, exist_ok=True)

    # Save the source stereo of the target
    source_file = lighthouse_dir/"source.png"
    image.save(source_file)
    log_output(logger, "source stereo saved to", source_file)

    target = resize_efficient(Image.open(target_file))
    source = resize_efficient(Image.open(source_file))
    edepth = resize_efficient(depth)
    rdepth = depth_to_ref(np.array(edepth), args.dmin, args.dmax)
    focal = fov_to_focal(args.fov, edepth.size)
    bundle = generate_bundle(target, source, rdepth, focal, obj_coords[0], E2P)

    logger.debug(f"ref_image shape: {bundle['ref_image'][0].shape}")
    logger.debug(f"src_image shape: {bundle['src_images'][0].shape}")
    logger.debug(f"ref_depth shape: {bundle['ref_depth'][0].shape}")
    logger.debug(f"focal length fx: {bundle['intrinsics'][0, 0, 0]}")
    logger.debug(f"focal length fy: {bundle['intrinsics'][0, 1, 1]}")
    logger.debug(f"principal point: {bundle['intrinsics'][0, 0:2, 2]}")
    logger.debug(f"env translation: {bundle['env_pose'][0, 0:3, 3]}")

    # Save the bundle
    bundle_file = lighthouse_dir/"bundle.npz"
    np.savez(bundle_file, **bundle)
    log_output(logger, "lighthouse bundle saved to", bundle_file)

    # Launch a subprocess to run lighthouse
    width, height = get_edim(image)
    lighthouse_args = [
        f'{py37_file}', '-m', 'lighthouse.interiornet_test', '--checkpoint_dir', 
        'lighthouse/model/', '--data_dir', f'{lighthouse_dir}', '--output_dir',
        f'{lighthouse_dir}', '--width', f'{width}', '--height', f'{height}']
    
    logger.info("[>] running Pratul et al.'s lighthouse...")
    result = subprocess.run(lighthouse_args, stderr=subprocess.DEVNULL)
    rc = result.returncode
    if rc != 0:
        logger.error(f"subprocess execution failed with return code {rc}")
        exit(rc)

    # Copy out lighthouse environment map
    house_src_file = lighthouse_dir/"00000.png"
    house_file = output_dir/"lighthouse.png"
    shutil.copy(house_src_file, house_file)
    log_output(logger, "Pratul et al.'s env map saved to", house_file)
