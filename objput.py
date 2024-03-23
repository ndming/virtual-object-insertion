import cv2, coloredlogs, logging, os, pyexr
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import OpenEXR as exr
from Imath import PixelType


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
coloredlogs.install(
    level='DEBUG', logger=logger, 
    fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'
)


def load_file(file: Path):
    if not file.exists():
        logger.error("could not find the file specified!")
        raise FileNotFoundError(file.resolve().relative_to(Path.cwd()))
    
    if file.suffix == ".exr":
        img = load_exr(file)
        logger.info(f"loaded {file.name} as EXR: {img.shape} - [{np.min(img)}, {np.max(img)}]")
        return img
    
    img = load_png(file)
    logger.info(f"loaded {file.name} as PNG: {img.shape} - [{np.min(img)}, {np.max(img)}]")
    return img


def load_png(file: Path):
    return cv2.imread(str(file)).astype(np.float32) / 255.0


def load_mask(file: Path):
    img = cv2.imread(str(file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask > 0


def load_exr(file: Path):
    exr_file = exr.InputFile(str(file))

    # Read R, G, and B channels
    r_raw = exr_file.channel('R', PixelType(PixelType.FLOAT))
    g_raw = exr_file.channel('G', PixelType(PixelType.FLOAT))
    b_raw = exr_file.channel('B', PixelType(PixelType.FLOAT))

    # Convert raw bytes to numpy arrays
    r_array = np.frombuffer(r_raw, dtype=np.float32)
    g_array = np.frombuffer(g_raw, dtype=np.float32)
    b_array = np.frombuffer(b_raw, dtype=np.float32)

    # Get image dimensions
    rows = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
    cols = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x

    # Reshape vectors into image arrays
    r = np.reshape(r_array, (rows, cols))
    g = np.reshape(g_array, (rows, cols))
    b = np.reshape(b_array, (rows, cols))

    return np.stack((b, g, r), axis=2)


def erode_mask(mask, radius):
    diameter = radius * 2
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.erode(mask.astype(float), es) > 0
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("target_image", help="path to the target image")
    parser.add_argument(
        "--resource-dir", type=str, required=True, 
        help="path to the directory containing all the resources"
    )
    parser.add_argument(
        "-exr", action='store_true',
        help="use EXR resources to perform the editing"
    )
    parser.add_argument(
        '--output', type=str, default="", 
        help="output file name"
    )

    args = parser.parse_args()
    res_dir = Path(os.path.abspath(args.resource_dir))

    # Load images
    i_file = Path(os.path.abspath(args.target_image))
    i = load_file(i_file)

    i_all_file = res_dir/"scene.exr" if args.exr else res_dir/"scene.png"
    i_all = load_file(i_all_file)

    i_pln_file = res_dir/"plane.exr" if args.exr else res_dir/"plane.png"
    i_pln = load_file(i_pln_file)

    m_all_file = res_dir/"mask_all.png"
    m_all = load_mask(m_all_file)
    logger.info(f"loaded mask {m_all_file.name}: {m_all.shape} - {m_all.dtype}")

    m_obj_file = res_dir/"mask_obj.png"
    m_obj = load_mask(m_obj_file)
    logger.info(f"loaded mask {m_obj_file.name}: {m_obj.shape} - {m_obj.dtype}")

    # Make the target image HDR if working with EXR resources
    if args.exr: i = i ** 2.2

    # Process masks
    m_all = erode_mask(m_all, 2)
    m_pln = np.clip(m_all.astype(float) - m_obj.astype(float), 0, None)
    m_pln = m_pln > 0

    m_all = np.stack([m_all, m_all, m_all], axis=2)
    m_pln = np.stack([m_pln, m_pln, m_pln], axis=2)
    m_obj = np.stack([m_obj, m_obj, m_obj], axis=2)

    k_sd = np.maximum(i_all, 1e-10) / np.maximum(i_pln, 1e-10) * m_pln
    k_sd = np.minimum(k_sd, 1)
    
    # Insert the object
    h = i * (1 - m_all) + i * k_sd * m_all
    h = i_all * m_obj + h * (1 - m_obj)
    logger.debug(f"result range: {np.min(h)} - {np.max(h)}")

    # Make a copy of the result and display
    p = h
    if args.exr:
        p = np.maximum(p, 0)
        p = h / np.max(p)
        p = p ** (1.0 / 2.2)

    rgb = (p * 255.0).astype(np.uint8)
    cv2.imshow('result', rgb)
    cv2.waitKey()

    # Write out the result
    if args.output:
        out_file = Path(os.path.abspath(args.output))
        if out_file.suffix == ".exr":
            if not args.exr: 
                h = h ** 2.2
            pyexr.write(str(out_file), np.flip(h, axis=2))
        elif out_file.suffix == ".png":
            cv2.imwrite(str(out_file), rgb)
        else:
            logger.error(f"unsupport output file format: {out_file.suffix}")
            exit(1)

        logger.info(f"result has been written to: {out_file.resolve().relative_to(Path.cwd())}")
