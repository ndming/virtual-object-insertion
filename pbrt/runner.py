from subprocess import PIPE, DEVNULL, Popen, run
from tqdm import tqdm


def run_pbrt(pbrt_file, render_file, gpu_enabled, logger):
    logger.info(f"[pbrt] invoking pbrt for: {render_file.name}")

    pbrt_cmd = f"{pbrt_file} {render_file}"
    if gpu_enabled: 
        pbrt_cmd += " --gpu"

    process = Popen(pbrt_cmd, stdout=DEVNULL, stderr=PIPE, universal_newlines=True)

    with tqdm(desc="Rendering") as pbar:
        while process.poll() is None:
            pbar.update(1)

    result = process.wait()

    if result != 0:
        logger.error(f"error rendering file: {render_file.name}")
        stderr_content = process.stderr.read()
        logger.error(stderr_content)
        exit(1)


def run_imgtool(cmd):
    return run(cmd, shell=True, capture_output=True, text=True)