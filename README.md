# Virtual object insertion

An implementation of virtual object insertion task based on deep learning.

## Pipeline overview
The pipeline for inserting a virtual sphere into a 2D image consists of 4 stages
in order:
- Generation of feature maps: this stage takes in a single 2D image and outputs
several feature maps like depth, normal, albedo, etc., together with files that
are relavent to subsequent stages.
- Generation of render scripts: this stage generates scene description files
that make use of the generated resources to orchestrate the rendering of the 
virtual object.
- Virtual object rendering: this stage automates the rendering instructed in the
generated scene description files. The implementation employs Matt Pharr's 
[pbrt](https://github.com/mmp/pbrt-v4) to handle the shading of the object.
- Virtual object insertion: this stages assembles the rendered scenes and
programmatically inserts the shaded object to an image.

## Environment setup
The implementation depends heavily on Python and some external packages like
OpenEXR for handling HDR contents. In addition, the workflow leverages different
deep learning models to estimate feature maps. Some of these models use outdated
libraries which make the dependency management task become cumbersome.

This section describes the steps to set up a working environment for this repo.

### CUDA Toolkit and OptiX SDK
- It is highly advisable to have 
[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) version 
`11.8` or `12.1` installed.
- [OptiX SDK](https://developer.nvidia.com/designworks/optix/downloads/legacy) 
will enable the rendering of `pbrt` on the GPU. Any OptiX version from `7.1` to
`7.7` is applicable.

### pbrt
`pbrt` must be built from source and the build instruction is detailed in their 
GitHub [homepage](https://github.com/mmp/pbrt-v4). The following scripts might 
help:

```
git clone --recursive https://github.com/mmp/pbrt-v4.git pbrt
cd pbrt
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

If GPU is compatible:
```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPBRT_OPTIX7_PATH="path/to/OptiX/v7.7.0"
```

The build can be invoked indirectly through CMake, if using Visual Studio or 
XCode:
```
cmake --build build --config Release --target ALL_BUILD
```

If using Makefile or Ninja:
```
cmake --build build
```

The build will generate several binaries, among which are the `pbrt` executable
and the `imgtool` program lying in the same folder.

### OpenEXR
This package helps streamline the management of HDR contents. Installation guide
can be found [here](https://openexr.com/en/latest/install.html) for all 
operating systems.

### Python virtual environments
There will be 2 separate Python environments required. The reason is that the 
implementation employs Pratul et al.'s 
[lighthouse](https://github.com/pratulsrinivasan/lighthouse) which based their
code heavily on TensorFlow `1.15.0`. Since the latest Python version supporting
TensorFlow `1.15.0` is `3.7.x`, some major features that are required by most 
packages used in this project are inapplicable. Therefore, a separate Python 
environment shall be set up to run lighthouse independently.

We will create a common Python environment which handles most executions and a
small one to take on lighthouse. The former will be referred to as `torch` and 
the latter `tensor`.

It is recommended to use Python's `venv` module to set up
[virtual environments](https://docs.python.org/3.10/library/venv.html). 
For example, to create a virtual environment named *torch*:
```
python -m venv torch
```

To activate the virtual environment on Windows (PowerShell):
```
torch\Scripts\activate
```

On Linux and MacOS (bash/zsh):
```
source torch/bin/activate
```

Deactivate an activated shell:
```
deactivate
```

Note that the Python version inside a virtual environment is the version of the
Python intepreter that was used to run the `venv` module.

#### torch environment
Despite not being the latest, Python `3.10` shall be used for this environment.
The main responsibilty of `torch` is to handle packages from 
[PyTorch](https://pytorch.org/get-started/locally/), for example to install
PyTorch with support for CUDA 12.1:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
 
PyTorch provides the lower layers for models maintained by the 
[Hugging Face](https://github.com/huggingface) community used in the implemenation:

```
pip install transformers diffusers[torch] xformers
```

Finally several handy packages:
```
pip install Pillow opencv-python scipy h5py matplotlib coloredlogs pyexr
```

A freezed snapshot of `torch` can be found in the `requirements-torch.txt` file.

#### tensor environment
Unlike `torch`, Python `3.7` must be used for this environment. This means there
must be 2 separate installations of Python on the machine.

```
pip install tensorflow==1.15.0 matplotlib==2.2.3 scipy==1.1.0 protobuf==3.20.3 numpy==1.16.0 absl-py
```

A snapshot of `tensor` can be found in the `requirements-tensor.txt` file.

## Prepare workspace
- First clone the project:
```
git clone https://github.com/ndming/virtual-object-insertion.git
cd virtual-object-insertion
```
- Download [checkpoints](https://drive.google.com/drive/folders/1VQjRpInmfspz0Rw0Dlm9RbdHX5ziFeDI) 
for Pratul et al.'s lighthouse and place them in `lighthouse/model`
- Download [checkpoints](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip)
for Li et al.'s inverse rendering and place them in `irois/models`

## Usage
The Python scripts that respectively govern the 4 stages of the pipeline are:
- `mapgen.py`: generates feature maps
- `pbrgen.py`: generates render scripts
- `pbrren.py`: automates rendering with `pbrt`
- `objput.py`: inserts the rendered object into a 2D image

To get the hint usage of these scripts:
```
python *.py --help
```

Note that the Python interpreter corresponding to the `torch` virtual enviroment
shall be used to execute these scripts.

### Generation of feature maps
`mapgen.py` requires the path to the image that we would like to insert an
object into and the path to the Python executable in the `tensor` environment.

For example, on Windows:
```
python mapgen.py -cuda --img path/to/some/im.png --py37 path/to/tensor.Scripts/python.exe
```

The `-cuda` option should only be passed if PyTorch was installed with CUDA.

The script will first prompt 4 coordinates that define a plane receiving shadow
cast be the object. The second prompt will be the position of the object in this 
plane. It will then generate all necessary resources, including ones estimated
by Li et al. and Pratul et al.

### Generation of render scripts
`pbrgen.py` only requires the path to the directory containing resources
generated by `mapgen.py`. It is advisable to change the weights applying
on Li et al. and Pratul et al's environment map, for example:
```
python pbrgen.py -upscale --res-dir path/to/gen --w-irois 0.5 --w-house 2
```

Note that if `-upscale` is specified, the script will supersample the Pratul et 
al.'s environment map 4 times its original size using 
[StableDiffusion](https://github.com/Stability-AI/stablediffusion). Depending
on the GPU capability, this process may take up to half an hour.

### Virtual object rendering
`pbrren.py` needs to know the folder containing the `pbrt` and `imgtool`
executables, and the path to the directory containing resources generated by
`pbrgen.py`. As an example:
```
python pbrren.py -gpu --pbrt-dir path/to/pbrt/folder --res-dir path/to/gen/pbrt
```

The `-gpu` option should only be passed if `pbrt` was built with GPU support.

### Virtual object insertion
Finally, `objput.py` takes the path to the directory containing resources
generated by `pbrren.py` and inserts the object into the the image specified
with the `--target` option. If `--target` is ommitted, the script inserts the
object to the `target.png` file sitting in the same folder.
```
python objput.py --res-dir path/to/gen/pbrt --target path/to/some/target.png
```

The insertion result will be saved to the file specified by the `--output` 
option. If this option is ommitted, the result is placed in the resource
directory. Note that this result is itself an EXR, for which an image viewer
like [tev](https://github.com/Tom94/tev) could be helpful.