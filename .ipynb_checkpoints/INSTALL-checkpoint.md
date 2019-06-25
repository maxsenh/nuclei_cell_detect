## Installation

### Requirements:
- PyTorch 1.0 from a nightly release. It **will not** work with 1.0 nor 1.0.1. Installation instructions can be found in https://pytorch.org/get-started/locally/
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV for the webcam demo


### Option 1: From working on Monod

```bash

conda create --name gpu_gen_env python=3.6.8 
source activate gpu_gen_env

# Install cuda python for GPUs
# Istall this before everything in order to have all the correct versions of the dependency of numpy/scikit-learn/scipy/blas/matplotlib/scipy
conda install numba cudatoolkit pyculib

conda install pip jupyter ipython dask
conda install -c conda-forge nodejs ipympl h5py scikit-image


# clone the general working env
conda create --name pt_env --clone gpu_gen_env
source activate pt_env
    
# install pytorch
conda install pytorch torchvision cuda100 -c pytorch


conda create --name pt_mask_Rcnn_env --clone pt_env
source activate pt_mask_Rcnn_env
    
# maskrcnn_benchmark and coco api dependencies
pip install yacs

# Create github directory
mkdir github
    
# install pycocotools
cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
    

# install PyTorch Detection
cd ~/github
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
```
