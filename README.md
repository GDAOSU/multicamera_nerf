# Multi-tiling neural radiance field (NeRF)—geometric assessment on large-scale aerial datasets 
Ningli Xu, Rongjun Qin, Debao Huang, Fabio Remondino
The Photogrammetric Record, 2024  

![alt text](comparison.png "Comparison between ours and SOTA MVS methods")

NeRF vs Multi-view stereo? We propose multi-camera tiling technique to enable NeRF on large-scale aerial datasets and further conduct experiment to compare their geometry reconstruction performance.

# Updates and To Do 
- [x] [06/05/2024]Release the sample datasets   
- [x] [06/05/2024]Speed up the rendering RGB,Depth,point cloud, mesh  
- [x] [08/30/2023] Release the code.  

# Install

## 1. Installation: Setup the environment
### Prerequisites
You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.3 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).
If you try to install pytorch with CUDA 11.7, it is not necessary to install CUDA individually.
### Create environment
Nerfstudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.
```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```
### Dependencies
Install pytorch with CUDA (this repo has been tested with CUDA 11.3 and CUDA 11.7) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
For CUDA 11.3:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
(optional) pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
For CUDA 11.7:
```bash
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
(optional) pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.
### Installing nerfstudio
```bash
git clone https://github.com/GDAOSU/multicamera_nerf.git
cd multicamera_nerf
pip install --upgrade pip setuptools
pip install -e .
```
### Installing mct module
```bash
cd mct
pip install -e .
```
# Datasets
Please down the demo data via this [link](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/xu_3961_buckeyemail_osu_edu/EQPWDZYSurRKj2pNDYfvdjAB-_lTkBTkGFbmEZJj66iprQ?e=KhaxWc). The data structure is shown below, where "/mct_data/data" contains images with poses in COLMAP format in 7 blocks and a whole dataset in dortmund, and "/mct_data/pcd" contains the generated dense point cloud by our methods.

Each block contains the pre-tiled image patches, modified camera intrinsic & extrinsic parameters, as described in `step1_preprocess.py`
```
mct_data
|-data
  |-dortmund_whole
  |-dortmund_1
  |-ra_40
  |-ra_311
  |- ...
|-pcd
  |-dortmund_1.ply
  |-ra_40.ply
  |-ra_311.ply
  |- ...
```

## Custom dataset
For a set of images, you can use any sfm softwares (e.g. Colmap, OpenSfM, OpenDroneMap) to 

0. You can take `mct_data/data/dortmund_whole` for example
1. calculate the intrinsic & extrinsic parameters (be sure they are in gravity-aligned direction, e.g. use colmap/model_aligner if GPS info is available)
2. perform undistortion
3. transform the undistroted images /w camera parameters into colmap format (sparse/images.txt,sparse/cameras.txt, images/*.jpg)
4. calculate the auxliary information (ground_range.txt, scene_bbox.txt)

# Training & Generating point cloud from a single block
Take demo data "ra_40" for example.
```
python mct/script/single_tile/process_single_tile.py
```


# Training & Rendering for a aerial dataset
## Step1: preprocess   
use multi-camera tiling to crop large high-res images into smaller images by spliting whole scene into many blocks  
```bash
python mct/script/batch_tiles/step1_preprocess.py
```

## Step2: block training  
training each blocks  
```bash
python mct/script/batch_tiles/step2_batch_train.py
```

## Step3: Rendering images or point cloud  
rendering the novel view or dense point cloud   
```bash
python mct/script/batch_tiles/step3_generate_pcd.py  
```
```bash
python mct/script/batch_tiles/step3_render_image.py  
```

## Citation
Please cite our paper via this link.
```bibtex
@article{xu2024multi,
  title={Multi-tiling neural radiance field (NeRF)—geometric assessment on large-scale aerial datasets},
  author={Xu, Ningli and Qin, Rongjun and Huang, Debao and Remondino, Fabio},
  journal={The Photogrammetric Record},
  year={2024},
  publisher={Wiley Online Library}
}
'''