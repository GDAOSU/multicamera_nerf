import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.exporter.exporter_utils import (
    generate_depth_rgb_all_mct,
    generate_point_cloud_all_mct,
)
from nerfstudio.exporter.tsdf_utils import TSDF
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup


def generate_mesh(config_file,out_dir,skip_img,downsample_factor):
    device_cpu=torch.device('cpu')
    use_bounding_box = False
    scene_scale=5
    bounding_box_min=[-1,-1,-0.5]
    bounding_box_max=[1,1,0.5]
    resolution=[256, 256, 200]

    ##
    _, pipeline, _, _ = eval_setup(Path(config_file),1024,test_mode='all',downsample_size=downsample_factor)
    device = pipeline.device
    scale_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_scale
    shift_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_shift
    shift=-shift_input2nerf.to(device)
    scale=(1/scale_input2nerf).to(device)

    depth_file=os.path.join(out_dir,"depths.pt")   
    rgb_file=os.path.join(out_dir,"rgbs.pt")
    Ks_file=os.path.join(out_dir,"Ks.pt")
    c2ws_file=os.path.join(out_dir,"c2ws.pt")

    if os.path.exists(depth_file) and os.path.exists(rgb_file) and os.path.exists(Ks_file) and os.path.exists(c2ws_file):
        depths=torch.load(os.path.join(out_dir,"depths.pt"))
        rgbs=torch.load(os.path.join(out_dir,"rgbs.pt"))
        c2ws=torch.load(os.path.join(out_dir,"c2ws.pt"))
        Ks=torch.load(os.path.join(out_dir,"Ks.pt"))
        # for i in range(len(depths)):
        #     depths[i]=depths[i].to(device_cpu)
        #     rgbs[i]=rgbs[i].to(device_cpu)
        #     c2ws[i]=c2ws[i].to(device_cpu)
        #     Ks[i]=Ks[i].to(device_cpu)
    else:
        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 10240
        pipeline.model.config.eval_num_rays_per_chunk=10000
        

        #offset=np.array([-self.offx,-self.offy,-self.offz])
        #shift+=offset
        depths,rgbs,c2ws,Ks = generate_depth_rgb_all_mct(
            pipeline=pipeline,
            rgb_output_name="rgb_fine",
            depth_output_name="depth_fine",
            output_dir=Path(out_dir),
            skip_image=skip_img,
            rgb_gt=False
        )
        torch.save(depths,os.path.join(out_dir,"depths.pt"))
        torch.save(rgbs,os.path.join(out_dir,"rgbs.pt"))
        torch.save(c2ws,os.path.join(out_dir,"c2ws.pt"))
        torch.save(Ks,os.path.join(out_dir,"Ks.pt"))

    print("Integrating the TSDF")

    # initialize the TSDF volume
    if not use_bounding_box:
        aabb = pipeline.datamanager.train_dataparser_outputs.scene_box.aabb
        aabb[:,0]/=scene_scale
        aabb[:,1]/=scene_scale

    else:
        aabb = torch.tensor([bounding_box_min, bounding_box_max])
    if isinstance(resolution, int):
        volume_dims = torch.tensor([resolution] * 3)
    elif isinstance(resolution, List):
        volume_dims = torch.tensor(resolution)
    else:
        raise ValueError("Resolution must be an int or a list.")
    tsdf = TSDF.from_aabb(aabb, volume_dims=volume_dims)
    tsdf.truncation_margin=3
    aabb.to(device)
    tsdf.to(device)
    for i in range(0, len(c2ws)):
        tsdf.integrate_tsdf(
            c2ws[i].unsqueeze(0),
            Ks[i].unsqueeze(0),
            depths[i].unsqueeze(0).unsqueeze(0),
            rgbs[i].unsqueeze(0)
        )


    print("Computing Mesh")
    mesh = tsdf.get_mesh()
    ## transform back to original coordiantes
    mesh.vertices*=scale
    mesh.vertices+=shift
    print("Saving TSDF Mesh")
    tsdf.export_mesh(mesh, filename=os.path.join(out_dir,"tsdf_mesh.ply"))
    torch.cuda.empty_cache()
                    
def generate_mesh_block(config_file,out_dir,skip_img,downsample_factor,block_split=2):
    device_cpu=torch.device('cpu')
    use_bounding_box = False
    scene_scale=5
    bounding_box_min=[-1,-1,-0.5]
    bounding_box_max=[1,1,0.5]
    resolution=[512, 512, 256]

    ##
    _, pipeline, _, _ = eval_setup(Path(config_file),1024,test_mode='all',downsample_size=downsample_factor)
    device = pipeline.device
    scale_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_scale
    shift_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_shift
    shift=-shift_input2nerf.to(device)
    scale=(1/scale_input2nerf).to(device)

    depth_file=os.path.join(out_dir,"depths.pt")   
    rgb_file=os.path.join(out_dir,"rgbs.pt")
    Ks_file=os.path.join(out_dir,"Ks.pt")
    c2ws_file=os.path.join(out_dir,"c2ws.pt")

    if os.path.exists(depth_file) and os.path.exists(rgb_file) and os.path.exists(Ks_file) and os.path.exists(c2ws_file):
        depths=torch.load(os.path.join(out_dir,"depths.pt"))
        rgbs=torch.load(os.path.join(out_dir,"rgbs.pt"))
        c2ws=torch.load(os.path.join(out_dir,"c2ws.pt"))
        Ks=torch.load(os.path.join(out_dir,"Ks.pt"))
        # for i in range(len(depths)):
        #     depths[i]=depths[i].to(device_cpu)
        #     rgbs[i]=rgbs[i].to(device_cpu)
        #     c2ws[i]=c2ws[i].to(device_cpu)
        #     Ks[i]=Ks[i].to(device_cpu)
    else:
        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 10240
        pipeline.model.config.eval_num_rays_per_chunk=10000
        

        #offset=np.array([-self.offx,-self.offy,-self.offz])
        #shift+=offset
        depths,rgbs,c2ws,Ks = generate_depth_rgb_all_mct(
            pipeline=pipeline,
            rgb_output_name="rgb_fine",
            depth_output_name="depth_fine",
            output_dir=Path(out_dir),
            skip_image=skip_img
        )
        torch.save(depths,os.path.join(out_dir,"depths.pt"))
        torch.save(rgbs,os.path.join(out_dir,"rgbs.pt"))
        torch.save(c2ws,os.path.join(out_dir,"c2ws.pt"))
        torch.save(Ks,os.path.join(out_dir,"Ks.pt"))

    print("Integrating the TSDF")

    # initialize the TSDF volume
    if not use_bounding_box:
        aabb = pipeline.datamanager.train_dataparser_outputs.scene_box.aabb
        aabb[:,0]/=scene_scale
        aabb[:,1]/=scene_scale

    else:
        aabb = torch.tensor([bounding_box_min, bounding_box_max])
    if isinstance(resolution, int):
        volume_dims = torch.tensor([resolution] * 3)
    elif isinstance(resolution, List):
        volume_dims = torch.tensor(resolution)
    else:
        raise ValueError("Resolution must be an int or a list.")
    
    x_len=aabb[1,0]-aabb[0,0]
    y_len=aabb[1,1]-aabb[0,1]
    aabb_min=aabb[0,:]
    aabb_max=aabb[1,:]
    for block_i in range(block_split):
        block_xlen=x_len/block_split
        for block_j in range(block_split):
            block_ylen=y_len/block_split
            block_aabb=aabb
            block_aabb[0,:]=aabb_min
            block_aabb[1,:]=aabb_max
            block_aabb[0,0]+=block_i*block_xlen
            block_aabb[1,0]=block_aabb[0,0]+block_xlen
            block_aabb[0,1]+=block_j*block_ylen
            block_aabb[1,1]=block_aabb[0,1]+block_ylen
            tsdf = TSDF.from_aabb(block_aabb, volume_dims=volume_dims)
            tsdf.truncation_margin=3
            tsdf.to(device)
            for i in range(0, len(c2ws)):
                tsdf.integrate_tsdf(
                    c2ws[i].unsqueeze(0),
                    Ks[i].unsqueeze(0),
                    depths[i].unsqueeze(0).unsqueeze(0),
                    rgbs[i].unsqueeze(0)
                )


            print("Computing Mesh")
            mesh = tsdf.get_mesh()
            ## transform back to original coordiantes
            mesh.vertices*=scale
            mesh.vertices+=shift
            print("Saving TSDF Mesh")
            tsdf.export_mesh(mesh, filename=os.path.join(out_dir,"tsdf_mesh_{}_{}.ply".format(block_i,block_j)))
    torch.cuda.empty_cache()
                    


# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area1\0\mct_mipnerf\0\config.yml',
#              r'J:\xuningli\cross-view\ns\nerfstudio\mesh\usc_area1\test',1,1)

# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area2\0\mct_mipnerf\250k\config.yml',
#              r'J:\xuningli\cross-view\ns\nerfstudio\mesh\usc_area2',1,1)
# generate_mesh_block('/research/GDA/xuningli/cross-view/ns/nerfstudio/outputs/usc_area2/0/mct_mipnerf/250k/config.yml',
#              '/research/GDA/xuningli/cross-view/ns/nerfstudio/mesh/usc_area2',1,1)

# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area3\0\mct_mipnerf\250k\config_win.yml',
#              r'J:\xuningli\cross-view\ns\nerfstudio\mesh\usc_area3',1,1)
# generate_mesh_block('/research/GDA/xuningli/cross-view/ns/nerfstudio/outputs/usc_area3/0/mct_mipnerf/250k/config.yml',
#              '/research/GDA/xuningli/cross-view/ns/nerfstudio/mesh/usc_area3',1,1)
# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area1\0\mct_mipnerf\250k\config.yml',
#              r'J:\xuningli\cross-view\ns\nerfstudio\mesh\usc_area1',1,1)

# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area1\0\mct_mipnerf\100k\config_win.yml',
#              r'J:\xuningli\cross-view\ns\nerfstudio\mesh\osu_jul22_area1',2,1)
# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area2\0\mct_mipnerf\100k\config_win.yml',
#              r'J:\xuningli\cross-view\ns\nerfstudio\mesh\osu_jul22_area2',2,1)
# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area3\0\mct_mipnerf\100k\config_win.yml',
#              r'J:\xuningli\cross-view\ns\nerfstudio\mesh\osu_jul22_area3',2,1)
# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area4\0\mct_mipnerf\100k\config_win.yml',
#              r'J:\xuningli\cross-view\ns\nerfstudio\mesh\osu_jul22_area4',2,1)

generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area2\0\mct_mipnerf\30k\config.yml',
r'J:\xuningli\cross-view\ns\nerfstudio\mesh\osu_jul22_area2',2,1)
generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area6\0\mct_mipnerf\30k\config.yml',
r'J:\xuningli\cross-view\ns\nerfstudio\mesh\osu_jul22_area6',2,1)
generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area7\0\mct_mipnerf\30k\config.yml',
r'J:\xuningli\cross-view\ns\nerfstudio\mesh\osu_jul22_area7',2,1)
# generate_mesh(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area7\0\mct_mipnerf\30k\config_linux.yml',
# r'J:\xuningli\cross-view\ns\nerfstudio\mesh\usc_area7',2,1)
# generate_mesh('/research/GDA/xuningli/cross-view/ns/nerfstudio/outputs/usc_area7/0/mct_mipnerf/30k/config_linux.yml',
# '/research/GDA/xuningli/cross-view/ns/nerfstudio/mesh/usc_area7',2,1)

# generate_mesh('/research/GDA/xuningli/cross-view/ns/nerfstudio/outputs/ra_area1/0/mct_mipnerf/30k/config.yml',
# '/research/GDA/xuningli/cross-view/ns/nerfstudio/mesh/ra_area1',1,1)
# generate_mesh('/research/GDA/xuningli/cross-view/ns/nerfstudio/outputs/ra_area2/0/mct_mipnerf/30k/config.yml',
# '/research/GDA/xuningli/cross-view/ns/nerfstudio/mesh/ra_area2',1,1)
# generate_mesh('/research/GDA/xuningli/cross-view/ns/nerfstudio/outputs/ra_area3/0/mct_mipnerf/30k/config.yml',
# '/research/GDA/xuningli/cross-view/ns/nerfstudio/mesh/ra_area3',1,1)
# generate_mesh('/research/GDA/xuningli/cross-view/ns/nerfstudio/outputs/ra_area4/0/mct_mipnerf/30k/config.yml',
# '/research/GDA/xuningli/cross-view/ns/nerfstudio/mesh/ra_area4',1,1)

