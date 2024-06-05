import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.exporter.exporter_utils import generate_point_cloud_all_mct
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
import argparse

### generate the dense point cloud from the trained model
## trained_model_dir: contraining the trained model
## timestamp: "mct_mipnerf/"+timestamp+"/config.yml"
## num_pts: #pts for the whole point clouds, the #pts for each block will be equally divided

def generate_pcd(trained_model_dir,timestamp,num_pts,out_dir,skip_img=1):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    trained_blocks=glob.glob(os.path.join(trained_model_dir,"*"))
    num_blks=len(trained_blocks)
    num_pts_per_blk=int(num_pts/num_blks)
    pcd_pts_list=[]
    pcd_rgb_list=[]
    for id,block in enumerate(trained_blocks):
        config, pipeline, _, _ = eval_setup(Path(os.path.join(block,"mct_mipnerf/"+timestamp+"/config.yml")),1024,test_mode='all')
        scene_bbox_txt=os.path.join(config.data,"dense/sparse/scene_bbox.txt")
        scene_bbox=np.loadtxt(scene_bbox_txt)
        scene_bbox_min=torch.tensor(scene_bbox[:3])
        scene_bbox_max=torch.tensor(scene_bbox[3:])
        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 10240
        pipeline.model.config.eval_num_rays_per_chunk=2000
        
        scale_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        shift_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_shift
        shift=-shift_input2nerf.numpy()

        scale=(1/scale_input2nerf).numpy()
        #offset=np.array([-self.offx,-self.offy,-self.offz])
        #shift+=offset
        points,rgbs = generate_point_cloud_all_mct(
            pipeline=pipeline,
            rgb_output_name="rgb_fine",
            depth_output_name="depth_fine",
            output_dir=Path(out_dir),
            num_pts=num_pts_per_blk,
            shiftx=shift[0],
            shifty=shift[1],
            shiftz=shift[2],
            scale=scale,
            skip_image=skip_img
        )
        points*=scale
        points+=shift.reshape(1,3)
        mask = torch.all(torch.concat([points > scene_bbox_min, points < scene_bbox_max], dim=-1), dim=-1)
        points = points[mask]
        rgbs = rgbs[mask]
        pcd_pts_list.append(points)
        pcd_rgb_list.append(rgbs)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        pcd.colors = o3d.utility.Vector3dVector(rgbs.numpy())
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
        o3d.t.io.write_point_cloud(os.path.join(out_dir, "block_{}.ply".format(id)), tpcd)
    points=torch.cat(pcd_pts_list)
    rgbs=torch.cat(pcd_rgb_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().numpy())
    tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    # The legacy PLY writer converts colors to UInt8,
    # let us do the same to save space.
    tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
    o3d.t.io.write_point_cloud(os.path.join(out_dir, "whole.ply"), tpcd)
    torch.cuda.empty_cache()

def generate_pcd_single_tile(trained_model_dir,block_id,timestamp,num_pts,out_dir,skip_img=1):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    #trained_blocks=glob.glob(os.path.join(trained_model_dir,"*"))
    trained_blocks=[os.path.join(trained_model_dir,"{}".format(block_id))]
    num_blks=1
    num_pts_per_blk=int(num_pts/num_blks)
    pcd_pts_list=[]
    pcd_rgb_list=[]
    for id,block in enumerate(trained_blocks):
        config, pipeline, _, _ = eval_setup(Path(os.path.join(block,"mct_mipnerf/"+timestamp+"/config.yml")),1024,test_mode='all')
        scene_bbox_txt=os.path.join(config.data,"dense/sparse/scene_bbox.txt")
        scene_bbox=np.loadtxt(scene_bbox_txt)
        scene_bbox_min=torch.tensor(scene_bbox[:3])
        scene_bbox_max=torch.tensor(scene_bbox[3:])
        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 10240
        pipeline.model.config.eval_num_rays_per_chunk=2000
        
        scale_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        shift_input2nerf=pipeline.datamanager.train_dataparser_outputs.dataparser_shift
        shift=-shift_input2nerf.numpy()

        scale=(1/scale_input2nerf).numpy()
        #offset=np.array([-self.offx,-self.offy,-self.offz])
        #shift+=offset
        points,rgbs = generate_point_cloud_all_mct(
            pipeline=pipeline,
            rgb_output_name="rgb_fine",
            depth_output_name="depth_fine",
            output_dir=Path(out_dir),
            num_pts=num_pts_per_blk,
            shiftx=shift[0],
            shifty=shift[1],
            shiftz=shift[2],
            scale=scale,
            skip_image=skip_img
        )
        points*=scale
        #points+=shift.reshape(1,3)
        # mask = torch.all(torch.concat([points > scene_bbox_min, points < scene_bbox_max], dim=-1), dim=-1)
        # points = points[mask]
        # rgbs = rgbs[mask]
        pcd_pts_list.append(points)
        pcd_rgb_list.append(rgbs)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points.numpy())
        # pcd.colors = o3d.utility.Vector3dVector(rgbs.numpy())
        # tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
        # o3d.t.io.write_point_cloud(os.path.join(out_dir, "block_{}.ply".format(id)), tpcd)
    points=torch.cat(pcd_pts_list)
    rgbs=torch.cat(pcd_rgb_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().numpy())
    tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    # The legacy PLY writer converts colors to UInt8,
    # let us do the same to save space.
    tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
    o3d.t.io.write_point_cloud(os.path.join(out_dir, "whole.ply"), tpcd)
    torch.cuda.empty_cache()
              
parser = argparse.ArgumentParser("Generate point cloud")
parser.add_argument("--trained_dir",type=str,required=True)
parser.add_argument("--",type=str,required=True)

parser.add_argument("counter", help="An integer will be increased by 1 and printed.", type=int)
args = parser.parse_args()

if __name__=='__main__':
    # generate_pcd_single_tile(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\demo_dortmund_metashape_center',0,'60000',10000000,
    #                          r'J:\xuningli\cross-view\ns\nerfstudio\pcd\demo_dortmund_metashape_center',1)
    # generate_pcd_single_tile(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\demo_dortmund_metashape_center_old',0,'60000',10000000,
    #                         r'J:\xuningli\cross-view\ns\nerfstudio\pcd\demo_dortmund_metashape_center_old',1)
    # generate_pcd_single_tile(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\demo_usc_area1',0,'60000',10000000,
    #                         r'J:\xuningli\cross-view\ns\nerfstudio\pcd\demo_usc_area1',1)
    # generate_pcd_single_tile(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\so_ra_0',0,'0',1000000,
    #                         r'J:\xuningli\cross-view\ns\nerfstudio\pcd\ra_so_1_test')
    # generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_dense_2_16_1','30000',10000000,
    #             r'J:\xuningli\cross-view\ns\nerfstudio\pcd\dortmund_dense2_16_1')
# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks_2_16','30k',10000000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\dortmund_dense2_blocks16')

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\boordaux_metashape_blocks_2_16','50000',10000000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\boordaux_metashape_blocks_2_16',1)

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks_2_36','50000',20000000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\dortmund_2_blocks36',1)

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\geomvs_test2','30000',40000000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\geomvs_test2',1)

# generate_pcd(r'/research/GDA/xuningli/cross-view/ns/nerfstudio/outputs/geomvs_test2','30000',100000000,
#              r'/research/GDA/xuningli/cross-view/ns/nerfstudio/pcd/geomvs_test2/new',1)


