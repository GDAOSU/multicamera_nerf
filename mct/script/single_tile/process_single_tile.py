import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from nerfstudio.exporter.exporter_utils import generate_point_cloud_all_mct
from nerfstudio.utils.eval_utils import eval_setup


def single_block_train(in_dir,out_dir,num_iters):
    block_path=in_dir
    cmd=["python","scripts/train.py","mct_mipnerf",
            "--data="+block_path,
            "--output-dir="+out_dir,
            "--pipeline.datamanager.dataparser.has_mask=False",
            "--pipeline.datamanager.train-num-images-to-sample-from=10",
            "--pipeline.datamanager.train-num-times-to-repeat-images=1000",
            "--pipeline.datamanager.eval-num-images-to-sample-from=1",
            "--pipeline.datamanager.eval-num-times-to-repeat-images=100",
            "--pipeline.datamanager.train_num_rays_per_batch=2000",
            "--pipeline.datamanager.dataparser.scene_scale","20",
            "--timestamp={}".format(num_iters),
            "--max-num-iterations={}".format(num_iters)]
    print(cmd)
    subprocess.call(cmd)

def generate_pcd_single_tile(trained_model_dir,timestamp,num_pts,out_dir,skip_img=1):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    name=os.path.basename(trained_model_dir)
    block=trained_model_dir
    num_blks=1
    num_pts_per_blk=int(num_pts/num_blks)

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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.numpy())
    tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
    o3d.t.io.write_point_cloud(os.path.join(out_dir, "{}.ply".format(name)), tpcd)

    torch.cuda.empty_cache()
         
if __name__=='__main__':
    #parameters
    data_dir=r'E:\data\mct_data\data\ra_40'  ### image&poses in colmap format, subfolder should be "dense"
    trained_dir=r'E:\data\mct_data\outputs'  ### dir where trained model stored
    num_iters=500                            ### #training iterations, usually from 15000-60000
    num_pts=1000000                          ### #points for the inference
    out_pcd_dir=r'E:\data\mct_data\pcd'      ### dir where point cloud stored
    skip_img=1                               ### when generating point cloud, how many images should skip, 1 means no skip
    trained_model_dir=os.path.join(trained_dir,os.path.basename(data_dir))

    #Step1: training
    single_block_train(data_dir,trained_dir,num_iters)
    #Step2: inference point cloud
    generate_pcd_single_tile(trained_model_dir,str(num_iters),num_pts,out_pcd_dir,skip_img)
