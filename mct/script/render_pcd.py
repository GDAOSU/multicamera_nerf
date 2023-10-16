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


def generate_pcd(config_file,num_pts,out_dir,skip_img):

    config, pipeline, _, _ = eval_setup(Path(config_file),1024,test_mode='all')

    # Increase the batchsize to speed up the evaluation.
    pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = 10240
    pipeline.model.config.eval_num_rays_per_chunk=10000


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
        num_pts=num_pts,
        shiftx=shift[0],
        shifty=shift[1],
        shiftz=shift[2],
        scale=scale,
        skip_image=skip_img
    )
    points*=scale
    points+=shift.reshape(1,3)

    scene_bbox_txt=os.path.join(config.data,"dense/sparse/scene_bbox.txt")
    scene_bbox=np.loadtxt(scene_bbox_txt)
    scene_bbox_min=torch.tensor(scene_bbox[:3])
    scene_bbox_max=torch.tensor(scene_bbox[3:])
    mask = torch.all(torch.concat([points > scene_bbox_min, points < scene_bbox_max], dim=-1), dim=-1)
    points = points[mask]
    rgbs = rgbs[mask]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.numpy())
    tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
    o3d.t.io.write_point_cloud(os.path.join(out_dir, "pcd.ply"), tpcd)


    torch.cuda.empty_cache()
                    


# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area1\0\mct_mipnerf\0\config.yml',
#              500000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\usc_area1',10)

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\geomvs_test2\0\mct_mipnerf\30000\config.yml',
#              500000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\geomvs_test2',10)

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area1\0\mct_mipnerf\250k\config.yml',
#             10000000,
#             r'J:\xuningli\cross-view\ns\nerfstudio\pcd\usc_area1\new',1)

generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area3\0\mct_mipnerf\250k\config.yml',
            10000000,
            r'J:\xuningli\cross-view\ns\nerfstudio\pcd\usc_area3\new',1)

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area2\0\mct_mipnerf\250k\config.yml',
#             10000000,
#             r'J:\xuningli\cross-view\ns\nerfstudio\pcd\usc_area2\new',1)

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area1\0\mct_mipnerf\100k\config_win.yml',
#             1000000,
#             r'J:\xuningli\cross-view\ns\nerfstudio\pcd\osu_jul22_area1',1)

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\osu_jull22_area1\0\mct_mipnerf\250k\config.yml',
#             5000000,
#             r'J:\xuningli\cross-view\ns\nerfstudio\pcd\osu_jul22_area1',5)

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area2\0\mct_mipnerf\0\config.yml',
#              5000000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\usc_area2')

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area3\0\mct_mipnerf\0\config.yml',
#              5000000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\usc_area3')

# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\usc_area4\0\mct_mipnerf\0\config.yml',
#              5000000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\usc_area4')


# generate_pcd(r'J:\xuningli\cross-view\ns\nerfstudio\outputs\geomvs_original_test1\0\mct_mipnerf\0\config.yml',
#              5000000,
#              r'J:\xuningli\cross-view\ns\nerfstudio\pcd\geomvs_original_test1')