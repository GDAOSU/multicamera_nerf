import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import argparse
import copy
import glob
import math
import os
import shutil

import cv2
import numpy as np
import torch
from rich.progress import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_text,
    read_images_text,
)

parser = argparse.ArgumentParser()
parser.add_argument("-i", required=True, help="input dir")
parser.add_argument("-num_split", required=True, help="input dir")
parser.add_argument("-o", default="", help="output dir")
parser.add_argument(
    "-scene_bbox", default="", help="optional, if given it will only generate block within this scene_bbox"
)
CONSOLE = Console(width=120)

"""num_tiles: in x/y direction"""

## scene_aabb: [xmin,ymin,zmin,xmax,ymax,zmax], img_pose: 4x4, w2c
def scenebbox_image_intersect(scene_aabb, img_pose, img_K, width, height):
    w2c = img_pose[:3, :]
    vertex_world = np.array(
        [
            [scene_aabb[0], scene_aabb[1], scene_aabb[2], 1],
            [scene_aabb[0], scene_aabb[4], scene_aabb[2], 1],
            [scene_aabb[3], scene_aabb[4], scene_aabb[2], 1],
            [scene_aabb[3], scene_aabb[1], scene_aabb[2], 1],
            [scene_aabb[0], scene_aabb[1], scene_aabb[5], 1],
            [scene_aabb[0], scene_aabb[4], scene_aabb[5], 1],
            [scene_aabb[3], scene_aabb[4], scene_aabb[5], 1],
            [scene_aabb[3], scene_aabb[1], scene_aabb[5], 1],
        ]
    )
    vertex_pix = img_K @ w2c @ vertex_world.transpose()
    vertex_pix[0, :] = vertex_pix[0, :] / vertex_pix[2, :]
    vertex_pix[1, :] = vertex_pix[1, :] / vertex_pix[2, :]
    vertex_pix[2, :] = vertex_pix[2, :] / vertex_pix[2, :]

    max_proj=np.max(vertex_pix)
    min_proj=np.min(vertex_pix)


    bbox_proj = [np.min(vertex_pix[0, :]), np.max(vertex_pix[0, :]), np.min(vertex_pix[1, :]), np.max(vertex_pix[1, :])]
    intersect_bbox = [
        int(max(0, bbox_proj[0])),
        int(min(width - 1, bbox_proj[1])),
        int(max(0, bbox_proj[2])),
        int(min(height - 1, bbox_proj[3])),
    ]
    if max_proj>15000 or min_proj < -15000:
        return False, intersect_bbox
    elif intersect_bbox[0] < intersect_bbox[1] and intersect_bbox[2] < intersect_bbox[3]:
        xlen = intersect_bbox[1] - intersect_bbox[0]
        ylen = intersect_bbox[3] - intersect_bbox[2]
        return True, intersect_bbox
        # if xlen * ylen / (width * height) > 0.8:
        #     return True, intersect_bbox
        # else:
        #     return False, intersect_bbox
    else:
        return False, intersect_bbox


def split_bbox(scene_bbox, num_tiles, expand_m=3):
    tiles = []
    x_len = scene_bbox[3] - scene_bbox[0]
    y_len = scene_bbox[4] - scene_bbox[1]

    x_tile_len = x_len / num_tiles
    y_tile_len = y_len / num_tiles
    expand_x = expand_m
    expand_y = expand_m
    for row in range(int(num_tiles)):
        for col in range(int(num_tiles)):
            xmin = col * x_tile_len + scene_bbox[0] - expand_x
            xmax = (col + 1) * x_tile_len + scene_bbox[0] + expand_x
            ymin = row * y_tile_len + scene_bbox[1] - expand_y
            ymax = (row + 1) * y_tile_len + scene_bbox[1] + expand_y
            tiles.append([xmin, ymin, scene_bbox[2], xmax, ymax, scene_bbox[5]])
    return tiles


def split_block(in_dir, num_tiles=2, out_dir="", scene_bbox_path=""):
    if out_dir == "":
        out_dir = os.path.join(in_dir, "block")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    poses = []

    cams = read_cameras_text(os.path.join(in_dir, "dense/sparse/cameras.txt"))
    imgs = read_images_text(os.path.join(in_dir, "dense/sparse/images.txt"))
    scene_bbox = np.loadtxt(os.path.join(in_dir, "dense/sparse/scene_bbox.txt"))
    ground_range = np.loadtxt(os.path.join(in_dir, "dense/sparse/ground_range.txt"))
    print(ground_range)
    tiles_bbox = []
    if os.path.exists(scene_bbox_path):
        tile_bbox = list(np.loadtxt(scene_bbox_path))
        ground_range[0] = tile_bbox[2]
        ground_range[1] = tile_bbox[5]
        tiles_bbox.append(tile_bbox)
    else:
        tiles_bbox = split_bbox(scene_bbox, num_tiles)
    poses = []
    fxs = []
    fys = []
    cxs = []
    cys = []
    heights = []
    widths = []
    Ks = []

    for _id, img in imgs.items():
        cam = cams[img.camera_id]
        pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
        pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
        poses.append(torch.linalg.inv(pose))
        heights.append(torch.tensor(cam.height))
        widths.append(torch.tensor(cam.width))
        fxs.append(torch.tensor(cam.params[0]))
        fys.append(torch.tensor(cam.params[1]))
        cxs.append(torch.tensor(cam.params[2]))
        cys.append(torch.tensor(cam.params[3]))
        Ks.append(torch.tensor([[cam.params[0], 0, cam.params[2]], [0, cam.params[1], cam.params[3]], [0, 0, 1]]))

    poses = torch.stack(poses).float()
    poses[..., 1:3] *= -1
    fxs = torch.stack(fxs).float()
    fys = torch.stack(fys).float()
    cxs = torch.stack(cxs).float()
    cys = torch.stack(cys).float()
    widths = torch.stack(widths).int()
    heights = torch.stack(heights).int()
    Ks = torch.stack(Ks).float()

    ## filter images in aoi_aabb
    for id, tile in enumerate(tiles_bbox):
        print(id)
        tile_w_ground = copy.deepcopy(tile)
        tile_w_ground[2] = ground_range[0]
        tile_w_ground[5] = ground_range[1]
        valid_ids = camera_utils.filter_images_aoi_aabb(poses, heights, widths, fxs, fys, cxs, cys, tile_w_ground)
        # valid_ids = [1, 2, 3]
        print("valid_ids: " + str(len(valid_ids)))
        os.mkdir(os.path.join(out_dir, str(id)))
        os.mkdir(os.path.join(out_dir, str(id), "dense"))
        os.mkdir(os.path.join(out_dir, str(id), "dense", "sparse"))
        os.mkdir(os.path.join(out_dir, str(id), "dense", "images"))
        ### write scene bbox txt
        np.savetxt(os.path.join(out_dir, str(id), "dense", "sparse", "scene_bbox.txt"), tile)
        ### write tile obj
        tile_obj_out = open(os.path.join(out_dir, str(id), "dense", "sparse", "scene_bbox.obj"), "w", encoding="utf-8")
        tile_obj_out.write("v {} {} {}\n".format(tile[0], tile[1], tile[2]))
        tile_obj_out.write("v {} {} {}\n".format(tile[0], tile[4], tile[2]))
        tile_obj_out.write("v {} {} {}\n".format(tile[3], tile[4], tile[2]))
        tile_obj_out.write("v {} {} {}\n".format(tile[3], tile[1], tile[2]))
        tile_obj_out.write("v {} {} {}\n".format(tile[0], tile[1], tile[5]))
        tile_obj_out.write("v {} {} {}\n".format(tile[0], tile[4], tile[5]))
        tile_obj_out.write("v {} {} {}\n".format(tile[3], tile[4], tile[5]))
        tile_obj_out.write("v {} {} {}\n".format(tile[3], tile[1], tile[5]))
        tile_obj_out.write("f 1 2 3\n")
        tile_obj_out.write("f 1 3 4\n")
        tile_obj_out.write("f 5 6 7\n")
        tile_obj_out.write("f 5 7 8\n")
        tile_obj_out.write("f 1 5 8\n")
        tile_obj_out.write("f 1 8 4\n")
        tile_obj_out.write("f 1 2 6\n")
        tile_obj_out.write("f 1 6 5\n")
        tile_obj_out.write("f 4 3 7\n")
        tile_obj_out.write("f 4 7 8\n")
        tile_obj_out.write("f 2 3 7\n")
        tile_obj_out.write("f 2 7 6\n")
        tile_obj_out.close()

        ## write colmap file
        cam_txt = os.path.join(out_dir, str(id), "dense", "sparse", "cameras.txt")
        img_txt = os.path.join(out_dir, str(id), "dense", "sparse", "images.txt")
        cam_out = open(cam_txt, "w", encoding="utf-8")
        img_out = open(img_txt, "w", encoding="utf-8")

        cnt = 1
        for i in valid_ids:
            cam = cams[i + 1]
            img = imgs[i + 1]
            cam_out.write(
                "{} {} {} {} {} {} {} {} {} {}\n".format(
                    cnt,
                    "PINHOLE",
                    cam.width,
                    cam.height,
                    cam.params[0],
                    cam.params[1],
                    cam.params[2],
                    cam.params[3],
                    cam.params[-2],
                    cam.params[-1],
                )
            )
            img_out.write(
                "{} {} {} {} {} {} {} {} {} {}\n".format(
                    cnt,
                    img.qvec[0],
                    img.qvec[1],
                    img.qvec[2],
                    img.qvec[3],
                    img.tvec[0],
                    img.tvec[1],
                    img.tvec[2],
                    cnt,
                    img.name,
                )
            )
            img_out.write("\n")
            shutil.copyfile(
                os.path.join(in_dir, "dense", "images", img.name),
                os.path.join(out_dir, str(id), "dense", "images", img.name),
            )
            cnt += 1
        cam_out.close()
        img_out.close()

### multi-camera tiling process
### in_dir: images and poses in colmap format
### num_tiles: num of tiles in x,y direction. e.g. num_tiles=4, then 4*4=16 blocks in total
### out_dir: output directory
### scene_bbox_path: if available, cropped image will only focus on that scene_bbox, if not, tiling will be based on num_tiles parameter.
def split_block_projection_new(in_dir, num_tiles=2, out_dir="", scene_bbox_path=""):
    musk_thresh=15 ## sum of r,g,b
    if out_dir == "":
        out_dir = os.path.join(in_dir, "block")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    poses = []

    cams = read_cameras_text(os.path.join(in_dir, "sparse/cameras.txt"))
    imgs = read_images_text(os.path.join(in_dir, "sparse/images.txt"))
    #scene_bbox = np.loadtxt(os.path.join(in_dir, "sparse/scene_bbox.txt"))
    #ground_range = np.loadtxt(os.path.join(in_dir, "sparse/ground_range.txt"))
    #print("ground range: {}".format(ground_range))
    tiles_bbox = []
    if os.path.exists(scene_bbox_path):
        tile_bbox = list(np.loadtxt(scene_bbox_path))
        ground_range=np.zeros(2)
        ground_range[0]=tile_bbox[2]
        ground_range[1]=tile_bbox[5]
        #tiles_bbox.append(tile_bbox)
        tiles_bbox=split_bbox(np.array(tile_bbox),num_tiles)
    else:
        scene_bbox = np.loadtxt(os.path.join(in_dir, "sparse/scene_bbox.txt"))
        ground_range = np.loadtxt(os.path.join(in_dir, "sparse/ground_range.txt"))
        tiles_bbox = split_bbox(scene_bbox, num_tiles)
    poses = []
    heights = []
    widths = []
    Ks = []
    img_names = []


    for _id, img in imgs.items():
        cam = cams[img.camera_id]
        pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
        pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)
        poses.append(pose)
        heights.append(torch.tensor(cam.height))
        widths.append(torch.tensor(cam.width))
        Ks.append(torch.tensor([[cam.params[0], 0, cam.params[2]], [0, cam.params[1], cam.params[3]], [0, 0, 1]]))
        img_names.append(img.name)

    poses = torch.stack(poses).float()
    widths = torch.stack(widths).int()
    heights = torch.stack(heights).int()
    Ks = torch.stack(Ks).float()

    block_validimg_ids = {}
    block_validimg_aoibbox = {}

    ## filter images in aoi_aabb
    for id, tile in enumerate(tiles_bbox):
        os.mkdir(os.path.join(out_dir, str(id)))
        os.mkdir(os.path.join(out_dir, str(id), "dense"))
        os.mkdir(os.path.join(out_dir, str(id), "dense", "sparse"))
        os.mkdir(os.path.join(out_dir, str(id), "dense", "images"))
        tile_w_ground = copy.deepcopy(tile)
        tile_w_ground[2] = ground_range[0]
        tile_w_ground[5] = ground_range[1]
        valid_ids = []
        aoi_bboxs = []
        for img_id in range(poses.shape[0]):
            inter_status, inter_bbox = scenebbox_image_intersect(
                tile_w_ground,
                poses[img_id, :, :].numpy(),
                Ks[img_id, :, :].numpy(),
                widths[img_id].numpy(),
                heights[img_id].numpy(),
            )
            if inter_status:
                aoi_bboxs.append(inter_bbox)
                valid_ids.append(img_id)
        block_validimg_ids[id] = valid_ids
        block_validimg_aoibbox[id] = aoi_bboxs
        print("block {} #valid_ids: {}".format(id, len(valid_ids)))

        ### write scene bbox txt
        np.savetxt(os.path.join(out_dir, str(id), "dense", "sparse", "scene_bbox.txt"), tile)
        ### write tile obj
        tile_obj_out = open(os.path.join(out_dir, str(id), "dense", "sparse", "scene_bbox.obj"), "w", encoding="utf-8")
        tile_obj_out.write("v {} {} {}\n".format(tile[0], tile[1], tile[2]))
        tile_obj_out.write("v {} {} {}\n".format(tile[0], tile[4], tile[2]))
        tile_obj_out.write("v {} {} {}\n".format(tile[3], tile[4], tile[2]))
        tile_obj_out.write("v {} {} {}\n".format(tile[3], tile[1], tile[2]))
        tile_obj_out.write("v {} {} {}\n".format(tile[0], tile[1], tile[5]))
        tile_obj_out.write("v {} {} {}\n".format(tile[0], tile[4], tile[5]))
        tile_obj_out.write("v {} {} {}\n".format(tile[3], tile[4], tile[5]))
        tile_obj_out.write("v {} {} {}\n".format(tile[3], tile[1], tile[5]))
        tile_obj_out.write("f 1 2 3\n")
        tile_obj_out.write("f 1 3 4\n")
        tile_obj_out.write("f 5 6 7\n")
        tile_obj_out.write("f 5 7 8\n")
        tile_obj_out.write("f 1 5 8\n")
        tile_obj_out.write("f 1 8 4\n")
        tile_obj_out.write("f 1 2 6\n")
        tile_obj_out.write("f 1 6 5\n")
        tile_obj_out.write("f 4 3 7\n")
        tile_obj_out.write("f 4 7 8\n")
        tile_obj_out.write("f 2 3 7\n")
        tile_obj_out.write("f 2 7 6\n")
        tile_obj_out.close()

    ## crop images
    for img_id in range(poses.shape[0]):
        print("processing img:{}".format(img_id))
        img_name=img_names[img_id]
        img_path = os.path.join(in_dir, "images/" + img_name)

        load=False
        for block_id in block_validimg_ids:
            valid_ids = block_validimg_ids[block_id]
            aoi_bboxs = block_validimg_aoibbox[block_id]
            if img_id in valid_ids:
                load=True
        if not load:
            continue
        img_arr = cv2.imread(img_path)
        for block_id in block_validimg_ids:
            valid_ids = block_validimg_ids[block_id]
            aoi_bboxs = block_validimg_aoibbox[block_id]
            if img_id in valid_ids:
                ind = np.where(np.array(valid_ids) == img_id)[0][0]
                aoi_bbox = aoi_bboxs[ind]
                img_aoi = img_arr[aoi_bbox[2] : aoi_bbox[3], aoi_bbox[0] : aoi_bbox[1], :]
                img_musk=(np.sum(img_aoi,axis=2)>musk_thresh)*255
                musk_name=img_name[:-4]+'_mask'+img_name[-4:]
                if np.sum(img_musk)==0:
                    valid_ids.remove(img_id)
                    aoi_bboxs.remove(aoi_bbox)
                    continue
                cv2.imwrite(os.path.join(out_dir, str(block_id), "dense", "images", musk_name), img_musk)
                cv2.imwrite(os.path.join(out_dir, str(block_id), "dense", "images", img_name), img_aoi)
    
    
    for id, tile in enumerate(tiles_bbox):       
        ## write colmap file
        cam_txt = os.path.join(out_dir, str(id), "dense", "sparse", "cameras.txt")
        img_txt = os.path.join(out_dir, str(id), "dense", "sparse", "images.txt")
        cam_out = open(cam_txt, "w", encoding="utf-8")
        img_out = open(img_txt, "w", encoding="utf-8")
        valid_ids=block_validimg_ids[id]
        aoi_bboxs=block_validimg_aoibbox[id]
        cnt = 1
        for id, i in enumerate(valid_ids):
            aoi_bbox = aoi_bboxs[id]
            width = aoi_bbox[1] - aoi_bbox[0]
            height = aoi_bbox[3] - aoi_bbox[2]
            cam = cams[i + 1]
            img = imgs[i + 1]
            cam_out.write(
                "{} {} {} {} {} {} {} {}\n".format(
                    cnt,
                    "PINHOLE",
                    width,
                    height,
                    cam.params[0],
                    cam.params[1],
                    cam.params[2] - float(aoi_bbox[0]),
                    cam.params[3] - float(aoi_bbox[2]),
                )
            )
            img_out.write(
                "{} {} {} {} {} {} {} {} {} {}\n".format(
                    cnt,
                    img.qvec[0],
                    img.qvec[1],
                    img.qvec[2],
                    img.qvec[3],
                    img.tvec[0],
                    img.tvec[1],
                    img.tvec[2],
                    cnt,
                    img.name,
                )
            )
            img_out.write("\n")
            cnt += 1
        cam_out.close()
        img_out.close()

if __name__=='__main__':
    in_dir=r'E:\data\mct_data\data\dortmund_whole'
    out_dir=r'E:\data\mct_data\data\dortmund_blocks'
    num_split=16
    scene_box_path=r''

    split_block_projection_new(in_dir, num_split, out_dir, scene_box_path)



