import glob
import os
import subprocess

import cv2


def train(id):
    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/osu_jul22/area"+str(id)+"/0",
               "--output-dir=outputs/osu_jull22_area"+str(id),
               "--experiment-name=0",
               "--timestamp=100k",
               "--pipeline.datamanager.dataparser.has_mask=True",
               "--pipeline.datamanager.dataparser.scene_scale=5",
               "--steps-per-eval-image=2000000",
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=100",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=4000",
               "--max-num-iterations=50000"]
    subprocess.call(cmd)

def train_usc(id):
    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/usc/area"+str(id)+"/0",
               "--output-dir=outputs/usc_area"+str(id),
               "--experiment-name=0",
               "--timestamp=30k",
               "--pipeline.datamanager.dataparser.has_mask=True",
               "--pipeline.datamanager.dataparser.scene_scale=5",
               "--steps-per-eval-image=2000000",
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=100",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=4000",
               "--max-num-iterations=30000"]
    subprocess.call(cmd)

def train_osu(id):
    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/osu_jul22/area"+str(id)+"/0",
               "--output-dir=outputs/osu_jull22_area"+str(id),
               "--experiment-name=0",
               "--timestamp=30k",
               "--pipeline.datamanager.dataparser.has_mask=True",
               "--pipeline.datamanager.dataparser.scene_scale=5",
               "--steps-per-eval-image=2000000",
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=100",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=4000",
               "--max-num-iterations=30000"]
    subprocess.call(cmd)

def train_ra(id):
    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/ra/area"+str(id)+"/0",
               "--output-dir=outputs/ra_area"+str(id),
               "--experiment-name=0",
               "--timestamp=30k",
               "--pipeline.datamanager.dataparser.has_mask=True",
               "--pipeline.datamanager.dataparser.scene_scale=5",
               "--steps-per-eval-image=2000000",
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=100",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=4000",
               "--max-num-iterations=30000"]
    subprocess.call(cmd)



def train_other():
    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/geomvs_original/test1/0",
               "--output-dir=outputs/test",
               "--experiment-name=0",
               "--timestamp=0",
               "--pipeline.datamanager.dataparser.has_mask=False",
               "--pipeline.datamanager.dataparser.scene_scale=5",
               "--steps-per-eval-image=200000",
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=1000",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=2000",
               "--max-num-iterations=200000"]
    subprocess.call(cmd)

def retrain():
    # cmd=["python","scripts/train.py","mct_mipnerf","--data=data/usc/area1/0",
    #            "--output-dir=outputs/usc_area1",
    #            "--experiment-name=0",
    #            "--timestamp=250k",
    #            "--load-dir","outputs/usc_area1/0/mct_mipnerf/0/nerfstudio_models",
    #            "--pipeline.datamanager.dataparser.scene_scale=5",
    #            "--steps-per-eval-image=2000000",
    #            "--pipeline.datamanager.train-num-images-to-sample-from=10",
    #            "--pipeline.datamanager.train-num-times-to-repeat-images=1000",
    #             "--pipeline.datamanager.eval-num-images-to-sample-from=1",
    #            "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
    #            "--pipeline.datamanager.train_num_rays_per_batch=2000",
    #            "--max-num-iterations=250000"]
    # subprocess.call(cmd)

    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/usc/area2/0",
               "--output-dir=outputs/usc_area2",
               "--experiment-name=0",
               "--timestamp=250k",
               "--load-dir","outputs/usc_area2/0/mct_mipnerf/0/nerfstudio_models",
               "--pipeline.datamanager.dataparser.scene_scale=5",
               "--steps-per-eval-image=2000000",
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=1000",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=2000",
               "--max-num-iterations=100000"]
    subprocess.call(cmd)

    cmd=["python","scripts/train.py","mct_mipnerf","--data=data/usc/area3/0",
               "--output-dir=outputs/usc_area3",
               "--experiment-name=0",
               "--timestamp=250k",
               "--load-dir","outputs/usc_area3/0/mct_mipnerf/0/nerfstudio_models",
               "--pipeline.datamanager.dataparser.scene_scale=5",
               "--steps-per-eval-image=2000000",
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=1000",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=500",
               "--pipeline.datamanager.train_num_rays_per_batch=2000",
               "--max-num-iterations=100000"]
    subprocess.call(cmd)

# train_osu(2)
# train_osu(6)
# train_osu(7)
# train_usc(7)

# train_ra(1)
train_ra(2)
# train_ra(3)
# train_ra(4)


# train(4)
# train(3)
#train_other()
#retrain()