import glob
import os
import subprocess


## training for each blocks
## in_dir: input direcotory containing the multi-camera tiling datasets
## out_dir: contraining the trained weight
## num_iters: number of iterations for each blocks
def batch_train(in_dir,out_dir,num_iters):
    blocks_paths=glob.glob(os.path.join(in_dir,"*"))
    for id,block_path in enumerate(blocks_paths):
        if not os.path.isdir(block_path):
            continue
        block_id=int(os.path.basename(block_path))
        pretrained_model_dir=os.path.join(out_dir,str(block_id)+"/mct_mipnerf/"+str(num_iters)+"/nerfstudio_models")
        if os.path.exists(pretrained_model_dir):
            continue
        cmd=["python","scripts/train.py","mct_mipnerf",
             "--data="+block_path,
               "--output-dir="+out_dir,
               "--pipeline.datamanager.train-num-images-to-sample-from=10",
               "--pipeline.datamanager.train-num-times-to-repeat-images=1000",
                "--pipeline.datamanager.eval-num-images-to-sample-from=1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=100",
               "--pipeline.datamanager.train_num_rays_per_batch=2000",
               "--pipeline.datamanager.dataparser.scene_scale","5",
               "--timestamp={}".format(num_iters),
               "--max-num-iterations={}".format(num_iters)]
        print(cmd)
        subprocess.call(cmd)

def batch_train_retrain(in_dir,out_dir):
    blocks_paths=glob.glob(os.path.join(in_dir,"*"))
    for block_path in blocks_paths:
        block_id=int(os.path.basename(block_path))
        pretrained_model_dir=os.path.join(out_dir,str(block_id)+"/mct_mipnerf/10k/nerfstudio_models")
        # if os.path.exists(pretrained_model_dir):
        #     continue
        cmd=["python","scripts/train.py","mct_mipnerf",
             "--data="+block_path,
               "--output-dir="+out_dir,
               "--load-dir",pretrained_model_dir,
               "--pipeline.datamanager.train-num-images-to-sample-from=-1",
               "--pipeline.datamanager.train-num-times-to-repeat-images=-1",
                "--pipeline.datamanager.eval-num-images-to-sample-from=-1",
               "--pipeline.datamanager.eval-num-times-to-repeat-images=-1",
               "--pipeline.datamanager.train_num_rays_per_batch=5000",
               "--timestamp=30k",
               "--max-num-iterations=20000"]
        print(cmd)
        subprocess.call(cmd)



# batch_train(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\blocks_2_36',
#            r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks_2_36',
#            50000)

# batch_train(r'J:\xuningli\cross-view\ns\nerfstudio\data\boordaux_metashape\blocks_2_16',
#            r'J:\xuningli\cross-view\ns\nerfstudio\outputs\boordaux_metashape_blocks_2_16',
#            50000)

# batch_train(r'J:\xuningli\cross-view\ns\nerfstudio\data\geomvs_original\test2',
#            r'J:\xuningli\cross-view\ns\nerfstudio\outputs\geomvs_test2',
#            25000)

batch_train(r'J:\xuningli\cross-view\ns\nerfstudio\data\geomvs_original\test2',
           r'J:\xuningli\cross-view\ns\nerfstudio\outputs\geomvs_test2',
           25000)

# batch_train_retrain(r'J:\xuningli\cross-view\ns\nerfstudio\data\dortmund_metashape\blocks_2_16',
#             r'J:\xuningli\cross-view\ns\nerfstudio\outputs\dortmund_metashape_blocks_2_16')

