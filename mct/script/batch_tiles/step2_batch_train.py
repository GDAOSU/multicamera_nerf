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
               "--pipeline.datamanager.dataparser.has_mask=True",
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

if __name__=='__main__':
    #parameters
    data_dir=r'E:\data\mct_data\data\dortmund_blocks'
    trained_dir=r'E:\data\mct_data\outputs\dortmund_blocks'
    num_iter=30000
    
    batch_train(data_dir,trained_dir,num_iters)
    
