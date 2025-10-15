import os
import sys
from Experiments.Transformation3 import *

# =====================================================================================================
# ============================================= References =============================================
# Pix2Pix source code from: https://github.com/Project-MONAI/GenerativeModels

# =====================================================================================================
# ========================================== USER PARAMETERS ==========================================
''' -------- Tunable --------  '''
''' Parameters in this section should be tuned by the user to obtain the best results'''

data_dir = '/home/jhubadmin/.ssh/Projects/3D_Segmentation/NSCLC_NEW'

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# data loader parameters
random_seed = 42
batch_size = 1

# model hyperparameters
num_train_timesteps = 1000
lr = 1e-6
n_epochs =16000
val_interval = 1

mode = 'train'  # No need to change this anymore

save_each_epoch = False
run_mode = "GPU"
# =====================================================================================================
# =====================================================================================================

args = {
    'image_dir': data_dir,
    'segmentation_dir': data_dir,
    'train_dir': train_dir,
    'val_dir': val_dir,
    'data_dir': data_dir,

    'random_seed': random_seed,
    'batch_size': batch_size,

    'num_train_timesteps': num_train_timesteps,
    'lr': lr,
    'n_epochs': n_epochs,
    'val_interval': val_interval,
    'save_each_epoch': save_each_epoch,

    'run_mode': run_mode,
}

exp = Experiment_transformation(args=args)
exp.prepare()
exp.train()  # This will now train and then test sequentially
