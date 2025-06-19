from easydict import EasyDict as edict
from configs.config import cfg
import os
import shutil


__E                                              = cfg

# Model config for different datasets
if __E.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __E.batch_size                               = 1
    __E.size_w                                   = 32
    __E.size_h                                   = 32


__E.verbose                                      = False
__E.serial_batches                               = True
__E.isTrain                                      = False
__E.image_out_path                               = './Images/' + __E.dataset_mode + '/' + __E.name
if not os.path.exists(__E.image_out_path):
    os.makedirs(__E.image_out_path)
else:
    shutil.rmtree(__E.image_out_path)
    os.makedirs(__E.image_out_path)

__E.num_test                                     = 500         # Number of images to test
__E.how_many_channel                             = 5           # Number of channel realizations per image
__E.epoch                                        = 'best'        # Each model to use for testing
__E.load_iter                                    = 0
