from easydict import EasyDict as edict
from configs.config import cfg

__T                                              = cfg

############################# Training configs ####################################

if __T.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __T.batch_size                               = 128        # Batch size
    __T.serial_batches                           = False      # The batches are continuous or randomly shuffled
    __T.n_epochs                                 = 150        # Number of epochs without lr decay
    __T.n_epochs_decay                           = 150        # Number of epochs with lr decay
    __T.lr_policy                                = 'linear'   # decay policy.  
    __T.beta1                                    = 0.5        # parameter for ADAM
    __T.lr                                       = 5e-4       # Initial learning rate
    __T.dataroot                                 = './data'
    __T.size_w                                   = 32
    __T.size_h                                   = 32

############################# Training configs ####################################

__T.repeat_times                                 = 5
__T.print_freq                                   = 100              # frequency of showing training results on console   
__T.save_latest_freq                             = 5000             #frequency of saving the latest results
__T.save_epoch_freq                              = 10               #frequency of saving checkpoints at the end of epochs
__T.save_by_iter                                 = False            #whether saves model by iteration
__T.continue_train                               = False            #continue training: load the latest model
__T.epoch_count                                  = 1                #the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
__T.verbose                                      = False
__T.isTrain                                      = True
        
       



