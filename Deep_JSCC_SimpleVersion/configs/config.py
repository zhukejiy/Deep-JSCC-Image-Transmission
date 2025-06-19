import numpy as np
from easydict import EasyDict as edict


__C                                              = edict()
cfg                                              = __C

############################# Basic settings ####################################

__C.name                                         = 'JSCC_MIC'  # Name of the experiment
__C.gpu_ids                                      = [0]  # GPUs to use
__C.dataset_mode                                 = 'CIFAR10'  # ['CIFAR10', 'CIFAR100', 'CelebA', 'OpenImage']
__C.checkpoints_dir                              = './Checkpoints/' + __C.dataset_mode  # Path to store the model
__C.model                                        = 'JSCCMIC'
__C.C_channel                                    = 16  # Number of channels for output latents (controls the communication rate)
                                                       # Calculation of the rate (channel usage per pixel):
                                                       #           C_channel / (3 x 2^(2 x n_downsample + 1))
                                                       # 8  1/12
                                                       # 16  1/6
__C.feedforward                                  = 'MIC-raw'  # Different schemes: MIC-raw
                                                       # MIC-CE-EQ: MMSE channel estimation and equalization without any subnets
                                                       # MIC-CE-sub-EQ: MMSE channel estimation and equalization with CE subnet
                                                       # MIC-CE-sub-EQ-sub: MMSE channel estimation and equalization with CE & EQ subnet
                                                       # MIC-feedback: pre-coding scheme with CSI feedback
__C.N_pilot                                      = 1  # Number of pilot symbols

__C.lam_h                                        = 50  # Weight for the channel reconstruction loss
__C.gan_mode                                     = 'none'  # ['wgangp', 'lsgan', 'vanilla', 'none']
__C.lam_G                                        = 0.02  # Weight for the adversarial loss
__C.lam_L2                                       = 100  # Weight for image reconstruction loss
# __C.lam_SSIM                                     = 50

############################# Model and training configs ####################################

if __C.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __C.n_layers_D                               = 3  # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 2  # Downsample times
    __C.n_blocks                                 = 2  # Numebr of residual blocks


elif __C.dataset_mode == 'CelebA':
    __C.n_layers_D                               = 3  # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3  # Downsample times
    __C.n_blocks                                 = 2  # Numebr of residual blocks


elif __C.dataset_mode == 'OpenImage':
    __C.n_layers_D                               = 4  # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3  # Downsample times
    __C.n_blocks                                 = 4  # Numebr of residual blocks

__C.norm_D                                       = 'instance' if __C.gan_mode == 'wgangp' else 'batch'  # Type of normalization in Discriminator
__C.norm_EG                                      = 'batch'    # Type of normalization in others
__C.norm_E                                       = 'batch'      # Encoder 使用 GDN 'gdn'
__C.norm_G                                       = 'batch'     # Generator 使用 IGDN 'igdn'

############################# Display and saving configs ####################################

# __C.name = f'C{__C.C_channel}_{__C.feedforward}_pilot_{__C.N_pilot}_hloss_{__C.lam_h}'
__C.name += f'_C{__C.C_channel}_{__C.feedforward}'
__C.name += f'_{__C.gan_mode}_{__C.lam_G}' if __C.gan_mode != 'none' else ''


############################# Electrical Characteristics ####################################
__C.N_S      = 15                                                    # turns of base station coil (source/transmitter)
__C.R_S      = 0.6                                                   # radius of base station coil (source) m
__C.N_D      = 30                                                    # turns of vehicle coil (destination/receiver)
__C.R_D      = 0.4                                                   # radius of vehicle coil (destination) m
__C.rho_w    = 0.0166                                                # unit length resistance Ω/m
__C.f0       = 10*1e3                                                # signal carrier frequency  Hz
__C.mu       = 4 * np.pi * 1e-7                                      # 介质磁导率 permeability H/m  干沙土环境约等于真空磁导率/自由空间磁导率 4π×10^-7 H/m
__C.R_cs     = (__C.N_S * 2 * np.pi * __C.R_S * __C.rho_w)           # resistors of the base station coil (source)
__C.R_cd     = (__C.N_D * 2 * np.pi * __C.R_D * __C.rho_w)           # resistors of the vehicle coil (destination)
__C.R_L      = __C.R_cd                                              # load resistors
__C.L_cs     = ((__C.N_S ** 2) * __C.R_S * __C.mu * np.pi / 2)       # inductances of the base station coil (source)
__C.L_cd     = ((__C.N_D ** 2) * __C.R_D * __C.mu * np.pi / 2)       # inductances of the vehicle coil (destination)
__C.C_cs     = 1 / (((2 * np.pi * __C.f0) ** 2) * __C.L_cs)          # capacitances of the base station (source)
__C.C_cd     = 1 / (((2 * np.pi * __C.f0) ** 2) * __C.L_cd)          # capacitances of the vehicle (destination)

__C.pwr_tx   = 5                                                     # 发送功率 5W
__C.symbol_rate = 250                                                # Symbol rate symbols/s
__C.bandwidth_signal = __C.symbol_rate                               # Signal bandwidth ~= Symbol rate Hz

############################# Environmental Characteristics ####################################
__C.epsilon_un              = 7 * 8.854 * 1e-12                                      # 介质介电常数 permittivity F/m 干沙土环境下等于真空介电常数epsilon_0 * 相对介电常数epsilon_r
__C.theta_un                = 0.01                                                   # 介质电导率 conductivity S/m 干沙土环境
__C.delta                   = np.sqrt(1 / (np.pi * __C.f0 * __C.mu * __C.theta_un))  # 介质趋肤深度 skin depth --> 趋肤效应 --> 涡流损耗
__C.noise_pwr_density_dBm   = -103                                                   # 噪声功率谱密度 忽略热噪声，只考虑环境噪声 单位dBm/2kHz
__C.noise_pwr_density       = 10 ** (__C.noise_pwr_density_dBm / 10) * 1e-3          # W/2kHz

__C.region_id               = 7
############################# Vehicle Movement Characteristics ####################################
__C.x_s = 0                                                          # Base station Position 原点（0, 0, 0）
__C.y_s = 0                                                          # Base station Position 原点（0, 0, 0）
__C.z_s = 0                                                          # Base station Position 原点（0, 0, 0）
__C.area_range = 50                                                  # XY 平面 100m * 100m 范围 [-50, 50] m
__C.phi_road = np.pi / 4                                             # 路面坡度 road gradient
__C.velocity = 2                                                     # 匀速行驶 设车辆沿着 west-east-tilting 坡度路面向下
__C.sigma = 0.4                                                      # average AVI - sigma^2 固定

__C.sample_freq_small = 500                                          # vibration frequency, small-scale fading (MI polarization gain) period
__C.sample_freq_large = 10                                           # update position related to large-scale fading (MI aligned gain)
