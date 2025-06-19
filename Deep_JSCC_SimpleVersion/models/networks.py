# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.optim import lr_scheduler
import functools
import numpy as np
# from compressai.layers import GDN
from .AFModule import AFModule

###############################################################################
# Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


# class IGDN(GDN):
#     def __init__(self, *args, **kwargs):
#         kwargs["inverse"] = True
#         super().__init__(*args, **kwargs)
#         self.__class__.__name__ = "IGDN"


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    # elif norm_type == 'gdn':
    #     def norm_layer(num_channels):
    #         return GDN(num_channels)
    # elif norm_type == 'igdn':
    #     def norm_layer(num_channels):
    #         return IGDN(num_channels)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('GDN') != -1 or classname.find('IGDN') != -1:
            # GDN/IGDN 有 gamma 和 beta 参数，需要初始化
            if hasattr(m, 'gamma') and m.gamma is not None:
                init.constant_(m.gamma.data, 1.0)
            if hasattr(m, 'beta') and m.beta is not None:
                init.constant_(m.beta.data, 1e-6)
        elif classname.find('AFModule') != -1:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if init_type == 'normal':
                        init.normal_(param.data, 0.0, init_gain)
                    elif init_type == 'xavier':
                        init.xavier_normal_(param.data, gain=init_gain)
                    elif init_type == 'kaiming':
                        init.kaiming_normal_(param.data, a=0, mode='fan_in')
                    elif init_type == 'orthogonal':
                        init.orthogonal_(param.data, gain=init_gain)
                elif 'bias' in name and param is not None:
                    init.constant_(param.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        # net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        if len(gpu_ids) > 1:  # multi-GPUs
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)
        else:
            net.to(gpu_ids[0])
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# class Normalize(nn.Module):
#   def forward(self, x, power):
#     N = x.shape[0]
#     pwr = torch.mean(x**2, (1,2,3), True)
#
#     return np.sqrt(power)*x/torch.sqrt(pwr)


# Initialization of encoder, generator and discriminator (optional)
def define_E(input_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='gdn', init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Encoder(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect")
    return init_net(net, init_type, init_gain, gpu_ids)


def define_G(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='igdn', init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect")
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_S(dim, dim_out, dim_in=64, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Subnet(dim=dim, dim_out=dim_out, dim_in = dim_in, padding_type='zero', norm_layer=norm_layer, use_dropout=False)
    return init_net(net, init_type, init_gain, gpu_ids)

# Encoder network
class Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=None, padding_type="reflect"):
        """Construct a Resnet-based encoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            max_ngf (int)       -- the maximum number of filters
            C_channel (int)     -- the number of channels of the output
            n_blocks (int)      -- the number of ResNet blocks
            n_downsampling      -- number of downsampling layers
            norm_layer          -- normalization layer
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_downsampling>=0)
        assert(n_blocks>=0)
        super(Encoder, self).__init__()

        if norm_layer is None:
            raise ValueError("You must specify a norm_layer when initializing Encoder.")

        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        if isinstance(norm_layer, functools.partial):
            norm_type = norm_layer.func
        else:
            norm_type = norm_layer

        use_bias = norm_type == nn.InstanceNorm2d

        # activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d((5-1)//2),
                 nn.Conv2d(input_nc, ngf, kernel_size=5, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU(ngf),
                 AFModule(ngf)
                 ]

        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), nn.PReLU(min(ngf * mult * 2, max_ngf)), AFModule(min(ngf * mult * 2, max_ngf))]

        self.model_down = nn.Sequential(*model)
        model= []
        # add ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(min(ngf * mult,max_ngf), padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]

        # self.model_res = nn.Sequential(*model)
        self.model_res = nn.ModuleList(model)

        self.projection = nn.Conv2d(min(ngf * mult,max_ngf), C_channel, kernel_size=3, padding=1, stride=1, bias=use_bias)

    # def forward(self, input, H=None):
    #
    #     z = self.model_down(input)
    #     z = self.model_res(z)
    #     return self.projection(z)

    def forward(self, input, snr_tensor):
        z = input
        for layer in self.model_down:
            if isinstance(layer, AFModule):
                z = layer(z, snr_tensor)
            else:
                z = layer(z)

        for block in self.model_res:
            z = block(z, snr_tensor)

        return self.projection(z)


# Generator network
class Generator(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=None, padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            output_nc (int)     -- the number of channels for the output image
            ngf (int)           -- the number of filters in the first conv layer
            max_ngf (int)       -- the maximum number of filters
            C_channel (int)     -- the number of channels of the input
            n_blocks (int)      -- the number of ResNet blocks
            n_downsampling      -- number of downsampling layers
            norm_layer          -- normalization layer
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks>=0)
        assert(n_downsampling>=0)

        super(Generator, self).__init__()

        if norm_layer is None:
            raise ValueError("You must specify a norm_layer when initializing Decoder.")

        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d
        if isinstance(norm_layer, functools.partial):
            norm_type = norm_layer.func
        else:
            norm_type = norm_layer

        use_bias = norm_type == nn.InstanceNorm2d

        # activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        model = [nn.Conv2d(C_channel,ngf_dim,kernel_size=3, padding=1 ,stride=1, bias=use_bias), AFModule(ngf_dim)]
        self.initial = nn.Sequential(*model)

        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]
        self.model_res = nn.ModuleList(model)

        model = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult,max_ngf), min(ngf * mult //2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult //2, max_ngf)),
                      nn.PReLU(min(ngf * mult //2, max_ngf)),
                      AFModule(min(ngf * mult //2, max_ngf))]

        model += [nn.ReflectionPad2d((5-1)//2), nn.Conv2d(ngf, output_nc, kernel_size=5, padding=0)]

        model +=[nn.Sigmoid()]

        self.model_up = nn.Sequential(*model)

    # def forward(self, input):
    #
    #     return 2*self.model(input)-1
    def forward(self, input, snr_tensor):
        z = input
        for layer in self.initial:
            if isinstance(layer, AFModule):
                z = layer(z, snr_tensor)
            else:
                z = layer(z)

        for block in self.model_res:
            z = block(z, snr_tensor)

        for layer in self.model_up:
            if isinstance(layer, AFModule):
                z = layer(z, snr_tensor)
            else:
                z = layer(z)
        return 2 * z - 1


# Defines the resnet block
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.af = AFModule(dim)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.PReLU(dim)]
        # if use_dropout:
        #     conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x, snr=None):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        if snr is not None:
            out = self.af(out, snr)
        return out


# Discriminator network
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]  # output 1 channel prediction map

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        """Standard forward."""
        res = [input]
        for n in range(self.n_layers+1):
            model = getattr(self, 'model'+str(n))
            res.append(model(res[-1]))

        model = getattr(self, 'model'+str(self.n_layers+1))
        out = model(res[-1])

        return res[1:], out


# Different types of adversarial losses
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'none']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label.to(torch.float32)
        else:
            target_tensor = self.fake_label.to(torch.float32)
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# Subnets
class Subnet(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, dim_out, dim_in, padding_type, norm_layer, use_dropout):

        super(Subnet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv_block = self.build_conv_block(dim, dim_out, dim_in, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, dim_out, dim_in, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim_in, kernel_size=5, padding=2, bias=use_bias), norm_layer(dim_in), nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim_in, dim_in, kernel_size=5, padding=2, bias=use_bias), norm_layer(dim_in), nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim_in, dim_out, kernel_size=5, padding=2, bias=use_bias)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)
