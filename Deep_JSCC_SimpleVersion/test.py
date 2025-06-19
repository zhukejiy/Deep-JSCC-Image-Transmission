import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from models import create_model
from configs.config_test import cfg
import util.util as util
from pytorch_msssim import ssim

if cfg.dataset_mode == 'CIFAR10':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
    dataset = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size,
                                              shuffle=False, num_workers=2)
    dataset_size = len(testset)
    print(f'testing images = {dataset_size}')

else:
    raise Exception('Not implemented yet')


# --- Create model ---
model = create_model(cfg)
model.setup(cfg)
model.eval()

PSNR_list = []
SSIM_list = []

for i, data in enumerate(dataset):
    if i >= cfg.num_test:  # only apply our model to opt.num_test images.
        break

    start_time = time.time()

    input_image = data[0].to(model.device)  # shape [1, C, H, W]
    input_batch = input_image.repeat(cfg.how_many_channel, 1, 1, 1)  # [N, C, H, W]

    model.set_input(input_batch)
    model.forward()
    fake = model.fake

    # # Get the int8 generated images
    # img_gen_numpy = fake.detach().cpu().float().numpy()
    # img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    # img_gen_int8 = img_gen_numpy.astype(np.uint8)
    #
    # origin_numpy = input_batch.detach().cpu().float().numpy()
    # origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    # origin_int8 = origin_numpy.astype(np.uint8)
    #
    # diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8)) ** 2, (1, 2, 3))
    #
    # PSNR = 10 * np.log10((255 ** 2) / diff)
    # PSNR_list.append(np.mean(PSNR))
    #
    # img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
    # origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()
    #
    # # ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(cfg.how_many_channel, 1, 1, 1), data_range=255,
    # #                 size_average=False)  # return (N,)
    # ssim_val = ssim(img_gen_tensor, origin_tensor, data_range=255, size_average=False)  # return (N,)
    # SSIM_list.append(torch.mean(ssim_val).item())

    # Compute PSNR
    img_gen = (fake.detach() + 1) / 2.0 * 255.0
    origin = (input_batch.detach() + 1) / 2.0 * 255.0

    mse = ((img_gen - origin) ** 2).view(img_gen.size(0), -1).mean(1)
    psnr = 10 * torch.log10((255 ** 2) / mse)
    PSNR_list.append(psnr.mean().item())

    # Compute SSIM
    ssim_val = ssim(((fake.detach() + 1) / 2.0), ((input_batch.detach() + 1) / 2.0), data_range=1.0, size_average=False)
    SSIM_list.append(ssim_val.mean().item())

    # Save the first sampled image
    save_path_gen = f'{cfg.image_out_path}/{i}_PSNR_{psnr[0]:.3f}_SSIM_{ssim_val[0]:.3f}.png'
    util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path_gen, aspect_ratio=1)

    save_path_ori = f'{cfg.image_out_path}/{i}_original.png'
    util.save_image(util.tensor2im(input_image), save_path_ori, aspect_ratio=1)

    # # Save the first sampled image
    # save_path = f'{cfg.image_out_path}/{i}_PSNR_{PSNR[0]:.3f}_SSIM_{ssim_val[0]:.3f}.png'
    # util.save_image(util.tensor2im(fake[0].unsqueeze(0)), save_path, aspect_ratio=1)
    #
    # save_path = f'{cfg.image_out_path}/{i}.png'
    # util.save_image(util.tensor2im(input[0].unsqueeze(0)), save_path, aspect_ratio=1)

    if i % 10 == 0:
        print(f"Tested {i} images")

print(f'Mean PSNR over test set: {np.mean(PSNR_list):.4f} dB')
print(f'Mean SSIM over test set: {np.mean(SSIM_list):.4f}')
