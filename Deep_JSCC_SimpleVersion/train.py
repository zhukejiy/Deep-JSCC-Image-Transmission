import time
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import Subset
from configs.config_train import cfg
from models import create_model
from util.visualizer import Visualizer

# Create dataloaders
if cfg.dataset_mode == 'CIFAR10':
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=5, pad_if_needed=True, fill=0, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # # torchvision.datasets.CIFAR10 类型的对象，它是一个可迭代的数据集对象，每次调用都会返回一张图片及其对应的标签
    # # len(full_dataset) 返回的是 full_dataset 里面样本（图像）的总数量
    # full_dataset = torchvision.datasets.CIFAR10(root=cfg.dataroot, train=True, download=True, transform=None)
    #
    # train_size = int(0.9 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    #
    # # 固定随机种子，确保每次划分一样
    # generator = torch.Generator().manual_seed(42)
    # train_subset, val_subset = torch.utils.data.random_split(
    #     full_dataset,
    #     [train_size, val_size],
    #     generator=generator
    # )
    #
    # base_dataset_train = torchvision.datasets.CIFAR10(
    #     root=cfg.dataroot, train=True, transform=transform_train
    # )
    # base_dataset_val = torchvision.datasets.CIFAR10(
    #     root=cfg.dataroot, train=True, transform=transform_val
    # )
    #
    # train_dataset = Subset(base_dataset_train, train_subset.indices)
    # val_dataset = Subset(base_dataset_val, val_subset.indices)
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.dataroot, train=True, download=False, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=cfg.dataroot, train=False, download=False, transform=transform_val
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    # DataLoader 所绑定的 Dataset
    print(f'training images = {len(train_loader.dataset)}, validation images = {len(val_loader.dataset)}')
    print(f'training batches = {len(train_loader)}, validation batches = {len(val_loader)}')
else:
    raise Exception('Not implemented yet')

model = create_model(cfg)
model.setup(cfg)
visualizer = Visualizer(cfg)

total_iters = 0
best_psnr = -1

for epoch in range(cfg.epoch_count, cfg.n_epochs + cfg.n_epochs_decay + 1):
    model.train()

    epoch_start_time = time.time()
    iter_data_time = time.time()

    epoch_iter = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()
        if total_iters % cfg.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += 1
        epoch_iter += 1

        input_batch = data[0]
        model.set_input(input_batch)
        model.optimize_parameters()

        # === 在训练主代码中，每个 batch 后记录 ===
        total_psnr += model.metric_PSNR
        total_ssim += model.metric_SSIM
        num_batches += 1

        if total_iters % cfg.print_freq == 0:
            losses = model.get_current_losses()
            metrics = model.get_current_metrics()
            t_comp = (time.time() - iter_start_time)
            visualizer.print_current_training(epoch, epoch_iter, losses, metrics, t_comp, t_data)

        iter_data_time = time.time()

    # === 验证阶段 ===
    model.eval()
    psnr_list = []
    with torch.no_grad():
        for data in val_loader:
            input_batch = data[0].to(model.device)
            input_repeat = input_batch.repeat_interleave(cfg.repeat_times, dim=0)  # [N*repeat_times, C, H, W]
            model.set_input(input_repeat)
            model.forward()
            fake = model.fake

            img_gen = (fake.detach() + 1) / 2.0 * 255.0
            origin = (input_repeat.detach() + 1) / 2.0 * 255.0

            mse = ((img_gen - origin) ** 2).view(img_gen.size(0), -1).mean(1)
            psnr = 10 * torch.log10((255 ** 2) / mse)

            psnr_list.append(psnr.mean().item())

    val_psnr = sum(psnr_list) / len(psnr_list)
    print(f"Epoch {epoch}: Validation PSNR = {val_psnr:.4f} dB")

    if val_psnr > best_psnr:
        print(f"Saving best model at the end of epoch {epoch}, iters {total_iters} with PSNR {val_psnr:.4f}")
        best_psnr = val_psnr
        model.save_networks('best')

    if epoch % cfg.save_epoch_freq == 0:
        print(f"Saving model at the end of epoch {epoch}, iters {total_iters}")
        model.save_networks('latest')
        model.save_networks(epoch)

    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    print(f"Epoch {epoch}: Avg PSNR = {avg_psnr:.4f} dB, Avg SSIM = {avg_ssim:.4f}")

    print('End of epoch %d / %d 	 Time Taken: %d sec' % (epoch, cfg.n_epochs + cfg.n_epochs_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
