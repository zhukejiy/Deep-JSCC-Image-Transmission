import numpy as np
import torch
import torch.nn as nn


def Normalize(x, pwr=1):
    """
    Normalization function
    """
    # torch.mean(..., (-2,-1), True)  (-2,-1): 沿着最后两个维度计算均值(符号维度 I/Q 复数维度）  True: 使返回的 power 保持 原始 x 的形状
    power = torch.mean(x**2, (-2,-1), True)
    alpha = np.sqrt(pwr/2)/torch.sqrt(power)
    return alpha*x


class SimpleChannel(nn.Module):
    """
    Simple channel model for coursework:
    - AWGN
    - Slow Rayleigh fading
    """
    def __init__(self, opt, device, snr_dB, fading):  # fading: 'awgn' or 'rayleigh'
        super(SimpleChannel, self).__init__()
        self.opt = opt
        self.device = device
        self.snr_dB = snr_dB
        self.fading = fading

    def forward(self, x):
        # x: [N, 2, S], real and imag parts
        N, _, S = x.shape

        # 神经网络输出标准化，不依赖能量
        # Normalize the input power per image
        x = Normalize(x, pwr=1)

        x_complex = x[:, 0, :] + 1j * x[:, 1, :]  # [N, S]
        signal_power = torch.mean(torch.abs(x_complex) ** 2)

        # 计算噪声功率
        snr_linear = 10 ** (self.snr_dB / 10)
        noise_power = signal_power / snr_linear

        # 生成噪声
        noise = torch.sqrt(noise_power / 2) * (
            torch.randn_like(x_complex, device=self.device) + 1j * torch.randn_like(x_complex, device=self.device)
        )

        if self.fading == 'awgn':
            y = x_complex + noise
        elif self.fading == 'rayleigh':
            # 每个样本一个恒定的复高斯增益
            h = (torch.randn(N, 1, device=self.device) + 1j * torch.randn(N, 1, device=self.device)) / np.sqrt(2)
            y = h * x_complex + noise
        else:
            raise NotImplementedError(f"Unknown fading type: {self.fading}")

        return torch.stack([y.real, y.imag], dim=1)  # [N, 2, S]

