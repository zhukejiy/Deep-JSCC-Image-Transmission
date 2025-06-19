%% preparation
clear     % 清除所有变量
clc       % 清空命令窗口
close all % 关闭所有图形窗口

clc; clear;

% SNR_test 横轴
SPP = [1/12, 1/8, 1/6];

% ADJSCC（SNR_train ∈ [0, 20]） With attention modules
% psnr_adjscc = [18.7, 23.6, 26.8, 29.9];
psnr_jscc = [29.2, 31.4, 33.0];

% BDJSCC（SNR_train = SNR_test）
% psnr_diag = [20.5, 23.6, 26.6, 31.4];
psnr_sscc = [22.2, 24.2, 26.4];

% BDJSCC（SNR_train = 5dB）
% psnr_train5 = [18.3, 23.6, 24.7, 27.6];

% BDJSCC（SNR_train = 10dB）
% psnr_train10 = [17.8, 22.6, 26.6, 28.4];

% 柔和色定义（RGB 值）
color_red      = [0.8, 0.3, 0.3];   % 柔和红
color_blue     = [0.3, 0.5, 0.8];   % 柔和蓝
% color_green    = [0.4, 0.7, 0.4];   % 柔和绿
% color_orange   = [0.85, 0.6, 0.3];  % 柔和橙

% 画图
figure;
plot(SPP, psnr_jscc, '-', 'Color', color_red, 'Marker', 'o', 'LineWidth', 2); hold on;
plot(SPP, psnr_sscc, '-', 'Color', color_blue, 'Marker', 's', 'LineWidth', 2);
% plot(snr_test, psnr_train5, '--', 'Color', color_green, 'Marker', '^', 'LineWidth', 2);
% plot(snr_test, psnr_train10, '--', 'Color', color_orange, 'Marker', 'x', 'LineWidth', 2);

% 设置图例
% legend('ADJSCC (SNR_{train} ∈ [0,20] dB)', ...
%        'BDJSCC (SNR_{train} = SNR_{test})', ...
%        'BDJSCC (SNR_{train} = 5 dB)', ...
%        'BDJSCC (SNR_{train} = 10 dB)', ...
%        'Location', 'best');
legend('Deep JSCC)', ...
       'SSCC (BPG+LDPC+QAM)', ...
       'Location', 'southeast');

% 设置坐标轴标签和标题
xlabel('SPP');
ylabel('PSNR (dB)');
title('Rayleigh Channel (SNR=20dB)');

% 坐标轴范围和网格
xlim([0.05 0.2]);
ylim([10 36]);
xticks(SPP);  % 设置 X 轴刻度位置
xticklabels({'1/12', '1/8', '1/6'});  % 设置对应刻度标签
grid on;
set(gca, 'FontSize', 12);
