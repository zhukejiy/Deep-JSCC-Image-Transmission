%% preparation
clear     % 清除所有变量
clc       % 清空命令窗口
close all % 关闭所有图形窗口

clc; clear;

% SNR_test 横轴
SPP = [1/12, 1/8, 1/6];

% ADJSCC（SNR_train ∈ [0, 20]） With attention modules
% ssim_adjscc = [0.57, 0.78, 0.87, 0.94];
ssim_jscc = [0.93, 0.95, 0.96];

% BDJSCC（SNR_train = SNR_test）
% ssim_diag = [0.62, 0.77, 0.87, 0.95];
ssim_sscc = [0.76, 0.83, 0.89];

% % BDJSCC（SNR_train = 5dB）
% ssim_train5 = [0.54, 0.77, 0.79, 0.84];
% 
% % BDJSCC（SNR_train = 10dB）
% ssim_train10 = [0.54, 0.74, 0.87, 0.88];

% 柔和、非重复颜色定义（相对于 PSNR 图）
color_jscc = [0.1, 0.45, 0.8];     % 深蓝色
% color_diag   = [0.2, 0.6, 0.2];      % 暗绿色
color_sscc    = [0.6, 0.3, 0.1];      % 深棕色
% color_10dB   = [0.5, 0.2, 0.5];      % 暗紫色

% 画图
figure;
plot(SPP, ssim_jscc, '-', 'Color', color_jscc, 'Marker', 'o', 'LineWidth', 2); hold on;
plot(SPP, ssim_sscc, '-', 'Color', color_sscc, 'Marker', 's', 'LineWidth', 2);
% plot(snr_test, ssim_train5, '--', 'Color', color_5dB, 'Marker', '^', 'LineWidth', 2);
% plot(snr_test, ssim_train10, '--', 'Color', color_10dB, 'Marker', 'x', 'LineWidth', 2);

% % 设置图例
% legend('ADJSCC (SNR_{train} ∈ [0,20] dB)', ...
%        'BDJSCC (SNR_{train} = SNR_{test})', ...
%        'BDJSCC (SNR_{train} = 5 dB)', ...
%        'BDJSCC (SNR_{train} = 10 dB)', ...
%        'Location', 'best');
% 设置图例
legend('Deep JSCC', ...
       'SSCC (BPG+LDPC+QAM)', ...
       'Location', 'southeast');


% 设置坐标轴标签和标题
xlabel('SPP');
ylabel('SSIM');
title('Rayleigh Channel (SNR=20dB)');

% 坐标轴范围和网格
xlim([0.05 0.2]);
ylim([0.7 1]);
xticks(SPP);  % 设置 X 轴刻度位置
xticklabels({'1/12', '1/8', '1/6'});  % 设置对应刻度标签
grid on;
set(gca, 'FontSize', 12);