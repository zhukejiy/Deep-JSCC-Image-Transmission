%% preparation
clear     % 清除所有变量
clc       % 清空命令窗口
close all % 关闭所有图形窗口

bpg_images_folder = 'cifar10_q34';
bpg_images_mat    = 'bpg_q34.mat';
quality = 34;

function [bpg_header, bpg_payload] = split_bpg_header_payload(bpg_image_data)
    idx = 1;
    % 跳过 file_magic（4字节）
    idx = idx + 4;
    % idx = 5
    % 下一个字节：pixel_format(3) + alpha1_flag(1) + bit_depth_minus_8(4)
    pixel_byte = bpg_image_data(idx);
    idx = idx + 1;
    % idx = 6
    % 下一个字节：color_space(4) + extension_present_flag(1) + alpha2_flag(1) + limited_range_flag(1) + animation_flag(1)
    flags_byte = bpg_image_data(idx);
    % bitget 取LSB过来的第k位
    extension_present_flag = bitget(flags_byte, 4);  % bit 4 是 extension_present_flag
    idx = idx + 1;
    % idx = 7
    % 读取 picture_width, picture_height, picture_data_length 都是 ue7()
    [~, len] = read_ue7(bpg_image_data, idx); idx = idx + len;
    [~, len] = read_ue7(bpg_image_data, idx); idx = idx + len;
    [~, len] = read_ue7(bpg_image_data, idx); idx = idx + len;

    % 如果有扩展字段，也要跳过 extension_data_length 和 extension_data()
    if extension_present_flag
        % len: extension_data_length 本身用了几个字节
        % ext_len: extension_data() 部分占用多少字节（其内容）
        [ext_len, len] = read_ue7(bpg_image_data, idx);
        idx = idx + len + ext_len;  % extension_data_length 字段长度 + 扩展数据字节
    end
    
    % Step 3: hevc_header_and_data() 开始处
    hevc_block_start = idx;

    
    bpg_header   = bpg_image_data(1 : hevc_block_start - 1);
    bpg_payload    = bpg_image_data(hevc_block_start : end);
    
    fprintf('BPG header length  : %d bytes\n', length(bpg_header));
    fprintf('BPG payload length : %d bytes\n', length(bpg_payload));
end

function [val, len] = read_ue7(data, start_idx)
    val = 0;
    len = 0;
    while true
        byte = data(start_idx + len);
        val = bitshift(val, 7) + bitand(byte, 127);  % 取后7位
        len = len + 1;
        if bitand(byte, 128) == 0  % 最高位是0，结束
            break;
        end
    end
end

num_images = 500;                  % 固定前500张图像

if exist(bpg_images_folder, 'dir')
    % 's' = 'subdirs'
    rmdir(bpg_images_folder, 's');  % 删除整个文件夹及其内容
end
mkdir(bpg_images_folder);  % 重新创建

%% bpg
% 1. 检查 bpgenc 安装
[bpgenc_status, bpgenc_cmdout] = system('bpgenc -h');
if contains(bpgenc_cmdout, 'BPG Image Encoder')
    % BPG Image Encoder version 0.9.8
    disp('bpgenc 安装成功，已加入 PATH！');
    % disp(bpgenc_cmdout);  % 显示 bpgenc 的帮助信息
else
    error('bpgenc 未正确安装或未加入系统 PATH！');
end

% 2. 设置文件路径和读取 CIFAR-10 测试集 bin 文件
raw_images_path = 'test_batch.bin';
raw_images_fid = fopen(raw_images_path, 'rb');
if raw_images_fid == -1
    error('无法打开文件，请检查路径！');
end
% raw_images_data: 3073 × 10000  double
% 每个样本 3073 字节（3073 字节：1 字节标签 + 3072 字节图像数据）
% data = fread(fid, size, precision)  size = [rows, cols]  fread 默认返回 double 类型
raw_images_data = fread(raw_images_fid, [3073, 10000], 'uint8');
fclose(raw_images_fid);

% raw_images: 32 x 32 x 3 x 10000  (HxWxCxN)  uint8
% 3. 提取图像数据并转换为 32x32x3 格式 HxWxC
% 3072 字节图像数据：[R(32×32), G(32×32), B(32×32)] —— 每个通道按行优先（Row-major）展开
% 但 matlab reshape 列优先（Column-major）即先填满第1维（32行） → 然后第2维（32列） → 然后第3维（通道） → 然后第4维（图像编号）
% WxHxCxN
raw_images = uint8(reshape(raw_images_data(2:end, :), [32, 32, 3, 10000]));
% 调整维度为 H×W×C×N
raw_images = permute(raw_images, [2, 1, 3, 4]);


% 初始化比特流存储
bpg_bits = cell(1, num_images);
bpg_header_bits = cell(1, num_images);
bpg_payload_bits = cell(1, num_images);
bpg_payload_bits_MSB = cell(1, num_images);
raw_size_total = 0;       % 原始像素矩阵大小
bpg_size_total = 0;       % BPG 文件大小

for image_idx = 1:num_images
    image_ind = image_idx;
    % raw_image: 32 x 32 x 3  (HxWxC)  uint8
    raw_image = raw_images(:, :, :, image_ind);
    % [height, width, channels] = size(raw_image);
    % fprintf('图像尺寸: %d × %d × %d\n', height, width, channels);
    % fprintf('图像数据类型: %s\n', class(raw_image));

    % PNG (Portable Network Graphics) 是 无损压缩格式，不会丢失图像数据，但是 PNG 使用了 DEFLATE 压缩算法来减小文件大小
    % PNG 的 DEFLATE 算法仅压缩图像数据的存储方式，不改变原始像素矩阵。
    png_image = fullfile(bpg_images_folder, sprintf('cifar10_%05d.png', image_idx));
    imwrite(raw_image, png_image);

    bpg_image = fullfile(bpg_images_folder, sprintf('cifar10_%05d.bpg', image_idx));

    % -q quality：设置 BPG 图像的质量因子
    % 取值范围：0 - 51 
    % quality 值越小 → 压缩率越高，图像失真越大；
    % quality 值越大 → 压缩更轻，图像保真度更好
    % quality = 40;
    system(sprintf('bpgenc -q %d -o %s %s', quality, bpg_image, png_image));

    % 计算原始图像大小
    raw_size = numel(raw_image); % 32x32x3 = 3072 字节
    raw_size_total = raw_size_total + raw_size;

    bpg_image_info = dir(bpg_image);  % 获取文件信息
    if ~isempty(bpg_image_info)      % 检查 BPG 文件是否存在
        bpg_size = bpg_image_info.bytes; % BPG 文件大小（字节）

        fprintf('bpg 原始字节: %d bytes\n', bpg_size);

        bpg_size_total = bpg_size_total + bpg_size;
    else
        warning('BPG 文件未生成: %s', bpg_image);
    end

    % 读取 BPG 文件数据并转换为比特流
    bpg_image_fid = fopen(bpg_image, 'rb');
    if bpg_image_fid == -1
        error('无法打开 BPG 文件：%s', bpg_image);
    end
    % bpg_image_data: bpg_size x 1  列向量  uint8
    % 读取 BPG 文件数据（fread 默认返回 double 类型）
    bpg_image_data = fread(bpg_image_fid, 'uint8');
    fclose(bpg_image_fid);
    % 转换为 uint8 格式
    bpg_image_data = uint8(bpg_image_data);

    % === 分离 header 和 payload ===
    % bpg_header, bpg_payload: size x 1  列向量  uint8
    [bpg_header, bpg_payload] = split_bpg_header_payload(bpg_image_data);

    % dec2bin(bpg_header, 8) 每个字节转8位bit N x 8
    % 转置后变为 8 × N
    % reshape()：拼接成 1D 字符串 
    % 1：将结果转换为行向量  
    % []：自动计算维度，使其合并为一个连续的字符串
    % 存储
    % 1 x bpg_header
    bpg_header_bits{image_idx} = reshape(dec2bin(bpg_header, 8)', 1, []);    % '1010...'
    % MSB first
    bpg_payload_bits_MSB{image_idx} = reshape(dec2bin(bpg_payload, 8)', 1, []);
    % LSB first
    % 1 x bpg_payload
    bpg_payload_bits{image_idx} = reshape(flip(dec2bin(bpg_payload, 8),2)', 1, []);
    bpg_bits{image_idx} = reshape(dec2bin(bpg_image_data, 8)', 1, []);

end

% 输出比特统计
fprintf('原始图像总大小: %.0f bits\n', raw_size_total * 8);
fprintf('BPG 编码后图像总大小: %.0f bits\n', bpg_size_total * 8);

save(bpg_images_mat, 'bpg_header_bits', 'bpg_payload_bits_MSB', 'bpg_payload_bits', '-v7.3');
