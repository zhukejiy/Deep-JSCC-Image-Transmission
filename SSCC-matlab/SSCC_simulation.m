%% preparation
clear     % 清除所有变量
clc       % 清空命令窗口
close all % 关闭所有图形窗口

bpg_images_folder = 'recv_q46';
bpg_images_mat    = 'bpg_q46.mat';
num_images = 500;                  % 固定前500张图像
num_repeats = 5;

block_length = 1944;        % WiFi LDPC block length: 648 / 1296 / 1944
rate = 1/2;                 % WiFi LDPC rate: 1/2, 2/3, 3/4, 5/6
mod_mode = 'qam16';

if exist(bpg_images_folder, 'dir')
    % 's' = 'subdirs'
    rmdir(bpg_images_folder, 's');  % 删除整个文件夹及其内容
end
mkdir(bpg_images_folder);  % 重新创建

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

fprintf('=== Generating WiFi LDPC H Matrix, block length = %d, rate = %.2f ===\n', block_length, rate);

ldpc_code = LDPCCode(block_length, block_length * rate);
ldpc_code.load_wifi_ldpc(block_length, rate);
modulation = Constellation(lower(mod_mode));

load(bpg_images_mat);

ldpc_bits = cell(1, num_images);
modulated_symbols = cell(1, num_images);

for image_idx = 1:num_images
    % 1 x bpg_payload  0/1 数组
    bpg_bits_image = char(bpg_payload_bits{image_idx});
    bpg_bits_image = bpg_bits_image - '0';

    % === Padding 补零到 info_length 的整数倍 ===
    % 1 x num_blocks*info_length  0/1 数组
    info_length = ldpc_code.K;
    pad_len = mod(-length(bpg_bits_image), info_length);
    if pad_len > 0
        bpg_bits_image = [bpg_bits_image, zeros(1, pad_len)];
    end

    % info_length x num_blocks
    bpg_blocks_image = reshape(bpg_bits_image, info_length, []);
    num_blocks = size(bpg_blocks_image, 2);
    % block_length x num_blocks
    ldpc_codewords_image = zeros(ldpc_code.N, num_blocks);

    % === LDPC 编码 ===
    for j = 1:num_blocks
        ldpc_codewords_image(:, j) = ldpc_code.encode_bits(bpg_blocks_image(:, j));
    end

    % ldpc_codewords_image(:) 展开成列向量
    % === BPSK or QPSK 调制 ===
    % modulated_symbols_image num_blocks*block_length x 1 列向量
    modulated_symbols_image = modulation.modulate(ldpc_codewords_image(:));

    % 存储
    ldpc_bits{image_idx} = ldpc_codewords_image;
    modulated_symbols{image_idx} = modulated_symbols_image;
end

rx_symbols = cell(num_images, num_repeats);
channel_gains = cell(num_images, num_repeats);

snr_dB = 20;                           % 信噪比 (可调)
snr_linear = 10^(snr_dB / 10);
channel_type = 'rayleigh';   % 或 'rayleigh'
tx_power = 1;            % 假设发射端单位功率归一化

for repeat_idx = 1:num_repeats
    for image_idx = 1:num_images
        tx = modulated_symbols{image_idx};           % 原始调制符号 [1 × N]，复数
        num_symbols = length(tx);

        % ==== 信道增益 ====
        switch channel_type
            case 'awgn'
                h = ones(num_symbols, 1);             % AWGN 信道: 不引入衰落
            case 'rayleigh'
                % 慢衰落：（块衰落）
                % num_blocks = ceil(num_symbols / block_len);
                % h = zeros(num_symbols, 1);
                % for b = 1:num_blocks
                %     h_val = (randn + 1j * randn) / sqrt(2);
                %     idx = (b-1)*block_len + 1 : min(b*block_len, num_symbols);
                %     h(idx) = h_val;
                % end
                h_single = (randn + 1j * randn) / sqrt(2);  % CN(0,1)
                h = h_single * ones(num_symbols, 1);
            otherwise
                error('Unsupported channel type: %s', channel_type);
        end

        % ==== 加性高斯噪声 ====
        noise_power = tx_power / snr_linear;
        noise_std = sqrt(noise_power / 2);
        noise = noise_std * (randn(num_symbols, 1) + 1j * randn(num_symbols, 1));

        tx = tx(:);            % 强制列向量 [3888×1]
        h = h(:);              % 强制列向量 [3888×1]
        noise = noise(:);      % 强制列向量 [3888×1]

        % % ==== 接收端符号 ====
        rx = tx .* h + noise;

        % disp(size(tx));
        % disp(size(h));
        % disp(size(noise));
        % disp(class(tx));

        % ==== 存储 ====
        rx_symbols{image_idx, repeat_idx} = rx;
        channel_gains{image_idx, repeat_idx} = h;
    end
end

ldpc_decoded_bits = cell(num_images, num_repeats);
bers = zeros(num_images, num_repeats);
psnr_all = zeros(num_images, num_repeats);
ssim_all = zeros(num_images, num_repeats);

for image_idx = 1:num_images
    for repeat_idx = 1:num_repeats
        % (num_blocks*block_length*sps + span*sps) x 1 列向量
        rx_sampled = rx_symbols{image_idx, repeat_idx};
        h_effective = channel_gains{image_idx, repeat_idx};

        rx_sampled = rx_sampled(:);       % 强制列向量
        h_effective = h_effective(:);     % 强制列向量

        % % (num_blocks*block_length) x 1 列向量
        % rx_sampled = rx_sampled(1:length(modulated_symbols{image_idx}));
        % h_effective = h_effective(1:length(modulated_symbols{image_idx}));

        % % 信道均衡
        % rx_eq = conj(h_effective) .* rx_sampled;
        % % === LLR计算，参考WiFi代码逻辑，AWGN近似 ===
        % llrs = 2 * real(rx_eq) ./ (noise_power * abs(h_effective).^2);
        % % [llrs, ~] = modulation.compute_llr(rx_eq, noise_power);

        % 均衡
        rx_eq = rx_sampled ./ h_effective;
        noise_eq = noise_power ./ abs(h_effective).^2;

        % % LLR计算
        % % (num_blocks*block_length) x 1 列向量
        % [llrs, ~] = modulation.compute_llr(rx_eq, noise_eq);
        % 
        % % % 假设 llrs_all 是全部LLR值的向量
        % % llrs_valid = llrs(~isinf(llrs));
        % % threshold = prctile(abs(llrs_valid), 99);  % 99分位数作为 clip 阈值
        % % 
        % % LLR_CLIP = min(threshold, 1000);  % 可设置最大不超过某个值
        % % llrs = max(min(llrs, LLR_CLIP), -LLR_CLIP);
        % 
        % fprintf('图像 %d 的 LLR 范围：min = %.4f, max = %.4f\n', image_idx, min(llrs), max(llrs));

        % LDPC 解码
        codeword_len = ldpc_code.N;
        info_len = ldpc_code.K;

        decoded_bits = [];
        num_symbols = length(rx_eq);  % QPSK: 每个 symbol 承载 2 bit
        n_bits = 4;
        num_blocks = num_symbols * n_bits / codeword_len;
        
        % 原始发送 bits（0/1）形式
        tx_bits = bpg_payload_bits{image_idx} - '0';

        total_error = 0;
        for j = 1:num_blocks
            start_sym = floor((j-1)*codeword_len / n_bits) + 1;
            end_sym   = floor(j*codeword_len / n_bits);
        
            rx_block = rx_eq(start_sym:end_sym);
            noise_block = noise_eq(start_sym:end_sym);
        
            [llrs_block, ~] = modulation.compute_llr(rx_block, noise_block);
            [decoded_codeword, ~] = ldpc_code.decode_llr(llrs_block, 50, 1);
            start_idx = (j - 1) * info_len + 1;
            end_idx = min(j * info_len, length(tx_bits));  % 不要越界

            tx_block = tx_bits(start_idx : end_idx);
            info_bits = double(decoded_codeword(1:(end_idx - start_idx + 1)));
            n_err_block = sum(info_bits(:) ~= tx_block(:));
        
            % if n_err_block > 0
            %     fprintf("图像 %d, block %d 出错，错误比特数 = %d\n", image_idx, j, n_err_block);
            % end

            total_error = total_error + n_err_block;
            decoded_bits = [decoded_bits; info_bits];
        end


        % num_blocks = length(llrs) / codeword_len;
        % % block_length x num_blocks
        % llrs_blocks = reshape(llrs, codeword_len, num_blocks);

        % bpg_payload x 1
        % decoded_bits = [];
        % for j = 1:num_blocks
        %     [info_bits, ~] = ldpc_code.decode_llr(llrs_blocks(:, j), 50, 1);
        %     decoded_bits = [decoded_bits; info_bits];
        % end

        % ori_payload_len = length(bpg_payload_bits{image_idx});
        % decoded_bits = decoded_bits(1:ori_payload_len);

        % 存储  字符串-行向量  % 1 x bpg_payload
        ldpc_decoded_bits{image_idx, repeat_idx} = char(decoded_bits' + '0');

        % === 计算 BER ===
        % 发送端payload_bits
        % tx_bits = bpg_payload_bits{image_idx} - '0';
        % n_err = sum(decoded_bits(:) ~= tx_bits(:));
        % bers(image_idx, repeat_idx) = n_err / length(tx_bits);
        bers(image_idx, repeat_idx) = total_error / length(tx_bits);

        fprintf('图像 %d: BER = %.5f (错误比特数=%d / 总比特=%d)\n', ...
            image_idx, bers(image_idx, repeat_idx), total_error, length(tx_bits));

        ori_header_bits = bpg_header_bits{image_idx};       % header
        decoded_payload_bits = ldpc_decoded_bits{image_idx, repeat_idx};

        % === 按照每8位，flip bit顺序（从LSB-first转MSB-first）===
        decoded_payload_bits = reshape(decoded_payload_bits, 8, []).'; % reshape成8列
        decoded_payload_bits = flip(decoded_payload_bits, 2);          % 每行内部翻转
        decoded_payload_bits = reshape(decoded_payload_bits.', 1, []); % 再reshape回1行

        % === 拼接 header + payload ===
        % 字符串-行向量
        % 1 x bpg_bits
        bpg_bits_total = [ori_header_bits, decoded_payload_bits];
        % 每8个比特转成uint8
        % .'	单纯转置（不共轭）
        bpg_bytes_total = uint8(bin2dec(reshape(bpg_bits_total, 8, []).'));

        % === 保存为 .bpg 文件 ===
        bpg_recover_file = fullfile(bpg_images_folder, sprintf('recv_%05d_r%d.bpg', image_idx, repeat_idx));
        recv_fid = fopen(bpg_recover_file, 'wb');
        fwrite(recv_fid, bpg_bytes_total, 'uint8');
        fclose(recv_fid);

        % === 使用bpgdec解码为png ===
        png_recover_file = fullfile(bpg_images_folder, sprintf('recv_%05d_r%d.png', image_idx, repeat_idx));
        system(sprintf('bpgdec -o %s %s', png_recover_file, bpg_recover_file));

        % === 读取原图与恢复图像 ===
        ori_image = raw_images(:, :, :, image_idx);

        try
            recover_image = imread(png_recover_file);
            ori_image = double(ori_image);
            recover_image = double(recover_image);

            % === 计算图像质量 ===
            psnr_all(image_idx, repeat_idx) = psnr(uint8(recover_image), uint8(ori_image));
            ssim_all(image_idx, repeat_idx) = ssim(uint8(recover_image), uint8(ori_image));

            % fprintf('图像 %d: PSNR = %.4f dB, SSIM = %.4f\n', ...
            %     image_idx, psnr_all(image_idx), ssim_all(image_idx));
        catch
            % Case2: BPG 解码失败，则用 Case1 的均值代替
            % fprintf('图像 %d: Could not decode image\n', image_idx);
        end
    end

    fprintf('图像 %d: PSNR = %.4f dB, SSIM = %.4f\n', ...
            image_idx, mean(psnr_all(image_idx, :)), mean(ssim_all(image_idx, :)));

end

% Step1: 判断 Case1 和 Case2
case1_idx = (bers > 0) & (psnr_all > 0);   % LDPC有误码 但 BPG成功解码
case2_idx = (bers > 0) & (psnr_all == 0);  % LDPC有误码 且 BPG解码失败

fprintf('Case1 图像个数 = %d\n', sum(case1_idx));
fprintf('Case2 图像个数 = %d\n', sum(case2_idx));

% Step2: 计算 Case1 的均值
if any(case1_idx)
    case1_psnr_mean = mean(psnr_all(case1_idx));
    case1_ssim_mean = mean(ssim_all(case1_idx));
else
    warning('没有Case1图像，Case2无法替代！');
    case1_psnr_mean = 10;
    case1_ssim_mean = 0.5;
end

% Step3: 将 Case2 替代为 Case1均值
psnr_all(case2_idx) = case1_psnr_mean;
ssim_all(case2_idx) = case1_ssim_mean;

% === Step4: 平均 ===
psnr_mean_final = mean(mean(psnr_all, 2));  % 每张图像5次平均 → 所有图像平均
ssim_mean_final = mean(mean(ssim_all, 2));

fprintf('平均PSNR = %.4f dB, 平均SSIM = %.4f\n', psnr_mean_final, ssim_mean_final);