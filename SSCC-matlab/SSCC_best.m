% Get BER以选择最佳LDPC+调制组合 ===
clear; clc; close all;

num_bits = 10000;            % 每次仿真的随机信息比特数
num_repeats = 5;
snr_dB = 5;
snr_linear = 10^(snr_dB / 10);
noise_power = 1 / snr_linear;
noise_std = sqrt(noise_power / 2);
channel_type = 'awgn';

ldpc_configs = {
    'QPSK',   1/2, 1944;
    'QPSK',   2/3, 1944;
    'QPSK',   3/4, 1944;
    'QPSK',   5/6, 1944;
    'QAM16',  1/2, 1944;
    'QAM16',  2/3, 1944;
    'QAM16',  3/4, 1944;
    'QAM16',  5/6, 1944;
    'QAM64',  1/2, 1944;
    'QAM64',  2/3, 1944;
    'QAM64',  3/4, 1944;
    'QAM64',  5/6, 1944;
};

% ber_threshold = 1e-2;
% best_idx = -1;

for idx = 1:size(ldpc_configs, 1)
    ber_all = zeros(num_repeats, 1);
    for repeat = 1:num_repeats
        mod_mode = ldpc_configs{idx, 1};
        rate = ldpc_configs{idx, 2};
        code_len = ldpc_configs{idx, 3};
    
        fprintf('\n=== 测试组合: %s, rate = %.2f, block length = %d ===\n', mod_mode, rate, code_len);
        ldpc_code = LDPCCode(code_len, code_len * rate);
        ldpc_code.load_wifi_ldpc(code_len, rate);
        modulation = Constellation(lower(mod_mode));
        mod_bits = modulation.n_bits;
    
        info_bits = randi([0 1], 1, num_bits);
        pad_len = mod(-length(info_bits), ldpc_code.K);
        info_padded = [info_bits, zeros(1, pad_len)];
        blocks = reshape(info_padded, ldpc_code.K, []);
    
        codewords = zeros(ldpc_code.N, size(blocks,2));
        for j = 1:size(blocks,2)
            codewords(:,j) = ldpc_code.encode_bits(blocks(:,j));
        end
    
        tx_bits = codewords(:);
        tx_syms = modulation.modulate(tx_bits);
    
        num_symbols = length(tx_syms);
    
        % 信道
        switch lower(channel_type)
            case 'awgn'
                h = ones(num_symbols, 1);
            case 'rayleigh'
                num_blocks_rayleigh = ceil(num_symbols / 500);
                h = zeros(num_symbols, 1);
                for b = 1:num_blocks_rayleigh
                    h_val = (randn + 1j * randn) / sqrt(2);
                    h_idx = (b-1)*500 + 1 : min(b*500, num_symbols);
                    h(h_idx) = h_val;
                end
            otherwise
            error('Unsupported channel type: %s', channel_type);
        end
    
        noise = noise_std * (randn(num_symbols, 1) + 1j * randn(num_symbols, 1));

        tx_syms = tx_syms(:);            
        h = h(:);             
        noise = noise(:);     

        rx = tx_syms .* h + noise;
        
        rx_eq = rx ./ h;
        noise_eq = noise_power ./ abs(h).^2;
    
        % LDPC 解码
        codeword_len = ldpc_code.N;
        info_len = ldpc_code.K;
    
        decoded_bits = [];
        num_blocks = length(tx_bits) / codeword_len;
    
        total_error = 0;

        for j = 1:num_blocks
            start_sym = floor((j-1)*codeword_len / mod_bits) + 1;
            end_sym   = floor(j*codeword_len / mod_bits);
        
            rx_block = rx_eq(start_sym:end_sym);
            noise_block = noise_eq(start_sym:end_sym);
        
            [llrs_block, ~] = modulation.compute_llr(rx_block, noise_block);
            [decoded_codeword, ~] = ldpc_code.decode_llr(llrs_block, 50, 1);
            start_idx = (j - 1) * info_len + 1;
            end_idx = min(j * info_len, length(info_bits));  % 不要越界
    
            tx_block = info_bits(start_idx : end_idx);
            payload_block = double(decoded_codeword(1:(end_idx - start_idx + 1)));
            n_err_block = sum(payload_block(:) ~= tx_block(:));
        
            % if n_err_block > 0
            %     fprintf("图像 %d, block %d 出错，错误比特数 = %d\n", image_idx, j, n_err_block);
            % end
    
            total_error = total_error + n_err_block;
            % decoded_bits = [decoded_bits; payload_bits];
        end
    
        ber_all(repeat) = total_error / length(info_bits);
    end
    ber = mean(ber_all);
    fprintf('BER = %.2e\n', ber);

    % if ber < ber_threshold && (best_idx == -1 || ldpc_configs{idx,2} > ldpc_configs{best_idx,2})
    %    best_idx = idx;
    % end
end

% if best_idx > 0
%     best_mod = ldpc_configs{best_idx, 1};
%     best_rate = ldpc_configs{best_idx, 2};
%     best_len = ldpc_configs{best_idx, 3};
%     fprintf('\n✅ 最佳组合: %s, rate = %.2f, block_len = %d\n', best_mod, best_rate, best_len);
% else
%     fprintf('\n❌ 无法在 SNR=%.1fdB 下满足 BER < %.1e\n', snr_dB, ber_threshold);
% end