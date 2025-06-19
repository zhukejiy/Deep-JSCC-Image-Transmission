classdef Constellation < handle
    % Deals with constellation transmit and receive
    
    properties
        
        constellation_points;
        
        n_bits;
        n_sym;
        bit_sym_map;

    end
    
    properties (Constant)
        
        % '0'	[0]	1
        % '1'	[1]	-1
        bpsk = [1 -1];
        
        ask4 = [-3, -1, 3, 1]/sqrt(5); % Gray mapping
        
        ask8 = [-7, -5, -1, -3, 7, 5, 1, 3]/sqrt(21); % Reflected Gray mapping

        % '00'	[0 0]  1+j
        % '01'	[1 0]  −1+j​
        % '10'	[0 1]  −1−j
        % '11'	[1 1]  1−j

        qpsk = [1+1j, -1+1j, -1-1j, 1-1j]/sqrt(2);

        qam16 = (repmat([-3 -1 3 1], 1, 4) + 1j * repelem([-3 -1 3 1], 4)) / sqrt(10);

        qam64 = (repmat([-7 -5 -1 -3 7 5 1 3], 1, 8) + 1j * repelem([-7 -5 -1 -3 7 5 1 3], 8)) / sqrt(42);               
    end
    
    methods
        
        function obj = Constellation ( constellion_name )
            if strcmp (constellion_name, 'bpsk')
                obj.constellation_points = obj.bpsk;
                obj.n_bits = 1;
            elseif strcmp(constellion_name, 'qpsk')
                obj.constellation_points = obj.qpsk;
                obj.n_bits = 2;
            elseif strcmp (constellion_name, 'ask4')
                obj.constellation_points = obj.ask4;
                obj.n_bits = 2;
            elseif strcmp (constellion_name, 'ask8')
                obj.constellation_points = obj.ask8;
                obj.n_bits = 3;
            elseif strcmp(constellion_name, 'qam16')
                obj.constellation_points = obj.qam16;
                obj.n_bits = 4;
            elseif strcmp(constellion_name, 'qam64')
                obj.constellation_points = obj.qam64;
                obj.n_bits = 6;
            else
                disp('Unsupported constellation');
            end
            
            % obj.n_bits = 2;
            % obj.n_sym = 2^obj.n_bits = 4;
            obj.n_sym = 2^obj.n_bits;

            obj.bit_sym_map = zeros(obj.n_sym, obj.n_bits);
            for sym_index = 0 : obj.n_sym - 1
                % sym_index = 0
                % dec2bin 十进制转二进制字符串
                % MSB -> LSB
                % a = dec2bin(0, 2) → '00'
                % a = dec2bin(1, 2) → '01'
                a = dec2bin(sym_index, obj.n_bits);
                for bit_index = 1 :obj.n_bits
                    if a(obj.n_bits + 1 - bit_index) == '1'
                        obj.bit_sym_map(sym_index + 1, bit_index) = 1;
                    end
                end
                % a = dec2bin(sym_index, obj.n_bits);
                % for bit_index = 1 : obj.n_bits
                %     if a(bit_index) == '1'   % LSB-first 顺序
                %         obj.bit_sym_map(sym_index + 1, bit_index) = 1;
                %     end
                % end
                % bit_sym_map = [1 0]
            end

            % obj.constellation_points = obj.constellation_points/sqrt(mean(obj.constellation_points.^2));
            obj.constellation_points = obj.constellation_points / sqrt(mean(abs(obj.constellation_points).^2));
        end
        
        function [sym] = modulate(obj, bits)
            % floor 保证整除（不足1组不调制）
            sym = zeros(floor(length(bits)/obj.n_bits), 1);
            for i = 1 : 1 : length(bits)/obj.n_bits
                symbol = 0;
                for j = 1 : obj.n_bits
                    % symbol_index
                    % 低位优先计算 symbol_index
                    % 调制:   LSB | bit0 bit1 bit2 bit3 bit4 bit5 bit6 bit7 | MSB
                    symbol = symbol + 2^(j-1) * bits((i-1)*obj.n_bits+j);
                end
                sym(i) = obj.constellation_points(symbol + 1);
            end
        end
        
        function [llr, p1] = compute_llr(obj, y, n_0)
            p0 = zeros(length(y) * obj.n_bits, 1);
            p1 = zeros(length(y) * obj.n_bits, 1);
            % 遍历每个接收符号
            for y_index = 1 : length(y)
                % 遍历所有可能的 constellation symbol
                for sym_index = 1  : 2^obj.n_bits
                    % p_sym = exp(-|y - s|^2 / 2 / n0)
                    if length(n_0) == 1
                        p_sym = exp(-abs(y(y_index) - obj.constellation_points(sym_index))^2/2/n_0);
                    else
                        p_sym = exp(-abs(y(y_index) - obj.constellation_points(sym_index))^2/2/n_0(y_index));
                    end
                    % 判断bit=0 or 1 累加对应 p_sym
                    for m_index = 1 : obj.n_bits
                        if obj.bit_sym_map(sym_index, m_index) == 0
                            p0((y_index-1) * obj.n_bits +  m_index) = p0((y_index-1) * obj.n_bits +  m_index) + p_sym;
                        else
                            p1((y_index-1) * obj.n_bits +  m_index) = p1((y_index-1) * obj.n_bits +  m_index) + p_sym;
                        end
                    end
                end
            end
            % 累加完，计算 LLR
            llr = log(p0./p1);
            p1 = p1./(p0 + p1);
        end
    end
end

