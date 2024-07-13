clc;
clear all;
close all;
set(0, 'defaultAxesXGrid', 'on', 'defaultAxesYGrid', 'on') % 打开网格
%% 基本参数
fs = 48000; % 采样频率
fl = 10e3; % (LFM)下限频率
B = 4e3; % 通信带宽
fh = fl + B; % (LFM)上限频率
f0 = (fl + fh) / 2; % 中心频率（单频带传输）==12KHz
Rb = 2000; % 符号率
N_up = fs / Rb; % 升采样点数
N_bit = 6000; % 发送的比特数
alpha = 1; % 滚降系数
N_filter = 512; % 滤波器阶数
PulseShape = rcosdesign(alpha, 1, N_up, 'sqrt'); % 脉冲成型滤波器
b1 = fir1(N_filter, 2 * [fl fh] / fs); % 带通滤波器
%% 信噪比范围
snr_range = -5:20; % 调整信噪比范围
%% 调制和码率组合
combinations = {
    struct('Mod', 2, 'rate', 1/2),
    struct('Mod', 2, 'rate', 1/3),
    struct('Mod', 2, 'rate', 1/4),
    struct('Mod', 4, 'rate', 1/2),
    struct('Mod', 4, 'rate', 1/3),
    struct('Mod', 4, 'rate', 1/4),
    struct('Mod', 8, 'rate', 1/2),
    struct('Mod', 8, 'rate', 1/3),
    struct('Mod', 8, 'rate', 1/4)
};
T_send = zeros(1,length(combinations));
N_s = zeros(1,length(combinations));
BER_results = cell(length(combinations), 1);
snr_at_target_BER = zeros(length(combinations), 1); % 初始化存储达到目标误码率的信噪比
for comb_idx = 1:length(combinations)
    Mod = combinations{comb_idx}.Mod;
    rate = combinations{comb_idx}.rate;
    bitnum_per = log2(Mod);
    N_BS = N_bit / bitnum_per; % 发送的符号数
    length_BS = N_BS * N_up * (1 / rate); % 使用卷积码需要乘以相应的因子
    %% 数据信号产生及编码
    load information.mat
    bit_generate = information(1 : N_bit);  
    %% 直接加扰 (scramble)
    rng(1); % 种子
    random_bits = randi([0, 1], 1, N_bit);
    scrambled_bits = xor(bit_generate, random_bits);
    %% 卷积码编码
    P = 0.1;
    % 卷积码的生成多项式
    if rate == 1
        trellis = poly2trellis(7, [131]);
    elseif rate == 1/2
        trellis = poly2trellis(7, [133 171]);
    elseif rate == 1/3
        trellis = poly2trellis(7, [133 171 165]);
    elseif rate == 1/4
        trellis = poly2trellis(7, [133 171 165 171]);
    end
    msg1 = convenc(scrambled_bits, trellis); % 卷积编码
    %% 进行交织
    cols = length(msg1) / 5; rows = 5;
    interleaved_data = matintrlv(msg1, rows, cols); % 解交织使用函数--matdeintrlv
    %% 调制映射
    [SymbolIn, Table] = Mapping(interleaved_data, Mod);
    %% 升采样脉冲成型--对信息进行调制
    signal_IQ = IQmodulate(PulseShape, N_up, SymbolIn, f0, fs);
    signal_IQ = signal_IQ ./ max(abs(signal_IQ));
    %% LFM信号参数设计
    T_syn = 0.1; K = B / T_syn; % LFM信号参数，B带宽，T脉宽，K调频斜率
    t = 0 : 1 / fs : T_syn - 1 / fs;
    signal_measure = cos(2 * pi * fl * t + pi * K * t.^2);
    length_measure = T_syn * fs;
    length_GI = 0.1 * fs; % 保护间隔
    signal_GI = zeros(1, length_GI);
    %% 发送信号构成
    signal_send = [signal_measure signal_GI signal_IQ signal_GI signal_measure signal_GI];
    %% 初始化误码率数组
    BER_nodc = zeros(1, length(snr_range));
    BER_dc = zeros(1, length(snr_range));
    for idx = 1:length(snr_range)
        SNR = snr_range(idx);
        m = 0;
        pass_by = Pass_Channel(fs, m, b1, length_measure, length_GI, length_BS, SNR, Rb, signal_send);
        %% --------------------接收机部分------------------------
        %% 时间同步
        signal_receive = pass_by;
        %% 带通滤波
        signal_bandpass = filter(b1, 1, [signal_receive zeros(1, fix(length(b1) / 2))]);
        signal_rec_pass = signal_bandpass(fix(length(b1) / 2) + 1:end);
        %% 多普勒测量
        Res_xcorr = corr_fun(signal_rec_pass, signal_measure);
        [~, pos1] = max(Res_xcorr(1 : length_measure + length(signal_GI)));
        [~, pos2] = max(Res_xcorr(length_GI + length_BS + 1 : end));
        pos2 = pos2 + length_GI + length_BS;
        del_rec = pos2 - pos1;
        del_send = length_measure + 2 * length_GI + length_BS;
        dup_det = (del_send - del_rec) / del_send;
%         fprintf(['多普勒因子测量值：' num2str(dup_det) '\n']);
        fs2 = fs * (1 - dup_det);
        fs2 = round(fs2 / factor_resample(fs)) * factor_resample(fs);
        signal_rec_dc = resample(signal_rec_pass, fs, fs2);
        %% 提取信息符号
        signal_rec_nodc_information = signal_rec_pass(length_measure + length_GI + 1 : length_measure + length_GI + length_BS);
        signal_rec_dc_information = signal_rec_dc(length_measure + length_GI + 1 : length_measure + length_GI + length_BS);
        %% 信道均衡
        Need_len = length(signal_rec_dc_information);
        signal_rec_dc_information = LTE_LMS_fun1(25, 0.01, Need_len / 2, Need_len / 2, signal_rec_dc_information, signal_rec_nodc_information);      
        %% 相干解调--IQ解调+下载波
        [symbol_demodulate_nodc] = IQdemodulate(signal_rec_nodc_information, fs, length_BS, f0, PulseShape, N_up);
        [symbol_demodulate_dc] = IQdemodulate(signal_rec_dc_information, fs, length_BS, f0, PulseShape, N_up);       
        %% 抽样判决
        symbol_decision_nodc = SymbolDecision(symbol_demodulate_nodc, Table);
        symbol_decision_dc = SymbolDecision(symbol_demodulate_dc, Table);
        %% 解映射
        bit_nodc = Demapping(symbol_decision_nodc, Table, Mod);
        bit_dc = Demapping(symbol_decision_dc, Table, Mod);
        %% 解交织
        deinterleaved_bit_nodc = matdeintrlv(bit_nodc, rows, cols);
        deinterleaved_bit_dc = matdeintrlv(bit_dc, rows, cols);
        %% 译码
        output_bit_nodc = vitdec(deinterleaved_bit_nodc, trellis, 7, 'trunc', 'hard');
        output_bit_dc = vitdec(deinterleaved_bit_dc, trellis, 7, 'trunc', 'hard');
        %% 解扰
        final_bit_nodc = xor(output_bit_nodc, random_bits);
        final_bit_dc = xor(output_bit_dc, random_bits);
        %% 计算误码
        BER_nodc(idx) = length(find(final_bit_nodc ~= bit_generate)) / N_bit;
        BER_dc(idx) = length(find(final_bit_dc ~= bit_generate)) / N_bit;

%         fprintf(['多普勒不补偿且信道不均衡误码率：' num2str(BER_nodc(idx)) '\n']);
%         fprintf(['多普勒补偿且信道均衡误码率：' num2str(BER_dc(idx)) '\n']);
    end
    % 找到第一个小于0.001的误码率对应的信噪比
    target_ber_threshold = 0.001;
    idx_target = find(BER_dc < target_ber_threshold, 1);
    if ~isempty(idx_target)
        snr_at_target_BER(comb_idx) = snr_range(idx_target);
        fprintf('组合 Mod=%d, Rate=%.2f 达到BER<0.001的信噪比为: %d dB\n', Mod, rate, snr_at_target_BER(comb_idx));
    else
        snr_at_target_BER(comb_idx) = NaN;
        fprintf('组合 Mod=%d, Rate=%.2f 未能达到BER<0.001\n', Mod, rate);
    end
    T_send(comb_idx) = length(signal_send) / fs;  % 发送时间
    N_s(comb_idx) = N_BS;  % 符号数
    BER_results{comb_idx} = struct('BER_nodc', BER_nodc, 'BER_dc', BER_dc);
end
% % 保存信噪比门限
save('snr_at_target_BER.mat', 'snr_at_target_BER');
% 计算并保存其他参数
xi = rate;  % 编码码率
BER = target_ber_threshold;  % 比特错误率
save('tx_params.mat','N_s', 'xi', 'T_send', 'Rb', 'BER');
close all;
%% 绘制误码率随信噪比变换的曲线
figure;
hold on;
for comb_idx = 1:length(combinations)
    BER_nodc = BER_results{comb_idx}.BER_nodc;
    BER_dc = BER_results{comb_idx}.BER_dc;
%     semilogy(snr_range, BER_nodc, '-o', 'LineWidth', 2, 'DisplayName', ['Mod=' num2str(combinations{comb_idx}.Mod) ', Rate=' num2str(combinations{comb_idx}.rate)]);
    semilogy(snr_range, BER_dc, '-s', 'LineWidth', 2, 'DisplayName', ['Mod=' num2str(combinations{comb_idx}.Mod) ', Rate=' num2str(combinations{comb_idx}.rate) ' (DC)']);
end
grid on;
xlabel('信噪比 (dB)');
ylabel('误码率');
legend;
title('误码率随信噪比变化曲线');

%% 辅助函数
function symbols = SymbolDecision(symbol_demod, Table)
    symbols = zeros(1, length(symbol_demod));
    for j = 1:length(symbol_demod)
        Distance_all = abs(symbol_demod(j) - Table);
        Tablemin = find(Distance_all == min(Distance_all));
        symbols(j) = Table(Tablemin(1));
    end
end
