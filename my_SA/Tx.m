clc;
clear all;
close all;
set(0, 'defaultAxesXGrid','on', 'defaultAxesYGrid', 'on') %打开网格
%% 第一次打开MATLAB需设置 Python 环境，后续运行可注释
% pyversion('E:\Anaconda\Data\envs\myenv\python.exe');
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
%% 基本参数
fs = 48000;                                                            % 采样频率
fl = 10e3;                                                             % (LFM)下限频率
B = 4e3;                                                               % 通信带宽
fh = fl+B;                                                              % (LFM)上限频率
f0 = (fl + fh) / 2;                                                        % 中心频率（单频带传输）==12KHz
Rb = 2000;                                                             % 符号率
N_up = fs / Rb;                                                        % 升采样点数
N_bit = 6000;                                                          % 发送的比特数
alpha = 1;                                                              % 滚降系数
N_filter = 512;                                                        % 滤波器阶数
% PulseShape = rcosfir(alpha, [ ], N_up, 1, 'sqrt');  % 脉冲成型滤波器（低通滤波器）
PulseShape = rcosdesign(alpha,1, N_up, 'sqrt');
b1 = fir1(N_filter, 2 * [fl fh] / fs);                               % 带通滤波器
%% --------------------发射机部分------------------------
%% 数据信号产生及编码
load information.mat
bit_generate = information(1 : N_bit);
%% 直接加扰?(scramble)
rng(1); % 种子
random_bits = randi([0, 1], 1, N_bit);
% 对bit_generate和随机数进行异或操作
scrambled_bits = xor(bit_generate, random_bits);
%% ------------利用信道预测结果的自适应调制------------
file_name = 'inpulse_informer_24';
% 准备输入数据
% data = load('./inpulse_informer_24/data.mat');
% channel_states = data.pred_data(:); % 转换为列向量
max_cross_time = 100; % 最大交叉变异次数
% 确保MATLAB找到Python路径
if count(py.sys.path, '') == 0
    insert(py.sys.path, int32(0), '');
end
% 导入Python模块
mod = py.importlib.import_module('algorithm_main');
py.importlib.reload(mod);
% 调用Python函数并处理结果
try
    % 将数据转换为Python可接受的类型
%     py_channel_states = py.numpy.array(channel_states);
%     py_channel_states即预测的信噪比，直接在python文件中读入了
    py_max_cross_time = int32(max_cross_time);
    % 调用Python函数
    result = mod.genetic_algorithm_main(file_name,py_max_cross_time);
    % 将结果转换为MATLAB类型
    best_modulation_schemes = cell(result);
%     disp('最佳调制和编码模式:');
%     disp(best_modulation_schemes);
catch ME
    disp('调用Python函数时出错:');
    disp(ME.message);
end
data = load(fullfile('.', file_name, 'data.mat'));
origin_data = data.origin_data(:);
num_points = length(origin_data);num_windows = length(best_modulation_schemes);window_size = floor(num_points/num_windows);
% 计算每24(window_size)个点的平均值赋值给SNR
SNR = mean(reshape(origin_data(1:num_windows*window_size), window_size, []));
% 初始化存储结果的变量
all_BER_dc = [];all_time_points = [];all_throughput = [];
for num_point = 1:length(best_modulation_schemes)
    switch char(best_modulation_schemes{num_point})
        case 'BPSK_1/2'
            Mod = combinations{1}.Mod;
            rate = combinations{1}.rate;
        case 'BPSK_1/3'
            Mod = combinations{2}.Mod;
            rate = combinations{2}.rate;
        case 'BPSK_1/4'
            Mod = combinations{3}.Mod;
            rate = combinations{3}.rate;
        case 'QPSK_1/2'
            Mod = combinations{4}.Mod;
            rate = combinations{4}.rate;
        case 'QPSK_1/3'
            Mod = combinations{5}.Mod;
            rate = combinations{5}.rate;
        case 'QPSK_1/4'
            Mod = combinations{6}.Mod;
            rate = combinations{6}.rate;
        case '8PSK_1/2'
            Mod = combinations{7}.Mod;
            rate = combinations{7}.rate;
        case '8PSK_1/3'
            Mod = combinations{8}.Mod;
            rate = combinations{8}.rate;
        case '8PSK_1/4'
            Mod = combinations{9}.Mod;
            rate = combinations{9}.rate;    
        otherwise
            error('Unknown modulation scheme');
    end
    disp(['Selected Modulation Scheme: ', char(best_modulation_schemes{num_point})]);
    bitnum_per = log2(Mod);
    % 更新调制模式相关参数
    N_BS = N_bit / bitnum_per; % 发送的符号数
    length_BS = N_BS * N_up * (1 / rate); % 使用卷积码需要乘以相应的因子
    %% 卷积码编码
    %卷积码的生成多项式
    if rate == 1
        tre1 = poly2trellis(7,[131]);
    elseif rate == 1/2
        tre1 = poly2trellis(7,[133 171]);   
    elseif rate == 1/3
        tre1 = poly2trellis(7, [133 171 165]);
    elseif rate == 1/4
        tre1 = poly2trellis(7, [133 171 165 171]);
    end
    msg1 = convenc(scrambled_bits,tre1); %卷积编码
    %% 进行交织
    cols = length(msg1)/5;rows =  5;
    interleaved_data = matintrlv(msg1, rows, cols); %解交织使用函数--matdeintrlv
    %% MPSK映射
    [SymbolIn, Table] = Mapping(interleaved_data, Mod);
    %% 升采样脉冲成型--对信息进行调制
    signal_IQ = IQmodulate(PulseShape, N_up, SymbolIn, f0, fs);
    signal_IQ = signal_IQ ./ max(abs(signal_IQ));
    %% LFM信号参数设计
    T_syn = 0.1;K = B / T_syn;  % LFM信号参数，B带宽，T脉宽，K调频斜率
    t = 0 : 1/fs : T_syn-1/fs;
    signal_measure = cos(2*pi*fl*t + pi*K*t.^2);                  
    length_measure = T_syn * fs;
    length_GI = 0.1 * fs;                                                %保护间隔
    signal_GI = zeros(1, length_GI);
    %% 发送信号构成
    signal_send = [signal_measure signal_GI signal_IQ signal_GI signal_measure signal_GI];    %信号结构[测量信号 保护间隔 调制信号 保护间隔 测量信号 保护间隔]
    m = 0; % 影响多普勒
    %% 过信道
    pass_by = Pass_Channel(fs,m,b1,length_measure,length_GI,length_BS,SNR(num_point),Rb,signal_send,num_point);
    %% --------------------接收机部分------------------------
    %% 时间同步
    signal_receive = pass_by;
    %% 带通滤波
    signal_bandpass = filter(b1, 1, [signal_receive zeros(1,fix(length(b1)/2))]);
    signal_rec_pass = signal_bandpass(fix(length(b1)/2)+1:end);

    %% 多普勒测量
    Res_xcorr = corr_fun(signal_rec_pass, signal_measure);
    [~, pos1] = max(Res_xcorr(1 : length_measure+length(signal_GI)));             
    [~, pos2] = max(Res_xcorr(length_GI+length_BS+1 : end));      
    pos2 = pos2 + length_GI + length_BS;                                      
    % 计算接收信号首尾LFM间隔，与发射间隔做对比
    del_rec = pos2 - pos1;
    del_send = length_measure + 2*length_GI + length_BS ;
    % 利用间隔变化做多普勒测量
    dup_det = (del_send - del_rec) / del_send;
%     fprintf(['多普勒因子测量值：' num2str(dup_det) '\n']);
    % 利用重采样进行多普勒补偿
    fs2 = fs*(1-dup_det);
    fs2 = round(fs2 / factor_resample(fs)) * factor_resample(fs);    %使其满足resample精度要求防止报错
    signal_rec_dc = resample(signal_rec_pass, fs, fs2);   %dc：Doppler compensation
    % 提取信息符号
    signal_rec_nodc_information = signal_rec_pass(length_measure+length_GI+1 : length_measure+length_GI+length_BS);
    signal_rec_dc_information = signal_rec_dc(length_measure+length_GI+1 : length_measure+length_GI+length_BS);
    signal_rec_origin_information = signal_send(length_measure+length_GI+1 : length_measure+length_GI+length_BS);
    %% 信道均衡
    Need_len =  length(signal_rec_dc_information);
%     signal_rec_dc_information = LTE_LMS_fun1(25,0.01,Need_len/2,Need_len/2,signal_rec_dc_information,signal_rec_origin_information);
    %% 相干解调--IQ解调+下载波
    [symbol_demodulate_nodc] = IQdemodulate(signal_rec_nodc_information, fs, length_BS, f0, PulseShape, N_up);
    [symbol_demodulate_dc] = IQdemodulate(signal_rec_dc_information, fs, length_BS, f0, PulseShape, N_up);
    %% 抽样判决
    for j = 1 : length(symbol_demodulate_nodc)
        Distance_all = abs(symbol_demodulate_nodc(j) - Table);
        Tablemin=find(Distance_all == min(Distance_all));
        symbol_decision_nodc(j) = Table(Tablemin(1));
    end
    for j = 1 : length(symbol_demodulate_dc)
        Distance_all = abs(symbol_demodulate_dc(j) - Table);
        Tablemin=find(Distance_all == min(Distance_all));
        symbol_decision_dc(j) = Table(Tablemin(1));
    end
    %% 解映射
    bit_nodc  = Demapping(symbol_decision_nodc , Table , Mod);
    bit_dc  = Demapping(symbol_decision_dc , Table , Mod);
    %% 解交织
    deinterleaved_bit_nodc = matdeintrlv(bit_nodc, rows, cols);
    deinterleaved_bit_dc = matdeintrlv(bit_dc, rows, cols);
    %% 译码
    output_bit_nodc = vitdec(deinterleaved_bit_nodc, tre1, 7, 'trunc', 'hard'); % 卷积码
    output_bit_dc = vitdec(deinterleaved_bit_dc, tre1, 7, 'trunc', 'hard'); % 卷积码
    %% 解扰
    final_bit_nodc = xor(output_bit_nodc, random_bits);
    final_bit_dc = xor(output_bit_dc, random_bits);
    %% 计算误码
    BER_nodc = length(find(final_bit_nodc ~= bit_generate)) ./ N_bit;
    BER_dc = length(find(final_bit_dc ~= bit_generate)) ./ N_bit;
%     fprintf(['多普勒不补偿且信道不均衡误码率：'  num2str(BER_nodc) '\n'] );
    fprintf(['多普勒补偿且信道均衡误码率：' num2str(BER_dc) '\n']);
%     scatterplot(symbol_demodulate_nodc);
%     title('未进行多普勒补偿且未进行信道均衡前')
%     scatterplot(symbol_demodulate_dc);
%     title('采用多普勒补偿和信道均衡后')
    % 存储BER_dc和时间点
    all_BER_dc = [all_BER_dc, BER_dc];
    all_time_points = [all_time_points, num_point];
    % 计算吞吐量
    T_send = length(signal_send) / fs;  % 每次因采取不同的调制方式发送时间不同
    throughput = (bitnum_per * N_BS * rate * (1 - BER_dc)) / T_send;
    all_throughput = [all_throughput throughput];
    % 除了以下变量不重置，其他全重置以避免干扰
    clearvars -except best_modulation_schemes N_bit N_up f0 fs PulseShape B fl b1 Rb random_bits bit_generate tre1 interleaved_data combinations scrambled_bits SNR  all_BER_dc all_time_points all_throughput
end
%% plot
figure;
hold on;
colors = [
    1, 0, 0;    % BPSK_1/2
    0, 1, 0;    % BPSK_1/3
    0, 0, 1;    % BPSK_1/4
    1, 1, 0;    % QPSK_1/2
    1, 0, 1;    % QPSK_1/3
    0, 1, 1;    % QPSK_1/4
    0.5, 0, 0;  % 8PSK_1/2
    0, 0.5, 0;  % 8PSK_1/3
    0, 0, 0.5;  % 8PSK_1/4
];
scheme_names = {'BPSK_1/2', 'BPSK_1/3', 'BPSK_1/4', 'QPSK_1/2', 'QPSK_1/3', 'QPSK_1/4', '8PSK_1/2', '8PSK_1/3', '8PSK_1/4'};
% 存储图例句柄
legend_handles = gobjects(length(scheme_names), 1);
for i = 1:length(all_time_points)
    scheme_str = char(best_modulation_schemes{i});  % 将其转换为 MATLAB 字符串
    scheme_index = find(strcmp(scheme_names, scheme_str));
    % 绘制3D散点图
    scatter3(all_time_points(i), SNR(i), all_BER_dc(i), 36, colors(scheme_index, :), 'filled');
%     scatter3(all_time_points(i), all_BER_dc(i), SNR(i), 36, colors(scheme_index, :), 'filled');
%     scatter3(SNR(i), all_BER_dc(i), all_time_points(i), 36, colors(scheme_index, :), 'filled');
end
% scatter3(all_time_points, SNR, all_BER_dc, 36, colors(scheme_index, :), 'filled');
% 绘制图例，只绘制一次
for j = 1:length(scheme_names)
    legend_handles(j) = scatter3(NaN, NaN, NaN, 36, colors(j, :), 'filled');
end
xlabel('时间点');
ylabel('信噪比 (SNR)');
zlabel('误码率 (BER\_dc)');
title('时间变化下的信噪比和误码率');
grid on;
% 添加图例
legend(legend_handles, scheme_names, 'Location', 'best');
hold off;
%% plot-吞吐量
figure;
plot(SNR,all_throughput,'o');
xlabel('SNR(dB)');
ylabel('Throughput(bps)');
title('SNR-Throughput');