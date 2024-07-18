clc;
clear;
close all;
% 读取Adaptive_modulation_throughput.mat文件
load('Adaptive_modulation_throughput.mat', 'SNR', 'all_throughput','all_BER_dc');
SNR_adaptive = SNR;
throughput_adaptive = all_throughput;
adaptive_BER_dc = all_BER_dc;
% 读取single_modulation_throughput.mat文件
load('single_modulation_throughput.mat', 'SNR', 'all_throughput', 'option','all_BER_dc');
SNR_single = SNR;
throughput_single = all_throughput;
single_BER_dc = all_BER_dc;
% 绘图
figure;
hold on;
plot(SNR_adaptive, throughput_adaptive, 'ro', 'DisplayName', 'Adaptive Modulation'); % 红色圆形标记
plot(SNR_single, throughput_single, 'bs', 'DisplayName', ['Single Modulation: ' option]); % 蓝色方形标记
hold off;
% 标签和图例
xlabel('SNR(dB)');
ylabel('Throughput(bps)');
% title('SNR-Throughput Comparison');
legend('show');
%%
% 筛选误码率小于0.001的数据
threshold = 0.001;
idx_adaptive = adaptive_BER_dc < threshold; % 自适应调制的索引
SNR_adaptive_filtered = SNR(idx_adaptive);
throughput_adaptive_filtered = throughput_adaptive(idx_adaptive);
idx_single = single_BER_dc < threshold; % 单一调制的索引
SNR_single_filtered = SNR(idx_single);
throughput_single_filtered = throughput_single(idx_single);
% 绘图
figure;
hold on;
plot(SNR_adaptive_filtered, throughput_adaptive_filtered, 'ro', 'DisplayName', 'Adaptive Modulation');
% plot(max(SNR_adaptive_filtered), max(throughput_adaptive_filtered), 'bs', 'DisplayName', ['Single Modulation: ' option]);
plot(SNR_single_filtered, throughput_single_filtered, 'bs', 'DisplayName', ['Single Modulation: ' option]);
hold off;
% 标签和图例
xlabel('SNR(dB)');
ylabel('Throughput(bps)');
% title('Throughput vs. SNR for BER < 0.001');
legend('show');