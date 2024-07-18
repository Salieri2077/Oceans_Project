clc;
clear all;
close all;
% 读取CSV文件
SNR = readmatrix('SNR.csv');
% 设置采样频率 fs，假设为1
fs = 100;  % 根据实际情况调整采样频率
% 生成时间向量
t = 0:1/fs:(length(SNR)-1)/fs;
% 绘制SNR图
figure;
plot(t, SNR);
xlabel('Time/s');
ylabel('SNR/dB');
grid on;