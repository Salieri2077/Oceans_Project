clc;
clear;
close all;

fs = 48000;
SNR = 12; % 12dB
N_filter = 512; % 滤波器阶数

%% LFM信号参数设计
B = 4e3;
fl = 10e3;
fh = fl + B;
T_syn = 0.1;
K = B / T_syn;  % LFM信号参数，B带宽，T脉宽，K调频斜率
t = 0 : 1/fs : T_syn-1/fs;
Signal = cos(2*pi*fl*t + pi*K*t.^2);
length_GI = 0.1 * fs; % 保护间隔
signal_GI = zeros(1, length_GI);
Signal_in = [signal_GI Signal signal_GI];
BandPass = fir1(N_filter, 2 * [fl fh] / fs); % 带通滤波器
s_begin = length_GI;
s_end = length_GI + length(Signal);

%% 加带内噪声
NumFilter = length(BandPass)-1;
if isequal(imag(Signal_in), zeros(1, length(Signal_in))) % 实数信号的接收噪声
    noise = normrnd(0, 1, 1, length(Signal_in));
else
    noise = sqrt(1/2)*normrnd(0, 1, 1, length(Signal_in)) + 1i*sqrt(1/2)*normrnd(0, 1, 1, length(Signal_in)); % 复数信号的接收噪声
end

% 生成随机噪声
NoiseAftFilter = filter(BandPass, 1, [noise zeros(1, NumFilter/2)]); % 滤波
NoiseAftFilter = NoiseAftFilter(NumFilter/2+1 : end);

% 功率计算
EnOfSignal = Signal_in(s_begin:s_end) * Signal_in(s_begin:s_end)';
EnOfNoise = NoiseAftFilter * NoiseAftFilter';
NorOfNoise = NoiseAftFilter / sqrt(EnOfNoise); % 噪声能量归一化

% SignalAftChannel包含了保护间隔和LFM信号，但实际我们只在乎符号的，所以按symbol的能量扩充到接收信号
AmpOfNoise = sqrt(10^(-SNR/10) * EnOfSignal * length(Signal_in) / (s_end - s_begin));
Noise = NorOfNoise * AmpOfNoise;
SignalAftNoise = Signal_in + Noise; % 信号加噪声

snrband = 20 * log10(std(Signal_in(s_begin:s_end)) / std(Noise)); % 带内信噪比测试 
fprintf('Input SNR: %.2f dB\n', SNR); % 输入的SNR
fprintf('Band SNR: %.2f dB\n', snrband); % 带内信噪比

% 计算并输出SNR
snr_estimated = calculate_snr(Signal_in, SignalAftNoise, s_begin, s_end);
fprintf('Estimated SNR: %.2f dB\n', snr_estimated);

% 根据接收到的信号直接计算SNR
function snr_estimated = calculate_snr(Signal_in, SignalAftNoise, s_begin, s_end)
    % 分离信号和噪声
    signal_segment = Signal_in(s_begin:s_end);
    noisy_signal_segment = SignalAftNoise(s_begin:s_end);
    noise_segment = noisy_signal_segment - signal_segment;
    % 计算信号和噪声的功率
    signal_power = mean(abs(signal_segment).^2);
    noise_power = mean(abs(noise_segment).^2);
    % 计算SNR
    snr_estimated = 10 * log10(signal_power / noise_power);
end
