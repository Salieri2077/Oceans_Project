import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

setting = '/SNR/True'

# setting = '/NOF1/inpulse_lstm_24'
# setting = '/NOF1/inpulse_gru_24'
# setting = '/NOF1/inpulse_VanillaRNN_24'
# setting = 'inpulse_VanillaRNN_48'
# setting = 'inpulse_VanillaRNN_96'

mtrs = np.load('./results/'+setting+'/metrics.npy')
preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')
# # [samples, pred_len, dimensions]
# preds.shape, trues.shape
num_tau = preds.shape[0]  # 窗体移动次数-384
num_time = preds.shape[2] # 不同时间的-20个不同信道
num_pred = preds.shape[1] # 向下预测的信道冲激响应-24
origin_data = np.zeros((num_tau+num_pred, preds.shape[2]))
pred_data = np.zeros((num_tau+num_pred, preds.shape[2]))
# origin = # num_time
for i in range(num_tau):
    if i == 0:
        origin_data[:num_pred,:] = trues[i,:num_pred,:]
        pred_data[:num_pred,:] = preds[i,:num_pred,:]
    else:
        origin_data[num_pred+i,:] = trues[i,-1,:] # 窗体每次移动1个单位
        pred_data[num_pred+i,:] = preds[i,-1,:]
origin_data = origin_data.T
pred_data = pred_data.T
# 上述是在多列信道下做的---单列的需要给它隔2048截一下
plt.figure(figsize=(10, 6))
plt.plot(origin_data.T, label='GroundTruth')
# plt.plot(pred_data.T, label='LSTM-Prediction')
plt.plot(pred_data.T, label='GRU-Prediction')
plt.legend()
plt.show()
# 绘制 origin_data 的伪彩图
plt.figure(figsize=(10, 6))
plt.imshow(origin_data, aspect='auto', cmap='viridis')
plt.colorbar(label='Value')
plt.title('Origin Data Pseudocolor Plot')
plt.xlabel('Delay')
plt.ylabel('Time')
plt.show()
# 绘制 pred_data 的伪彩图
plt.figure(figsize=(10, 6))
plt.imshow(pred_data, aspect='auto', cmap='viridis')
plt.colorbar(label='Value')
# plt.title('LSTM-Predicted Data Pseudocolor Plot')
plt.title('GRU-Predicted Data Pseudocolor Plot')
plt.xlabel('Delay')
plt.ylabel('Time')
plt.show()

# draw OT prediction
output_folder = 'plots'
# 确保保存图像的文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 对每个样本进行绘图并保存  1777-24
for i in range(1700,1800):
    plt.figure()
    plt.plot(trues[i,:,-1], label='GroundTruth')
    plt.plot(preds[i,:,-1], label='Prediction')
    plt.legend()
    plt.title(f'Sample {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)   
    # 保存图像到文件夹
    plt.savefig(os.path.join(output_folder, f'sample_{i+1}.png'))
    plt.close()  # 关闭图像以释放内存，避免太多图像同时打开  
print("Plots saved successfully.")

# # draw HUFL prediction
# plt.figure()
# plt.plot(trues[-1,:,0], label='GroundTruth')
# plt.plot(preds[-1,:,0], label='Prediction')
# plt.legend()
# plt.show()

# # 读取CSV文件
# data = pd.read_csv('./data/ETT/ETTh1.csv')

# # 提取OT列的数据的17397行到17421行的数据
# ot_data = data['OT'].iloc[17396:17421]

# # 创建图像并绘制曲线
# plt.figure(figsize=(10, 6))
# plt.plot(ot_data, label='OT')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('OT Data')
# plt.legend()
# plt.grid(True)
# plt.show()
