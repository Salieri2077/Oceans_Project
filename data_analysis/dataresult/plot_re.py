import re
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_name):
    hss = np.load('./'+file_name+'/metrics.npy')
    preds = np.load('./'+file_name+'/pred.npy')
    trues = np.load('./'+file_name+'/true.npy')
    return trues, preds

def plot_result(trues, preds, model_name, color):
    num_tau = preds.shape[0]
    num_pred = preds.shape[1]
    origin_data = np.zeros((num_tau+num_pred, preds.shape[2]))
    pred_data = np.zeros((num_tau+num_pred, preds.shape[2]))

    for i in range(num_tau):
        if i == 0:
            origin_data[:num_pred,:] = trues[i,:num_pred,:]
            pred_data[:num_pred,:] = preds[i,:num_pred,:]
        else:
            origin_data[num_pred+i,:] = trues[i,-1,:]
            pred_data[num_pred+i,:] = preds[i,-1,:]

    origin_data = origin_data.T
    pred_data = pred_data.T
    time_duration = np.arange(0,len(pred_data.T))/100
    plt.plot(time_duration,pred_data.T, label=model_name+' Prediction', color=color)
    plt.xlabel('Time /ms')
    plt.ylabel('Channel State Information (SNR)')
    plt.title('24 point Prediction')
    plt.grid(True)
    return  origin_data,time_duration
def moving_average(data, window_size):
    data = np.array(data).flatten()  # 转换为 NumPy 数组
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data

plt.figure(figsize=(10, 6))
trues, preds = load_data('inpulse_tran_24')
origin_plot,time_duration = plot_result(trues, preds, 'Transformer-24', color='red')
# 使用移动平均平滑处理 trues 数据
smoothed_trues = moving_average(origin_plot.T, window_size=5)  # 可以根据需要调整窗口大小
plt.plot(np.arange(0,len(smoothed_trues))/100,smoothed_trues, label='GroundTruth(smoothed)', color='blue')  
# plt.plot(time_duration,origin_plot.T, label='GroundTruth', color='blue')  
trues, preds = load_data('inpulse_lstm_24')
plot_result(trues, preds, 'LSTM-24', color='green')
trues, preds = load_data('inpulse_gru_24')
plot_result(trues, preds, 'GRU-24', color='orange')
# trues, preds = load_data('inpulse_VanillaRNN_24')
# plot_result(trues, preds, 'RNN-24', color='purple')
trues, preds = load_data('inpulse_informer_24')
plot_result(trues, preds, 'Informer-24', color='purple')
plt.legend()
# plt.xlim(106, 136)
plt.savefig('train_line_24.svg', format='svg')
plt.show()

# plt.figure(figsize=(10, 6))
# trues, preds = load_data('inpulse_tran_36')
# origin_plot,time_duration = plot_result(trues, preds, 'Transformer-36', color='red')
# # 使用移动平均平滑处理 trues 数据
# smoothed_trues = moving_average(origin_plot.T, window_size=5)  # 可以根据需要调整窗口大小
# plt.plot(np.arange(0,len(smoothed_trues))*(128/2048),smoothed_trues, label='GroundTruth(smoothed)', color='blue')  
# # plt.plot(time_duration,origin_plot.T, label='GroundTruth', color='blue')  
# trues, preds = load_data('inpulse_lstm_36')
# plot_result(trues, preds, 'LSTM-36', color='green')
# trues, preds = load_data('inpulse_gru_36')
# plot_result(trues, preds, 'GRU-36', color='orange')
# trues, preds = load_data('inpulse_VanillaRNN_36')
# plot_result(trues, preds, 'RNN-36', color='purple')
# # trues, preds = load_data('inpulse_informer_36')
# # plot_result(trues, preds, 'Informer-36', color='purple')
# plt.legend()
# plt.xlim(106, 136)
# plt.savefig('train_line_36.svg', format='svg')
# plt.show()

# plt.figure(figsize=(10, 6))
# trues, preds = load_data('inpulse_tran_48')
# origin_plot,time_duration = plot_result(trues, preds, 'Transformer-48', color='red')
# # 使用移动平均平滑处理 trues 数据
# smoothed_trues = moving_average(origin_plot.T, window_size=5)  # 可以根据需要调整窗口大小
# plt.plot(np.arange(0,len(smoothed_trues))*(128/2048),smoothed_trues, label='GroundTruth(smoothed)', color='blue')  
# # plt.plot(time_duration,origin_plot.T, label='GroundTruth', color='blue')  
# trues, preds = load_data('inpulse_lstm_48')
# plot_result(trues, preds, 'LSTM-48', color='green')
# trues, preds = load_data('inpulse_gru_48')
# plot_result(trues, preds, 'GRU-48', color='orange')
# trues, preds = load_data('inpulse_VanillaRNN_48')
# plot_result(trues, preds, 'RNN-48', color='purple')
# # trues, preds = load_data('inpulse_informer_48')
# # plot_result(trues, preds, 'Informer-48', color='purple')
# plt.legend()
# plt.xlim(106, 136)
# plt.savefig('train_line_48.svg', format='svg')
# plt.show()

# plt.figure(figsize=(10, 6))
# trues, preds = load_data('inpulse_tran_72')
# origin_plot,time_duration = plot_result(trues, preds, 'Transformer-72', color='red')
# # 使用移动平均平滑处理 trues 数据
# smoothed_trues = moving_average(origin_plot.T, window_size=5)  # 可以根据需要调整窗口大小
# plt.plot(np.arange(0,len(smoothed_trues))*(128/2048),smoothed_trues, label='GroundTruth(smoothed)', color='blue')  
# # plt.plot(time_duration,origin_plot.T, label='GroundTruth', color='blue')  
# trues, preds = load_data('inpulse_lstm_72')
# plot_result(trues, preds, 'LSTM-72', color='green')
# trues, preds = load_data('inpulse_gru_72')
# plot_result(trues, preds, 'GRU-72', color='orange')
# trues, preds = load_data('inpulse_VanillaRNN_72')
# plot_result(trues, preds, 'RNN-72', color='purple')
# # trues, preds = load_data('inpulse_informer_72')
# # plot_result(trues, preds, 'Informer-72', color='purple')
# plt.legend()
# plt.xlim(106, 136)
# plt.savefig('train_line_72.svg', format='svg')
# plt.show()

# plt.figure(figsize=(10, 6))
# trues, preds = load_data('inpulse_tran_96')
# origin_plot,time_duration = plot_result(trues, preds, 'Transformer-96', color='red')
# # 使用移动平均平滑处理 trues 数据
# smoothed_trues = moving_average(origin_plot.T, window_size=5)  # 可以根据需要调整窗口大小
# plt.plot(np.arange(0,len(smoothed_trues))*(128/2048),smoothed_trues, label='GroundTruth(smoothed)', color='blue')  
# # plt.plot(time_duration,origin_plot.T, label='GroundTruth', color='blue')  
# trues, preds = load_data('inpulse_lstm_96')
# plot_result(trues, preds, 'LSTM-96', color='green')
# trues, preds = load_data('inpulse_gru_96')
# plot_result(trues, preds, 'GRU-96', color='orange')
# trues, preds = load_data('inpulse_VanillaRNN_96')
# plot_result(trues, preds, 'RNN-96', color='purple')
# # trues, preds = load_data('inpulse_informer_24')
# # plot_result(trues, preds, 'Informer-24', color='purple')
# plt.legend()
# plt.xlim(106, 136)
# plt.savefig('train_line_96.svg', format='svg')
# plt.show()