def moving_average(data, window_size):
    data = np.array(data)  # 转换为 NumPy 数组
    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data



plt.figure(figsize=(10, 6))
trues, preds = load_data('inpulse_tran_72')

origin_plot,time_duration = plot_result(trues, preds, 'Transformer-72', color='red')

# 使用移动平均平滑处理 trues 数据
smoothed_trues = moving_average(origin_plot.T, window_size=5)  # 可以根据需要调整窗口大小
plt.plot(time_duration,smoothed_trues, label='GroundTruth', color='olive')  