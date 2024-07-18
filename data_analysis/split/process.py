import re
import matplotlib.pyplot as plt
import numpy as np

# def plot_train_loss(file_name, model_name):
#     with open(file_name, 'r') as file:
#         data = file.read()
    
#     pattern = r'Epoch: (\d+), Steps: \d+ \| Train Loss: (\d+\.\d+)'
#     matches = re.findall(pattern, data)
#     iterations = [int(match[0]) for match in matches]
#     losses = [float(match[1]) for match in matches]

#     plt.plot(iterations, losses)
#     plt.xlabel('Epoch')
#     plt.ylabel('Train Loss (dB)')
#     plt.title(f'{model_name} Train Loss over Epochs')
#     plt.grid(True)
#     # 修改保存的文件名，将空格替换为下划线，并添加后缀 '.svg'
#     save_name = model_name.replace(' ', '_') + '.svg'
#     plt.savefig(save_name, format='svg')  # 保存为 SVG 格式
#     plt.show()

# 绘制 Transformer 不同配置的 Train Loss 曲线
# plot_train_loss('transformer_24.txt', 'Transformer-24')
# plot_train_loss('transformer_48.txt', 'Transformer-48')
# plot_train_loss('transformer_96.txt', 'Transformer-96')
# plot_train_loss('LSTM_24.txt', 'LSTM-24')
# plot_train_loss('LSTM_48.txt', 'LSTM-48')
# plot_train_loss('LSTM_96.txt', 'LSTM-96')
# plot_train_loss('GRU_24.txt', 'GRU-24')
# plot_train_loss('GRU_48.txt', 'GRU-48')
# plot_train_loss('GRU_96.txt', 'GRU-96')
# plot_train_loss('informer_24.txt', 'Informer-24')
# plot_train_loss('informer_48.txt', 'Informer-48')
# plot_train_loss('informer_96.txt', 'Informer-96')
########################################################

def plot_train_loss(file_name, model_name):
    with open(file_name, 'r') as file:
        data = file.read()
    
    pattern = r'Epoch: (\d+), Steps: \d+ \| Train Loss: (\d+\.\d+)'
    matches = re.findall(pattern, data)
    iterations = [int(match[0]) for match in matches]
    losses = [float(match[1]) for match in matches]
    
    # 转换为分贝单位
    losses_db = 10 * np.log10(losses)
    
    plt.plot(iterations, losses_db, label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss (dB)')
    plt.title('Train Loss over Epochs')
    plt.grid(True)
    plt.legend()

# 创建一个图表，将不同模型的曲线放在同一张图上
plt.figure(figsize=(10, 6))
# 绘制 Transformer 不同配置的 Train Loss 曲线
plot_train_loss('transformer_24.txt', 'Transformer-24')
plot_train_loss('LSTM_24.txt', 'LSTM-24')
plot_train_loss('GRU_24.txt', 'GRU-24')
# plot_train_loss('informer_24.txt', 'Informer-24')
plot_train_loss('VanRNN_24.txt', 'VanRNN-24')
# 显示图例和保存图表
plt.legend()
plt.savefig('train_loss_24.svg', format='svg')
plt.show()

# 创建一个图表，将不同模型的曲线放在同一张图上
plt.figure(figsize=(10, 6))
# 绘制 Transformer 不同配置的 Train Loss 曲线
plot_train_loss('transformer_48.txt', 'Transformer-48')
plot_train_loss('LSTM_48.txt', 'LSTM-48')
plot_train_loss('GRU_48.txt', 'GRU-48')
# plot_train_loss('informer_48.txt', 'Informer-48')
plot_train_loss('VanRNN_48.txt', 'VanRNN-48')
# 显示图例和保存图表
plt.legend()
plt.savefig('train_loss_48.svg', format='svg')
plt.show()

# 创建一个图表，将不同模型的曲线放在同一张图上
plt.figure(figsize=(10, 6))
# 绘制 Transformer 不同配置的 Train Loss 曲线
plot_train_loss('transformer_96.txt', 'Transformer-96')
plot_train_loss('LSTM_96.txt', 'LSTM-96')
plot_train_loss('GRU_96.txt', 'GRU-96')
# plot_train_loss('informer_96.txt', 'Informer-96')
plot_train_loss('VanRNN_96.txt', 'VanRNN-96')
# 显示图例和保存图表
plt.legend()
plt.savefig('train_loss_96.svg', format='svg')
plt.show()