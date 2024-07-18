import re
import matplotlib.pyplot as plt

# 读取文本文件
with open('result_24.txt', 'r') as file:
    data = file.read()

# 使用正则表达式匹配 Transformer 每次迭代的 Train Loss 数据
pattern = r'Epoch: (\d+), Steps: \d+ \| Train Loss: (\d+\.\d+)'
matches = re.findall(pattern, data)

# 提取迭代次数和对应的 Train Loss
iterations = [int(match[0]) for match in matches]
losses = [float(match[1]) for match in matches]

# 绘制图表
plt.plot(iterations, losses)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Transformer Train Loss over Epochs')
plt.grid(True)
plt.show()