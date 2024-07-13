import numpy as np
import random
import scipy.io
import os
from collections import Counter

# 读取MATLAB生成的参数
def load_tx_params(file_name=r'.\tx_params.mat'):
    params = scipy.io.loadmat(file_name)
    N_s = params['N_s'].flatten()
    xi = params['xi'][0][0]
    T_send = params['T_send'].flatten()
    Rb = params['Rb'][0][0]
    BER = params['BER'][0][0]
    return N_s, xi, T_send, Rb, BER
# 读取MATLAB生成的信噪比门限
def load_snr_thresholds(file_name=r'.\snr_at_target_BER.mat'):
    params = scipy.io.loadmat(file_name)
    snr_thresholds = params['snr_at_target_BER'].flatten()
    return snr_thresholds
# 生成mcs_configs字典
def generate_mcs_configs(N_s, xi, T_send, Rb, BER, snr_thresholds):
    modulations = ['BPSK', 'QPSK', '8PSK']
    coding_rates = ['1/2', '1/3', '1/4']
    mcs_configs = {}
    index = 1
    for modulation in modulations:
        for coding_rate in coding_rates:
            if modulation == 'BPSK':
                bitnum_per = 1
            elif modulation == 'QPSK':
                bitnum_per = 2
            elif modulation == '8PSK':
                bitnum_per = 3
            else:
                raise ValueError(f"Unsupported modulation: {modulation}")
            
            threshold = snr_thresholds[index - 1] if index <= len(snr_thresholds) else float('inf')
            rate = (bitnum_per * eval(coding_rate) * N_s[index - 1] * (1 - BER)) / T_send[index - 1]
            mcs_configs[f'MCS{index}'] = {
                'modulation': modulation,
                'coding_rate': coding_rate,
                'threshold': threshold,
                'rate': rate
            }
            index += 1

    return mcs_configs
# 预测数据读入
def load_data(file_name):
    hss = np.load('./'+file_name+'/metrics.npy')
    preds = np.load('./'+file_name+'/pred.npy')
    trues = np.load('./'+file_name+'/true.npy')
    return trues, preds

def generate_channel_state(trues, preds, model_name):
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
    return origin_data, pred_data

# 定义调制编码方案的信噪比门限和传输速率
# 其中传输速率为 rate = bitnum_per*coding_rate*Rb*(1-BER)/T;
# 其中bitnum_per=log2(Mod)其中BPSK的Mod是2,QPSK是4,8PSK是8,Rb是matlab中的Rb,T是matlab中发送信号的持续时间
# mcs_configs = {
#     'MCS1': {'modulation': 'BPSK', 'coding_rate': '1/2', 'threshold': 11, 'rate': 1},
#     'MCS2': {'modulation': 'BPSK', 'coding_rate': '1/3', 'threshold': 6, 'rate': 1},
#     'MCS3': {'modulation': 'BPSK', 'coding_rate': '1/4', 'threshold': 5, 'rate': 1},
#     'MCS4': {'modulation': 'QPSK', 'coding_rate': '1/2', 'threshold': 30, 'rate': 1},
#     'MCS5': {'modulation': 'QPSK', 'coding_rate': '1/3', 'threshold': 25, 'rate': 1},
#     'MCS6': {'modulation': 'QPSK', 'coding_rate': '1/4', 'threshold': 10, 'rate': 1},
#     'MCS7': {'modulation': '8PSK', 'coding_rate': '1/2', 'threshold': float('inf'), 'rate': 1},
#     'MCS8': {'modulation': '8PSK', 'coding_rate': '1/3', 'threshold': float('inf'), 'rate': 1},
#     'MCS9': {'modulation': '8PSK', 'coding_rate': '1/4', 'threshold': float('inf'), 'rate': 1}
# }
# 读取MATLAB生成的参数
N_s, xi, T_send, Rb, BER = load_tx_params()
snr_thresholds = load_snr_thresholds() 
# 生成mcs_configs字典
mcs_configs = generate_mcs_configs(N_s, xi, T_send, Rb, BER, snr_thresholds)
# 定义目标误码率
target_ber = 10**-2

# 初始化种群
def initialize_population(size):
    population = []
    modulation_schemes = ['BPSK', 'QPSK', '8PSK']
    coding_rates = ['1/2','1/3', '1/4']
    for i in range(size):
        modulation = modulation_schemes[i % len(modulation_schemes)]
        coding_rate = coding_rates[(i // len(modulation_schemes)) % len(coding_rates)]
        individual = {
            'modulation': modulation,
            'coding_rate': coding_rate
        }
        population.append(individual)
    
    while len(population) < size:
        individual = {
            'modulation': random.choice(modulation_schemes),
            'coding_rate': random.choice(coding_rates)
        }
        population.append(individual)
    
    random.shuffle(population)  # 打乱初始种群顺序
    return population

# 计算适应度函数
def fitness(individual, snrs):
    total_rate = 0
    for snr in snrs:
        for mcs, config in mcs_configs.items():
            if config['modulation'] == individual['modulation'] and config['coding_rate'] == individual['coding_rate']:
                if snr >= config['threshold']: # 当信噪比大于设定的门限时，误码率会小于target_ber
                    total_rate += config['rate'] # !! rate为通信速率 !!
                    break
    return total_rate

# 选择操作
def select(population, fitnesses, K):
    selected_indices = np.argsort(fitnesses)[-K:]
    selected_indices = selected_indices.astype(int)  # 确保selected_indices为整数数组
    selected_population = [population[int(i)] for i in selected_indices]
    # 确保 selected_population 的长度为偶数
    if len(selected_population) % 2 != 0:
        selected_population.append(selected_population[np.random.randint(0, len(selected_population))])
    return selected_population

# 交叉操作
def crossover(parent1, parent2):
    child1 = {'modulation': parent1['modulation'], 'coding_rate': parent2['coding_rate']}
    child2 = {'modulation': parent2['modulation'], 'coding_rate': parent1['coding_rate']}
    return child1, child2

# 变异操作
def mutate(chromosome, mutation_rate=0.1):  # 增大变异率
    modulation_schemes = ['BPSK', 'QPSK', '8PSK']
    coding_rates = ['1/2','1/3', '1/4']
    if np.random.rand() < mutation_rate:
        chromosome['modulation'] = np.random.choice(modulation_schemes)
    if np.random.rand() < mutation_rate:
        chromosome['coding_rate'] = np.random.choice(coding_rates)
    return chromosome

# 主算法
def genetic_algorithm(snrs, max_cross_time, population_size=50, K=25):
    num_time_slots = len(snrs)
    population = initialize_population(population_size)
    
    cross_time = 0
    
    while cross_time < max_cross_time:
        fitnesses = np.array([fitness(chrom, snrs) for chrom in population], dtype=object)
        selected_population = select(population, fitnesses, K)
        next_generation = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            offspring1, offspring2 = crossover(parent1, parent2)
            next_generation.append(mutate(offspring1))
            next_generation.append(mutate(offspring2))
        population = next_generation
        cross_time += 1
        
        # 调试信息，打印每一代的最佳适应度
        # best_fitness = max(fitnesses)
        # print(f"Generation {cross_time}, Best Fitness: {best_fitness}")
    
    best_chromosome = max(population, key=lambda chrom: fitness(chrom, snrs))
    return best_chromosome

# 封装的主函数
def genetic_algorithm_main(file_name, max_cross_time):
    trues, preds = load_data(file_name)
    origin_data, channel_states = generate_channel_state(trues, preds, 'Informer-24')
    snrs = channel_states # 信道状态为预测的信噪比
    # 保存数据为 .mat 文件
    output_dir = './' + file_name
    save_path = os.path.join(output_dir, 'data.mat')
    scipy.io.savemat(save_path, {'origin_data': origin_data, 'pred_data': channel_states})
    print("Data saved to data.mat")
    
    best_modulation_schemes = []

    # 每24个点确定一种调制方式(取24个点中调制方式最多的那一种)
    for i in range(0, len(snrs), 24):
        window_snrs = snrs[i:i+24]
        if len(window_snrs) == 24:
            best_modulation_scheme = genetic_algorithm( window_snrs, max_cross_time)
            modulation_counter = Counter([f"{best_modulation_scheme['modulation']}_{best_modulation_scheme['coding_rate']}"])
            most_common_modulation = modulation_counter.most_common(1)[0][0]
            best_modulation_schemes.append(most_common_modulation)
    
    return best_modulation_schemes

# 示例调用
if __name__ == "__main__":
    file_name = 'inpulse_informer_24'
    max_cross_time = 100  # 设定的交叉变异轮数
    best_modulation_schemes = genetic_algorithm_main(file_name, max_cross_time)
    print("最佳调制和编码模式:", best_modulation_schemes)
