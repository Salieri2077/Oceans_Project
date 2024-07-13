# from locale import nl_langinfo
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.gru = nn.GRU(configs.enc_in, configs.d_model, batch_first=True, dropout=configs.dropout)
        self.linear = nn.Linear(configs.d_model, configs.pred_len*configs.c_out)

    def forward(self, x):
        # x shape: [batch_size, seq_length, features]
        gru_out, _ = self.gru(x)
        # 只取序列的最后一个时间点的输出
        last_time_step = gru_out[:, -1, :]
        output = self.linear(last_time_step)
        # 将output的形状转换为[batch_size, pred_len, c_out]
        output = output.reshape(-1, self.pred_len, self.c_out)
        return output