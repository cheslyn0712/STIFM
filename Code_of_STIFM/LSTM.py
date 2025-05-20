import torch
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device("cuda:0")

def preprocess_data(train_data, train_labels):
    """
    处理 train_data 和 train_labels，使其符合训练的需求
    train_data: 原始数据，形状为 (batch_size, N, M)
    train_labels: 原始标签，形状为 (batch_size, L, M)
    
    返回处理后的 train_data 和 train_labels
    """
    # 处理 train_data: 提取每个 batch 的第 (N+1)//2 个点，形状变为 (batch_size, 1, M)
    batch_size, N, M = train_data.size()  # 使用 .size() 获取维度

    processed_data = train_data[:, 0:1, :].to(device) 
    
    # 处理 train_labels: 逐个 batch 提取最后一列，并去除第一行
    processed_labels = []
    _, L, M = train_labels.size()  # 获取 train_labels 的维度 (batch_size, L, M)
    
    for i in range(batch_size):
        # 获取当前 batch 的 L*M 矩阵
        label_matrix = train_labels[i]  # 形状为 (L, M)

        # 提取最后一列，形状为 (L, 1)
        last_column = label_matrix[:, -1:]  # 形状为 (L, 1)

        # 去除第一行，形状变为 (L-1, 1)
        processed_label = last_column[1:, :]  # 提取后面的 L-1 行，形状为 (L-1, 1)

        # 重新调整形状为 (1, L-1)
        processed_labels.append(processed_label.T)  # 形状为 (1, L-1)

    # 将所有 processed_labels 拼接为 (batch_size, 1, L-1)
    processed_labels = torch.stack(processed_labels, dim=0).to(device)

    return processed_data, processed_labels

class LSTMModel(nn.Module): 
    def __init__(self, time_step, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        # 输入的 input_size 是 1，因为每个时间步只有一个特征
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        # 全连接层，将 LSTM 输出的 hidden_size 映射到目标输出尺寸
        self.fc = nn.Linear(hidden_size, output_size)  # output_size 应该是 L-1

    def forward(self, x):
        # 输入 x 的形状是 (1, time_step)，即每个 batch 是 1，time_step 是时间序列长度
        x = x.unsqueeze(2)  # 扩展为 (1, time_step, 1)，即加上特征维度
        
        # LSTM 层处理，输出 lstm_out 形状为 (1, time_step, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 取最后一个时间步的输出，形状是 (1, hidden_size)
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出

        # 输出形状是 (1, L-1)，即通过全连接层映射到目标时间序列长度
        return out

# 定义训练过程
def train_model(params,train_data, train_labels):
    # 从params中获取超参数

    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)  # 转换并移到 device
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)  # 转换并移到 device

    train_data, train_labels=preprocess_data(train_data, train_labels)

    hidden_size = params['hidden_layers']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']
    
    # 获取train_data和train_labels的维度
    batch_size, _, time_step = train_data.size()  # 输入数据维度 (batch_size, time_step, input_size)
    _, _, L = train_labels.size()  # 标签数据维度 (batch_size, L, M)

    output_size = L   # 期望的输出维度

    # 创建LSTM模型
    model = LSTMModel(time_step, hidden_size, output_size).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 逐batch训练
    avg_loss=0.0
    for epoch in range(epochs):
        model.train()
        
        for i in range(batch_size):
            # 获取当前的训练数据和标签
            batch_data = train_data[i].to(device)  # (time_step, 1)
            batch_labels = train_labels[i].to(device)  # ( 1, L-1)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            output = model(batch_data)  # 输出形状为 (time_step, L-1)
            
            # 计算损失
            loss = criterion(output, batch_labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            avg_loss += loss.item()
        
        # 打印每个epoch的损失
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    avg_loss /= epochs  # 计算平均损失
    # 保存模型
    model_name = model.__class__.__name__
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"{model_name}.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # 如果目录不存在，则创建
    torch.save(model, model_save_path)
    
    return avg_loss, model_save_path
