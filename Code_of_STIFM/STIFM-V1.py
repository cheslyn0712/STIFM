import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DLinear(nn.Module):
    def __init__(self, time_steps, window_size):
        super(DLinear, self).__init__()
        self.window_size = window_size
        self.trend_linear = nn.Linear(time_steps, time_steps)    
        self.seasonal_linear = nn.Linear(time_steps, time_steps) 

    def forward(self, x):

        batch_size, num_points, time_steps = x.size()
        trend = torch.zeros_like(x)

        # 计算趋势性分解
        for j in range(time_steps):
            if j < self.window_size:
                trend[:, :, j] = torch.mean(x[:, :, :j+1], dim=2)
            else:
                trend[:, :, j] = torch.mean(x[:, :, j-self.window_size+1:j+1], dim=2)

        seasonal = x - trend

        trend_output = self.trend_linear(trend)
        seasonal_output = self.seasonal_linear(seasonal)


        combined_output = trend_output + seasonal_output
        return combined_output

class DNN(nn.Module):
    def __init__(self, num_points, hidden_size, output_points, drop_out, num_heads=1):
        super(DNN, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.fc1 = nn.Linear(num_points, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

  
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.attn_out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(drop_out)


        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)


        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.ln4 = nn.LayerNorm(hidden_size // 2)


        self.fc5 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.ln5 = nn.LayerNorm(hidden_size // 2)


        self.fc6 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.ln6 = nn.LayerNorm(hidden_size // 2)


        self.fc_output = nn.Linear(hidden_size // 2, output_points)

        self.saved_attention_weights = None

    def forward(self, x):
        
        x = x.permute(0, 2, 1)  

        x1 = self.fc1(x)          
        x1 = self.ln1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        x2 = self.fc2(x1)          
        x2 = self.ln2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)


        Q = self.query(x2)        
        K = self.key(x2)           
        V = self.value(x2)         


        batch_size, seq_length, hidden_size = Q.size()
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale 
        attention_weights = F.softmax(scores, dim=-1)              
        self.saved_attention_weights = attention_weights
        attention_weights = F.dropout(attention_weights, p=self.dropout.p, training=self.training)

        attention_output = torch.matmul(attention_weights, V)     

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)  

        attention_output = self.attn_out(attention_output)        
        attention_output = self.attn_dropout(attention_output)

        x2 = x2 + attention_output  

     
        x3 = self.fc3(x2)         
        x3 = self.ln3(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

  
        x4 = self.fc4(x3)         
        x4 = self.ln4(x4)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = x4 + x3               


        x5 = self.fc5(x4)        
        x5 = self.ln5(x5)
        x5 = self.relu(x5)
        x5 = self.dropout(x5)

     
        x6 = self.fc6(x5)        
        x6 = self.ln6(x6)
        x6 = self.relu(x6)
        x6 = self.dropout(x6)
        x6 = x6 + x5                

        output = self.fc_output(x6)  


        output = output.permute(0, 2, 1)

        return output

class SeaTemperatureModel(nn.Module):

    def __init__(self, time_steps, num_points, hidden_size, output_points, window_size, drop_out):
        super(SeaTemperatureModel, self).__init__()
        self.dlinear = DLinear(time_steps, window_size)
        self.dnn = DNN(num_points, hidden_size, output_points, drop_out)

    def forward(self, x):
    
        
        batch_size, num_points, time_steps = x.size()
        y = x[:, 0, -1]*0.8 
        
     
        x_linear = self.dlinear(x)  
        x_dnn = self.dnn(x_linear)   

        y_expanded = y.unsqueeze(1).unsqueeze(2).repeat(1,x_dnn.shape[1],time_steps)  
        
        output = x_dnn+y_expanded  
        return output

def compute_rmse(outputs, targets):
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs, dtype=torch.float32).to(device)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32).to(device)

    mse = torch.mean((outputs - targets) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

def anti_diagonal_loss(outputs):

    batch_size, L, M = outputs.size()
    
    device = outputs.device
    indices = torch.arange(L, device=device).unsqueeze(1).repeat(1, M) + torch.arange(M, device=device).unsqueeze(0).repeat(L, 1)
    
    indices = indices.unsqueeze(0).repeat(batch_size, 1, 1)
    
    unique_k = torch.unique(indices)
    
    loss = 0.0
    for k in unique_k:
        mask = (indices == k) 
        selected = outputs[mask].view(batch_size, -1)  

        mean = selected.mean(dim=1, keepdim=True)
        var = ((selected - mean) ** 2).mean(dim=1) 
        loss += var.mean()  
    
    loss = loss / unique_k.size(0)
    return loss


def train_model(params, train_data, train_labels):

    # 定义模型
    model = SeaTemperatureModel(
        time_steps=params['M'],
        num_points=train_data.shape[1],
        hidden_size=params['hidden_layers'],
        output_points=params['L'],
        window_size=params['window_size'],
        drop_out=params['drop_out']
    )

    model.to(device)

    model_name = model.__class__.__name__
    model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{model_name}.pth")

    mse_criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    avg_loss = 0.0
    avg_rmse = 0.0

    lambda_diag = params['lambda_diag'] 
    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0.0
        total_rmse = 0.0

        for batch_data, batch_labels in train_loader:

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            
            mse_loss = mse_criterion(outputs, batch_labels)
            
            diag_loss = anti_diagonal_loss(outputs)
            
            loss = mse_loss + lambda_diag * diag_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_current = batch_data.size(0)
            total_loss += loss.item() * batch_size_current
            rmse = compute_rmse(outputs.detach(), batch_labels.detach())
            total_rmse += rmse * batch_size_current


        avg_loss = total_loss / len(train_loader.dataset)
        avg_rmse = total_rmse / len(train_loader.dataset)

        print(f'Epoch [{epoch+1}/{params["epochs"]}], Loss: {avg_loss:.4f}, RMSE: {avg_rmse:.4f}')

    torch.save(model, model_save_path)
    print(f"model is saved at: {model_save_path}")

    return avg_loss, model_save_path