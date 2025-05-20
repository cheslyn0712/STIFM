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
        """
        :param x: 输入张量，形状为 (batch_size, num_points, time_steps)
        :return: 输出张量，形状为 (batch_size, num_points, time_steps)
        """
        batch_size, num_points, time_steps = x.size()
        trend = torch.zeros_like(x)

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
        
        self.fc_output = nn.Linear(hidden_size, output_points)

    def forward(self, x):

        x = x.permute(0, 2, 1)  
        x1 = self.fc1(x)         
    
        output = self.fc_output(x1) 

        output = output.permute(0, 2, 1)

        return output

class SeaTemperatureModel(nn.Module):
    def __init__(self, time_steps, num_points, hidden_size, output_points, window_size, drop_out):
        super(SeaTemperatureModel, self).__init__()
        self.dlinear = DLinear(time_steps, window_size)
        self.dnn = DNN(num_points, hidden_size, output_points, drop_out)

    def forward(self, x):

        x = self.dlinear(x) 
        x = self.dnn(x)     
        return x

def compute_rmse(outputs, targets):
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs, dtype=torch.float32).to(device)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32).to(device)

    mse = torch.mean((outputs - targets) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

def train_model(params, train_data, train_labels):

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

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    avg_loss = 0.0
    avg_rmse = 0.0

    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0.0
        total_rmse = 0.0

        for batch_data, batch_labels in train_loader:

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

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
    print(f"Modle saved at: {model_save_path}")

    return avg_loss, model_save_path