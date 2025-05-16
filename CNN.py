import torch
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device("cuda:0")

def preprocess_data(train_data, train_labels):

    batch_size, N, M = train_data.size()  

    processed_data = train_data[:, 0:1, :].to(device) 
    
    processed_labels = []
    _, L, M = train_labels.size()  
    for i in range(batch_size):
        label_matrix = train_labels[i]  

        last_column = label_matrix[:, -1:]  

        processed_label = last_column[1:, :]  
        processed_labels.append(processed_label.T)  
    processed_labels = torch.stack(processed_labels, dim=0).to(device)

    return processed_data, processed_labels

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, time_step, output_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flattened_size = 64 * (time_step // 2) 

        self.fc1 = nn.Linear(self.flattened_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  
        
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.pool(x)  
        

        x = x.view(x.size(0), -1)  
        
        x = self.fc1(x) 
        
        return x
    
def train_model(params,train_data, train_labels):

    train_data = torch.tensor(train_data, dtype=torch.float32).to(device) 
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device) 

    train_data, train_labels=preprocess_data(train_data, train_labels)

    hidden_size = params['hidden_layers']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']
    
    batch_size, _, time_step = train_data.size()  
    _, _, L = train_labels.size()  

    output_size = L  

    model = CNN(time_step,output_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    avg_loss=0.0
    for epoch in range(epochs):
        model.train()
        
        for i in range(batch_size):
            batch_data = train_data[i].to(device)  
            batch_labels = train_labels[i].to(device) 

            optimizer.zero_grad()

            output = model(batch_data)  
            
            loss = criterion(output, batch_labels)
            
            loss.backward()
            
            optimizer.step()
            avg_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    avg_loss /= epochs 
    model_name = model.__class__.__name__
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f"{model_name}.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True) 
    torch.save(model, model_save_path)
    
    return avg_loss, model_save_path
