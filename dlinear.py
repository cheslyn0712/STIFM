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

class DLinear(nn.Module):
    def __init__(self, input_size, window_size, output_size):
        super(DLinear, self).__init__()
        self.window_size = window_size
        self.trend_linear = nn.Linear(input_size, output_size)  
        self.seasonal_linear = nn.Linear(input_size, output_size) 

    def forward(self, x):
        batch_size, seq_len = x.size()

        trend = torch.zeros_like(x)

        for i in range(batch_size):  
            for j in range(seq_len):
                if j < self.window_size:
                    trend[i, j] = torch.mean(x[i, :j+1])
                else:
                    trend[i, j] = torch.mean(x[i, j-self.window_size+1:j+1])

        seasonal = x - trend

        trend_output = self.trend_linear(trend)
        seasonal_output = self.seasonal_linear(seasonal)

        combined_output = trend_output + seasonal_output

        return combined_output

def train_model(params,train_data, train_labels):

    train_data = torch.tensor(train_data, dtype=torch.float32).to(device) 
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device) 
    train_data, train_labels=preprocess_data(train_data, train_labels)

    window_size = params['window_size']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']
    
    batch_size, _, time_step = train_data.size()  
    _, _, L = train_labels.size()  

    output_size = L   

    model = DLinear(time_step,window_size, output_size).to(device)
    
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
