import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import os
import numpy as np

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

class XGBoostModel:
    def __init__(self, time_step, hidden_size, output_size):
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = None  

    def train(self, batch_data, batch_labels):

        X_batch = batch_data.detach().cpu().numpy()
        y_batch = batch_labels.detach().cpu().numpy()

        params = {
            'objective': 'reg:squarederror',   
            'max_depth': 6,                   
            'learning_rate': 0.1,              
            'n_estimators': 3,               
            'eval_metric': 'rmse',            
            'silent': 1,                       
            'num_round': 50                     
        }

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            n_estimators=params['num_round']
        )

        self.model.fit(X_batch, y_batch)

        y_pred = self.model.predict(X_batch)
        
        batch_loss = ((y_pred - y_batch) ** 2).mean()  
        
        return batch_loss  

    def save_model(self, model_save_path):

        if self.model is not None:
            self.model.save_model(model_save_path)
        else:
            print("Model is not trained yet!")
    
def train_model(params, train_data, train_labels):
    learning_rate = params['learning_rate']
    epochs = params['epochs']

    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)  
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)  

    train_data, train_labels=preprocess_data(train_data, train_labels)

    
    batch_size, _, time_step = train_data.size()  
    _, _, L = train_labels.size() 

    output_size = L   


    model = XGBoostModel(time_step, params['hidden_layers'], output_size)

    avg_loss = 0.0
    for i in range(batch_size):
   
        batch_data = train_data[i].to(device)  
        batch_labels = train_labels[i].to(device)  

        batch_loss = model.train(batch_data, batch_labels)

        avg_loss += batch_loss

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'xgboost_model.json')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path) 

    print("Training Average Loss",avg_loss)

    return avg_loss, model_save_path