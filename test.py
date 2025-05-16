import torch
import os
import json
import numpy as np
import csv
import xgboost as xgb

def compute_rmse(outputs, targets):

    outputs = torch.tensor(outputs, dtype=torch.float32) if not isinstance(outputs, torch.Tensor) else outputs
    targets = torch.tensor(targets, dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets


    mse = torch.mean((outputs - targets) ** 2)  
    rmse = torch.sqrt(mse)  
    return rmse.item()


def compute_elementwise_rmse(outputs, targets):

    outputs = torch.tensor(outputs, dtype=torch.float32) if not isinstance(outputs, torch.Tensor) else outputs
    targets = torch.tensor(targets, dtype=torch.float32) if not isinstance(targets, torch.Tensor) else targets


    error = outputs - targets

    elementwise_rmse = torch.sqrt(error ** 2)
    
    return elementwise_rmse

def compute_elementwise_rmse(outputs, targets):

    error = outputs - targets

    elementwise_rmse = torch.sqrt(error ** 2)
    return elementwise_rmse

def persistent_model(test_label, target_label):

    rmse_batch = []
    elementwise_rmse_batch = []


    batch_size = test_label.size(0)

    for i in range(batch_size):
        pred = test_label[i]
        true = target_label[i]
        

        rmse_1 = compute_rmse(pred, true)
        rmse_batch.append(rmse_1)  


        elementwise_rmse_1 = compute_elementwise_rmse(pred, true)
        elementwise_rmse_batch.append(elementwise_rmse_1.cpu().numpy()) 

    return rmse_batch, elementwise_rmse_batch

def model_test(model_path, test_data, test_labels, parameters):
    
    if parameters['model_name']=='XGboost':
        model=xgb.XGBRegressor()  
        model.load_model(model_path)
    else:
        model = torch.load(model_path)  
        model.eval()  
        model.to('cpu')
    
    if parameters['model_name']!='STIFM-V1' and parameters['model_name']!='STIFM':
        test_data=test_data[:,0:1,:]

    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to('cpu')
    test_labels = torch.tensor(test_labels, dtype=torch.float32).to('cpu')

    L = test_labels.shape[1]

    print(test_labels.shape)
    last_elements = test_labels[:, 0].unsqueeze(1).repeat(1, L - 1)  

    modified_labels = test_labels[:, 1:]  

    predict_result = []
    true_data = []
    persistent_data=[]


    for i in range(test_data_tensor.size(0)):

        batch_data = test_data_tensor[i]
        batch_labels = modified_labels[i]
        last_data=last_elements[i]

        with torch.no_grad():
            if parameters['model_name']=='XGboost':
                output = model.predict(batch_data)
            else:
                if parameters['model_name']=='STIFM-V1' or parameters['model_name']=='STIFM'  :
                    batch_data=batch_data.unsqueeze(0)
                output = model(batch_data)  

            if parameters['model_name']=='STIFM-V1' or parameters['model_name']=='STIFM' :
                output = output[0,:,-1]
                output = output[1:]
            else:
                output=output[0,:]

            if parameters['model_name']!='XGboost':
                predict_result.append(output.cpu().numpy()) 
            else:
                predict_result.append(output)
                
            true_data.append(batch_labels.cpu().numpy())  
            persistent_data.append(last_data.cpu().numpy())

    return persistent_data,predict_result,true_data
