#本代码可以作为项目的main文件
#基本结构如下
import os
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

params = {
    'center_lat': 10,  
    'center_lon':115, 
    'sampling_window':0,
    't_span': 50,
    'test_span':10, 
    't_train': 20131001,  
    'L': 15,  
    'M': 30,  
    't_gap':5,
    'data_directory': str(Path(__file__).parent.parent / "SST Data" / "SST"),
    'window_size': 30,
    'hidden_layers': 128,
    'epochs': 500,  
    'learning_rate': 0.001,
    'lat_lon_window_size': 2,
    'lambda_diag':20,
    'drop_out':0.1,
    'batch_size':50,
    'model_name':'STIFM-V1'
}


def data_extract(params):

    from extract import integration

    train_data,train_labels,test_data,test_labels=integration(params)

    return train_data,train_labels,test_data,test_labels

#模型训练

def model_train(model_name,train_data,train_labels, parameters):

    module = __import__(model_name) 
    

    avg_loss, model_save_path= module.train_model(parameters, train_data, train_labels)

    return avg_loss, model_save_path

#模型测试
def model_test(model_path,test_data,test_labels,parameters):
 
    from test import model_test

    persist,predict,label=model_test(model_path,test_data,test_labels,parameters)
    
    return persist,predict,label



def result_process(persistent_data, predict_data, label_data):

    from result_process import plot_process

    plot_process(persistent_data, predict_data, label_data)
    return
train_data,train_labels,test_data,test_labels=data_extract(params) 

S=train_data.shape[0]

persist_all=[]
predict_all=[]
label_all=[]

for i in range(S):
    current_train_data=train_data[i]
    current_train_labels=train_labels[i]
    current_test_data=test_data[i]
    current_test_labels=test_labels[i]
    
    avg_loss,model_save_path=model_train(params['model_name'],current_train_data,current_train_labels, params)

    print("The {:,}th Training average Loss is".format(i+1),avg_loss)

    persist,predict,label=model_test(model_save_path,current_test_data,current_test_labels,params)
    persist_all.append(persist)
    predict_all.append(predict)
    label_all.append(label)

persist_all=np.array(persist_all)
predict_all=np.array(predict_all)
label_all=np.array(label_all)


result_process(persist_all,predict_all,label_all)