import numpy as np
import matplotlib.pyplot as plt
import csv
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def compute_rmse(predictions, labels):

    return np.sqrt(np.mean((predictions - labels) ** 2))

def calculate_rmse_per_time_step(persistent_data, predict_data, label_data):

    _, _, time_steps = persistent_data.shape
    
    rmse_persistent = np.zeros(time_steps)
    rmse_predict = np.zeros(time_steps)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'data')
    output_file = os.path.join(output_dir, 'rmse_output.csv')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time Step', 'Persistent RMSE', 'Prediction RMSE'])
        
        for i in range(time_steps):

            persistent_slice = persistent_data[:, :, i]
            predict_slice = predict_data[:, :, i]
            label_slice = label_data[:, :, i]
            
            persistent_flat = persistent_slice.flatten()
            predict_flat = predict_slice.flatten()
            label_flat = label_slice.flatten()
            
            rmse_persistent[i] = compute_rmse(persistent_flat, label_flat)
            rmse_predict[i] = compute_rmse(predict_flat, label_flat)
            
            writer.writerow([i + 1, rmse_persistent[i], rmse_predict[i]])

    
    print(f"RMSE data has been written to {output_file}")
    
    mean_rmse_persistent = np.mean(rmse_persistent)
    mean_rmse_predict = np.mean(rmse_predict)
    
    return rmse_persistent, rmse_predict, mean_rmse_persistent, mean_rmse_predict

def plot_rmse(time_step_rmse_persistent, time_step_rmse_predict):

    time_steps = len(time_step_rmse_persistent)
    x = np.arange(1, time_steps + 1)
    
    plt.figure(figsize=(14, 8))
    
    plt.plot(x, time_step_rmse_persistent, label='Persistent RMSE', color='blue', marker='o')
    
    plt.plot(x, time_step_rmse_predict, label='Prediction RMSE', color='orange', marker='x')
    
    for i in range(time_steps):
        plt.text(x[i], time_step_rmse_persistent[i], str(i + 1), color='blue', fontsize=8, ha='center', va='bottom')
        plt.text(x[i], time_step_rmse_predict[i], str(i + 1), color='orange', fontsize=8, ha='center', va='bottom')
    
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison between Persistent and Prediction over Time Steps')
    
    plt.ylim(0, 4)
    plt.yticks(np.arange(0, 4.1, 0.2))  
    
    plt.xticks(x)
    
    plt.legend(loc='best')
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.show()

def plot_process(persistent_data, predict_data, label_data):

    if not (persistent_data.shape == predict_data.shape == label_data.shape):
        raise ValueError("All input data must have the same shape.")
    
    rmse_persistent, rmse_predict, mean_rmse_persistent, mean_rmse_predict = calculate_rmse_per_time_step(persistent_data, predict_data, label_data)
    
    print(f"Overall Persistent RMSE Mean: {mean_rmse_persistent:.4f}")
    print(f"Overall Prediction RMSE Mean: {mean_rmse_predict:.4f}")
    
    plot_rmse(rmse_persistent, rmse_predict)