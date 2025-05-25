import os
import numpy as np
from scipy import io
import torch
import time
import time
start_time = time.time()

#  train
mat_dir = '/data/yangchangfan/DiffusionPDE/data/training/TE_heat/' 
output_file = '/home/yangchangfan/CODE/neuraloperator-main/neuralop/data/datasets/data/TE_heat_train_128.pt'  
os.makedirs(os.path.dirname(output_file), exist_ok=True)


num_samples = 10000         
shape = (128, 128)       


x_data = np.zeros((num_samples, *shape), dtype=np.float64)
y_data = np.zeros((num_samples, 3, *shape), dtype=np.float64)



for idx in range(num_samples):
    mater_path = os.path.join(mat_dir, f'mater/{idx+1}.mat')
    mater_data = io.loadmat(mater_path)['mater'].astype(np.float64)
    
    mater_in = (mater_data >= 1e11) & (mater_data <= 3e11)
    mater_out = (mater_data >= 10) & (mater_data <= 20)

    normal_datamater = np.where(mater_in, (mater_data - 1e11) / (3e11 - 1e11) * 0.8 + 0.1, (mater_data - 10) / (20 - 10) * 0.8 - 0.9)
    # 边上和内部设置为parm.Sigma_Si_coef(0.1,0.9)，其他设置为normal_Pho_Al(-0.9,-0.1)

    x_data[idx] = normal_datamater


# output material  Re(Ez) Im(Ez) T
    # load max_abs_Ez
max_abs_Ez_path = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/Ez/max_abs_Ez.mat"
max_abs_Ez = io.loadmat(max_abs_Ez_path)['max_abs_Ez']

print(max_abs_Ez)

# load T
range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/T/range_allT.mat"
range_allT = io.loadmat(range_allT_paths)['range_allT']

max_T = range_allT[0,1]
min_T = range_allT[0,0]


for idx in range(num_samples):

    Ez_path = os.path.join(mat_dir, f'Ez/{idx+1}.mat')
    Ez_data = io.loadmat(Ez_path)['export_Ez'].astype(np.complex64)
    
    Ez_normalized = Ez_data / max_abs_Ez * 0.9  # 保持相位不变  [0,0.9]
    real_Ez_normalized = np.real(Ez_normalized)
    imag_Ez_normalized = np.imag(Ez_normalized)

    y_data[idx, 0] = real_Ez_normalized.astype(np.float64)
    y_data[idx, 1] = imag_Ez_normalized.astype(np.float64)

    # T 
    T_path = os.path.join(mat_dir, f'T/{idx+1}.mat')
    T_data = io.loadmat(T_path)['export_T'].astype(np.float64)
    T_normalized = (T_data - min_T) / (max_T - min_T) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 2] = T_normalized.astype(np.float64)

    if idx % 200 == 0:
        print(f"Saved combined array for index {idx} to {output_file }")
        print("Min:", y_data.min(), "Max:", y_data.max())

# 转换为PyTorch张量
x_tensor = torch.from_numpy(x_data)
y_tensor = torch.from_numpy(y_data)

# 保存为pt文件
torch.save({'x': x_tensor, 'y': y_tensor}, output_file)

print(f"数据已成功保存为 {output_file}")
print(f"输入形状: {x_tensor.shape}")
print(f"输出形状: {y_tensor.shape}")

print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    




#  test  归一化
mat_dir = '/data/yangchangfan/DiffusionPDE/data/testing/TE_heat/' 
output_file = '/home/yangchangfan/CODE/neuraloperator-main/neuralop/data/datasets/data/TE_heat_test_128.pt'  
os.makedirs(os.path.dirname(output_file), exist_ok=True)


num_samples = 1000        
shape = (128, 128)       


x_data = np.zeros((num_samples, *shape), dtype=np.float64)
y_data = np.zeros((num_samples, 3, *shape), dtype=np.float64)



for idx in range(num_samples):
    mater_path = os.path.join(mat_dir, f'mater/{idx+10001}.mat')
    mater_data = io.loadmat(mater_path)['mater'].astype(np.float64)
    
    mater_in = (mater_data >= 1e11) & (mater_data <= 3e11)
    mater_out = (mater_data >= 10) & (mater_data <= 20)

    normal_datamater = np.where(mater_in, (mater_data - 1e11) / (3e11 - 1e11) * 0.8 + 0.1, (mater_data - 10) / (20 - 10) * 0.8 - 0.9)
    # 边上和内部设置为parm.Sigma_Si_coef(0.1,0.9)，其他设置为normal_Pho_Al(-0.9,-0.1)

    x_data[idx] = normal_datamater


# output material  Re(Ez) Im(Ez) T
    # load max_abs_Ez
max_abs_Ez_path = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/Ez/max_abs_Ez.mat"
max_abs_Ez = io.loadmat(max_abs_Ez_path)['max_abs_Ez']

print(max_abs_Ez)

# load T
range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/T/range_allT.mat"
range_allT = io.loadmat(range_allT_paths)['range_allT']

max_T = range_allT[0,1]
min_T = range_allT[0,0]


for idx in range(num_samples):

    Ez_path = os.path.join(mat_dir, f'Ez/{idx+10001}.mat')
    Ez_data = io.loadmat(Ez_path)['export_Ez'].astype(np.complex64)
    
    Ez_normalized = Ez_data / max_abs_Ez * 0.9  # 保持相位不变  [0,0.9]
    real_Ez_normalized = np.real(Ez_normalized)
    imag_Ez_normalized = np.imag(Ez_normalized)

    y_data[idx, 0] = real_Ez_normalized.astype(np.float64)
    y_data[idx, 1] = imag_Ez_normalized.astype(np.float64)

    # T 
    T_path = os.path.join(mat_dir, f'T/{idx+10001}.mat')
    T_data = io.loadmat(T_path)['export_T'].astype(np.float64)
    T_normalized = (T_data - min_T) / (max_T - min_T) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 2] = T_normalized.astype(np.float64)


    if idx % 200 == 0:
        print(f"Saved combined array for index {idx} to {output_file }")
        print("Min:", y_data.min(), "Max:", y_data.max())

# 转换为PyTorch张量
x_tensor = torch.from_numpy(x_data)
y_tensor = torch.from_numpy(y_data)

# 保存为pt文件
torch.save({'x': x_tensor, 'y': y_tensor}, output_file)

print(f"数据已成功保存为 {output_file}")
print(f"输入形状: {x_tensor.shape}")
print(f"输出形状: {y_tensor.shape}")

print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    




