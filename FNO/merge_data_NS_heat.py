import os
import numpy as np
import scipy.io as io
import torch
import time
import time
start_time = time.time()

#  train
# mat_dir = '/data/yangchangfan/DiffusionPDE/data/training/NS_heat/'
# output_file_train = '/home/yangchangfan/CODE/neuraloperator-main/neuralop/data/datasets/data/NS_heat_train_128.pt'
# os.makedirs(os.path.dirname(output_file_train), exist_ok=True)
#
#
# num_samples_train = 10000
# shape = (128, 128)
#
# x_data = np.zeros((num_samples_train, *shape), dtype=np.float64)
# y_data = np.zeros((num_samples_train, 3, *shape), dtype=np.float64)


# input Q_heat
# load max_Q_heat  min_Q_heat
range_allQ_heat_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/Q_heat/range_allQ_heat.mat"
range_allQ_heat = io.loadmat(range_allQ_heat_paths)['range_allQ_heat']

max_Q_heat = range_allQ_heat[0,1]
min_Q_heat = range_allQ_heat[0,0]

# output  u_u, u_v, T
# load max_u_u min_u_u
range_allu_u_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_u/range_allu_u.mat"
range_allu_u = io.loadmat(range_allu_u_paths)['range_allu_u']

max_u_u = range_allu_u[0,1]
min_u_u = range_allu_u[0,0]

# load max_u_v min_u_v
range_allu_v_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_v/range_allu_v.mat"
range_allu_v = io.loadmat(range_allu_v_paths)['range_allu_v']

max_u_v = range_allu_v[0,1]
min_u_v = range_allu_v[0,0]


# load max_T min_T
range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/T/range_allT.mat"
range_allT = io.loadmat(range_allT_paths)['range_allT']

max_T = range_allT[0,1]
min_T = range_allT[0,0]


# for idx in range(num_samples_train):
#     # Q_heat
#     path_Q_heat = os.path.join(f"/data/yangchangfan/DiffusionPDE/data/training/NS_heat/Q_heat/", f'{idx+1}.mat')
#     Q_heat = io.loadmat(path_Q_heat)['export_Q_heat']
#     Q_heat_normalized = (Q_heat - min_Q_heat) / (max_Q_heat - min_Q_heat) * 1.8 - 0.9 # [-0.9,0.9]
#
#     x_data[idx] = Q_heat_normalized
#
#
#     # u_u
#     path_u_u = os.path.join(f"/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_u/", f'{idx+1}.mat')
#     u_u = io.loadmat(path_u_u)['export_u_u']
#     u_u_normalized = (u_u - min_u_u) / (max_u_u - min_u_u) * 1.8 - 0.9 # [-0.9,0.9]
#
#     # u_v
#     path_u_v = os.path.join(f"/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_v/", f'{idx+1}.mat')
#     u_v = io.loadmat(path_u_v)['export_u_v']
#     u_v_normalized = (u_v - min_u_v) / (max_u_v - min_u_v) * 1.8 - 0.9 # [-0.9,0.9]
#
#     y_data[idx, 0] = u_u_normalized.astype(np.float64)
#     y_data[idx, 1] = u_v_normalized.astype(np.float64)
#
#
#     # T
#     path_T = os.path.join(f"/data/yangchangfan/DiffusionPDE/data/training/NS_heat/T/", f'{idx+1}.mat')
#     T = io.loadmat(path_T)['export_T']
#     T_normalized = (T - min_T) / (max_T - min_T) * 1.8 - 0.9 # [-0.9,0.9]
#
#     y_data[idx, 2] = T_normalized.astype(np.float64)
#
#
#     if idx % 200 == 0:
#         print("train: Min_x:", x_data[idx].min(), " Max_x:", x_data[idx].max())
#         print("train: Min_y:", y_data[idx,:,:].min(), "Max_y:", y_data[idx,:,:].max())
#
#
# # 转换为PyTorch张量
# x_tensor = torch.from_numpy(x_data)
# y_tensor = torch.from_numpy(y_data)
#
# # 保存为pt文件
# torch.save({'x': x_tensor, 'y': y_tensor}, output_file_train)
#
# print(f"数据已成功保存为 {output_file_train}")
# print(f"输入形状: {x_tensor.shape}")
# print(f"输出形状: {y_tensor.shape}")
#
# print("Finished processing all files.")
# print(f"运行时间: {time.time() - start_time} 秒")




#  test  归一化
output_file_test = './NS_heat_test_128.pt'
num_samples_test = 1000        
shape = (128, 128)       


x_data = np.zeros((num_samples_test, *shape), dtype=np.float64)
y_data = np.zeros((num_samples_test, 3, *shape), dtype=np.float64)


for idx in range(num_samples_test):
    # Q_heat 
    path_Q_heat = os.path.join(f"/data/yangchangfan/DiffusionPDE/data/testing/NS_heat/Q_heat/", f'{idx+10001}.mat')
    Q_heat = io.loadmat(path_Q_heat)['export_Q_heat']
    Q_heat_normalized = (Q_heat - min_Q_heat) / (max_Q_heat - min_Q_heat) * 1.8 - 0.9 # [-0.9,0.9]

    x_data[idx] = Q_heat_normalized


    # u_u
    path_u_u = os.path.join(f"/data/yangchangfan/DiffusionPDE/data/testing/NS_heat/u_u/", f'{idx+10001}.mat')
    u_u = io.loadmat(path_u_u)['export_u_u']
    u_u_normalized = (u_u - min_u_u) / (max_u_u - min_u_u) * 1.8 - 0.9 # [-0.9,0.9]

    # u_v
    path_u_v = os.path.join(f"/data/yangchangfan/DiffusionPDE/data/testing/NS_heat/u_v/", f'{idx+10001}.mat')
    u_v = io.loadmat(path_u_v)['export_u_v']
    u_v_normalized = (u_v - min_u_v) / (max_u_v - min_u_v) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 0] = u_u_normalized.astype(np.float64)
    y_data[idx, 1] = u_v_normalized.astype(np.float64)


    # T 
    path_T = os.path.join(f"/data/yangchangfan/DiffusionPDE/data/testing/NS_heat/T/", f'{idx+10001}.mat')
    T = io.loadmat(path_T)['export_T']
    T_normalized = (T - min_T) / (max_T - min_T) * 1.8 - 0.9 # [-0.9,0.9]

    y_data[idx, 2] = T_normalized.astype(np.float64)

    if idx % 200 == 0:
        print("test: Min_x:", x_data[idx].min(), " Max_x:", x_data[idx].max())
        print("test: Min_y:", y_data[idx,:,:].min(), "Max_y:", y_data[idx,:,:].max())


# 转换为PyTorch张量
x_tensor = torch.from_numpy(x_data)
y_tensor = torch.from_numpy(y_data)

# 保存为pt文件
torch.save({'x': x_tensor, 'y': y_tensor}, output_file_test)

print(f"数据已成功保存为 {output_file_test}")
print(f"输入形状: {x_tensor.shape}")
print(f"输出形状: {y_tensor.shape}")

print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    




