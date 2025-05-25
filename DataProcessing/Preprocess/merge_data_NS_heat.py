import numpy as np
import os
import scipy.io as sio


output_base_path = "../../bench_dataset/MHD-merged/merge_{}.npy"
# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

# set timer
import time
start_time = time.time()

number = 10000


# input Q_heat
# load max_Q_heat  min_Q_heat
range_allQ_heat_paths = "../../raw_data/training/NS_heat/Q_heat/range_allQ_heat.mat"
range_allQ_heat = sio.loadmat(range_allQ_heat_paths)['range_allQ_heat']

max_Q_heat = range_allQ_heat[0,1]
min_Q_heat = range_allQ_heat[0,0]


# output  u_u, u_v, T
# load max_u_u min_u_u
range_allu_u_paths = "../../raw_data/training/NS_heat/u_u/range_allu_u.mat"
range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']

max_u_u = range_allu_u[0,1]
min_u_u = range_allu_u[0,0]

# load max_u_v min_u_v
range_allu_v_paths = "../../raw_data/training/NS_heat/u_v/range_allu_v.mat"
range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']

max_u_v = range_allu_v[0,1]
min_u_v = range_allu_v[0,0]


# load max_T min_T
range_allT_paths = "../../raw_data/training/NS_heat/T/range_allT.mat"
range_allT = sio.loadmat(range_allT_paths)['range_allT']

max_T = range_allT[0,1]
min_T = range_allT[0,0]


# nomalization
for i in range(number):
    # Q_heat 
    path_Q_heat = os.path.join(f"../../raw_data/training/NS_heat/Q_heat/", f'{i+1}.mat')
    Q_heat = sio.loadmat(path_Q_heat)['export_Q_heat']
    Q_heat_normalized = (Q_heat - min_Q_heat) / (max_Q_heat - min_Q_heat) * 1.8 - 0.9 # [-0.9,0.9]

    # u_u
    path_u_u = os.path.join(f"../../raw_data/training/NS_heat/u_u/", f'{i+1}.mat')
    u_u = sio.loadmat(path_u_u)['export_u_u']
    u_u_normalized = (u_u - min_u_u) / (max_u_u - min_u_u) * 1.8 - 0.9 # [-0.9,0.9]

    # u_v
    path_u_v = os.path.join(f"../../raw_data/training/NS_heat/u_v/", f'{i+1}.mat')
    u_v = sio.loadmat(path_u_v)['export_u_v']
    u_v_normalized = (u_v - min_u_v) / (max_u_v - min_u_v) * 1.8 - 0.9 # [-0.9,0.9]

    # T 
    path_T = os.path.join(f"../../raw_data/training/NS_heat/T/", f'{i+1}.mat')
    T = sio.loadmat(path_T)['export_T']
    T_normalized = (T - min_T) / (max_T - min_T) * 1.8 - 0.9 # [-0.9,0.9]


    # Combine them into a new array with a shape [H, W, 4]
    combined = np.stack((Q_heat_normalized, u_u_normalized, u_v_normalized, T_normalized), axis=-1)

    # Save the combined array to a new .npy file
    output_file_path = output_base_path.format(i+1)
    np.save(output_file_path, combined)

    assert combined.shape == (128, 128, 4)
    if i % 200 == 0:
      print(f"Saved combined array for index {i} to {output_file_path}")
      print("Min:", combined.min(), "Max:", combined.max())

print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    

