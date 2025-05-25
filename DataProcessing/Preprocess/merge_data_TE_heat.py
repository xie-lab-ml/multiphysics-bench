import numpy as np
import os
import scipy.io as sio


output_base_path = "../../bench_dataset/TE_heat-merged/merge_{}.npy"
# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

# set timer
import time
start_time = time.time()

number = 10000

mater_normalized = np.zeros((number,128, 128))

# input material
for i in range(number):
    normal_datamater = np.zeros((128, 128))
    # load mater
    # load poly
    path_mater = os.path.join(f"../../raw_data/training/TE_heat/mater/", f'{i+1}.mat')
    mater = sio.loadmat(path_mater)['mater']

    mater_in = (mater >= 1e11) & (mater <= 3e11)
    mater_out = (mater >= 10) & (mater <= 20)
    normal_datamater = np.where(mater_in, (mater - 1e11) / (3e11 - 1e11) * 0.8 + 0.1, (mater - 10) / (20 - 10) * 0.8 - 0.9)

    # 边上和内部设置为parm.Sigma_Si_coef(0.1,0.9)，其他设置为normal_Pho_Al(-0.9,-0.1)

    mater_normalized[i,:,:] = normal_datamater
    

# output material  Re(Ez) Im(Ez) T
    # load max_abs_Ez
max_abs_Ez_path = "../../raw_data/training/TE_heat/Ez/max_abs_Ez.mat"
max_abs_Ez = sio.loadmat(max_abs_Ez_path)['max_abs_Ez']

print(max_abs_Ez)

    # load T
range_allT_paths = "../../raw_data/training/TE_heat/T/range_allT.mat"
range_allT = sio.loadmat(range_allT_paths)['range_allT']

max_T = range_allT[0,1]
min_T = range_allT[0,0]


# nomalization
for i in range(number):
    # Ez 
    path_Ez = os.path.join(f"../../raw_data/training/TE_heat/Ez/", f'{i+1}.mat')
    Ez = sio.loadmat(path_Ez)['export_Ez']
    Ez_normalized = Ez / max_abs_Ez * 0.9  # 保持相位不变  [0,0.9]
    real_Ez_normalized = np.real(Ez_normalized)
    imag_Ez_normalized = np.imag(Ez_normalized)

    # T 
    path_T = os.path.join(f"../../raw_data/training/TE_heat/T/", f'{i+1}.mat')
    T = sio.loadmat(path_T)['export_T']
    T_normalized = (T - min_T) / (max_T - min_T) * 1.8 - 0.9 # [-0.9,0.9]


    # Combine them into a new array with a shape [H, W, 4]
    combined = np.stack((mater_normalized[i,:,:], real_Ez_normalized, imag_Ez_normalized, T_normalized), axis=-1)

    # Save the combined array to a new .npy file
    output_file_path = output_base_path.format(i+1)
    np.save(output_file_path, combined)

    assert combined.shape == (128, 128, 4)
    if i % 200 == 0:
      print(f"Saved combined array for index {i} to {output_file_path}")
      print("Min:", combined.min(), "Max:", combined.max())

print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    

