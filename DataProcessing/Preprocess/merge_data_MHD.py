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


# input Br
# load max_Br  min_Br
range_allBr_paths = "../../raw_data/training/MHD/Br/range_allBr.mat"
range_allBr = sio.loadmat(range_allBr_paths)['range_allBr']

max_Br = range_allBr[0,1]
min_Br = range_allBr[0,0]


# output Jx, Jy, Jz, u_u, u_v

# load max_Jx min_Jx
range_allJx_paths = "../../raw_data/training/MHD/Jx/range_allJx.mat"
range_allJx = sio.loadmat(range_allJx_paths)['range_allJx']

max_Jx = range_allJx[0,1]
min_Jx = range_allJx[0,0]

# load max_Jy min_Jy
range_allJy_paths = "../../raw_data/training/MHD/Jy/range_allJy.mat"
range_allJy = sio.loadmat(range_allJy_paths)['range_allJy']

max_Jy = range_allJy[0,1]
min_Jy = range_allJy[0,0]

# load max_Jz min_Jz
range_allJz_paths = "../../raw_data/training/MHD/Jz/range_allJz.mat"
range_allJz = sio.loadmat(range_allJz_paths)['range_allJz']

max_Jz = range_allJz[0,1]
min_Jz = range_allJz[0,0]

# load max_u_u min_u_u
range_allu_u_paths = "../../raw_data/training/MHD/u_u/range_allu_u.mat"
range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']

max_u_u = range_allu_u[0,1]
min_u_u = range_allu_u[0,0]

# load max_u_v min_u_v
range_allu_v_paths = "../../raw_data/training/MHD/u_v/range_allu_v.mat"
range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']

max_u_v = range_allu_v[0,1]
min_u_v = range_allu_v[0,0]


# nomalization
for i in range(number):
    # Br 
    path_Br = os.path.join(f"../../raw_data/training/MHD/Br/", f'{i+1}.mat')
    Br = sio.loadmat(path_Br)['export_Br']
    Br_normalized = (Br - min_Br) / (max_Br - min_Br) * 1.8 - 0.9 # [-0.9,0.9]

    # Jx 
    path_Jx = os.path.join(f"../../raw_data/training/MHD/Jx/", f'{i+1}.mat')
    Jx = sio.loadmat(path_Jx)['export_Jx']
    Jx_normalized = (Jx - min_Jx) / (max_Jx - min_Jx) * 1.8 - 0.9 # [-0.9,0.9]

    # Jy 
    path_Jy = os.path.join(f"../../raw_data/training/MHD/Jy/", f'{i+1}.mat')
    Jy = sio.loadmat(path_Jy)['export_Jy']
    Jy_normalized = (Jy - min_Jy) / (max_Jy - min_Jy) * 1.8 - 0.9 # [-0.9,0.9]

    # Jz 
    path_Jz = os.path.join(f"../../raw_data/training/MHD/Jz/", f'{i+1}.mat')
    Jz = sio.loadmat(path_Jz)['export_Jz']
    Jz_normalized = (Jz - min_Jz) / (max_Jz - min_Jz) * 1.8 - 0.9 # [-0.9,0.9]

    # u_u
    path_u_u = os.path.join(f"../../raw_data/training/MHD/u_u/", f'{i+1}.mat')
    u_u = sio.loadmat(path_u_u)['export_u']
    u_u_normalized = (u_u - min_u_u) / (max_u_u - min_u_u) * 1.8 - 0.9 # [-0.9,0.9]

    # u_v
    path_u_v = os.path.join(f"../../raw_data/training/MHD/u_v/", f'{i+1}.mat')
    u_v = sio.loadmat(path_u_v)['export_v']
    u_v_normalized = (u_v - min_u_v) / (max_u_v - min_u_v) * 1.8 - 0.9 # [-0.9,0.9]


    # Combine them into a new array with a shape [H, W, 4]
    combined = np.stack((Br_normalized, Jx_normalized, Jy_normalized, Jz_normalized, u_u_normalized, u_v_normalized), axis=-1)

    # Save the combined array to a new .npy file
    output_file_path = output_base_path.format(i+1)
    np.save(output_file_path, combined)

    assert combined.shape == (128, 128, 6)
    if i % 200 == 0:
      print(f"Saved combined array for index {i} to {output_file_path}")
      print("Min:", combined.min(), "Max:", combined.max())
      # print("Br Min:", Br_normalized.min(), "Max:", Br_normalized.max())
      # print("Jx Min:", Jx_normalized.min(), "Max:", Jx_normalized.max())
      # print("Jy Min:", Jy_normalized.min(), "Max:", Jy_normalized.max())
      # print("Jz Min:", Jz_normalized.min(), "Max:", Jz_normalized.max())
      # print("u_u Min:", u_u_normalized.min(), "Max:", u_u_normalized.max())
      # print("u_v Min:", u_v_normalized.min(), "Max:", u_v_normalized.max())


print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    

