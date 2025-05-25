import numpy as np
import os
import scipy.io as sio


output_base_path = "../../bench_dataset/E_flow-merged/merge_{}.npy"
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

# set timer
import time
start_time = time.time()

number = 10000


# input kappa
# load max_kappa  min_kappa
range_allkappa_paths = "../../raw_data/training/E_flow/kappa/range_allkappa.mat"
range_allkappa = sio.loadmat(range_allkappa_paths)['range_allkappa']

max_kappa = range_allkappa[0,1]
min_kappa = range_allkappa[0,0]


# output ec_V, u_flow, v_flow

# load max_ec_V min_ec_V
range_allec_V_paths = "../../raw_data/training/E_flow/ec_V/range_allec_V.mat"
range_allec_V = sio.loadmat(range_allec_V_paths)['range_allec_V']

max_ec_V = range_allec_V[0,1]
min_ec_V = range_allec_V[0,0]

# load max_u_flow min_u_flow
range_allu_flow_paths = "../../raw_data/training/E_flow/u_flow/range_allu_flow.mat"
range_allu_flow = sio.loadmat(range_allu_flow_paths)['range_allu_flow']

max_u_flow = range_allu_flow[0,1]
min_u_flow = range_allu_flow[0,0]

# load max_v_flow min_v_flow
range_allv_flow_paths = "../../raw_data/data/training/E_flow/v_flow/range_allv_flow.mat"
range_allv_flow = sio.loadmat(range_allv_flow_paths)['range_allv_flow']

max_v_flow = range_allv_flow[0,1]
min_v_flow = range_allv_flow[0,0]


# nomalization
for i in range(number):
    # kappa 
    path_kappa = os.path.join(f"../../raw_data/training/E_flow/kappa/", f'{i+1}.mat')
    kappa = sio.loadmat(path_kappa)['export_kappa']
    kappa_normalized = (kappa - min_kappa) / (max_kappa - min_kappa) * 1.8 - 0.9 # [-0.9,0.9]

    # ec_V
    path_ec_V = os.path.join(f"../../raw_data/training/E_flow/ec_V/", f'{i+1}.mat')
    ec_V = sio.loadmat(path_ec_V)['export_ec_V']
    ec_V_normalized = (ec_V - min_ec_V) / (max_ec_V - min_ec_V) * 1.8 - 0.9 # [-0.9,0.9]

    # u_flow
    path_u_flow = os.path.join(f"../../raw_data/training/E_flow/u_flow/", f'{i+1}.mat')
    u_flow = sio.loadmat(path_u_flow)['export_u_flow']
    u_flow_normalized = (u_flow - min_u_flow) / (max_u_flow - min_u_flow) * 1.8 - 0.9 # [-0.9,0.9]

    # v_flow
    path_v_flow = os.path.join(f"../../raw_data/training/E_flow/v_flow/", f'{i+1}.mat')
    v_flow = sio.loadmat(path_v_flow)['export_v_flow']
    v_flow_normalized = (v_flow - min_v_flow) / (max_v_flow - min_v_flow) * 1.8 - 0.9 # [-0.9,0.9]


    # Combine them into a new array with a shape [H, W, 4]
    combined = np.stack((kappa_normalized, ec_V_normalized, u_flow_normalized, v_flow_normalized), axis=-1)

    # Save the combined array to a new .npy file
    output_file_path = output_base_path.format(i+1)
    np.save(output_file_path, combined)

    assert combined.shape == (128, 128, 4)
    if i % 200 == 0:
      print(f"Saved combined array for index {i} to {output_file_path}")
      print("Min:", combined.min(), "Max:", combined.max())
      # print("kappa Min:", kappa_normalized.min(), "Max:", kappa_normalized.max())
      # print("ec_V Min:", ec_V_normalized.min(), "Max:", ec_V_normalized.max())
      # print("u_flow Min:", u_flow_normalized.min(), "Max:", u_flow_normalized.max())
      # print("v_flow Min:", v_flow_normalized.min(), "Max:", v_flow_normalized.max())


print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    

