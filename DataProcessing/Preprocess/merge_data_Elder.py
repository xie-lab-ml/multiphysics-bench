import numpy as np
import os
import scipy.io as sio
from scipy.io import loadmat
# set timer
import time
start_time = time.time()

num_samples = 1000
time_steps = 11
H, W, C = 128, 128, 34

output_base_path = "../../bench_dataset/Elder-merged/merge_{}.npy"
# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

data_base_path = "../../raw_data/training/Elder/"

# -------------------- 加载归一化范围 --------------------
range_allS_c = sio.loadmat(os.path.join(data_base_path, "S_c/range_S_c_t.mat"))['range_S_c_t'][0]
range_allu_u = sio.loadmat(os.path.join(data_base_path, "u_u/range_u_u_t_999.mat"))['range_u_u_t_999']
range_allu_v = sio.loadmat(os.path.join(data_base_path, "u_v/range_u_v_t_99.mat"))['range_u_v_t_99']
range_allc_flow = sio.loadmat(os.path.join(data_base_path, "c_flow/range_c_flow_t_99.mat"))['range_c_flow_t_99']

ranges = {
    'S_c': range_allS_c,
    'u_u': range_allu_u,
    'u_v': range_allu_v,
    'c_flow': range_allc_flow,
}

# print(range_allu_u)
# exit()

def minmax_normalize(x, min_val, max_val):
    return -0.9 + 1.8 * (x - min_val) / (max_val - min_val)

# -------------------- 主循环 --------------------
for i in range(1, num_samples + 1):
    combined_data = np.zeros((H, W, C), dtype=np.float32)

    # ---------- S_c ----------
    path_Sc = os.path.join(data_base_path, 'S_c', str(i), '0.mat')
    Sc = loadmat(path_Sc)
    Sc_data = list(Sc.values())[-1]
    Sc_n = minmax_normalize(Sc_data, *ranges['S_c'])
    combined_data[:, :, 0] = Sc_n

    # ---------- u_u, u_v, c_flow ----------
    var_names = ['u_u', 'u_v', 'c_flow']

    for t in range(0, time_steps ):
        # 时域场 t=0~10
        for var_idx, var in enumerate(var_names):
            min_val, max_val = ranges[var][t]

            path_t = os.path.join(data_base_path, var, str(i), f'{t}.mat')
            data_t = loadmat(path_t)
            data_t = list(data_t.values())[-1]
            data_t_n = minmax_normalize(data_t, min_val, max_val)
            ch_idx = 1 + var_idx * time_steps + t
            combined_data[:, :, ch_idx] = data_t_n

    # ---------- 保存 .npy ----------
    out_path = output_base_path.format(i)
    np.save(out_path, combined_data)

    # 打印部分信息
    if i % 200 == 0:
        print(f"[{i}] saved: {out_path}, shape: {combined_data.shape}, min: {combined_data.min():.4f}, max: {combined_data.max():.4f}")


print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    

