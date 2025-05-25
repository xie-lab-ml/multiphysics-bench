import numpy as np
import os
import scipy.io as sio
import time
start_time = time.time()


base_path = "../../raw_data/training/VA"
output_base_path = "../../bench_dataset/VA-merged/merge_{}.npy"
os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

variables = ['p_t', 'Sxx', 'Sxy', 'Syy', 'x_u', 'x_v']
number = 10000


def load_ranges(base_path,variables):
    ranges = {}
    
    # rho_water
    rho_data = sio.loadmat(f"{base_path}/rho_water/range_allrho_water.mat")['range_allrho_water']
    ranges['rho_water'] = {'max': rho_data[0,1], 'min': rho_data[0,0]}
    
    # 加载其他变量的范围
    for var in variables:
        real_data = sio.loadmat(f"{base_path}/{var}/range_allreal_{var}.mat")[f'range_allreal_{var}']
        imag_data = sio.loadmat(f"{base_path}/{var}/range_allimag_{var}.mat")[f'range_allimag_{var}']
        
        ranges[var] = {
            'max_real': real_data[0,1],
            'min_real': real_data[0,0],
            'max_imag': imag_data[0,1],
            'min_imag': imag_data[0,0]
        }
    
    return ranges

def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val) * 1.8 - 0.9


range_data = load_ranges(base_path,variables)


# nomalization

for i in range(number):
    # rho_water
    rho_water = sio.loadmat(f"{base_path}/rho_water/{i+1}.mat")['export_rho_water']
    rho_water_normalized = normalize(rho_water, range_data['rho_water']['min'], range_data['rho_water']['max'])
    components = [rho_water_normalized]

    for var in variables:
        data = sio.loadmat(f"{base_path}/{var}/{i+1}.mat")[f"export_{var}"]
        components.append(normalize(data.real, range_data[var]['min_real'], range_data[var]['max_real']))
        components.append(normalize(data.imag, range_data[var]['min_imag'], range_data[var]['max_imag']))
    
    
    
    # Combine them into a new array with a shape [H, W, 4]
    combined = np.stack(components, axis=-1)
    # Save the combined array to a new .npy file
    output_file_path = output_base_path.format(i+1)
    np.save(output_file_path, combined)

    assert combined.shape == (128, 128, 13)
    if i % 200 == 0:
      print(f"Saved combined array for index {i+1} to {output_file_path}")
      print("Min:", combined.min(), "Max:", combined.max())
      # print("rho_water Min:", rho_water_normalized.min(), "Max:", rho_water_normalized.max())
      # print("ec_V Min:", ec_V_normalized.min(), "Max:", ec_V_normalized.max())
      # print("u_flow Min:", u_flow_normalized.min(), "Max:", u_flow_normalized.max())
      # print("v_flow Min:", v_flow_normalized.min(), "Max:", v_flow_normalized.max())


print("Finished processing all files.")
print(f"运行时间: {time.time() - start_time} 秒")    

