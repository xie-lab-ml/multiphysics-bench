import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io
import os
import scipy.io as sio
import pandas as pd
import math
import numpy as np
from shapely.geometry import Polygon, Point
import json
import tqdm


def random_index(k, grid_size, seed=None, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size] grid.'''
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.choice(grid_size**2, k, replace=False)
    indices_2d = np.unravel_index(indices, (grid_size, grid_size))
    indices_list = list(zip(indices_2d[0], indices_2d[1]))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask


def identify_mater_ddp(elliptic_params, device=torch.device('cuda')):
    # elliptic_params: 椭圆参数，shape为 [batchsize, 1, 3]
    batch_size = elliptic_params.shape[0]
    mater_iden = torch.zeros(batch_size, 128, 128, device=device)
    
    # 椭圆参数
    e_a = elliptic_params[:, 0, 0]  # 长轴 [batchsize]
    e_b = elliptic_params[:, 0, 1]  # 短轴 [batchsize]
    angles = torch.deg2rad(elliptic_params[:, 0, 2])  # 旋转角度，转换为弧度 [batchsize]
    
    center_x = 0
    center_y = 0
    
    # 创建坐标网格
    y0, x0 = torch.meshgrid(torch.arange(-63.5, 64.5, 1, device=device),
                           torch.arange(-63.5, 64.5, 1, device=device))
    
    # 对每个样本进行处理
    for i in range(batch_size):
        angle = angles[i]
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # 旋转坐标系，将椭圆旋转到标准位置
        x_rot = (x0 - center_x) * cos_angle + (y0 - center_y) * sin_angle
        y_rot = -(x0 - center_x) * sin_angle + (y0 - center_y) * cos_angle
        
        # 判断点是否在椭圆内
        ellipse_eq = (x_rot / e_a[i]) ** 2 + (y_rot / e_b[i]) ** 2
        mater_iden[i] = torch.where(ellipse_eq <= 1, 1.0, -1.0)

    return mater_iden

    
def identify_mater(elliptic_params, device=torch.device('cuda')):

    # elliptic_params: 椭圆参数，格式为 [长轴, 短轴, 旋转角度（度）]
    mater_iden = torch.zeros(128, 128, device=device)
    
    # 椭圆参数
    e_a = elliptic_params[0,0]  # 长轴
    e_b = elliptic_params[0,1]  # 短轴
    angle = math.radians(elliptic_params[0,2])  # 旋转角度，转换为弧度

    center_x = 0
    center_y = 0

    for j in range(128):
        for k in range(128):
            x0 = -63.5 + j
            y0 = -63.5 + k

            # 旋转坐标系，将椭圆旋转到标准位置
            x_rot = (x0 - center_x) * math.cos(angle) + (y0 - center_y) * math.sin(angle)
            y_rot = -(x0 - center_x) * math.sin(angle) + (y0 - center_y) * math.cos(angle)

            # 判断点是否在椭圆内
            if (x_rot / e_a) ** 2 + (y_rot / e_b) ** 2 <= 1:
                mater_iden[j, k] = 1  # 椭圆内部或边界
            else:
                mater_iden[j, k] = -1  # 椭圆外部

    return mater_iden


def generate_separa_mater(mater, T, mater_iden, device=torch.device('cuda')):
    f = 4e9
    k_0 = 2 * np.pi * f / 3e8
    omega = 2 * np.pi * f
    q = 1.602
    miu_r = 1
    eps_0 = 8.854e-12
    kB = 8.6173e-5
    Eg = 1.12

    sigma_coef_map = torch.where(mater_iden > 1e-5, mater, 0)
    sigma_map = q * sigma_coef_map * torch.exp(- Eg / (kB * T))   #与T有关
    sigma_map = torch.where(mater_iden > 1e-5, sigma_map, 1e-7)
    pho_map = torch.where(mater_iden > 1e-5, 70, mater)
    eps_r = torch.where(mater_iden > 1e-5, 11.7, 1)
    K_map = miu_r * k_0**2 * (eps_r - 1j * sigma_map/(omega * eps_0))



    # scipy.io.savemat('sigma_map.mat', {'sigma_map': sigma_map.cpu().detach().numpy()})
    # scipy.io.savemat('sigma_coef_map.mat', {'sigma_coef_map': sigma_coef_map.cpu().detach().numpy()})
    # scipy.io.savemat('T.mat', {'T': T.cpu().detach().numpy()})

    return sigma_map, pho_map, K_map


def get_TE_heat_loss(mater, Ez, T, mater_GT, Ez_GT, T_GT, mater_mask, Ez_mask, T_mask, mater_iden, device=torch.device('cuda')):
    """Return the loss of the TE_heat equation and the observation loss."""

    sigma, pho, K_E = generate_separa_mater(mater, T, mater_iden)

    delta_x = 128/128*1e-3 # 1mm
    delta_y = 128/128*1e-3 # 1mm
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    deriv_x_complex = torch.complex(deriv_x, torch.zeros_like(deriv_x))
    deriv_y_complex = torch.complex(deriv_y, torch.zeros_like(deriv_y))

    # E_filed
    grad_x_next_x_E = F.conv2d(Ez, deriv_x_complex, padding=(0, 1))
    grad_x_next_y_E = F.conv2d(Ez, deriv_y_complex, padding=(1, 0))
    Laplac_E = F.conv2d(grad_x_next_x_E, deriv_x_complex, padding=(0, 1)) + F.conv2d(grad_x_next_y_E, deriv_y_complex, padding=(1, 0))
    result_E = Laplac_E + K_E * Ez
    # T_filed
    grad_x_next_x_T = F.conv2d(T, deriv_x, padding=(0, 1))
    grad_x_next_y_T = F.conv2d(T, deriv_y, padding=(1, 0))
    Laplac_T = F.conv2d(grad_x_next_x_T, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_T, deriv_y, padding=(1, 0))
    result_T = pho * Laplac_T + 0.5 * sigma * Ez * torch.conj(Ez)

    pde_loss_E = result_E
    pde_loss_T = result_T



    pde_loss_E = pde_loss_E.squeeze()
    pde_loss_T = pde_loss_T.squeeze()
    
    # scipy.io.savemat('pde_loss_E.mat', {'pde_loss_E': pde_loss_E.cpu().detach().numpy()})
    # scipy.io.savemat('pde_loss_T.mat', {'pde_loss_T': pde_loss_T.cpu().detach().numpy()})
    # scipy.io.savemat('result_E.mat', {'result_E': result_E.cpu().detach().numpy()})
    # scipy.io.savemat('Laplac_E.mat', {'Laplac_E': Laplac_E.cpu().detach().numpy()})
    # scipy.io.savemat('result_T.mat', {'result_T': result_T.cpu().detach().numpy()})


    observation_loss_mater = (mater - mater_GT).squeeze()
    observation_loss_mater = observation_loss_mater * mater_mask  
    observation_loss_Ez = (Ez - Ez_GT).squeeze()
    observation_loss_Ez = observation_loss_Ez * Ez_mask  
    observation_loss_T = (T - T_GT).squeeze()
    observation_loss_T = observation_loss_T * T_mask

    pde_loss_E = pde_loss_E/1000000
    pde_loss_T = pde_loss_T/1000000
    
    return pde_loss_E, pde_loss_T, observation_loss_mater, observation_loss_Ez, observation_loss_T



def generate_TE_heat(config):
    """Generate TE_heat equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset'][0]
    device = config['generate']['device']

    mater_GT_path = os.path.join(datapath, "mater", f"{offset}.mat")
    mater_GT = sio.loadmat(mater_GT_path)['mater']
    mater_GT = torch.tensor(mater_GT, dtype=torch.float64, device=device)

    Ez_GT_path = os.path.join(datapath, "Ez", f"{offset}.mat")
    Ez_GT = sio.loadmat(Ez_GT_path)['export_Ez']
    Ez_GT = torch.tensor(Ez_GT, dtype=torch.complex128, device=device)
    
    T_GT_path = os.path.join(datapath, "T", f"{offset}.mat")
    T_GT = sio.loadmat(T_GT_path)['export_T']
    T_GT = torch.tensor(T_GT, dtype=torch.float64, device=device)

    elliptic_GT_path = os.path.join(datapath, "ellipticcsv", f"{offset}.csv")
    elliptic_GT = pd.read_csv(elliptic_GT_path, header=None)
    elliptic_GT = torch.tensor(elliptic_GT.values, dtype=torch.float64)
    
    batch_size = config['generate']['batch_size']
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)
    
    ############################ Set up EDM latent ############################
    print(f'Generating {batch_size} samples...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    
    sigma_min = config['generate']['sigma_min']
    sigma_max = config['generate']['sigma_max']
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    num_steps = config['test']['iterations']
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    rho = config['generate']['rho']
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
    
    def sigma_func(t):
        return ((sigma_max ** (1 / rho) + t * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)))
            ** rho)

    x_next = latents.to(torch.float64) * sigma_t_steps[0]
    known_index_mater = random_index(500, 128, seed=2)
    known_index_Ez = random_index(500, 128, seed=1)
    known_index_T = random_index(500, 128, seed=0)
    
    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # Euler step

        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        # 2nd order correction
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        
        # Scale the data back
        mater_N = x_N[:,0,:,:].unsqueeze(0)
        real_Ez_N = x_N[:,1,:,:].unsqueeze(0)
        imag_Ez_N = x_N[:,2,:,:].unsqueeze(0)
        T_N = x_N[:,3,:,:].unsqueeze(0)

        mater_iden = identify_mater(elliptic_GT)
        val_in = ((mater_N - 0.1) * (3e11 - 1e11) / 0.8 + 1e11).to(torch.float64)
        val_out = ((mater_N + 0.9) * (20 - 10) / 0.8 + 10).to(torch.float64)
        mater_N = torch.where(mater_iden > 1e-5, val_in, val_out)
        mater_N_np = mater_N[0, 0, :, :].cpu().detach().numpy()
        np.savetxt('mater_N.txt', mater_N_np, fmt='%.6f')
        scipy.io.savemat('mater_N.mat', {'mater_N': mater_N.cpu().detach().numpy()})

        # inv_normalization
        max_abs_Ez_path = "/data/training/TE_heat/Ez/max_abs_Ez.mat"
        max_abs_Ez = sio.loadmat(max_abs_Ez_path)['max_abs_Ez']
        max_abs_Ez = torch.tensor(max_abs_Ez, device=device)

        range_allT_paths = "/data/training/TE_heat/T/range_allT.mat"
        range_allT = sio.loadmat(range_allT_paths)['range_allT']
        range_allT = torch.tensor(range_allT, device=device)

        max_T = range_allT[0,1]
        min_T = range_allT[0,0]

        real_Ez_N = (real_Ez_N*max_abs_Ez/0.9).to(torch.float64)
        imag_Ez_N = (imag_Ez_N*max_abs_Ez/0.9).to(torch.float64)
        T_N = ((T_N+0.9)/1.8 *(max_T - min_T) + min_T).to(torch.float64)

        complex_Ez_N = torch.complex(real_Ez_N, imag_Ez_N)
      
        # Compute the loss
        # with torch.enable_grad():
        pde_loss_E, pde_loss_T, observation_loss_mater, observation_loss_Ez, observation_loss_T = get_TE_heat_loss(mater_N, complex_Ez_N, T_N, mater_GT, Ez_GT, T_GT, known_index_mater, known_index_Ez, known_index_T, mater_iden, device=device)
        L_pde_E = torch.norm(pde_loss_E, 2)/(128*128)
        L_pde_T = torch.norm(pde_loss_T, 2)/(128*128)
        
        nan_pde_loss_E = torch.isnan(pde_loss_E)
        if nan_pde_loss_E.any():
            print( torch.nonzero(nan_pde_loss_E))
            print(pde_loss_E[nan_pde_loss_E])

        
        L_obs_mater = torch.norm(observation_loss_mater, 2)/500
        L_obs_real_Ez = torch.norm(observation_loss_Ez.real, 2)/500  # 拆成实部和虚部
        L_obs_imag_Ez = torch.norm(observation_loss_Ez.imag, 2)/500  # 拆成实部和虚部
        L_obs_T = torch.norm(observation_loss_T, 2)/500


        output_file_path = "inference_losses.jsonl"
        if i % 5 == 0:
            log_entry = {
              "step": i,
              "L_pde_E": L_pde_E.tolist(),
              "L_pde_T": L_pde_T.tolist(),
               "L_obs_mater": L_obs_mater.tolist(),
               "L_obs_real_Ez": L_obs_real_Ez.tolist(),
               "L_obs_imag_Ez": L_obs_imag_Ez.tolist(),
              "L_obs_T": L_obs_T.tolist()
           }
            with open(output_file_path, "a") as file:
                json.dump(log_entry, file)
                file.write("\n")  


        grad_x_cur_obs_mater = torch.autograd.grad(outputs=L_obs_mater, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_real_Ez = torch.autograd.grad(outputs=L_obs_real_Ez, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_imag_Ez = torch.autograd.grad(outputs=L_obs_imag_Ez, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_T = torch.autograd.grad(outputs=L_obs_T, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_E = torch.autograd.grad(outputs=L_pde_E, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_T = torch.autograd.grad(outputs=L_pde_T, inputs=x_cur)[0]

       
        zeta_obs_mater = 10
        zeta_obs_real_Ez = 10
        zeta_obs_imag_Ez = 10
        zeta_obs_T = 10
        zeta_pde_E = 10
        zeta_pde_T = 10

    # scale zeta
        norm_mater = torch.norm(zeta_obs_mater * grad_x_cur_obs_mater)
        scale_factor = 10 / norm_mater
        zeta_obs_mater = zeta_obs_mater * scale_factor

        norm_real_Ez = torch.norm(zeta_obs_real_Ez * grad_x_cur_obs_real_Ez)
        scale_factor = 1.0 / norm_real_Ez
        zeta_obs_real_Ez = zeta_obs_real_Ez * scale_factor

        norm_imag_Ez = torch.norm(zeta_obs_imag_Ez * grad_x_cur_obs_imag_Ez)
        scale_factor = 1.0 / norm_imag_Ez
        zeta_obs_imag_Ez = zeta_obs_imag_Ez * scale_factor

        norm_T = torch.norm(zeta_obs_T * grad_x_cur_obs_T)
        scale_factor = 10.0 / norm_T
        zeta_obs_T = zeta_obs_T * scale_factor
    

        if i <= 0.5 * num_steps:

            x_next = x_next - zeta_obs_mater * grad_x_cur_obs_mater

            norm_value = torch.norm(zeta_obs_mater * grad_x_cur_obs_mater).item()
            norm_value_2 = torch.norm(x_next).item()
            print(norm_value, norm_value_2)
        else:
            norm_pde_E = torch.norm(zeta_pde_E * grad_x_cur_pde_E)
            scale_factor = 1 / norm_pde_E
            zeta_pde_E = zeta_pde_E * scale_factor

            norm_pde_T = torch.norm(zeta_pde_T * grad_x_cur_pde_E)
            scale_factor = 0.00005 / norm_pde_T
            zeta_pde_T = zeta_pde_T * scale_factor

            x_next = x_next - (zeta_pde_E * grad_x_cur_pde_E + zeta_pde_T * grad_x_cur_pde_T)

            norm_value = torch.norm(grad_x_cur_pde_E).item()
            print(norm_value)
            norm_value = torch.norm(grad_x_cur_pde_T).item()
            print(norm_value)

    ############################ Save the data ############################
    x_final = x_next
    mater_final = x_final[:,0,:,:].unsqueeze(0)
    real_Ez_final = x_final[:,1,:,:].unsqueeze(0)
    imag_Ez_final = x_final[:,2,:,:].unsqueeze(0)
    T_final = x_final[:,3,:,:].unsqueeze(0)    
    

    mater_iden = identify_mater(elliptic_GT)
    val_in = ((mater_final - 0.1) * (3e11 - 1e11) / 0.8 + 1e11).to(torch.float64)  
    val_out = ((mater_final + 0.9) * (20 - 10) / 0.8 + 10).to(torch.float64) 
    mater_final = torch.where(mater_iden > 1e-5, val_in, val_out)


    real_Ez_final = (real_Ez_final * max_abs_Ez / 0.9).to(torch.float64)
    imag_Ez_final = (imag_Ez_final * max_abs_Ez / 0.9).to(torch.float64)
    complex_Ez_final = torch.complex(real_Ez_final, imag_Ez_final)
    T_final = ((T_final+0.9)/1.8 *(max_T - min_T) + min_T).to(torch.float64)


    relative_error_mater = torch.norm(mater_final - mater_GT, 2) / torch.norm(mater_GT, 2)
    relative_error_real_Ez = torch.norm(real_Ez_final - Ez_GT.real, 2) / torch.norm(Ez_GT.real, 2)
    relative_error_imag_Ez = torch.norm(imag_Ez_final - Ez_GT.imag, 2) / torch.norm(Ez_GT.imag, 2)
    relative_error_T = torch.norm(T_final - T_GT, 2) / torch.norm(T_GT, 2)  


    print(f'Relative error of mater: {relative_error_mater}')
    print(f'Relative error of real_Ez: {relative_error_real_Ez}')
    print(f'Relative error of imag_Ez: {relative_error_imag_Ez}')
    print(f'Relative error of T: {relative_error_T}')

    mater_final = mater_final.detach().cpu().numpy()
    complex_Ez_final = complex_Ez_final.detach().cpu().numpy()
    T_final = T_final.detach().cpu().numpy()

    scipy.io.savemat('TE_heat_results.mat', {'mater': mater_final, 'Ez': complex_Ez_final, 'T': T_final})
    print('Done.')
