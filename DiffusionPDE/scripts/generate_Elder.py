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
from scipy.io import loadmat
import numpy as np
from shapely.geometry import Polygon, Point
import json


def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size] grid.'''
    np.random.seed(seed)
    indices = np.random.choice(grid_size**2, k, replace=False)
    indices_2d = np.unravel_index(indices, (grid_size, grid_size))
    indices_list = list(zip(indices_2d[0], indices_2d[1]))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float64).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask


def invnormalize(data, min_val, max_val):
    return (data + 0.9) / 1.8 * (max_val - min_val )

        
def get_Elder_loss(S_c, u_u, u_v, c_flow, S_c_GT, u_u_GT, u_v_GT, c_flow_GT, S_c_mask, u_u_mask, u_v_mask, c_flow_mask, device=torch.device('cuda')):
    """Return the loss of the Elder equation and the observation loss."""

    rho_0 = 1000
    beta = 200
    rho = rho_0+beta*c_flow  # [T, H, W]
    T, H, W = rho.shape

    delta_x = (300/128) # 1m
    delta_y = (150/128) # 1m
    delta_t = 2 * 365 * 24 * 60 *60 # 2 a

    # 空间导数核 (for conv2d)
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # 时间导数核 (for conv1d)
    deriv_t = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 3) / (2 * delta_t)

    # Darcy
    # 时间导数 d(rho)/dt
    rho_t = rho.permute(1, 2, 0).reshape(-1, 1, T)     # [H*W, 1, T]
    d_rho_dt = F.conv1d(rho_t, deriv_t, padding=1)     # [H*W, 1, T]
    d_rho_dt = d_rho_dt.squeeze(1).reshape(H, W, T).permute(2, 0, 1)  # [T, H, W]

    rho_u = rho * u_u
    rho_u_2d = rho_u.unsqueeze(1)  # [T, 1, H, W]
    d_rho_u_dx = F.conv2d(rho_u_2d, deriv_x, padding=(0, 1)).squeeze(1)  # [T, H, W]

    rho_v = rho * u_v
    rho_v_2d = rho_v.permute(0, 1, 2).unsqueeze(1)
    d_rho_v_dy = F.conv2d(rho_v_2d, deriv_y, padding=(1, 0)).squeeze(1)

    result_Darcy = 0.1 * d_rho_dt + d_rho_u_dx + d_rho_v_dy

    # TDS
    c_t = c_flow.permute(1, 2, 0).reshape(-1, 1, T)
    dc_dt = F.conv1d(c_t, deriv_t, padding=1).squeeze(1).reshape(H, W, T).permute(2, 0, 1)

    c_2d = c_flow.unsqueeze(1)  # [T, 1, H, W]
    dc_dx = F.conv2d(c_2d, deriv_x, padding=(0, 1)).squeeze(1)
    dc_dy = F.conv2d(c_2d, deriv_y, padding=(1, 0)).squeeze(1)

    laplace_c = F.conv2d(dc_dx.unsqueeze(1), deriv_x, padding=(0, 1)).squeeze(1) + F.conv2d(dc_dy.unsqueeze(1), deriv_y, padding=(1, 0)).squeeze(1)

    result_TDS = 0.1 * dc_dt + u_u * dc_dx + u_v * dc_dy - 0.1 * 3.56e-6 * laplace_c - S_c

    # scipy.io.savemat('result_Darcy.mat', {'result_Darcy': result_Darcy.cpu().detach().numpy()})
    # scipy.io.savemat('result_TDS.mat', {'result_TDS': result_TDS.cpu().detach().numpy()})

    pde_loss_Darcy = result_Darcy
    pde_loss_TDS = result_TDS

    pde_loss_Darcy = pde_loss_Darcy.squeeze()
    pde_loss_TDS = pde_loss_TDS.squeeze()

    # pde_loss_Darcy = pde_loss_Darcy/1
    # pde_loss_TDS = pde_loss_TDS/1

    observation_loss_S_c = (S_c - S_c_GT).squeeze()
    observation_loss_S_c = observation_loss_S_c * S_c_mask  
    observation_loss_u_u = (u_u - u_u_GT).squeeze()
    observation_loss_u_u = observation_loss_u_u * u_u_mask
    observation_loss_u_v = (u_v - u_v_GT).squeeze()
    observation_loss_u_v = observation_loss_u_v * u_v_mask  
    observation_loss_c_flow = (c_flow - c_flow_GT).squeeze()
    observation_loss_c_flow = observation_loss_c_flow * c_flow_mask  


    return pde_loss_Darcy, pde_loss_TDS, observation_loss_S_c, observation_loss_u_u, observation_loss_u_v, observation_loss_c_flow



def generate_Elder(config):
    """Generate E_flow equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset'][0]
    time_steps = config['data']['time_steps'][0]
    device = config['generate']['device']
    C, H, W = 34, 128, 128
    data_test_path = "/data/testing/Elder/"
    combined_data_GT = np.zeros((C, H, W), dtype=np.float64)

    # ---------- 读取 S_c ----------
    path_Sc = os.path.join(data_test_path, 'S_c', str(offset), '0.mat')
    Sc_data = loadmat(path_Sc)
    Sc = list(Sc_data.values())[-1]
    combined_data_GT[0, :, :] = Sc

    # ---------- 读取初始场 + 时域场 ----------
    var_names = ['u_u', 'u_v', 'c_flow']
    for var_idx, var in enumerate(var_names):

       # 时域场 t=0~10（通道 4 ~ 33）
        for t in range(0, time_steps):
            path_t = os.path.join(data_test_path, var, str(offset), f'{t}.mat')
            data_t = loadmat(path_t)
            data_t = list(data_t.values())[-1]
            ch_idx = 1 + var_idx * time_steps + t
            combined_data_GT[ch_idx, :, : ] = data_t

    combined_data_GT = torch.tensor(combined_data_GT, dtype=torch.float64, device=device)

    # S_c_GT = combined_data_GT[0].unsqueeze(0).expand(11, -1, -1)
    # u_u_GT = torch.stack([combined_data_GT[i] for i in [1] + list(range(4, 14))], dim=0)  # [11, H, W]
    # u_v_GT = torch.stack([combined_data_GT[i] for i in [2] + list(range(14, 24))], dim=0)
    # c_flow_GT = torch.stack([combined_data_GT[i] for i in [3] + list(range(24, 34))], dim=0)

    # S_c_GT = combined_data_GT[0].unsqueeze(0).expand(2, -1, -1)
    # u_u_GT = torch.stack([combined_data_GT[i] for i in [1] + list(range(4, 5))], dim=0)  # [2, H, W]
    # u_v_GT = torch.stack([combined_data_GT[i] for i in [2] + list(range(5, 6))], dim=0)
    # c_flow_GT = torch.stack([combined_data_GT[i] for i in [3] + list(range(6, 7))], dim=0)

    S_c_GT = combined_data_GT[0].unsqueeze(0)
    u_u_GT = combined_data_GT[1:12]
    u_v_GT = combined_data_GT[12:23]
    c_flow_GT = combined_data_GT[23:34]

    
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
    
    x_next = latents.to(torch.float64) * sigma_t_steps[0]

    known_index_S_c = random_index(500, 128, seed=3)
    known_index_u_u = random_index(500, 128, seed=2)
    known_index_u_v = random_index(500, 128, seed=1)
    known_index_c_flow = random_index(500, 128, seed=0)
    
    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # Euler step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)

        # print("x_N requires grad:", x_N.requires_grad, "grad_fn:", x_N.grad_fn)


        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        # 2nd order correction
        if i < num_steps - 1:
            x_N = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        

        # Scale the data back
        # S_c_N = x_N[0 ,0,:,:].unsqueeze(0).expand(11, -1, -1)
        # u_u_N = x_N[0, [1] + list(range(4, 14)), :, :]  # shape: [11, 128, 128]
        # u_v_N = x_N[0, [2] + list(range(14, 24)), :, :] 
        # c_flow_N = x_N[0, [3] + list(range(24, 34)), :, :] 

        # # Scale the data back
        # S_c_N = x_N[0 ,0,:,:].unsqueeze(0).expand(2, -1, -1)
        # u_u_N = x_N[0, [1] + list(range(4, 5)), :, :]  # shape: [2, 128, 128]
        # u_v_N = x_N[0, [2] + list(range(5, 6)), :, :] 
        # c_flow_N = x_N[0, [3] + list(range(6, 7)), :, :] 

        # Scale the data back
        S_c_N = x_N[0 ,0,:,:].unsqueeze(0)
        u_u_N = x_N[0, 1:12, :, :]  # shape: [2, 128, 128]
        u_v_N = x_N[0, 12:23, :, :] 
        c_flow_N = x_N[0, 23:34, :, :] 

        data_base_path = "/data/training/Elder/"

         # -------------------- 加载归一化范围 --------------------
        range_allS_c = sio.loadmat(os.path.join(data_base_path, "S_c/range_S_c_t.mat"))['range_S_c_t']
        range_allu_u = sio.loadmat(os.path.join(data_base_path, "u_u/range_u_u_t_999.mat"))['range_u_u_t_999']
        range_allu_v = sio.loadmat(os.path.join(data_base_path, "u_v/range_u_v_t_99.mat"))['range_u_v_t_99']
        range_allc_flow = sio.loadmat(os.path.join(data_base_path, "c_flow/range_c_flow_t_99.mat"))['range_c_flow_t_99']

        ranges = {
            'S_c': range_allS_c,
            'u_u': range_allu_u,
            'u_v': range_allu_v,
            'c_flow': range_allc_flow,
        }

        S_c_N[0,:,:] = invnormalize(S_c_N[0,:,:], *ranges['S_c'][0,:]).to(torch.float64)
        for t in range(0, time_steps ):

            u_u_N[t,:,:] = invnormalize(u_u_N[t,:,:], *ranges['u_u'][t,:]).to(torch.float64)
            u_v_N[t,:,:] = invnormalize(u_v_N[t,:,:], *ranges['u_v'][t,:]).to(torch.float64)
            c_flow_N[t,:,:] = invnormalize(c_flow_N[t,:,:], *ranges['c_flow'][t,:]).to(torch.float64)

        # Compute the loss
        pde_loss_Darcy, pde_loss_TDS, observation_loss_S_c, observation_loss_u_u, observation_loss_u_v, observation_loss_c_flow = get_Elder_loss(S_c_N, u_u_N, u_v_N, c_flow_N, S_c_GT, u_u_GT, u_v_GT, c_flow_GT, known_index_S_c, known_index_u_u, known_index_u_v, known_index_c_flow, device=device)

        

        L_pde_Darcy = torch.norm(pde_loss_Darcy, 2)/(128*128)
        L_pde_TDS = torch.norm(pde_loss_TDS, 2)/(128*128)

        L_obs_S_c = torch.norm(observation_loss_S_c, 2)/500

        u_loss_list = []
        v_loss_list = []
        c_flow_loss_list = []
        u_grad_list = []
        v_grad_list = []
        c_flow_grad_list = []
        for t in range(0, time_steps ):
            L_obs_u_u = torch.norm(observation_loss_u_u[t,:,:], 2)/500
            L_obs_u_v = torch.norm(observation_loss_u_v[t,:,:], 2)/500
            L_obs_c_flow = torch.norm(observation_loss_c_flow[t,:,:], 2)/500

            u_loss_list.append(L_obs_u_u)
            v_loss_list.append(L_obs_u_v)
            c_flow_loss_list.append(L_obs_c_flow)
    
            grad_x_cur_obs_u_u = torch.autograd.grad(outputs=L_obs_u_u, inputs=x_cur, retain_graph=True)[0]
            grad_x_cur_obs_u_v = torch.autograd.grad(outputs=L_obs_u_v, inputs=x_cur, retain_graph=True)[0]
            grad_x_cur_obs_c_flow = torch.autograd.grad(outputs=L_obs_c_flow, inputs=x_cur, retain_graph=True)[0]
            u_grad_list.append(grad_x_cur_obs_u_u)
            v_grad_list.append(grad_x_cur_obs_u_v)
            c_flow_grad_list.append(grad_x_cur_obs_c_flow)


        output_file_path = "inference_losses.jsonl"
        if i % 10 == 0:
            log_entry = {
              "step": i,
              "L_pde_Darcy": L_pde_Darcy .tolist(),
              "L_pde_TDS": L_pde_TDS.tolist(),

               "L_obs_S_c": L_obs_S_c.tolist(),
            #    "L_obs_u_u": L_obs_u_u.tolist(),
            #    "L_obs_u_v": L_obs_u_v.tolist(),
            #    "L_obs_c_flow": L_obs_c_flow.tolist(), 
           }
            with open(output_file_path, "a") as file:
                json.dump(log_entry, file)
                file.write("\n")  

        grad_x_cur_obs_S_c = torch.autograd.grad(outputs=L_obs_S_c, inputs=x_cur, retain_graph=True)[0]

        grad_x_cur_pde_Darcy = torch.autograd.grad(outputs=L_pde_Darcy, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_TDS = torch.autograd.grad(outputs=L_pde_TDS, inputs=x_cur, retain_graph=True)[0]
        
       
        zeta_obs_S_c = 10
        zeta_obs_u_u = 10
        zeta_obs_u_v = 10
        zeta_obs_c_flow = 10

        zeta_pde_Darcy = 10
        zeta_pde_TDS = 10

        zeta_obs_u_u_list = []
        zeta_obs_u_v_list = []
        zeta_obs_c_flow_list = []
    # scale zeta
        norm_S_c = torch.norm(zeta_obs_S_c * grad_x_cur_obs_S_c)
        scale_factor = 10 / norm_S_c
        zeta_obs_S_c = zeta_obs_S_c * scale_factor

        for t in range(0, time_steps ):
            norm_u_u = torch.norm(zeta_obs_u_u * grad_x_cur_obs_u_u)
            scale_factor = 0.5 / norm_u_u
            zeta_obs_u_u = zeta_obs_u_u * scale_factor

            norm_u_v = torch.norm(zeta_obs_u_v * grad_x_cur_obs_u_v)
            scale_factor = 0.5 / norm_u_v
            zeta_obs_u_v = zeta_obs_u_v * scale_factor

            norm_c_flow = torch.norm(zeta_obs_c_flow * grad_x_cur_obs_c_flow)
            scale_factor = 0.5 / norm_c_flow
            zeta_obs_c_flow = zeta_obs_c_flow * scale_factor

            zeta_obs_u_u_list.append(zeta_obs_u_u)
            zeta_obs_u_v_list.append(zeta_obs_u_v)
            zeta_obs_c_flow_list.append(zeta_obs_c_flow)

        if i <= 0.5 * num_steps:
            
            x_next = (x_next - zeta_obs_S_c * grad_x_cur_obs_S_c)


            for t in  range(0, 1 ):
                x_next = x_next - zeta_obs_u_u_list[t]*u_grad_list[t]
                x_next = x_next - zeta_obs_u_v_list[t]*v_grad_list[t]
                x_next = x_next - zeta_obs_c_flow_list[t]*c_flow_grad_list[t]

        else:
            
            norm_pde_Darcy = torch.norm(zeta_pde_Darcy * grad_x_cur_pde_Darcy)
            scale_factor = 10 / norm_pde_Darcy
            zeta_pde_Darcy = zeta_pde_Darcy * scale_factor

            norm_pde_TDS = torch.norm(zeta_pde_TDS * grad_x_cur_pde_TDS)
            scale_factor = 20 / norm_pde_TDS
            zeta_pde_TDS = zeta_pde_TDS * scale_factor


            x_next = (x_next - zeta_obs_S_c * grad_x_cur_obs_S_c)
            for t in  range(0, 1):
                x_next = x_next - zeta_obs_u_u_list[t]*u_grad_list[t]
                x_next = x_next - zeta_obs_u_v_list[t]*v_grad_list[t]
                x_next = x_next - zeta_obs_c_flow_list[t]*c_flow_grad_list[t]


            x_next = (x_next - 1* (zeta_pde_Darcy * grad_x_cur_pde_Darcy + zeta_pde_TDS * grad_x_cur_pde_TDS))


    ############################ Save the data ############################
    x_final = x_next

    S_c_final = x_final[0, 0,:,:].unsqueeze(0)
    u_u_final = x_final[0, 1:12, :, :].unsqueeze(0)  # shape: [2, 128, 128]
    u_v_final = x_final[0, 12:23, :, :].unsqueeze(0)
    c_flow_final = x_final[0, 23:34, :, :].unsqueeze(0)
    
    S_c_final[0,:,:] = invnormalize(S_c_final[0,:,:], *ranges['S_c'][0,:]).to(torch.float64)
    for t in range(0, time_steps ):
        u_u_final[0,t,:,:] = invnormalize(u_u_final[0,t,:,:], *ranges['u_u'][t,:]).to(torch.float64)
        u_v_final[0,t,:,:] = invnormalize(u_v_final[0,t,:,:], *ranges['u_v'][t,:]).to(torch.float64)
        c_flow_final[0,t,:,:] = invnormalize(c_flow_final[0,t,:,:], *ranges['c_flow'][t,:]).to(torch.float64)


    relative_error_S_c = torch.norm(S_c_final - S_c_GT, 2) / torch.norm(S_c_GT, 2)
    relative_error_u_u = torch.norm(u_u_final - u_u_GT, 2) / torch.norm(u_u_GT, 2)
    relative_error_u_v = torch.norm(u_v_final - u_v_GT, 2) / torch.norm(u_v_GT, 2)
    relative_error_c_flow = torch.norm(c_flow_final - c_flow_GT, 2) / torch.norm(c_flow_GT, 2)
    

    print(f'Relative error of S_c: {relative_error_S_c}')
    print(f'Relative error of u_u: {relative_error_u_u}')
    print(f'Relative error of u_v: {relative_error_u_v}')
    print(f'Relative error of c_flow: {relative_error_c_flow}')


    S_c_final = S_c_final.detach().cpu().numpy()
    u_u_final = u_u_final.detach().cpu().numpy()
    u_v_final = u_v_final.detach().cpu().numpy()
    c_flow_final = c_flow_final.detach().cpu().numpy()

    scipy.io.savemat('Elder_results.mat', {'S_c': S_c_final, 'u_u': u_u_final, 'u_v': u_v_final, 'c_flow': c_flow_final})
    print('Done.')