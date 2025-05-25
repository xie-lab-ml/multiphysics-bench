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


def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val ) * 1.8 - 0.9



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

        

def get_VA_loss(rho_water, p_t_real, p_t_imag, Sxx_real, Sxx_imag, Sxy_real, Sxy_imag, Syy_real, Syy_imag, x_u_real, x_u_imag, x_v_real, 
                x_v_imag, rho_water_GT, p_t_GT, Sxx_GT, Sxy_GT, Syy_GT, x_u_GT, x_v_GT, rho_water_mask, p_t_mask, Sxx_mask, Sxy_mask, 
                Syy_mask, x_u_mask, x_v_mask, device=torch.device('cuda')):
    
    """Return the loss of the VA equation and the observation loss."""

    omega = torch.tensor(np.pi * 1e5, dtype=torch.float64, device=device)

    c_ac = 1.48144e3
    

    delta_x = (40/128)*1e-3 # 1mm
    delta_y = (40/128)*1e-3 # 1mm
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # Continuity_acoustic real
    grad_x_next_x_p_t_real = F.conv2d(p_t_real, deriv_x, padding=(0, 1))
    grad_x_next_y_p_t_real = F.conv2d(p_t_real, deriv_y, padding=(1, 0))
    Laplace_p_t_real = F.conv2d(grad_x_next_x_p_t_real/rho_water, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_p_t_real/rho_water, deriv_y, padding=(1, 0))
    result_AC_real = Laplace_p_t_real + omega**2*p_t_real/(rho_water*c_ac**2)

    # Continuity_acoustic imag
    grad_x_next_x_p_t_imag = F.conv2d(p_t_imag, deriv_x, padding=(0, 1))
    grad_x_next_y_p_t_imag = F.conv2d(p_t_imag, deriv_y, padding=(1, 0))
    Laplace_p_t_imag = F.conv2d(grad_x_next_x_p_t_imag/rho_water, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_p_t_imag/rho_water, deriv_y, padding=(1, 0))
    result_AC_imag = Laplace_p_t_imag + omega**2*p_t_imag/(rho_water*c_ac**2)


    # Continuity_structure real_x imag_x
    grad_x_next_x_Sxx_real = F.conv2d(Sxx_real, deriv_x, padding=(0, 1))
    grad_x_next_y_Sxy_real = F.conv2d(Sxy_real, deriv_y, padding=(1, 0))
    result_structure_real_x = grad_x_next_x_Sxx_real + grad_x_next_y_Sxy_real + x_u_real
    
    grad_x_next_x_Sxx_imag = F.conv2d(Sxx_imag, deriv_x, padding=(0, 1))
    grad_x_next_y_Sxy_imag = F.conv2d(Sxy_imag, deriv_y, padding=(1, 0))
    result_structure_imag_x = grad_x_next_x_Sxx_imag + grad_x_next_y_Sxy_imag + x_u_imag

    # Continuity_structure real_y imag_y
    grad_x_next_x_Sxy_real = F.conv2d(Sxy_real, deriv_x, padding=(0, 1))
    grad_x_next_y_Syy_real = F.conv2d(Syy_real, deriv_y, padding=(1, 0))
    result_structure_real_y = grad_x_next_x_Sxy_real + grad_x_next_y_Syy_real + x_v_real

    grad_x_next_x_Sxy_imag = F.conv2d(Sxy_imag, deriv_x, padding=(0, 1))
    grad_x_next_y_Syy_imag = F.conv2d(Syy_imag, deriv_y, padding=(1, 0))
    result_structure_imag_y = grad_x_next_x_Sxy_imag + grad_x_next_y_Syy_imag + x_v_imag

    pde_loss_AC_real = result_AC_real
    pde_loss_AC_imag = result_AC_imag

    pde_loss_structure_real_x = result_structure_real_x
    pde_loss_structure_imag_x = result_structure_imag_x

    pde_loss_structure_real_y = result_structure_real_y
    pde_loss_structure_imag_y = result_structure_imag_y


    pde_loss_AC_real = pde_loss_AC_real.squeeze()
    pde_loss_AC_imag = pde_loss_AC_imag.squeeze()

    pde_loss_structure_real_x = pde_loss_structure_real_x.squeeze()
    pde_loss_structure_imag_x = pde_loss_structure_imag_x.squeeze()
    pde_loss_structure_real_y = pde_loss_structure_real_y.squeeze()
    pde_loss_structure_imag_y = pde_loss_structure_imag_y.squeeze()
    
    pde_loss_AC_real = pde_loss_AC_real/1000000
    pde_loss_AC_imag = pde_loss_AC_imag/1000000

    pde_loss_structure_real_x = pde_loss_structure_real_x/1000
    pde_loss_structure_imag_x = pde_loss_structure_imag_x/1000
    pde_loss_structure_real_y = pde_loss_structure_real_y/1000
    pde_loss_structure_imag_y = pde_loss_structure_imag_y/1000

    p_t_complex = torch.complex(p_t_real, p_t_imag)
    Sxx_complex = torch.complex(Sxx_real, Sxx_imag)
    Sxy_complex = torch.complex(Sxy_real, Sxy_imag)
    Syy_complex = torch.complex(Syy_real, Syy_imag)
    x_u_complex = torch.complex(x_u_real, x_u_imag)
    x_v_complex = torch.complex(x_v_real, x_v_imag)

    observation_loss_rho_water = (rho_water - rho_water_GT).squeeze()
    observation_loss_rho_water = observation_loss_rho_water * rho_water_mask  
    observation_loss_p_t = (p_t_complex - p_t_GT).squeeze()
    observation_loss_p_t = observation_loss_p_t * p_t_mask
    observation_loss_Sxx = (Sxx_complex - Sxx_GT).squeeze()
    observation_loss_Sxx = observation_loss_Sxx * Sxx_mask  
    observation_loss_Sxy = (Sxy_complex - Sxy_GT).squeeze()
    observation_loss_Sxy = observation_loss_Sxy * Sxy_mask  
    observation_loss_Syy = (Syy_complex - Syy_GT).squeeze()
    observation_loss_Syy = observation_loss_Syy * Syy_mask  
    observation_loss_x_u = (x_u_complex - x_u_GT).squeeze()
    observation_loss_x_u = observation_loss_x_u * x_u_mask  
    observation_loss_x_v = (x_v_complex - x_v_GT).squeeze()
    observation_loss_x_v = observation_loss_x_v * x_v_mask  
    

    return pde_loss_AC_real, pde_loss_AC_imag, pde_loss_structure_real_x, pde_loss_structure_imag_x, pde_loss_structure_real_y, pde_loss_structure_imag_y, observation_loss_rho_water, observation_loss_p_t, observation_loss_Sxx, observation_loss_Sxy, observation_loss_Syy, observation_loss_x_u, observation_loss_x_v


def generate_VA(config):
    """Generate E_flow equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset'][0]
    device = config['generate']['device']

    rho_water_GT_path = os.path.join(datapath, "rho_water", f"{offset}.mat")
    rho_water_GT = sio.loadmat(rho_water_GT_path)['export_rho_water']
    rho_water_GT = torch.tensor(rho_water_GT, dtype=torch.float64, device=device)

    p_t_GT_path = os.path.join(datapath, "p_t", f"{offset}.mat")
    p_t_GT = sio.loadmat(p_t_GT_path)['export_p_t']
    p_t_GT = torch.tensor(p_t_GT, dtype=torch.complex128, device=device)

    Sxx_GT_path = os.path.join(datapath, "Sxx", f"{offset}.mat")
    Sxx_GT = sio.loadmat(Sxx_GT_path)['export_Sxx']
    Sxx_GT = torch.tensor(Sxx_GT, dtype=torch.complex128, device=device)

    Sxy_GT_path = os.path.join(datapath, "Sxy", f"{offset}.mat")
    Sxy_GT = sio.loadmat(Sxy_GT_path)['export_Sxy']
    Sxy_GT = torch.tensor(Sxy_GT, dtype=torch.complex128, device=device)

    Syy_GT_path = os.path.join(datapath, "Syy", f"{offset}.mat")
    Syy_GT = sio.loadmat(Syy_GT_path)['export_Syy']
    Syy_GT = torch.tensor(Syy_GT, dtype=torch.complex128, device=device)


    x_u_GT_path = os.path.join(datapath, "x_u", f"{offset}.mat")
    x_u_GT = sio.loadmat(x_u_GT_path)['export_x_u']
    x_u_GT = torch.tensor(x_u_GT, dtype=torch.complex128, device=device)

    x_v_GT_path = os.path.join(datapath, "x_v", f"{offset}.mat")
    x_v_GT = sio.loadmat(x_v_GT_path)['export_x_v']
    x_v_GT = torch.tensor(x_v_GT, dtype=torch.complex128, device=device)

    
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

    known_index_rho_water = random_index(500, 128, seed=6)
    known_index_p_t = random_index(500, 128, seed=5)
    known_index_Sxx = random_index(500, 128, seed=4)
    known_index_Sxy = random_index(500, 128, seed=3)
    known_index_Syy = random_index(500, 128, seed=2)
    known_index_x_u = random_index(500, 128, seed=1)
    known_index_x_v = random_index(500, 128, seed=0)
    
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
        rho_water_N = x_N[:,0,:,:].unsqueeze(0)
        real_p_t_N = x_N[:,1,:,:].unsqueeze(0)
        imag_p_t_N = x_N[:,2,:,:].unsqueeze(0)
        real_Sxx_N = x_N[:,3,:,:].unsqueeze(0)
        imag_Sxx_N = x_N[:,4,:,:].unsqueeze(0)
        real_Sxy_N = x_N[:,5,:,:].unsqueeze(0)
        imag_Sxy_N = x_N[:,6,:,:].unsqueeze(0)
        real_Syy_N = x_N[:,7,:,:].unsqueeze(0)
        imag_Syy_N = x_N[:,8,:,:].unsqueeze(0)
        real_x_u_N = x_N[:,9,:,:].unsqueeze(0)
        imag_x_u_N = x_N[:,10,:,:].unsqueeze(0)
        real_x_v_N = x_N[:,11,:,:].unsqueeze(0)
        imag_x_v_N = x_N[:,12,:,:].unsqueeze(0)

        # inv_normalization
        range_allrho_water_paths = "/data/training/VA/rho_water/range_allrho_water.mat"
        range_allrho_water = sio.loadmat(range_allrho_water_paths)['range_allrho_water']
        range_allrho_water = torch.tensor(range_allrho_water, device=device)
        max_rho_water = range_allrho_water[0,1]
        min_rho_water = range_allrho_water[0,0]


        base_path = "/data/training/VA"
        variables = ['p_t', 'Sxx', 'Sxy', 'Syy', 'x_u', 'x_v']

        range_data = load_ranges(base_path,variables)

        rho_water_N = ((rho_water_N+0.9)/1.8 *(max_rho_water - min_rho_water) + min_rho_water).to(torch.float64)
        real_p_t_N = ((real_p_t_N+0.9)/1.8 *(range_data['p_t']['max_real'] - range_data['p_t']['min_real']) + range_data['p_t']['min_real']).to(torch.float64)
        imag_p_t_N = ((imag_p_t_N+0.9)/1.8 *(range_data['p_t']['max_imag'] - range_data['p_t']['min_imag']) + range_data['p_t']['min_imag']).to(torch.float64)
        real_Sxx_N = ((real_Sxx_N+0.9)/1.8 *(range_data['Sxx']['max_real'] - range_data['Sxx']['min_real']) + range_data['Sxx']['min_real']).to(torch.float64)
        imag_Sxx_N = ((imag_Sxx_N+0.9)/1.8 *(range_data['Sxx']['max_imag'] - range_data['Sxx']['min_imag']) + range_data['Sxx']['min_imag']).to(torch.float64)
        real_Sxy_N = ((real_Sxy_N+0.9)/1.8 *(range_data['Sxy']['max_real'] - range_data['Sxy']['min_real']) + range_data['Sxy']['min_real']).to(torch.float64)
        imag_Sxy_N = ((imag_Sxy_N+0.9)/1.8 *(range_data['Sxy']['max_imag'] - range_data['Sxy']['min_imag']) + range_data['Sxy']['min_imag']).to(torch.float64)
        real_Syy_N = ((real_Syy_N+0.9)/1.8 *(range_data['Syy']['max_real'] - range_data['Syy']['min_real']) + range_data['Syy']['min_real']).to(torch.float64)
        imag_Syy_N = ((imag_Syy_N+0.9)/1.8 *(range_data['Syy']['max_imag'] - range_data['Syy']['min_imag']) + range_data['Syy']['min_imag']).to(torch.float64)
        real_x_u_N = ((real_x_u_N+0.9)/1.8 *(range_data['x_u']['max_real'] - range_data['x_u']['min_real']) + range_data['x_u']['min_real']).to(torch.float64)
        imag_x_u_N = ((imag_x_u_N+0.9)/1.8 *(range_data['x_u']['max_imag'] - range_data['x_u']['min_imag']) + range_data['x_u']['min_imag']).to(torch.float64)
        real_x_v_N = ((real_x_v_N+0.9)/1.8 *(range_data['x_v']['max_real'] - range_data['x_v']['min_real']) + range_data['x_v']['min_real']).to(torch.float64)
        imag_x_v_N = ((imag_x_v_N+0.9)/1.8 *(range_data['x_v']['max_imag'] - range_data['x_v']['min_imag']) + range_data['x_v']['min_imag']).to(torch.float64)
        

        # Compute the loss

        (pde_loss_AC_real, pde_loss_AC_imag, pde_loss_structure_real_x, pde_loss_structure_imag_x, pde_loss_structure_real_y, pde_loss_structure_imag_y, 
          observation_loss_rho_water, observation_loss_p_t, observation_loss_Sxx, observation_loss_Sxy, observation_loss_Syy, observation_loss_x_u, 
          observation_loss_x_v) = get_VA_loss(rho_water_N, real_p_t_N, imag_p_t_N, real_Sxx_N, imag_Sxx_N, real_Sxy_N, imag_Sxy_N, real_Syy_N, imag_Syy_N, 
                                              real_x_u_N, imag_x_u_N, real_x_v_N, imag_x_v_N, rho_water_GT, p_t_GT, Sxx_GT, Sxy_GT, Syy_GT, x_u_GT, x_v_GT, 
                                              known_index_rho_water, known_index_p_t, known_index_Sxx, known_index_Sxy, known_index_Syy, known_index_x_u, 
                                              known_index_x_v, device=device)
        

        L_pde_AC_real = torch.norm(pde_loss_AC_real, 2)/(128*128)
        L_pde_AC_imag = torch.norm(pde_loss_AC_imag, 2)/(128*128)
        L_pde_structure_real_x = torch.norm(pde_loss_structure_real_x, 2)/(128*128)
        L_pde_structure_imag_x = torch.norm(pde_loss_structure_imag_x, 2)/(128*128)
        L_pde_structure_real_y = torch.norm(pde_loss_structure_real_y, 2)/(128*128)
        L_pde_structure_imag_y = torch.norm(pde_loss_structure_imag_y, 2)/(128*128)
        
        L_obs_rho_water = torch.norm(observation_loss_rho_water, 2)/500
        L_obs_p_t_real = torch.norm(observation_loss_p_t.real, 2)/500
        L_obs_p_t_imag = torch.norm(observation_loss_p_t.imag, 2)/500
        L_obs_Sxx_real = torch.norm(observation_loss_Sxx.real, 2)/500
        L_obs_Sxx_imag = torch.norm(observation_loss_Sxx.imag, 2)/500
        L_obs_Sxy_real = torch.norm(observation_loss_Sxy.real, 2)/500
        L_obs_Sxy_imag = torch.norm(observation_loss_Sxy.imag, 2)/500
        L_obs_Syy_real = torch.norm(observation_loss_Syy.real, 2)/500
        L_obs_Syy_imag = torch.norm(observation_loss_Syy.imag, 2)/500
        L_obs_x_u_real = torch.norm(observation_loss_x_u.real, 2)/500
        L_obs_x_u_imag = torch.norm(observation_loss_x_u.imag, 2)/500
        L_obs_x_v_real = torch.norm(observation_loss_x_v.real, 2)/500
        L_obs_x_v_imag = torch.norm(observation_loss_x_v.imag, 2)/500


        output_file_path = "inference_losses.jsonl"
        if i % 10 == 0:
            log_entry = {
              "step": i,
              "L_pde_AC_real": L_pde_AC_real .tolist(),
              "L_pde_AC_imag": L_pde_AC_imag.tolist(),
              "L_pde_structure_real_x": L_pde_structure_real_x .tolist(),
              "L_pde_structure_imag_x": L_pde_structure_imag_x.tolist(),
              "L_pde_structure_real_y": L_pde_structure_real_y.tolist(),
              "L_pde_structure_imag_y": L_pde_structure_imag_y.tolist(),

               "L_obs_rho_water": L_obs_rho_water.tolist(),
               "L_obs_p_t_real": L_obs_p_t_real.tolist(),
               "L_obs_p_t_imag": L_obs_p_t_imag.tolist(),
               "L_obs_Sxx_real": L_obs_Sxx_real.tolist(),
               "L_obs_Sxx_imag": L_obs_Sxx_imag.tolist(),
               "L_obs_Sxy_real": L_obs_Sxy_real.tolist(),
               "L_obs_Sxy_imag": L_obs_Sxy_imag.tolist(),
               "L_obs_Syy_real": L_obs_Syy_real.tolist(),
               "L_obs_Syy_imag": L_obs_Syy_imag.tolist(),
               "L_obs_x_u_real": L_obs_x_u_real.tolist(),
               "L_obs_x_u_imag": L_obs_x_u_imag.tolist(),
               "L_obs_x_v_real": L_obs_x_v_real.tolist(),             
               "L_obs_x_v_imag": L_obs_x_v_imag.tolist(),    
           }
            with open(output_file_path, "a") as file:
                json.dump(log_entry, file)
                file.write("\n")  

        grad_x_cur_obs_rho_water = torch.autograd.grad(outputs=L_obs_rho_water, inputs=x_cur, retain_graph=True)[0]

        grad_x_cur_pde_AC_real = torch.autograd.grad(outputs=L_pde_AC_real, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_AC_imag = torch.autograd.grad(outputs=L_pde_AC_imag, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_structure_real_x = torch.autograd.grad(outputs=L_pde_structure_real_x, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_structure_imag_x = torch.autograd.grad(outputs=L_pde_structure_imag_x, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_structure_real_y = torch.autograd.grad(outputs=L_pde_structure_real_y, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_structure_imag_y = torch.autograd.grad(outputs=L_pde_structure_imag_y, inputs=x_cur, retain_graph=True)[0]
    
       
        zeta_obs_rho_water = 10

        zeta_pde_AC_real = 10
        zeta_pde_AC_imag = 10
        zeta_pde_structure_real_x = 10
        zeta_pde_structure_imag_x = 10
        zeta_pde_structure_real_y = 10
        zeta_pde_structure_imag_y = 10

    # scale zeta
        norm_rho_water = torch.norm(zeta_obs_rho_water * grad_x_cur_obs_rho_water)
        scale_factor = 30.0 / norm_rho_water
        zeta_obs_rho_water = zeta_obs_rho_water * scale_factor

        if i <= 0.9 * num_steps:

            x_next = x_next - zeta_obs_rho_water * grad_x_cur_obs_rho_water


        else:
            
            norm_pde_AC_real = torch.norm(zeta_pde_AC_real * grad_x_cur_pde_AC_real)
            scale_factor = 13 / norm_pde_AC_real
            zeta_pde_AC_real = zeta_pde_AC_real * scale_factor
            norm_pde_AC_imag = torch.norm(zeta_pde_AC_imag * grad_x_cur_pde_AC_imag)
            scale_factor = 13 / norm_pde_AC_imag
            zeta_pde_AC_imag = zeta_pde_AC_imag * scale_factor

            norm_pde_structure_real_x = torch.norm(zeta_pde_structure_real_x * grad_x_cur_pde_structure_real_x)
            scale_factor = 10 / norm_pde_structure_real_x
            zeta_pde_structure_real_x = zeta_pde_structure_real_x * scale_factor

            norm_pde_structure_imag_x = torch.norm(zeta_pde_structure_imag_x * grad_x_cur_pde_structure_imag_x)
            scale_factor = 10 / norm_pde_structure_imag_x
            zeta_pde_structure_imag_x = zeta_pde_structure_imag_x * scale_factor

            norm_pde_structure_real_y = torch.norm(zeta_pde_structure_real_y * grad_x_cur_pde_structure_real_y)
            scale_factor = 20 / norm_pde_structure_real_y
            zeta_pde_structure_real_y = zeta_pde_structure_real_y * scale_factor

            norm_pde_structure_imag_y = torch.norm(zeta_pde_structure_imag_y * grad_x_cur_pde_structure_imag_y)
            scale_factor = 10 / norm_pde_structure_imag_y
            zeta_pde_structure_imag_y = zeta_pde_structure_imag_y * scale_factor

            x_next = (x_next -  (zeta_obs_rho_water * grad_x_cur_obs_rho_water ) -
                     1 * (zeta_pde_AC_real * grad_x_cur_pde_AC_real + zeta_pde_AC_imag * grad_x_cur_pde_AC_imag
                           + zeta_pde_structure_real_x * grad_x_cur_pde_structure_real_x + zeta_pde_structure_imag_x * grad_x_cur_pde_structure_imag_x
                           + zeta_pde_structure_real_y * grad_x_cur_pde_structure_real_y + zeta_pde_structure_imag_y * grad_x_cur_pde_structure_imag_y))
            

    ############################ Save the data ############################
    x_final = x_next
    rho_water_final = x_final[:,0,:,:].unsqueeze(0)
    p_t_real_final = x_final[:,1,:,:].unsqueeze(0)
    p_t_imag_final = x_final[:,2,:,:].unsqueeze(0)
    Sxx_real_final = x_final[:,3,:,:].unsqueeze(0)
    Sxx_imag_final = x_final[:,4,:,:].unsqueeze(0)
    Sxy_real_final = x_final[:,5,:,:].unsqueeze(0)
    Sxy_imag_final = x_final[:,6,:,:].unsqueeze(0)
    Syy_real_final = x_final[:,7,:,:].unsqueeze(0)
    Syy_imag_final = x_final[:,8,:,:].unsqueeze(0)
    x_u_real_final = x_final[:,9,:,:].unsqueeze(0)
    x_u_imag_final = x_final[:,10,:,:].unsqueeze(0)
    x_v_real_final = x_final[:,11,:,:].unsqueeze(0)
    x_v_imag_final = x_final[:,12,:,:].unsqueeze(0)
    

    rho_water_final = ((rho_water_final+0.9)/1.8 *(max_rho_water - min_rho_water) + min_rho_water).to(torch.float64)
    p_t_real_final = ((p_t_real_final+0.9)/1.8 *(range_data['p_t']['max_real'] - range_data['p_t']['min_real']) + range_data['p_t']['min_real']).to(torch.float64)
    p_t_imag_final = ((p_t_imag_final+0.9)/1.8 *(range_data['p_t']['max_imag'] - range_data['p_t']['min_imag']) + range_data['p_t']['min_imag']).to(torch.float64)
    Sxx_real_final = ((Sxx_real_final+0.9)/1.8 *(range_data['Sxx']['max_real'] - range_data['Sxx']['min_real']) + range_data['Sxx']['min_real']).to(torch.float64)
    Sxx_imag_final = ((Sxx_imag_final+0.9)/1.8 *(range_data['Sxx']['max_imag'] - range_data['Sxx']['min_imag']) + range_data['Sxx']['min_imag']).to(torch.float64)
    Sxy_real_final = ((Sxy_real_final+0.9)/1.8 *(range_data['Sxy']['max_real'] - range_data['Sxy']['min_real']) + range_data['Sxy']['min_real']).to(torch.float64)
    Sxy_imag_final = ((Sxy_imag_final+0.9)/1.8 *(range_data['Sxy']['max_imag'] - range_data['Sxy']['min_imag']) + range_data['Sxy']['min_imag']).to(torch.float64)
    Syy_real_final = ((Syy_real_final+0.9)/1.8 *(range_data['Syy']['max_real'] - range_data['Syy']['min_real']) + range_data['Syy']['min_real']).to(torch.float64)
    Syy_imag_final = ((Syy_imag_final+0.9)/1.8 *(range_data['Syy']['max_imag'] - range_data['Syy']['min_imag']) + range_data['Syy']['min_imag']).to(torch.float64)
    x_u_real_final = ((x_u_real_final+0.9)/1.8 *(range_data['x_u']['max_real'] - range_data['x_u']['min_real']) + range_data['x_u']['min_real']).to(torch.float64)
    x_u_imag_final = ((x_u_imag_final+0.9)/1.8 *(range_data['x_u']['max_imag'] - range_data['x_u']['min_imag']) + range_data['x_u']['min_imag']).to(torch.float64)
    x_v_real_final = ((x_v_real_final+0.9)/1.8 *(range_data['x_v']['max_real'] - range_data['x_v']['min_real']) + range_data['x_v']['min_real']).to(torch.float64)
    x_v_imag_final = ((x_v_imag_final+0.9)/1.8 *(range_data['x_v']['max_imag'] - range_data['x_v']['min_imag']) + range_data['x_v']['min_imag']).to(torch.float64)

    p_t_final = torch.complex(p_t_real_final, p_t_imag_final)
    Sxx_final = torch.complex(Sxx_real_final, Sxx_imag_final)
    Sxy_final = torch.complex(Sxy_real_final, Sxy_imag_final)
    Syy_final = torch.complex(Syy_real_final, Syy_imag_final)
    x_u_final = torch.complex(x_u_real_final, x_u_imag_final)
    x_v_final = torch.complex(x_v_real_final, x_v_imag_final)

    relative_error_rho_water = torch.norm(rho_water_final - rho_water_GT, 2) / torch.norm(rho_water_GT, 2)
    relative_error_p_t = torch.norm(p_t_final - p_t_GT, 2) / torch.norm(p_t_GT, 2)
    relative_error_Sxx = torch.norm(Sxx_final - Sxx_GT, 2) / torch.norm(Sxx_GT, 2)
    relative_error_Sxy = torch.norm(Sxy_final - Sxy_GT, 2) / torch.norm(Sxy_GT, 2)
    relative_error_Syy = torch.norm(Syy_final - Syy_GT, 2) / torch.norm(Syy_GT, 2)
    relative_error_x_u = torch.norm(x_u_final - x_u_GT, 2) / torch.norm(x_u_GT, 2)
    relative_error_x_v = torch.norm(x_v_final - x_v_GT, 2) / torch.norm(x_v_GT, 2)

    print(f'Relative error of rho_water: {relative_error_rho_water}')
    print(f'Relative error of p_t: {relative_error_p_t}')
    print(f'Relative error of Sxx: {relative_error_Sxx}')
    print(f'Relative error of Sxy: {relative_error_Sxy}')
    print(f'Relative error of Syy: {relative_error_Syy}')
    print(f'Relative error of x_u: {relative_error_x_u}')
    print(f'Relative error of x_v: {relative_error_x_v}')

    rho_water_final = rho_water_final.detach().cpu().numpy()
    p_t_final = p_t_final.detach().cpu().numpy()
    Sxx_final = Sxx_final.detach().cpu().numpy()
    Sxy_final = Sxy_final.detach().cpu().numpy()
    Syy_final = Syy_final.detach().cpu().numpy()
    x_u_final = x_u_final.detach().cpu().numpy()
    x_v_final = x_v_final.detach().cpu().numpy()

    scipy.io.savemat('VA_results.mat', {'rho_water': rho_water_final, 'p_t': p_t_final, 'Sxx': Sxx_final, 'Sxy': Sxy_final, 'Syy': Syy_final, 'x_u': x_u_final, 'x_v': x_v_final})
    print('Done.')