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
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask


def identify_mater(circle_params, device=torch.device('cuda')):
    mater_iden = torch.zeros( 128, 128, device=device)
    circle_params = circle_params.squeeze(0)
    cx, cy, r = map(float, circle_params) 
    coords = (torch.arange(128, device=device) - 63.5) * 0.001
    xx, yy = torch.meshgrid(coords, coords, indexing='ij')


    mater_iden = torch.where((xx-cx)**2 + (yy-cy)**2 <= r**2, 1, -1)
    # in 1, out -1
    return mater_iden
    

def generate_separa_PDE_mater(mater_iden, device=torch.device('cuda')):

    rho_air = 1.24246
    rho_copper = 8960
    Crho_air = 1005.10779
    Crho_copper = 385
    kappa_air = 0.02505
    kappa_copper = 400

    rho = torch.where(mater_iden > 1e-5, rho_copper, rho_air)
    Crho = torch.where(mater_iden > 1e-5, Crho_copper, Crho_air)
    kappa = torch.where(mater_iden > 1e-5, kappa_copper, kappa_air)

    rho = rho.permute(1, 0)
    Crho = Crho.permute(1, 0)
    kappa = kappa.permute(1, 0)

    # scipy.io.savemat('pho.mat', {'pho': pho.cpu().detach().numpy()})
    # scipy.io.savemat('Cpho.mat', {'Cpho': Cpho.cpu().detach().numpy()})
    # scipy.io.savemat('kappa.mat', {'kappa': kappa.cpu().detach().numpy()})

    return rho, Crho, kappa

def get_NS_heat_loss(Q_heat, u_u, u_v, T, Q_heat_GT, u_u_GT, u_v_GT, T_GT, Q_heat_mask, u_u_mask, u_v_mask, T_mask, mater_iden, device=torch.device('cuda')):
    """Return the loss of the NS_heat equation and the observation loss."""

    rho, Crho, kappa = generate_separa_PDE_mater(mater_iden)

    delta_x = 0.128/128 # 1mm
    delta_y = 0.128/128 # 1mm
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # Continuity_NS
    grad_x_next_x_NS = F.conv2d(u_u, deriv_x, padding=(0, 1))
    grad_x_next_y_NS = F.conv2d(u_v, deriv_y, padding=(1, 0))
    result_NS = grad_x_next_x_NS + grad_x_next_y_NS

    # T_filed
    grad_x_next_x_T = F.conv2d(T, deriv_x, padding=(0, 1))
    grad_x_next_y_T = F.conv2d(T, deriv_y, padding=(1, 0))
    Laplac_T = F.conv2d(grad_x_next_x_T, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_T, deriv_y, padding=(1, 0))

    result_heat = rho * Crho * (u_u * grad_x_next_x_T + u_v * grad_x_next_y_T) - kappa * Laplac_T - Q_heat
    
    pde_loss_NS = result_NS
    pde_loss_heat = result_heat

    pde_loss_NS = pde_loss_NS.squeeze()
    pde_loss_heat = pde_loss_heat.squeeze()
    
    pde_loss_heat = pde_loss_heat/1000000
    pde_loss_NS = pde_loss_NS/1000
    
    scipy.io.savemat('test_rho.mat', {'rho': rho.cpu().detach().numpy()})
    scipy.io.savemat('test_Crho.mat', {'Crho': Crho.cpu().detach().numpy()})
    scipy.io.savemat('test_kappa.mat', {'kappa': kappa.cpu().detach().numpy()})
    scipy.io.savemat('test_Laplac_T.mat', {'Laplac_T': Laplac_T.cpu().detach().numpy()})
    scipy.io.savemat('test_Q_heat.mat', {'Q_heat': Q_heat.cpu().detach().numpy()})
    scipy.io.savemat('test_u_u.mat', {'u_u': u_u.cpu().detach().numpy()})
    scipy.io.savemat('test_u_v.mat', {'u_u': u_v.cpu().detach().numpy()})


    observation_loss_Q_heat = (Q_heat - Q_heat_GT).squeeze()
    observation_loss_Q_heat = observation_loss_Q_heat * Q_heat_mask  
    observation_loss_u_u = (u_u - u_u_GT).squeeze()
    observation_loss_u_u = observation_loss_u_u * u_u_mask  
    observation_loss_u_v = (u_v - u_v_GT).squeeze()
    observation_loss_u_v = observation_loss_u_v * u_v_mask  
    observation_loss_T = (T - T_GT).squeeze()
    observation_loss_T = observation_loss_T * T_mask

    return pde_loss_NS, pde_loss_heat, observation_loss_Q_heat, observation_loss_u_u, observation_loss_u_v, observation_loss_T




def generate_NS_heat(config):
    """Generate NS_heat equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset'][0]
    device = config['generate']['device']

    Q_heat_GT_path = os.path.join(datapath, "Q_heat", f"{offset}.mat")
    # print(Q_heat_GT_path)
    Q_heat_GT = sio.loadmat(Q_heat_GT_path)['export_Q_heat']
    Q_heat_GT = torch.tensor(Q_heat_GT, dtype=torch.float64, device=device)

    u_u_GT_path = os.path.join(datapath, "u_u", f"{offset}.mat")
    u_u_GT = sio.loadmat(u_u_GT_path)['export_u_u']
    u_u_GT = torch.tensor(u_u_GT, device=device)
    
    u_v_GT_path = os.path.join(datapath, "u_v", f"{offset}.mat")
    u_v_GT = sio.loadmat(u_v_GT_path)['export_u_v']
    u_v_GT = torch.tensor(u_v_GT, dtype=torch.float64, device=device)

    T_GT_path = os.path.join(datapath, "T", f"{offset}.mat")
    T_GT = sio.loadmat(T_GT_path)['export_T']
    T_GT = torch.tensor(T_GT, dtype=torch.float64, device=device)
    
    circle_GT_path = os.path.join(datapath, "circlecsv", f"{offset}.csv")
    circle_GT = pd.read_csv(circle_GT_path, header=None)
    circle_GT = torch.tensor(circle_GT.values, dtype=torch.float64)

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
    known_index_Q_heat = random_index(500, 128, seed=3)
    known_index_u_u = random_index(500, 128, seed=2)
    known_index_u_v = random_index(500, 128, seed=1)
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
        Q_heat_N = x_N[:,0,:,:].unsqueeze(0)
        u_u_N = x_N[:,1,:,:].unsqueeze(0)
        u_v_N = x_N[:,2,:,:].unsqueeze(0)
        T_N = x_N[:,3,:,:].unsqueeze(0)

        circle_iden = identify_mater(circle_GT)

        # inv_normalization
        range_allQ_heat_paths = "/data/training/NS_heat/Q_heat/range_allQ_heat.mat"
        range_allQ_heat = sio.loadmat(range_allQ_heat_paths)['range_allQ_heat']
        range_allQ_heat = torch.tensor(range_allQ_heat, device=device)

        max_Q_heat = range_allQ_heat[0,1]
        min_Q_heat = range_allQ_heat[0,0]

        range_allu_u_paths = "/data/training/NS_heat/u_u/range_allu_u.mat"
        range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']
        range_allu_u = torch.tensor(range_allu_u, device=device)

        max_u_u = range_allu_u[0,1]
        min_u_u = range_allu_u[0,0]

        range_allu_v_paths = "/data/training/NS_heat/u_v/range_allu_v.mat"
        range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']
        range_allu_v = torch.tensor(range_allu_v, device=device)

        max_u_v = range_allu_v[0,1]
        min_u_v = range_allu_v[0,0]

        range_allT_paths = "/data/training/NS_heat/T/range_allT.mat"
        range_allT = sio.loadmat(range_allT_paths)['range_allT']
        range_allT = torch.tensor(range_allT, device=device)

        max_T = range_allT[0,1]
        min_T = range_allT[0,0]

        Q_heat_N = ((Q_heat_N+0.9)/1.8 *(max_Q_heat - min_Q_heat) + min_Q_heat).to(torch.float64)
        u_u_N = ((u_u_N+0.9)/1.8 *(max_u_u - min_u_u) + min_u_u).to(torch.float64)
        u_v_N = ((u_v_N+0.9)/1.8 *(max_u_v - min_u_v) + min_u_v).to(torch.float64)
        T_N = ((T_N+0.9)/1.8 *(max_T - min_T) + min_T).to(torch.float64)


        # Compute the loss

        pde_loss_NS, pde_loss_heat, observation_loss_Q_heat, observation_loss_u_u, observation_loss_u_v, observation_loss_T = get_NS_heat_loss(Q_heat_N, u_u_N, u_v_N, T_N, Q_heat_GT, u_u_GT, u_v_GT, T_GT, known_index_Q_heat, known_index_u_u, known_index_u_v, known_index_T, circle_iden, device=device)
        
        L_pde_NS = torch.norm(pde_loss_NS, 2)/(128*128)
        L_pde_heat = torch.norm(pde_loss_heat, 2)/(128*128)

        L_obs_Q_heat = torch.norm(observation_loss_Q_heat, 2)/500
        L_obs_u_u = torch.norm(observation_loss_u_u, 2)/500
        L_obs_u_v = torch.norm(observation_loss_u_v, 2)/500
        L_obs_T = torch.norm(observation_loss_T, 2)/500

        output_file_path = "inference_losses.jsonl"
        if i % 5 == 0:
            log_entry = {
              "step": i,
              "L_pde_NS": L_pde_NS.tolist(),
              "L_pde_heat": L_pde_heat.tolist(),
               "L_obs_Q_heat": L_obs_Q_heat.tolist(),
               "L_obs_u_u": L_obs_u_u.tolist(),
               "L_obs_u_v": L_obs_u_v.tolist(),
              "L_obs_T": L_obs_T.tolist()
           }
            with open(output_file_path, "a") as file:
                json.dump(log_entry, file)
                file.write("\n")  

        grad_x_cur_obs_Q_heat = torch.autograd.grad(outputs=L_obs_Q_heat, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u_u = torch.autograd.grad(outputs=L_obs_u_u, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u_v = torch.autograd.grad(outputs=L_obs_u_v, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_T = torch.autograd.grad(outputs=L_obs_T, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_NS = torch.autograd.grad(outputs=L_pde_NS, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_heat = torch.autograd.grad(outputs=L_pde_heat, inputs=x_cur)[0]


        zeta_obs_Q_heat = 10
        zeta_obs_u_u = 10
        zeta_obs_u_v = 10
        zeta_obs_T = 10
        zeta_pde_NS = 10
        zeta_pde_heat = 10

    # scale zeta
        norm_Q_heat = torch.norm(zeta_obs_Q_heat * grad_x_cur_obs_Q_heat)
        scale_factor =28/ norm_Q_heat
        zeta_obs_Q_heat = zeta_obs_Q_heat * scale_factor

        

        if i <= 0 * num_steps:

            x_next = x_next - zeta_obs_Q_heat * grad_x_cur_obs_Q_heat
      
        else:
            
            norm_pde_NS = torch.norm(zeta_pde_NS * grad_x_cur_pde_NS)
            scale_factor = 0.05 / norm_pde_NS   
            zeta_pde_NS = zeta_pde_NS * scale_factor


            norm_pde_heat = torch.norm(zeta_pde_heat * grad_x_cur_pde_heat)
            scale_factor = 0.0005 / norm_pde_heat
            zeta_pde_heat = zeta_pde_heat * scale_factor

            x_next = x_next - 1*(zeta_obs_Q_heat * grad_x_cur_obs_Q_heat )- 1* (zeta_pde_NS * grad_x_cur_pde_NS + zeta_pde_heat * grad_x_cur_pde_heat)

    ############################ Save the data ############################
    x_final = x_next
    Q_heat_final = x_final[:,0,:,:].unsqueeze(0)
    u_u_final = x_final[:,1,:,:].unsqueeze(0)
    u_v_final = x_final[:,2,:,:].unsqueeze(0)
    T_final = x_final[:,3,:,:].unsqueeze(0)    
    

    Q_heat_final = ((Q_heat_final+0.9)/1.8 *(max_Q_heat - min_Q_heat) + min_Q_heat).to(torch.float64)
    u_u_final = ((u_u_final+0.9)/1.8 *(max_u_u - min_u_u) + min_u_u).to(torch.float64)
    u_v_final = ((u_v_final+0.9)/1.8 *(max_u_v - min_u_v) + min_u_v).to(torch.float64)
    T_final = ((T_final+0.9)/1.8 *(max_T - min_T) + min_T).to(torch.float64)


    relative_error_Q_heat = torch.norm(Q_heat_final - Q_heat_GT, 2) / torch.norm(Q_heat_GT, 2)
    relative_error_u_u = torch.norm(u_u_final - u_u_GT, 2) / torch.norm(u_u_GT, 2)
    relative_error_u_v = torch.norm(u_v_final - u_v_GT, 2) / torch.norm(u_v_GT, 2)
    relative_error_T = torch.norm(T_final - T_GT, 2) / torch.norm(T_GT, 2)  
    

    print(f'Relative error of Q_heat: {relative_error_Q_heat}')
    print(f'Relative error of u_u: {relative_error_u_u}')
    print(f'Relative error of u_v: {relative_error_u_v}')
    print(f'Relative error of T: {relative_error_T}')

    Q_heat_final = Q_heat_final.detach().cpu().numpy()
    u_u_final = u_u_final.detach().cpu().numpy()
    u_v_final = u_v_final.detach().cpu().numpy()
    T_final = T_final.detach().cpu().numpy()

    scipy.io.savemat('NS_heat_results.mat', {'Q_heat': Q_heat_final, 'u_u': u_u_final, 'u_v': u_v_final, 'T': T_final})
    print('Done.')


