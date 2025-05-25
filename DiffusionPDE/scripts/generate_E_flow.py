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

    

def get_E_flow_loss(kappa, ec_V, u_flow, v_flow, kappa_GT, ec_V_GT, u_flow_GT, v_flow_GT, kappa_mask, ec_V_mask, u_flow_mask, v_flow_mask, device=torch.device('cuda')):
    """Return the loss of the E_flow equation and the observation loss."""

    delta_x = 1.28e-3/128 # 1mm
    delta_y = 1.28e-3/128 # 1mm
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # Continuity_NS
    grad_x_next_x_NS = F.conv2d(u_flow, deriv_x, padding=(0, 1))
    grad_x_next_y_NS = F.conv2d(v_flow, deriv_y, padding=(1, 0))
    result_NS = grad_x_next_x_NS + grad_x_next_y_NS

    # Continuity_J
    grad_x_next_x_V = F.conv2d(ec_V, deriv_x, padding=(0, 1))
    grad_x_next_y_V = F.conv2d(ec_V, deriv_y, padding=(1, 0))

    grad_x_next_x_J = F.conv2d(kappa*grad_x_next_x_V, deriv_x, padding=(0, 1))
    grad_x_next_y_J = F.conv2d(kappa*grad_x_next_y_V, deriv_y, padding=(1, 0))

    result_J = grad_x_next_x_J + grad_x_next_y_J
    
    pde_loss_NS = result_NS
    pde_loss_J = result_J

    pde_loss_NS = pde_loss_NS.squeeze()
    pde_loss_J = pde_loss_J.squeeze()
    
    pde_loss_NS = pde_loss_NS/1000
    pde_loss_J = pde_loss_J/1000000

    pde_loss_NS[0, :] = 0
    pde_loss_NS[-1, :] = 0
    pde_loss_NS[:, 0] = 0
    pde_loss_NS[:, -1] = 0

    pde_loss_J[0, :] = 0
    pde_loss_J[-1, :] = 0
    pde_loss_J[:, 0] = 0
    pde_loss_J[:, -1] = 0

    pde_loss_J[(pde_loss_J > 0.5) | (pde_loss_J < -0.5)] = 0
    pde_loss_NS[(pde_loss_NS > 0.05) | (pde_loss_NS < -0.05)] = 0

    observation_loss_kappa = (kappa - kappa_GT).squeeze()
    observation_loss_kappa = observation_loss_kappa * kappa_mask  
    observation_loss_ec_V = (ec_V - ec_V_GT).squeeze()
    observation_loss_ec_V = observation_loss_ec_V * ec_V_mask
    observation_loss_u_flow = (u_flow - u_flow_GT).squeeze()
    observation_loss_u_flow = observation_loss_u_flow * u_flow_mask  
    observation_loss_v_flow = (v_flow - v_flow_GT).squeeze()
    observation_loss_v_flow = observation_loss_v_flow * v_flow_mask  

    return pde_loss_NS, pde_loss_J, observation_loss_kappa, observation_loss_ec_V, observation_loss_u_flow, observation_loss_v_flow


def generate_E_flow(config):
    """Generate E_flow equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset'][0]
    device = config['generate']['device']

    kappa_GT_path = os.path.join(datapath, "kappa", f"{offset}.mat")
    # print(kappa_GT_path)
    kappa_GT = sio.loadmat(kappa_GT_path)['export_kappa']
    kappa_GT = torch.tensor(kappa_GT, dtype=torch.float64, device=device)

    ec_V_GT_path = os.path.join(datapath, "ec_V", f"{offset}.mat")
    ec_V_GT = sio.loadmat(ec_V_GT_path)['export_ec_V']
    ec_V_GT = torch.tensor(ec_V_GT, dtype=torch.float64, device=device)

    u_flow_GT_path = os.path.join(datapath, "u_flow", f"{offset}.mat")
    u_flow_GT = sio.loadmat(u_flow_GT_path)['export_u_flow']
    u_flow_GT = torch.tensor(u_flow_GT, dtype=torch.float64, device=device)
    
    v_flow_GT_path = os.path.join(datapath, "v_flow", f"{offset}.mat")
    v_flow_GT = sio.loadmat(v_flow_GT_path)['export_v_flow']
    v_flow_GT = torch.tensor(v_flow_GT, dtype=torch.float64, device=device)

    
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
    known_index_kappa = random_index(500, 128, seed=3)
    known_index_ec_V = random_index(500, 128, seed=2)
    known_index_u_flow = random_index(500, 128, seed=1)
    known_index_v_flow = random_index(500, 128, seed=0)
    
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
        kappa_N = x_N[:,0,:,:].unsqueeze(0)
        ec_V_N = x_N[:,1,:,:].unsqueeze(0)
        u_flow_N = x_N[:,2,:,:].unsqueeze(0)
        v_flow_N = x_N[:,3,:,:].unsqueeze(0)

        # inv_normalization
        range_allkappa_paths = "data/training/E_flow/kappa/range_allkappa.mat"
        range_allkappa = sio.loadmat(range_allkappa_paths)['range_allkappa']
        range_allkappa = torch.tensor(range_allkappa, device=device)

        max_kappa = range_allkappa[0,1]
        min_kappa = range_allkappa[0,0]

        range_allec_V_paths = "data/training/E_flow/ec_V/range_allec_V.mat"
        range_allec_V = sio.loadmat(range_allec_V_paths)['range_allec_V']
        range_allec_V = torch.tensor(range_allec_V, device=device)

        max_ec_V = range_allec_V[0,1]
        min_ec_V = range_allec_V[0,0]

        range_allu_flow_paths = "data/training/E_flow/u_flow/range_allu_flow.mat"
        range_allu_flow = sio.loadmat(range_allu_flow_paths)['range_allu_flow']
        range_allu_flow = torch.tensor(range_allu_flow, device=device)

        max_u_flow = range_allu_flow[0,1]
        min_u_flow = range_allu_flow[0,0]

        range_allv_flow_paths = "data/training/E_flow/v_flow/range_allv_flow.mat"
        range_allv_flow = sio.loadmat(range_allv_flow_paths)['range_allv_flow']
        range_allv_flow = torch.tensor(range_allv_flow, device=device)

        max_v_flow = range_allv_flow[0,1]
        min_v_flow = range_allv_flow[0,0]


        kappa_N = ((kappa_N+0.9)/1.8 *(max_kappa - min_kappa) + min_kappa).to(torch.float64)
        ec_V_N = ((ec_V_N+0.9)/1.8 *(max_ec_V - min_ec_V) + min_ec_V).to(torch.float64)
        u_flow_N = ((u_flow_N+0.9)/1.8 *(max_u_flow - min_u_flow) + min_u_flow).to(torch.float64)
        v_flow_N = ((v_flow_N+0.9)/1.8 *(max_v_flow - min_v_flow) + min_v_flow).to(torch.float64)


        # Compute the loss

        pde_loss_NS, pde_loss_J, observation_loss_kappa, observation_loss_ec_V, observation_loss_u_flow, observation_loss_v_flow = get_E_flow_loss(kappa_N, ec_V_N, u_flow_N, v_flow_N, kappa_GT, ec_V_GT, u_flow_GT, v_flow_GT, known_index_kappa, known_index_ec_V, known_index_u_flow, known_index_v_flow, device=device)
        
        L_pde_NS = torch.norm(pde_loss_NS, 2)/(128*128)
        L_pde_J = torch.norm(pde_loss_J, 2)/(128*128)
        
        L_obs_kappa = torch.norm(observation_loss_kappa, 2)/500
        L_obs_ec_V = torch.norm(observation_loss_ec_V, 2)/500
        L_obs_u_flow = torch.norm(observation_loss_u_flow, 2)/500
        L_obs_v_flow = torch.norm(observation_loss_v_flow, 2)/500

        # print(L_pde_NS)
        # print(L_pde_J)

        # print(L_obs_Ez)
        # print(L_obs_T)

        output_file_path = "inference_losses.jsonl"
        if i % 10 == 0:
            log_entry = {
              "step": i,
              "L_pde_NS": L_pde_NS.tolist(),
              "L_pde_J": L_pde_J.tolist(),
               "L_obs_kappa": L_obs_kappa.tolist(),
               "L_obs_ec_V": L_obs_ec_V.tolist(),
               "L_obs_u_flow": L_obs_u_flow.tolist(),
               "L_obs_v_flow": L_obs_v_flow.tolist(),
           }
            with open(output_file_path, "a") as file:
                json.dump(log_entry, file)
                file.write("\n")  

        grad_x_cur_obs_kappa = torch.autograd.grad(outputs=L_obs_kappa, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_ec_V = torch.autograd.grad(outputs=L_obs_ec_V, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u_flow = torch.autograd.grad(outputs=L_obs_u_flow, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_v_flow = torch.autograd.grad(outputs=L_obs_v_flow, inputs=x_cur, retain_graph=True)[0]

        grad_x_cur_pde_NS = torch.autograd.grad(outputs=L_pde_NS, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_J = torch.autograd.grad(outputs=L_pde_J, inputs=x_cur, retain_graph=True)[0]

        # zeta_obs_mater = config['generate']['zeta_obs_mater']
        # zeta_obs_Ez = config['generate']['zeta_obs_Ez']
        # zeta_obs_T = 1e3*config['generate']['zeta_obs_T']
        # zeta_pde = config['generate']['zeta_pde']
       
        zeta_obs_kappa = 10
        zeta_obs_ec_V = 10
        zeta_obs_u_flow = 10
        zeta_obs_v_flow = 10

        zeta_pde_NS = 10
        zeta_pde_J = 10

    # scale zeta
        norm_kappa = torch.norm(zeta_obs_kappa * grad_x_cur_obs_kappa)
        scale_factor = 5 / norm_kappa
        zeta_obs_kappa = zeta_obs_kappa * scale_factor

        # norm_ec_V = torch.norm(zeta_obs_ec_V * grad_x_cur_obs_ec_V)
        # scale_factor = 1.0 / norm_ec_V
        # zeta_obs_ec_V = zeta_obs_ec_V * scale_factor

        # norm_u_flow = torch.norm(zeta_obs_u_flow * grad_x_cur_obs_u_flow)
        # scale_factor = 1.0 / norm_u_flow
        # zeta_obs_u_flow = zeta_obs_u_flow * scale_factor

        # norm_v_flow = torch.norm(zeta_obs_v_flow * grad_x_cur_obs_v_flow)
        # scale_factor = 1.0 / norm_v_flow
        # zeta_obs_v_flow = zeta_obs_v_flow * scale_factor

    
        if i <= 0.5 * num_steps:
        # if i <= 1 * num_steps:
            # x_next = x_next - zeta_obs_kappa * grad_x_cur_obs_kappa - zeta_obs_ec_V * grad_x_cur_obs_ec_V - zeta_obs_u_flow * grad_x_cur_obs_u_flow - zeta_obs_v_flow * grad_x_cur_obs_v_flow
           
            x_next = x_next - zeta_obs_kappa * grad_x_cur_obs_kappa
      
            # norm_value = torch.norm(zeta_obs_kappa * grad_x_cur_obs_kappa).item()
            # print(norm_value)

            # x_next = x_next

        else:
            
            norm_pde_NS = torch.norm(zeta_pde_NS * grad_x_cur_pde_NS)
            scale_factor = 0.4 / norm_pde_NS
            zeta_pde_NS = zeta_pde_NS * scale_factor

            norm_pde_J = torch.norm(zeta_pde_J * grad_x_cur_pde_J)
            scale_factor = 0.3 / norm_pde_J
            zeta_pde_J = zeta_pde_J * scale_factor

            # x_next = x_next - 0.1 * (zeta_obs_mater * grad_x_cur_obs_mater + zeta_obs_Ez * grad_x_cur_obs_Ez + zeta_obs_T * grad_x_cur_obs_T) - zeta_pde_E * grad_x_cur_pde_E - zeta_pde_T * grad_x_cur_pde_T

            # x_next = x_next - 0.8*(zeta_obs_kappa * grad_x_cur_obs_kappa + zeta_obs_ec_V * grad_x_cur_obs_ec_V + zeta_obsu_flow_ * grad_x_cur_obs_u_flow + zeta_obs_v_flow * grad_x_cur_obs_v_flow ) - 0.2* (zeta_pde_NS * grad_x_cur_pde_NS + zeta_pde_J * grad_x_cur_pde_J)
            
            x_next = x_next - zeta_obs_kappa * grad_x_cur_obs_kappa - (zeta_pde_NS * grad_x_cur_pde_NS + zeta_pde_J * grad_x_cur_pde_J)
           

            # norm_value = torch.norm(zeta_pde_NS * grad_x_cur_pde_NS).item()
            # print(norm_value)

    ############################ Save the data ############################
    x_final = x_next
    kappa_final = x_final[:,0,:,:].unsqueeze(0)
    ec_V_final = x_final[:,1,:,:].unsqueeze(0)
    u_flow_final = x_final[:,2,:,:].unsqueeze(0)
    v_flow_final = x_final[:,3,:,:].unsqueeze(0)
 
    

    kappa_final = ((kappa_final+0.9)/1.8 *(max_kappa - min_kappa) + min_kappa).to(torch.float64)
    ec_V_final = ((ec_V_final+0.9)/1.8 *(max_ec_V - min_ec_V) + min_ec_V).to(torch.float64)
    u_flow_final = ((u_flow_final+0.9)/1.8 *(max_u_flow - min_u_flow) + min_u_flow).to(torch.float64)
    v_flow_final = ((v_flow_final+0.9)/1.8 *(max_v_flow - min_v_flow) + min_v_flow).to(torch.float64)


    relative_error_kappa = torch.norm(kappa_final - kappa_GT, 2) / torch.norm(kappa_GT, 2)
    relative_error_ec_V = torch.norm(ec_V_final - ec_V_GT, 2) / torch.norm(ec_V_GT, 2)
    relative_error_u_flow = torch.norm(u_flow_final - u_flow_GT, 2) / torch.norm(u_flow_GT, 2)
    relative_error_v_flow = torch.norm(v_flow_final - v_flow_GT, 2) / torch.norm(v_flow_GT, 2)


    print(f'Relative error of kappa: {relative_error_kappa}')
    print(f'Relative error of ec_V: {relative_error_ec_V}')
    print(f'Relative error of u_flow: {relative_error_u_flow}')
    print(f'Relative error of v_flow: {relative_error_v_flow}')

    kappa_final = kappa_final.detach().cpu().numpy()
    ec_V_final = ec_V_final.detach().cpu().numpy()
    u_flow_final = u_flow_final.detach().cpu().numpy()
    v_flow_final = v_flow_final.detach().cpu().numpy()

    scipy.io.savemat('E_flow_results.mat', {'kappa': kappa_final, 'ec_V': ec_V_final, 'u_flow': u_flow_final, 'v_flow': v_flow_final})
    print('Done.')