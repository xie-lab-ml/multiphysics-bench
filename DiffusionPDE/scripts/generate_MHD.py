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

    

def get_MHD_loss(Br, Jx, Jy, Jz, u_u, u_v, Br_GT, Jx_GT, Jy_GT, Jz_GT, u_u_GT, u_v_GT, Br_mask, Jx_mask, Jy_mask, Jz_mask, u_u_mask, u_v_mask, device=torch.device('cuda')):
    """Return the loss of the MHD equation and the observation loss."""

    delta_x = 8e-2/128 # 1cm
    delta_y = 2.75e-2/128 # 1cm
    
    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

    # Continuity_NS
    grad_x_next_x_NS = F.conv2d(u_u, deriv_x, padding=(0, 1))
    grad_x_next_y_NS = F.conv2d(u_v, deriv_y, padding=(1, 0))
    result_NS = grad_x_next_x_NS + grad_x_next_y_NS

    # Continuity_J
    grad_x_next_x_J = F.conv2d(Jx, deriv_x, padding=(0, 1))
    grad_x_next_y_J = F.conv2d(Jy, deriv_y, padding=(1, 0))
    result_J = grad_x_next_x_J + grad_x_next_y_J
    
    pde_loss_NS = result_NS
    pde_loss_J = result_J

    pde_loss_NS = pde_loss_NS.squeeze()
    pde_loss_J = pde_loss_J.squeeze()
    
    pde_loss_NS = pde_loss_NS/100
    pde_loss_J = pde_loss_J/100

    pde_loss_NS[0, :] = 0
    pde_loss_NS[-1, :] = 0
    pde_loss_NS[:, 0] = 0
    pde_loss_NS[:, -1] = 0


    pde_loss_J[(pde_loss_J > 5) | (pde_loss_J < -5)] = 0

    # scipy.io.savemat('test_rho.mat', {'rho': rho.cpu().detach().numpy()})
    # scipy.io.savemat('test_Crho.mat', {'Crho': Crho.cpu().detach().numpy()})
    # scipy.io.savemat('test_kappa.mat', {'kappa': kappa.cpu().detach().numpy()})
    # scipy.io.savemat('test_Laplac_T.mat', {'Laplac_T': Laplac_T.cpu().detach().numpy()})
    # scipy.io.savemat('test_Q_heat.mat', {'Q_heat': Q_heat.cpu().detach().numpy()})
    # scipy.io.savemat('test_u_u.mat', {'u_u': u_u.cpu().detach().numpy()})
    # scipy.io.savemat('test_u_v.mat', {'u_u': u_v.cpu().detach().numpy()})


    observation_loss_Br = (Br - Br_GT).squeeze()
    observation_loss_Br = observation_loss_Br * Br_mask  
    observation_loss_Jx = (Jx - Jx_GT).squeeze()
    observation_loss_Jx = observation_loss_Jx * Jx_mask
    observation_loss_Jy = (Jy - Jy_GT).squeeze()
    observation_loss_Jy = observation_loss_Jy * Jy_mask
    observation_loss_Jz = (Jz - Jz_GT).squeeze()
    observation_loss_Jz = observation_loss_Jz * Jz_mask
    observation_loss_u_u = (u_u - u_u_GT).squeeze()
    observation_loss_u_u = observation_loss_u_u * u_u_mask  
    observation_loss_u_v = (u_v - u_v_GT).squeeze()
    observation_loss_u_v = observation_loss_u_v * u_v_mask  

    return pde_loss_NS, pde_loss_J, observation_loss_Br, observation_loss_Jx, observation_loss_Jy, observation_loss_Jz, observation_loss_u_u, observation_loss_u_v



def generate_MHD(config):
    """Generate MHD equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset'][0]
    device = config['generate']['device']

    Br_GT_path = os.path.join(datapath, "Br", f"{offset}.mat")
    # print(Br_GT_path)
    Br_GT = sio.loadmat(Br_GT_path)['export_Br']
    Br_GT = torch.tensor(Br_GT, dtype=torch.float64, device=device)

    Jx_GT_path = os.path.join(datapath, "Jx", f"{offset}.mat")
    Jx_GT = sio.loadmat(Jx_GT_path)['export_Jx']
    Jx_GT = torch.tensor(Jx_GT, dtype=torch.float64, device=device)

    Jy_GT_path = os.path.join(datapath, "Jy", f"{offset}.mat")
    Jy_GT = sio.loadmat(Jy_GT_path)['export_Jy']
    Jy_GT = torch.tensor(Jy_GT, dtype=torch.float64, device=device)

    Jz_GT_path = os.path.join(datapath, "Jz", f"{offset}.mat")
    Jz_GT = sio.loadmat(Jz_GT_path)['export_Jz']
    Jz_GT = torch.tensor(Jz_GT, dtype=torch.float64, device=device)

    u_u_GT_path = os.path.join(datapath, "u_u", f"{offset}.mat")
    u_u_GT = sio.loadmat(u_u_GT_path)['export_u']
    u_u_GT = torch.tensor(u_u_GT, dtype=torch.float64, device=device)
    
    u_v_GT_path = os.path.join(datapath, "u_v", f"{offset}.mat")
    u_v_GT = sio.loadmat(u_v_GT_path)['export_v']
    u_v_GT = torch.tensor(u_v_GT, dtype=torch.float64, device=device)

    
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
    known_index_Br = random_index(500, 128, seed=5)
    known_index_Jx = random_index(500, 128, seed=4)
    known_index_Jy = random_index(500, 128, seed=3)
    known_index_Jz = random_index(500, 128, seed=2)
    known_index_u_u = random_index(500, 128, seed=1)
    known_index_u_v = random_index(500, 128, seed=0)
    
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
        Br_N = x_N[:,0,:,:].unsqueeze(0)
        Jx_N = x_N[:,1,:,:].unsqueeze(0)
        Jy_N = x_N[:,2,:,:].unsqueeze(0)
        Jz_N = x_N[:,3,:,:].unsqueeze(0)
        u_u_N = x_N[:,4,:,:].unsqueeze(0)
        u_v_N = x_N[:,5,:,:].unsqueeze(0)

        # inv_normalization
        range_allBr_paths = "/data/training/MHD/Br/range_allBr.mat"
        range_allBr = sio.loadmat(range_allBr_paths)['range_allBr']
        range_allBr = torch.tensor(range_allBr, device=device)

        max_Br = range_allBr[0,1]
        min_Br = range_allBr[0,0]

        range_allJx_paths = "/data/training/MHD/Jx/range_allJx.mat"
        range_allJx = sio.loadmat(range_allJx_paths)['range_allJx']
        range_allJx = torch.tensor(range_allJx, device=device)

        max_Jx = range_allJx[0,1]
        min_Jx = range_allJx[0,0]

        range_allJy_paths = "/data/training/MHD/Jy/range_allJy.mat"
        range_allJy = sio.loadmat(range_allJy_paths)['range_allJy']
        range_allJy = torch.tensor(range_allJy, device=device)

        max_Jy = range_allJy[0,1]
        min_Jy = range_allJy[0,0]

        range_allJz_paths = "/data/training/MHD/Jz/range_allJz.mat"
        range_allJz = sio.loadmat(range_allJz_paths)['range_allJz']
        range_allJz = torch.tensor(range_allJz, device=device)

        max_Jz = range_allJz[0,1]
        min_Jz = range_allJz[0,0]

        range_allu_u_paths = "/data/training/MHD/u_u/range_allu_u.mat"
        range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']
        range_allu_u = torch.tensor(range_allu_u, device=device)

        max_u_u = range_allu_u[0,1]
        min_u_u = range_allu_u[0,0]

        range_allu_v_paths = "/data/training/MHD/u_v/range_allu_v.mat"
        range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']
        range_allu_v = torch.tensor(range_allu_v, device=device)

        max_u_v = range_allu_v[0,1]
        min_u_v = range_allu_v[0,0]


        Br_N = ((Br_N+0.9)/1.8 *(max_Br - min_Br) + min_Br).to(torch.float64)
        Jx_N = ((Jx_N+0.9)/1.8 *(max_Jx - min_Jx) + min_Jx).to(torch.float64)
        Jy_N = ((Jy_N+0.9)/1.8 *(max_Jy - min_Jy) + min_Jy).to(torch.float64)
        Jz_N = ((Jz_N+0.9)/1.8 *(max_Jz - min_Jz) + min_Jz).to(torch.float64)
        u_u_N = ((u_u_N+0.9)/1.8 *(max_u_u - min_u_u) + min_u_u).to(torch.float64)
        u_v_N = ((u_v_N+0.9)/1.8 *(max_u_v - min_u_v) + min_u_v).to(torch.float64)


        # Compute the loss

        pde_loss_NS, pde_loss_J, observation_loss_Br, observation_loss_Jx, observation_loss_Jy, observation_loss_Jz, observation_loss_u_u, observation_loss_u_v = get_MHD_loss(Br_N, Jx_N, Jy_N, Jz_N, u_u_N, u_v_N, Br_GT, Jx_GT, Jy_GT, Jz_GT, u_u_GT, u_v_GT, known_index_Br, known_index_Jx, known_index_Jy, known_index_Jz, known_index_u_u, known_index_u_v, device=device)
        
        L_pde_NS = torch.norm(pde_loss_NS, 2)/(128*128)
        L_pde_J = torch.norm(pde_loss_J, 2)/(128*128)
        
        L_obs_Br = torch.norm(observation_loss_Br, 2)/500
        L_obs_Jx = torch.norm(observation_loss_Jx, 2)/500
        L_obs_Jy = torch.norm(observation_loss_Jy, 2)/500
        L_obs_Jz = torch.norm(observation_loss_Jz, 2)/500
        L_obs_u_u = torch.norm(observation_loss_u_u, 2)/500
        L_obs_u_v = torch.norm(observation_loss_u_v, 2)/500


        output_file_path = "inference_losses.jsonl"
        if i % 5 == 0:
            log_entry = {
              "step": i,
            #   "L_pde_NS": L_pde_NS.tolist(),
            #   "L_pde_heat": L_pde_heat.tolist(),
               "L_obs_Br": L_obs_Br.tolist(),
               "L_obs_Jx": L_obs_Jx.tolist(),
               "L_obs_Jy": L_obs_Jy.tolist(),
               "L_obs_Jz": L_obs_Jz.tolist(),
               "L_obs_u_u": L_obs_u_u.tolist(),
               "L_obs_u_v": L_obs_u_v.tolist(),
           }
            with open(output_file_path, "a") as file:
                json.dump(log_entry, file)
                file.write("\n")  

        grad_x_cur_obs_Br = torch.autograd.grad(outputs=L_obs_Br, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_Jx = torch.autograd.grad(outputs=L_obs_Jx, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_Jy = torch.autograd.grad(outputs=L_obs_Jy, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_Jz = torch.autograd.grad(outputs=L_obs_Jz, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u_u = torch.autograd.grad(outputs=L_obs_u_u, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u_v = torch.autograd.grad(outputs=L_obs_u_v, inputs=x_cur, retain_graph=True)[0]

        grad_x_cur_pde_NS = torch.autograd.grad(outputs=L_pde_NS, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde_J = torch.autograd.grad(outputs=L_pde_J, inputs=x_cur, retain_graph=True)[0]

       
        zeta_obs_Br = 10
        zeta_obs_Jx = 10
        zeta_obs_Jy = 10
        zeta_obs_Jz = 10
        zeta_obs_u_u = 10
        zeta_obs_u_v = 10

        zeta_pde_NS = 10
        zeta_pde_J = 10

    # scale zeta
        norm_Br = torch.norm(zeta_obs_Br * grad_x_cur_obs_Br)
        scale_factor = 0.0005 / norm_Br
        zeta_obs_Br = zeta_obs_Br * scale_factor
    
        if i <= 0.5 * num_steps:

            x_next = x_next - zeta_obs_Br * grad_x_cur_obs_Br
      
        else:
            
            norm_pde_NS = torch.norm(zeta_pde_NS * grad_x_cur_pde_NS)
            scale_factor = 0.1 / norm_pde_NS
            zeta_pde_NS = zeta_pde_NS * scale_factor

            norm_pde_J = torch.norm(zeta_pde_J * grad_x_cur_pde_J)
            scale_factor = 0.0005 / norm_pde_J
            zeta_pde_J = zeta_pde_J * scale_factor

            x_next = x_next - 1* (zeta_pde_NS * grad_x_cur_pde_NS + zeta_pde_J * grad_x_cur_pde_J)


    ############################ Save the data ############################
    x_final = x_next
    Br_final = x_final[:,0,:,:].unsqueeze(0)
    Jx_final = x_final[:,1,:,:].unsqueeze(0)
    Jy_final = x_final[:,2,:,:].unsqueeze(0)
    Jz_final = x_final[:,3,:,:].unsqueeze(0)
    u_u_final = x_final[:,4,:,:].unsqueeze(0)
    u_v_final = x_final[:,5,:,:].unsqueeze(0)
 
    

    Br_final = ((Br_final+0.9)/1.8 *(max_Br - min_Br) + min_Br).to(torch.float64)
    Jx_final = ((Jx_final+0.9)/1.8 *(max_Jx - min_Jx) + min_Jx).to(torch.float64)
    Jy_final = ((Jy_final+0.9)/1.8 *(max_Jy - min_Jy) + min_Jy).to(torch.float64)
    Jz_final = ((Jz_final+0.9)/1.8 *(max_Jz - min_Jz) + min_Jz).to(torch.float64)
    u_u_final = ((u_u_final+0.9)/1.8 *(max_u_u - min_u_u) + min_u_u).to(torch.float64)
    u_v_final = ((u_v_final+0.9)/1.8 *(max_u_v - min_u_v) + min_u_v).to(torch.float64)


    relative_error_Br = torch.norm(Br_final - Br_GT, 2) / torch.norm(Br_GT, 2)
    relative_error_Jx = torch.norm(Jx_final - Jx_GT, 2) / torch.norm(Jx_GT, 2)
    relative_error_Jy = torch.norm(Jy_final - Jy_GT, 2) / torch.norm(Jy_GT, 2)
    relative_error_Jz = torch.norm(Jz_final - Jz_GT, 2) / torch.norm(Jz_GT, 2)
    relative_error_u_u = torch.norm(u_u_final - u_u_GT, 2) / torch.norm(u_u_GT, 2)
    relative_error_u_v = torch.norm(u_v_final - u_v_GT, 2) / torch.norm(u_v_GT, 2)
    

    print(f'Relative error of Br: {relative_error_Br}')
    print(f'Relative error of Jx: {relative_error_Jx}')
    print(f'Relative error of Jy: {relative_error_Jy}')
    print(f'Relative error of Jz: {relative_error_Jz}')
    print(f'Relative error of u_u: {relative_error_u_u}')
    print(f'Relative error of u_v: {relative_error_u_v}')

    Br_final = Br_final.detach().cpu().numpy()
    Jx_final = Jx_final.detach().cpu().numpy()
    Jy_final = Jy_final.detach().cpu().numpy()
    Jz_final = Jz_final.detach().cpu().numpy()
    u_u_final = u_u_final.detach().cpu().numpy()
    u_v_final = u_v_final.detach().cpu().numpy()

    scipy.io.savemat('MHD_results.mat', {'Br': Br_final, 'Jx': Jx_final, 'Jy': Jy_final, 'Jz': Jz_final, 'u_u': u_u_final, 'u_v': u_v_final})
    print('Done.')