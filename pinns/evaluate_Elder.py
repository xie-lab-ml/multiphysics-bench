import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import torch.nn.functional as F
import scipy.io as sio
from shapely.geometry import Polygon, Point
import pandas as pd
import math
import json
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
from scipy.io import savemat

def invnormalize(data, min_val, max_val):
    return (data + 0.9) / 1.8 * (max_val - min_val)  + min_val


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
    return sigma_map, pho_map, K_map


def identify_mater(elliptic_params, device=torch.device('cuda')):
    circle_params = elliptic_params.squeeze()
    coords = (torch.arange(128, device=device) - 63.5) * 0.001
    xx, yy = torch.meshgrid(coords, coords, indexing='ij')  # [128, 128]
    xx = xx.to(device)
    yy = yy.to(device)

    # 提取 cx, cy, r 并扩展维度以支持广播 [64, 1, 1]
    cx = circle_params[:, 0].view(-1, 1, 1)  # [64, 1, 1]
    cy = circle_params[:, 1].view(-1, 1, 1)  # [64, 1, 1]
    r = circle_params[:, 2].view(-1, 1, 1)  # [64, 1, 1]

    distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2  # [64, 128, 128]
    mater_iden = torch.where(distance_sq <= r ** 2, 1, -1)  # [64, 128, 128]
    return mater_iden

class FieldDataset(Dataset):
    def __init__(self, input_data, output_data, poly_csv_path,is_test):
        self.input_data = input_data  # (N, 1, H, W)
        self.output_data = output_data  # (N, 3, H, W)
        if poly_csv_path is not None:
            self.poly_res = self.read_polycsv(poly_csv_path,is_test)
        else:
            self.poly_res = None

    def __len__(self):
        return len(self.input_data)

    def read_polycsv(self,path_dir,is_test=True):
        dir_list = os.listdir(path_dir)
        poly_res = []
        for offset in range(len(dir_list)):
            if is_test:
                if offset < 1000:
                    break
                poly_GT_path = os.path.join(path_dir, f"{offset+10001}.csv")
            else:
                poly_GT_path = os.path.join(path_dir, f"{offset+1}.csv")


            poly_GT = pd.read_csv(poly_GT_path, header=None)
            poly_GT = torch.tensor(poly_GT.values, dtype=torch.float64)
            poly_res.append(poly_GT)
        return poly_res

    def __getitem__(self, idx):
        x = self.input_data[idx]  # (1, 128, 128)
        y = self.output_data[idx]  # (3, 128, 128)

        x_coord = torch.linspace(-1, 1, 128).view(1, 128, 1).expand(1, 128, 128)
        y_coord = torch.linspace(-1, 1, 128).view(1, 1, 128).expand(1, 128, 128)
        coords = torch.cat([x_coord, y_coord], dim=0)
        return x, coords, y, idx


def evaluate(model, dataloader):
    res_dict = {
        'RMSE': {}, 'nRMSE': {}, 'MaxError': {}, 'fRMSE_low': {}, 'fRMSE_middle': {}, 'fRMSE_high': {}, 'bRMSE': {}
    }

    for metric in res_dict:
        for key in ['u_u', 'u_v', 'c_flow']:
            res_dict[metric][key] = {}
            for t in range(10):
                res_dict[metric][key][f'{t}'] = 0

    def get_RMSE():
        for t in range(10):
            u_metric = torch.sqrt(torch.mean((u_u_N[:,t,:,:] - u_u_gt[:,t,:,:]) ** 2, dim=(1, 2)))
            v_metric = torch.sqrt(torch.mean((u_v_N[:,t,:,:] - u_v_gt[:,t,:,:]) ** 2, dim=(1, 2)))
            c_flow_metric = torch.sqrt(torch.mean((c_flow_N[:,t,:,:] - c_flow_gt[:,t,:,:]) ** 2, dim=(1, 2)))
            res_dict['RMSE']['u_u'][f'{t}'] += u_metric.sum()
            res_dict['RMSE']['u_v'][f'{t}'] += v_metric.sum()
            res_dict['RMSE']['c_flow'][f'{t}'] += c_flow_metric.sum()

    def get_nRMSE():
        for t in range(10):
            u_metric = torch.norm(u_u_N[:,t,:,:] - u_u_gt[:,t,:,:], 2, dim=(1, 2)) / torch.norm(u_u_gt[:,t,:,:], 2, dim=(1, 2))
            v_metric = torch.norm(u_v_N[:,t,:,:] - u_v_gt[:,t,:,:], 2, dim=(1, 2)) / torch.norm(u_v_gt[:,t,:,:], 2, dim=(1, 2))
            c_flow_metric = torch.norm(c_flow_N[:,t,:,:] - c_flow_gt[:,t,:,:], 2, dim=(1, 2)) / torch.norm(c_flow_gt[:,t,:,:], 2, dim=(1, 2))
            res_dict['nRMSE']['u_u'][f'{t}'] += u_metric.sum()
            res_dict['nRMSE']['u_v'][f'{t}'] += v_metric.sum()
            res_dict['nRMSE']['c_flow'][f'{t}'] += c_flow_metric.sum()

    def get_MaxError():
        for t in range(10):
            u_metric = torch.abs(u_u_N[:,t,:,:] - u_u_gt[:,t,:,:]).flatten(1).max(dim=1)[0]  # 先展平再求max
            v_metric = torch.abs(u_v_N[:,t,:,:] - u_v_gt[:,t,:,:]).flatten(1).max(dim=1)[0]  # 先展平再求max
            c_flow_metric = torch.abs(c_flow_N[:,t,:,:] - c_flow_gt[:,t,:,:]).flatten(1).max(dim=1)[0]  # 先展平再求max

            res_dict['MaxError']['u_u'][f'{t}'] += u_metric.sum()
            res_dict['MaxError']['u_v'][f'{t}'] += v_metric.sum()
            res_dict['MaxError']['c_flow'][f'{t}'] += c_flow_metric.sum()

    def get_bRMSE():
        for t in range(10):
            boundary_mask = torch.zeros_like(u_u_N[:, 0, :, :], dtype=bool)
            boundary_mask[:, 0, :] = True  # 上边界
            boundary_mask[:, -1, :] = True  # 下边界
            boundary_mask[:, :, 0] = True  # 左边界
            boundary_mask[:, :, -1] = True  # 右边界

            u_boundary_pred = u_u_N[:,t,:,:][boundary_mask].view(u_u_N[:,t,:,:].shape[0], -1)
            u_boundary_true = u_u_gt[:,t,:,:][boundary_mask].view(u_u_gt[:,t,:,:].shape[0], -1)
            u_metric = torch.sqrt(torch.mean((u_boundary_pred - u_boundary_true) ** 2, dim=1))
            res_dict['bRMSE']['u_u'][f'{t}'] += u_metric.sum()

            v_boundary_pred = u_v_N[:, t, :, :][boundary_mask].view(u_v_N[:, t, :, :].shape[0], -1)
            v_boundary_true = u_v_gt[:, t, :, :][boundary_mask].view(u_v_gt[:, t, :, :].shape[0], -1)
            v_metric = torch.sqrt(torch.mean((v_boundary_pred - v_boundary_true) ** 2, dim=1))
            res_dict['bRMSE']['u_v'][f'{t}'] += v_metric.sum()

            c_flow_boundary_pred = c_flow_N[:, t, :, :][boundary_mask].view(c_flow_N[:, t, :, :].shape[0], -1)
            c_flow_boundary_true = c_flow_gt[:, t, :, :][boundary_mask].view(c_flow_gt[:, t, :, :].shape[0], -1)
            c_flow_metric = torch.sqrt(torch.mean((c_flow_boundary_pred - c_flow_boundary_true) ** 2, dim=1))
            res_dict['bRMSE']['c_flow'][f'{t}'] += c_flow_metric.sum()

    def get_fRMSE():
        freq_bands = {
            'low': (0, 4),  # k_min=0, k_max=4
            'middle': (5, 12),  # k_min=5, k_max=12
            'high': (13, None)  # k_min=13, k_max=∞ (实际取Nyquist频率)
        }
        def compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W):
            """计算指定频段的fRMSE"""
            # 生成频段掩码
            kx = torch.arange(H, device=pred_fft.device)
            ky = torch.arange(W, device=pred_fft.device)
            kx, ky = torch.meshgrid(kx, ky, indexing='ij')

            # 计算径向波数 (避免重复计算0和Nyquist频率)
            r = torch.sqrt(kx ** 2 + ky ** 2)
            if k_max is None:
                mask = (r >= k_min)
                k_max = max(H // 2, W // 2) #nyquist
            else:
                mask = (r >= k_min) & (r <= k_max)

            # 计算误差
            diff_fft = torch.abs(pred_fft - true_fft) ** 2
            band_error = diff_fft[:, mask].sum(dim=1)  # 对空间维度
            band_error = torch.sqrt(band_error) / (k_max - k_min + 1)
            return band_error

        for t in range(10):
            for channel_idx, (pred_ch, true_ch, name) in enumerate([
                (u_u_N[:,t,:,:], u_u_gt[:,t,:,:], 'u_u'),
                (u_v_N[:,t,:,:], u_v_gt[:,t,:,:], 'u_v'),
                (c_flow_N[:,t,:,:], c_flow_gt[:,t,:,:], 'c_flow')
            ]):
                # 傅里叶变换 (shift后低频在中心)
                pred_fft = torch.fft.fft2(pred_ch)
                true_fft = torch.fft.fft2(true_ch)
                H, W = pred_ch.shape[-2], pred_ch.shape[-1]

                # 计算各频段
                for band, (k_min, k_max) in freq_bands.items():
                    error = compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W)
                    res_dict[f'fRMSE_{band}'][f'{name}'][f'{t}'] += error.sum()

    sample_total = 0
    for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            coords = coords.to(device)
            outputs = outputs.to(device)
            pred_outputs = model.forward(inputs, coords)

            u_u_N = pred_outputs[:, 0:10, :, :]
            u_v_N = pred_outputs[:, 10:20, :, :]
            c_flow_N = pred_outputs[:, 20:30, :, :]

            u_u_gt = outputs[:, 0:10, :, :]
            u_v_gt = outputs[:, 10:20, :, :]
            c_flow_gt = outputs[:, 20:30, :, :]

        for t in range(10):
            u_u_N[:,t,:,:] = invnormalize(u_u_N[:,t,:,:], *model.ranges['u_u'][t,:]).to(torch.float64)
            u_v_N[:,t,:,:] = invnormalize(u_v_N[:,t,:,:], *model.ranges['u_v'][t,:]).to(torch.float64)
            c_flow_N[:,t,:,:] = invnormalize(c_flow_N[:,t,:,:], *model.ranges['c_flow'][t,:]).to(torch.float64)

            u_u_gt[:, t, :, :] = invnormalize(u_u_gt[:, t, :, :], *model.ranges['u_u'][t, :]).to(torch.float64)
            u_v_gt[:, t, :, :] = invnormalize(u_v_gt[:, t, :, :], *model.ranges['u_v'][t, :]).to(torch.float64)
            c_flow_gt[:, t, :, :] = invnormalize(c_flow_gt[:, t, :, :], *model.ranges['c_flow'][t, :]).to(torch.float64)

        get_RMSE()
        get_nRMSE()
        get_MaxError()
        get_bRMSE()
        get_fRMSE()
        sample_total += outputs.shape[0]

    for metric in res_dict:
        for var in res_dict[metric]:
            avg, count = 0, 0
            for t in res_dict[metric][var]:
                res_dict[metric][var][t] /= sample_total
                res_dict[metric][var][t] = res_dict[metric][var][t].item()
                avg += res_dict[metric][var][t]
                count += 1
            res_dict[metric][var]['avg'] = avg / count
    return res_dict


class MultiPhysicsPINN(nn.Module):
    """处理电热耦合的PINN模型"""
    def __init__(self):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(6, 128),
            nn.LayerNorm(128)
        )
        # 残差块1
        self.res_block1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.LayerNorm(128)
        )
        # 残差块2 (与块1结构相同)
        self.res_block2 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.LayerNorm(128)
        )
        # 输出层
        self.output_layers = nn.ModuleList([nn.Linear(128, 1) for i in range(30)])

        train_data_base_path = "/data/yangchangfan/DiffusionPDE/data/training/Elder/"
        # -------------------- 加载归一化范围 --------------------
        range_allS_c = sio.loadmat(os.path.join(train_data_base_path, "S_c/range_S_c_t.mat"))['range_S_c_t'][0]
        range_allu_u = sio.loadmat(os.path.join(train_data_base_path, "u_u/range_u_u_t_999.mat"))['range_u_u_t_999'][1:]
        range_allu_v = sio.loadmat(os.path.join(train_data_base_path, "u_v/range_u_v_t_99.mat"))['range_u_v_t_99'][1:]
        range_allc_flow = sio.loadmat(os.path.join(train_data_base_path, "c_flow/range_c_flow_t_99.mat"))['range_c_flow_t_99'][1:]

        # range_allu_u = sio.loadmat(os.path.join(train_data_base_path, "u_u/range_u_u_t.mat"))['range_u_u_t'][1:]
        # range_allu_v = sio.loadmat(os.path.join(train_data_base_path, "u_v/range_u_v_t.mat"))['range_u_v_t'][1:]
        # range_allc_flow = sio.loadmat(os.path.join(train_data_base_path, "c_flow/range_c_flow_t.mat"))[
        #                       'range_c_flow_t'][1:]

        self.S_c_ranges = range_allS_c
        self.ranges = {
            'u_u': range_allu_u,
            'u_v': range_allu_v,
            'c_flow': range_allc_flow,
        }

    def forward(self, inputs, coords):
        batch_size = inputs.shape[0]
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, 4)

        coords = coords.permute(0, 2, 3, 1).reshape(-1, 2)

        input_data = torch.cat([inputs, coords], dim=1)

        # 输入处理
        x = self.initial_layer(input_data)

        # 残差块1
        residual = x
        x = self.res_block1(x)
        x += residual  # 添加残差连接

        # 残差块2
        residual = x
        x = self.res_block2(x)
        x += residual  # 添加残差连接

        # 输出层
        x = nn.Tanh()(x)  # 最后加一个激活函数

        res = []
        for idx, layer in enumerate(self.output_layers):
            res.append(layer(x))
        output = torch.cat(res, dim=-1)

        output = output.view(batch_size, 128, 128, 30).permute(0, 3, 1, 2)
        return output

    def compute_loss(self, inputs, coords, true_outputs):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        pred_outputs = self.forward(inputs, coords)
        data_loss = torch.nn.MSELoss()(pred_outputs,true_outputs)

        S_c_N = inputs[:,0,:,:]

        u_u_N = pred_outputs[:, 0:10, :, :]
        u_v_N = pred_outputs[:, 10:20, :, :]
        c_flow_N = pred_outputs[:, 20:30, :, :]

        u_u_gt = true_outputs[:, 0:10, :, :]  # shape: [2, 128, 128]
        u_v_gt = true_outputs[:, 10:20, :, :]
        c_flow_gt = true_outputs[:, 20:30, :, :]

        u_u_N_unnorm = torch.empty_like(u_u_N, dtype=torch.float64)
        u_v_N_unnorm = torch.empty_like(u_v_N, dtype=torch.float64)
        c_flow_N_unnorm = torch.empty_like(c_flow_N, dtype=torch.float64)

        u_u_gt_unnorm = torch.empty_like(u_u_gt, dtype=torch.float64)
        u_v_gt_unnorm = torch.empty_like(u_v_gt, dtype=torch.float64)
        c_flow_gt_unnorm = torch.empty_like(c_flow_gt, dtype=torch.float64)

        S_c_N = invnormalize(S_c_N, *model.S_c_ranges).to(torch.float64)
        for t in range(0, 10):
            u_u_N_unnorm[:, t, :, :] = invnormalize(u_u_N[:, t, :, :], *model.ranges['u_u'][t, :])
            u_v_N_unnorm[:, t, :, :] = invnormalize(u_v_N[:, t, :, :], *model.ranges['u_v'][t, :])
            c_flow_N_unnorm[:, t, :, :] = invnormalize(c_flow_N[:, t, :, :], *model.ranges['c_flow'][t, :])

            u_u_gt_unnorm[:, t, :, :] = invnormalize(u_u_gt[:, t, :, :], *model.ranges['u_u'][t, :])
            u_v_gt_unnorm[:, t, :, :] = invnormalize(u_v_gt[:, t, :, :], *model.ranges['u_v'][t, :])
            c_flow_gt_unnorm[:, t, :, :] = invnormalize(c_flow_gt[:, t, :, :], *model.ranges['c_flow'][t, :])

        u_u_N_unnorm = u_u_N_unnorm.to(torch.float64)
        u_v_N_unnorm = u_v_N_unnorm.to(torch.float64)
        c_flow_N_unnorm = c_flow_N_unnorm.to(torch.float64)

        pde_loss_Darcy, pde_loss_TDS = self.get_Elder_loss(S_c_N, u_u_N_unnorm, u_v_N_unnorm, c_flow_N_unnorm, device=device)
        return pde_loss_Darcy, pde_loss_TDS, data_loss

    def get_Elder_loss(self,S_c, u_u, u_v, c_flow, device=torch.device('cuda')):
        rho_0 = 1000
        beta = 200
        rho = rho_0 + beta * c_flow  # [T, H, W]
        B, T, H, W = rho.shape

        # 输入: rho [B, T, H, W], u_u [B, T, H, W], u_v [B, T, H, W]
        delta_x = (300 / 128)  # 1m
        delta_y = (150 / 128)  # 1m
        delta_t = 2 * 365 * 24 * 60 * 60  # 2 a

        # 导数核
        deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).reshape(1, 1, 1, 3) / (2 * delta_x)
        deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).reshape(1, 1, 3, 1) / (2 * delta_y)
        deriv_t = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).reshape(1, 1, 3) / (2 * delta_t)

        # 时间导数 d(rho)/dt (向量化)
        rho_t = rho.permute(0, 2, 3, 1).reshape(-1, 1, T)  # [B*H*W, 1, T]
        d_rho_dt = F.conv1d(rho_t, deriv_t, padding=1).reshape(B, H, W, T).permute(0, 3, 1, 2)  # [B, T, H, W]

        # x方向导数 d(rho*u)/dx (向量化)
        rho_u = rho * u_u
        rho_u_2d = rho_u.reshape(B * T, 1, H, W)  # [T, 1, H, W]
        d_rho_u_dx = F.conv2d(rho_u_2d, deriv_x, padding=(0, 1)).squeeze(1)  # [T, H, W]
        d_rho_u_dx = d_rho_u_dx.reshape(B, T, H, W)

        rho_v = rho * u_v
        rho_v_2d = rho_v.reshape(B * T, 1, H, W)  # [T, 1, H, W]
        d_rho_v_dy = F.conv2d(rho_v_2d, deriv_y, padding=(1, 0)).squeeze(1)  # [T, H, W]
        d_rho_v_dy = d_rho_v_dy.reshape(B, T, H, W)

        result_Darcy = 0.1 * d_rho_dt + d_rho_u_dx + d_rho_v_dy


        B, T, H, W = c_flow.shape  # c_flow: [B, T, H, W]
        # 1. 时间导数 dc/dt (保持原向量化实现)
        c_t = c_flow.permute(0, 2, 3, 1).reshape(-1, 1, T)  # [B*H*W, 1, T]
        dc_dt = F.conv1d(c_t, deriv_t, padding=1)  # [B*H*W, 1, T]
        dc_dt = dc_dt.reshape(B, H, W, T).permute(0, 3, 1, 2)  # [B, T, H, W]

        # 2. 空间导数 (向量化实现)
        # 将时间维度T作为通道维度处理
        c_flow_4d = c_flow.reshape(B * T, 1, H, W)  # [B*T, 1, H, W]

        # 计算空间导数
        dc_dx = F.conv2d(c_flow_4d, deriv_x, padding=(0, 1))  # [B*T, 1, H, W]
        dc_dy = F.conv2d(c_flow_4d, deriv_y, padding=(1, 0))  # [B*T, 1, H, W]

        # 3. 拉普拉斯算子 (向量化实现)
        laplace_c = F.conv2d(dc_dx, deriv_x, padding=(0, 1)).squeeze(1) + F.conv2d(dc_dy,deriv_y,padding=(1, 0)).squeeze(1)
        dc_dx = dc_dx.reshape(B, T, H, W)  # 恢复原始维度
        dc_dy = dc_dy.reshape(B, T, H, W)
        laplace_c = laplace_c.reshape(B, T, H, W)

        # 4. 最终计算 (确保所有输入维度一致)
        result_TDS = (
                0.1 * dc_dt +
                u_u * dc_dx +
                u_v * dc_dy -
                0.1 * 3.56e-6 * laplace_c -
                S_c.unsqueeze(1))  # 假设S_c需要广播

        pde_loss_Darcy = result_Darcy
        pde_loss_TDS = result_TDS

        pde_loss_Darcy = pde_loss_Darcy.squeeze()
        pde_loss_TDS = pde_loss_TDS.squeeze()
        return pde_loss_Darcy,pde_loss_TDS



def eval_model(model, save_dir, device='cuda'):
    plot_results(model, test_dataset, savepath=f'/data/bailichen/PDE/PDE/paint/data/Elder/PINNs', sample_idx=1082)
    exit()

    res_dict = evaluate(model, test_dataloader)
    print('-' * 20)
    print(f'metric:')
    for metric in res_dict:
        for var in res_dict[metric]:
            print(f'{metric}\t\t{var}:\t\t{res_dict[metric][var]}')

    # TODO 保存log
    with open(os.path.join(save_dir, f'log_final.json'), "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False)

    data = res_dict
    res = []
    for metric in data:
        for var in data[metric]:
            res.append(data[metric][var]['avg'])
            print(f"{metric} {var}: {data[metric][var]['avg']}")

    output_file = os.path.join(save_dir, './exp.csv')
    frmse_df = pd.DataFrame(res)
    frmse_df.to_csv(output_file, index=False, encoding="utf-8", float_format="%.16f")


def plot_results(model, dataset, savepath, sample_idx=0):
    sample_idx = sample_idx - 1001

    if not os.path.exists(os.path.join(savepath, str(sample_idx + 1001))):
        os.mkdir(os.path.join(savepath, str(sample_idx + 1001)))

    inputs, coords, outputs, polygt_idx = dataset[sample_idx]
    inputs = inputs.unsqueeze(0).to(device)
    coords = coords.unsqueeze(0).to(device)
    outputs = outputs.to(device).unsqueeze(0)
    with torch.no_grad():
        pred_outputs = model.forward(inputs, coords)


    u_u_N = pred_outputs[:, 0:10, :, :]
    u_v_N = pred_outputs[:, 10:20, :, :]
    c_flow_N = pred_outputs[:, 20:30, :, :]

    u_u_gt = outputs[:, 0:10, :, :]
    u_v_gt = outputs[:, 10:20, :, :]
    c_flow_gt = outputs[:, 20:30, :, :]

    res_data_pred = {}
    res_data_gt = {}
    for t in range(10):
        u_u_N[:, t, :, :] = invnormalize(u_u_N[:, t, :, :], *model.ranges['u_u'][t, :]).to(torch.float64)
        u_v_N[:, t, :, :] = invnormalize(u_v_N[:, t, :, :], *model.ranges['u_v'][t, :]).to(torch.float64)
        c_flow_N[:, t, :, :] = invnormalize(c_flow_N[:, t, :, :], *model.ranges['c_flow'][t, :]).to(torch.float64)

        u_u_gt[:, t, :, :] = invnormalize(u_u_gt[:, t, :, :], *model.ranges['u_u'][t, :]).to(torch.float64)
        u_v_gt[:, t, :, :] = invnormalize(u_v_gt[:, t, :, :], *model.ranges['u_v'][t, :]).to(torch.float64)
        c_flow_gt[:, t, :, :] = invnormalize(c_flow_gt[:, t, :, :], *model.ranges['c_flow'][t, :]).to(torch.float64)

        res_data_gt[f'u_{t}'] = u_u_gt[:, t, :, :].squeeze().to('cpu')
        res_data_gt[f'v_{t}'] = u_v_gt[:, t, :, :].squeeze().to('cpu')
        res_data_gt[f'c_flow_{t}'] = c_flow_gt[:, t, :, :].squeeze().to('cpu')

        res_data_pred[f'u_{t}'] = u_u_N[:, t, :, :].squeeze().to('cpu')
        res_data_pred[f'v_{t}'] = u_v_N[:, t, :, :].squeeze().to('cpu')
        res_data_pred[f'c_flow_{t}'] = c_flow_N[:, t, :, :].squeeze().to('cpu')

    data = []
    titles = []

    for i in range(5):
        data.extend([u_u_gt[0, (i)*2+1].cpu().numpy(), u_u_N[0, (i)*2].cpu().numpy()])
    for i in range(5):
        titles.extend([f'gt (t={(i+1)*2})', f'pred (t={(i+1)*2})'])

    # 绘制结果
    fig, axes = plt.subplots(5, 2, figsize=(6, 14))
    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(data[i], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, str(sample_idx+1001), "sample_good_norm.png"))
    # plt.savefig(os.path.join(savepath, str(sample_idx+1001), "sample_bad_norm.png"))

    savemat(os.path.join(savepath, str(sample_idx + 1001), 'pred_good_norm.mat'), res_data_pred)
    savemat(os.path.join(savepath, str(sample_idx + 1001), 'gt_good_norm.mat'), res_data_gt)

if __name__ == '__main__':
    base_dir = './data/Elder_new'
    # base_dir = './data/Elder_new_bad_norm'
    if not os.path.exists(os.path.join(base_dir,'fig')):
        os.mkdir(os.path.join(base_dir,'fig'))
    if not os.path.exists(os.path.join(base_dir,'ckpt')):
        os.mkdir(os.path.join(base_dir,'ckpt'))
    if not os.path.exists(os.path.join(base_dir,'fig_loss')):
        os.mkdir(os.path.join(base_dir,'fig_loss'))

    test_data = torch.load('/data/bailichen/PDE/PDE/pinns/data/Elder_new/Elder_test_128_new.pt')
    test_polycsv = None
    test_dataset = FieldDataset(test_data['x'], test_data['y'],
                                poly_csv_path=None,
                                is_test=True)

    batch_size = 32
    num_workers = 16  # 多进程加载数
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )

    device = 'cuda'
    model = torch.load(os.path.join(base_dir,'ckpt','model_49.pth')).to(device)  # 或 'model.pth'
    eval_model(model, base_dir, device)
