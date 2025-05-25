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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

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
        self.poly_res = self.read_polycsv(poly_csv_path,is_test)

    def __len__(self):
        return len(self.input_data)

    def read_polycsv(self,path_dir,is_test=True):
        dir_list = os.listdir(path_dir)
        poly_res = []
        for offset in range(len(dir_list)):
            if is_test:
                if offset < 1000:
                    break
                poly_GT_path = os.path.join(path_dir, f"{offset+30001}.csv")
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
    u_metric_total, v_metric_total, T_metric_total, sample_total = 0,0,0,0
    res_dict = {
        'RMSE': {'u':0,'v':0,'T':0},
        'nRMSE': {'u': 0, 'v': 0, 'T': 0},
        'MaxError': {'u': 0, 'v': 0, 'T': 0},
        'fRMSE': {},
        'bRMSE': {'u': 0, 'v': 0, 'T': 0},
    }
    def get_nRMSE():
        u,v,T = pred
        u_metric = torch.norm(u - outputs[:, 0, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 0, :, :], 2, dim=(1, 2))
        v_metric = torch.norm(v - outputs[:, 1, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 1, :, :], 2, dim=(1, 2))
        T_metric = torch.norm(T - outputs[:, 2, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 2, :, :], 2, dim=(1, 2))
        res_dict['nRMSE']['u'] += u_metric.sum()
        res_dict['nRMSE']['v'] += v_metric.sum()
        res_dict['nRMSE']['T'] += T_metric.sum()

    def get_RMSE():
        u, v, T = pred  # pred是模型预测值 (B, C, H, W)
        # 计算各通道RMSE（按batch和空间维度平均）
        u_metric = torch.sqrt(torch.mean((u - outputs[:, 0, :, :]) ** 2, dim=(1, 2)))
        v_metric = torch.sqrt(torch.mean((v - outputs[:, 1, :, :]) ** 2, dim=(1, 2)))
        T_metric = torch.sqrt(torch.mean((T - outputs[:, 2, :, :]) ** 2, dim=(1, 2)))
        # 累加到结果字典
        res_dict['RMSE']['u'] += u_metric.sum()
        res_dict['RMSE']['v'] += v_metric.sum()
        res_dict['RMSE']['T'] += T_metric.sum()

    def get_MaxError():
        u, v, T = pred
        # 计算各通道的绝对误差最大值（沿空间维度）
        u_metric = torch.abs(u - outputs[:, 0, :, :]).flatten(1).max(dim=1)[0]  # 先展平再求max
        v_metric = torch.abs(v - outputs[:, 1, :, :]).flatten(1).max(dim=1)[0]  # 先展平再求max
        T_metric = torch.abs(T - outputs[:, 2, :, :]).flatten(1).max(dim=1)[0]  # 先展平再求max
        # 累加结果
        res_dict['MaxError']['u'] += u_metric.sum()
        res_dict['MaxError']['v'] += v_metric.sum()
        res_dict['MaxError']['T'] += T_metric.sum()

    def get_bRMSE():
        u, v, T = pred
        # 提取边界像素（上下左右各1像素）
        boundary_mask = torch.zeros_like(outputs[:, 0, :, :], dtype=bool)
        boundary_mask[:, 0, :] = True  # 上边界
        boundary_mask[:, -1, :] = True  # 下边界
        boundary_mask[:, :, 0] = True  # 左边界
        boundary_mask[:, :, -1] = True  # 右边界

        # 计算边界RMSE
        u_boundary_pred = u[boundary_mask].view(u.shape[0], -1)
        u_boundary_true = outputs[:, 0, :, :][boundary_mask].view(u.shape[0], -1)
        u_metric = torch.sqrt(torch.mean((u_boundary_pred - u_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['u'] += u_metric.sum()

        v_boundary_pred = v[boundary_mask].view(v.shape[0], -1)
        v_boundary_true = outputs[:, 1, :, :][boundary_mask].view(v.shape[0], -1)
        v_metric = torch.sqrt(torch.mean((v_boundary_pred - v_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['v'] += v_metric.sum()

        T_boundary_pred = T[boundary_mask].view(T.shape[0], -1)
        T_boundary_true = outputs[:, 2, :, :][boundary_mask].view(T.shape[0], -1)
        T_metric = torch.sqrt(torch.mean((T_boundary_pred - T_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['T'] += T_metric.sum()

    def get_fRMSE():
        u, v, T = pred  # pred形状: (Batch, Channel, Height, Width)

        # 初始化结果存储
        for freq_band in ['low', 'middle', 'high']:
            res_dict['fRMSE'][f'u_{freq_band}'] = 0.0
            res_dict['fRMSE'][f'v_{freq_band}'] = 0.0
            res_dict['fRMSE'][f'T_{freq_band}'] = 0.0

        # 定义频段范围 (基于论文设置)
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

        # 对每个通道计算fRMSE
        for channel_idx, (pred_ch, true_ch, name) in enumerate([
            (u, outputs[:, 0, :, :], 'u'),
            (v, outputs[:, 1, :, :], 'v'),
            (T, outputs[:, 2, :, :], 'T')
        ]):
            # 傅里叶变换 (shift后低频在中心)
            pred_fft = torch.fft.fft2(pred_ch)
            true_fft = torch.fft.fft2(true_ch)
            H, W = pred_ch.shape[-2], pred_ch.shape[-1]

            # 计算各频段
            for band, (k_min, k_max) in freq_bands.items():
                error = compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W)
                res_dict['fRMSE'][f'{name}_{band}'] += error.sum()

    for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            coords = coords.to(device)
            outputs = outputs.to(device)
            pred_outputs = model.forward(inputs, coords)

            # GT 反归一化
            outputs[:, 0, :, :] = (outputs[:, 0, :, :] * model.max_abs_Ez / 0.9).to(torch.float64)
            outputs[:, 1, :, :] = (outputs[:, 1, :, :] * model.max_abs_Ez / 0.9).to(torch.float64)
            outputs[:, 2, :, :] = ((outputs[:, 2, :, :] + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(torch.float64)

            #Pred 反归一化
            E_real_pred, E_imag_pred, T_pred = pred_outputs
            E_real_pred = (E_real_pred * model.max_abs_Ez / 0.9).to(torch.float64)
            E_imag_pred = (E_imag_pred * model.max_abs_Ez / 0.9).to(torch.float64)
            T_pred = ((T_pred + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(torch.float64)
            pred = (E_real_pred,E_imag_pred,T_pred)

        get_RMSE()
        get_nRMSE()
        get_MaxError()
        get_bRMSE()
        get_fRMSE()
        sample_total += outputs.shape[0]

    for metric in res_dict:
        for var in res_dict[metric]:
            res_dict[metric][var] /= sample_total
            res_dict[metric][var] = res_dict[metric][var].item()
    return res_dict

class MultiPhysicsPINN(nn.Module):
    """处理电热耦合的PINN模型"""
    def __init__(self):
        super().__init__()
        # 初始层
        self.initial_layer = nn.Sequential(
            nn.Linear(3, 128),
            nn.LayerNorm(128)
        )

        # 残差块1
        self.res_block1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

        # 残差块2 (与块1结构相同)
        self.res_block2 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

        # 输出层
        self.output_layer = nn.Linear(128, 3)  # 输出: E_real, E_imag, T

        max_abs_Ez_path = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/Ez/max_abs_Ez.mat"
        max_abs_Ez = sio.loadmat(max_abs_Ez_path)['max_abs_Ez']
        self.max_abs_Ez = torch.tensor(max_abs_Ez, device=device)

        range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/T/range_allT.mat"
        range_allT = sio.loadmat(range_allT_paths)['range_allT']
        range_allT = torch.tensor(range_allT, device=device)

        self.max_T = range_allT[0, 1]
        self.min_T = range_allT[0, 0]

        self.num_fourier_features = 32

    def get_L_BC_heat(self, T, pho, boundary_mask, device=torch.device('cuda')):
        T_ext = 293.15
        h_trans = 15  # 读取
        T = T.detach().clone().requires_grad_(True)

        delta_x = 128 / 128 * 1e-3  # 1mm
        delta_y = 128 / 128 * 1e-3  # 1mm

        deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
        deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

        dT_dx = F.conv2d(T.unsqueeze(1), deriv_x, padding=(0, 1))
        dT_dy = F.conv2d(T.unsqueeze(1), deriv_y, padding=(1, 0))

        # 边界法向导数：只保留边界方向（这里是硬编码矩形边界）
        dT_dn = torch.zeros_like(dT_dy).squeeze()  # 和 dT_dy 保持维度一致
        T_s = T.squeeze()
        dT_dn[:, 0, :] = -(T_s[:, 1, :] - T_s[:, 0, :]) / delta_y  # 上边界 (y=0)
        dT_dn[:, -1, :] = (T_s[:, -1, :] - T_s[:, -2, :]) / delta_y  # 下边界 (y=127)
        dT_dn[:, :, 0] = -(T_s[:,:, 1] - T_s[:,:, 0]) / delta_x  # left
        dT_dn[:,:, -1] = (T_s[:,:, -1] - T_s[:,:, -2]) / delta_x  # right

        T_boundary = T.squeeze()
        result_BC_T = pho * dT_dn + h_trans * (T_boundary - T_ext)
        result_BC_T = result_BC_T * boundary_mask

        return result_BC_T

    def fourier_feature_mapping(self, coords, num_features=64, scale=10.0):
        d = coords.size(-1)  # 坐标维度 (d)
        B = torch.randn((d, num_features), device=coords.device) * scale  # 频率矩阵 [d, m]
        # 投影并拼接正弦/余弦特征
        proj = 2 * torch.pi * torch.matmul(coords, B)  # [..., m]
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [..., 2*m]
        return features

    def forward(self, inputs, coords):
        batch_size = inputs.shape[0]

        inputs = inputs.reshape(-1, 1)

        coords = coords.permute(0, 2, 3, 1)
        coords = coords.reshape(-1, 2)

        # fourier_features = self.fourier_feature_mapping(coords, self.num_fourier_features)


        # input_data = torch.cat([inputs, coords, fourier_features], dim=1)
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
        output = self.output_layer(x)

        output = output.view(batch_size, 128, 128, 3).permute(0, 3, 1, 2)
        return output[:, 0], output[:, 1], output[:, 2]  # E_real, E_imag, T

    def get_boundary_points(self,grid_size, device=torch.device('cuda')):
        boundary_mask = torch.zeros((grid_size, grid_size), device=device)

        boundary_mask[0, :] = 1  # top
        boundary_mask[-1, :] = 1  # bottom
        boundary_mask[:, 0] = 1  # left
        boundary_mask[:, -1] = 1  # right

        return boundary_mask

    def get_TE_heat_loss(self, mater, Ez, T, mater_iden):
        sigma, pho, K_E = generate_separa_mater(mater, T, mater_iden)
        delta_x = 1e-3  # 1mm
        delta_y = 1e-3  # 1mm

        deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
        deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

        deriv_x_complex = torch.complex(deriv_x, torch.zeros_like(deriv_x))
        deriv_y_complex = torch.complex(deriv_y, torch.zeros_like(deriv_y))

        grad_x_next_x_E = F.conv2d(Ez.unsqueeze(1), deriv_x_complex, padding=(0, 1))
        grad_x_next_y_E = F.conv2d(Ez.unsqueeze(1), deriv_y_complex, padding=(1, 0))

        Laplac_E = F.conv2d(grad_x_next_x_E, deriv_x_complex, padding=(0, 1)) + F.conv2d(grad_x_next_y_E,
                                                                                         deriv_y_complex,
                                                                                         padding=(1, 0))
        result_E = Laplac_E.squeeze() + K_E * Ez

        # T_filed
        grad_x_next_x_T = F.conv2d(T.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_T = F.conv2d(T.unsqueeze(1), deriv_y, padding=(1, 0))
        Laplac_T = F.conv2d(grad_x_next_x_T, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_T, deriv_y,
                                                                                 padding=(1, 0))
        result_T = pho * Laplac_T.squeeze() + 0.5 * sigma * Ez * torch.conj(Ez)

        pde_loss_E = result_E
        pde_loss_T = result_T

        pde_loss_E = pde_loss_E.squeeze()
        pde_loss_T = pde_loss_T.squeeze()

        pde_loss_E = pde_loss_E / 1e6
        pde_loss_T = pde_loss_T / 1e6
        return pde_loss_E, pde_loss_T

    def compute_loss(self, inputs, coords, poly_GT, E_real_true,E_imag_true,T_true):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        E_real_pred, E_imag_pred, T_pred = self.forward(inputs, coords)

        data_loss = torch.nn.MSELoss()(E_real_pred,E_real_true) + torch.nn.MSELoss()(E_imag_pred,E_imag_true) + torch.nn.MSELoss()(T_pred,T_true)

        #inv normalization
        E_real_pred = (E_real_pred * self.max_abs_Ez / 0.9).to(torch.float64)
        E_imag_pred = (E_imag_pred * self.max_abs_Ez / 0.9).to(torch.float64)
        T_N = ((T_pred+0.9)/1.8 *(self.max_T - self.min_T) + self.min_T).to(torch.float64)
        complex_Ez_N = torch.complex(E_real_pred, E_imag_pred)

        mater_iden = torch.tensor(identify_mater(poly_GT)).to(torch.float64).to(inputs.device)

        val_in = ((inputs - 0.1) * (3e11 - 1e11) / 0.8 + 1e11).to(torch.float64)
        val_out = ((inputs + 0.9) * (20 - 10) / 0.8 + 10).to(torch.float64)
        mater_N = torch.where(mater_iden > 1e-5, val_in, val_out)

        pde_loss_E, pde_loss_T = self.get_TE_heat_loss(mater_N, complex_Ez_N, T_N, mater_iden)

        return pde_loss_E ,pde_loss_T, data_loss

def train_model(model, dataloader, train_dataset, test_dataset, optimizer, scheduler, Epoch, clip_value, save_dir, device='cuda'):
    for epoch in range(Epoch):
        total_loss = 0.0
        for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            coords = coords.to(device)
            polygt = [train_dataset.poly_res[i] for i in polygt_idx]
            polygt = torch.stack(polygt, dim=0).to(device)
            E_real_true = outputs[:, 0].to(device)
            E_imag_true = outputs[:, 1].to(device)
            T_true = outputs[:, 2].to(device)

            optimizer.zero_grad()
            pde_loss_E,pde_loss_T,data_loss = model.compute_loss(inputs, coords, polygt, E_real_true,E_imag_true,T_true)
            L_pde_E = torch.norm(pde_loss_E, 2) / (128 * 128)
            L_pde_T = torch.norm(pde_loss_T, 2) / (128 * 128)

            print(f'E_Loss: {L_pde_E.item()}, \t T_Loss: {L_pde_T.item()} \t Data_Loss: {data_loss.item()}')
            L_pde_E = L_pde_E * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_E.clone().detach()))
            L_pde_T = L_pde_T * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_T.clone().detach()))
            loss = 0.001 * (L_pde_E + L_pde_T) + data_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # 梯度裁剪

            optimizer.step()
            total_loss += loss.item()

        if True:
            plot_results(model, train_dataset, savepath=os.path.join(save_dir, 'fig', f'show_{epoch}.png'), sample_idx=0)
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4e}")

            res_dict = evaluate(model, test_dataloader)
            print('-' * 20)
            print(f'metric:')
            for metric in res_dict:
                for var in res_dict[metric]:
                    print(f'{metric}\t\t{var}:\t\t{res_dict[metric][var]}')
            print('-' * 20)

            torch.save(model, os.path.join(save_dir, 'ckpt', f'model_{epoch}.pth'))

        scheduler.step()
    return model


def plot_results(model, dataset, savepath, sample_idx=0):
    inputs, coords, outputs, polygt_idx = dataset[sample_idx]
    inputs = inputs.unsqueeze(0).to(device)
    coords = coords.unsqueeze(0).to(device)
    outputs = outputs.to(device)

    with torch.no_grad():
        E_real_pred, E_imag_pred, T_pred = model.forward(inputs, coords)
    #Pred 归一化
    E_real_pred = (E_real_pred * model.max_abs_Ez / 0.9).to(torch.float64)
    E_imag_pred = (E_imag_pred * model.max_abs_Ez / 0.9).to(torch.float64)
    T_pred = ((T_pred + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(torch.float64)

    #Ground Truth 归一化
    E_real_true = outputs[0]
    E_real_true = (E_real_true * model.max_abs_Ez / 0.9).to(torch.float64).cpu().numpy()
    E_imag_true = outputs[1]
    E_imag_true = (E_imag_true * model.max_abs_Ez / 0.9).to(torch.float64).cpu().numpy()
    T_true = outputs[2]
    T_true = ((T_true + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(torch.float64).cpu().numpy()

    E_real_pred = E_real_pred.squeeze().cpu().numpy()
    E_imag_pred = E_imag_pred.squeeze().cpu().numpy()
    T_pred = T_pred.squeeze().cpu().numpy()

    # 绘制结果
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    titles = ['E_real (True)', 'E_real (Pred)', 'E_imag (True)', 'E_imag (Pred)', 'T (True)', 'T (Pred)']
    data = [E_real_true, E_real_pred, E_imag_true, E_imag_pred, T_true, T_pred]

    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(data[i], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(savepath)

if __name__ == '__main__':
    base_dir = './data/Scale_TE_res_30000'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(os.path.join(base_dir,'fig')):
        os.mkdir(os.path.join(base_dir,'fig'))
    if not os.path.exists(os.path.join(base_dir,'ckpt')):
        os.mkdir(os.path.join(base_dir,'ckpt'))
    if not os.path.exists(os.path.join(base_dir,'fig_loss')):
        os.mkdir(os.path.join(base_dir,'fig_loss'))

    # train_data = torch.load('/data/bailichen/PDE/PDE/pinns/data/te_heat/TE_heat_train_128.pt')
    train_data = torch.load('/data/bailichen/PDE/PDE/DeepONet/TE_Heat/data/TE_heat_train_128_3w.pt')
    train_dataset = FieldDataset(train_data['x'][:30000], train_data['y'][:30000],
                                 poly_csv_path='/data/yangchangfan/DiffusionPDE/data/training/TE_heat_3w/ellipticcsv',
                                 is_test=False)

    test_data = torch.load('/data/bailichen/PDE/PDE/DeepONet/TE_Heat/data/TE_heat_test_128_3w.pt')
    test_polycsv = None
    test_dataset = FieldDataset(test_data['x'], test_data['y'],
                                poly_csv_path='/data/yangchangfan/DiffusionPDE/data/testing/TE_heat/ellipticcsv/',
                                is_test=True)

    batch_size = 64
    num_workers = 16  # 多进程加载数
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )

    device = 'cuda'
    Epoch = 20
    model = MultiPhysicsPINN().type(torch.float64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max = Epoch)
    train_model(model,train_dataloader,train_dataset, test_dataset, optimizer,scheduler,Epoch, 1.0 , base_dir, device)