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


def load_ranges(base_path, variables):
    ranges = {}

    # # rho_water
    # rho_data = sio.loadmat(f"{base_path}/rho_water/range_allrho_water.mat")['range_allrho_water']
    # ranges['rho_water'] = {'max': rho_data[0, 1], 'min': rho_data[0, 0]}

    # 加载其他变量的范围
    for var in variables:
        real_data = sio.loadmat(f"{base_path}/{var}/range_allreal_{var}.mat")[f'range_allreal_{var}']
        imag_data = sio.loadmat(f"{base_path}/{var}/range_allimag_{var}.mat")[f'range_allimag_{var}']

        ranges[var] = {
            'max_real': real_data[0, 1],
            'min_real': real_data[0, 0],
            'max_imag': imag_data[0, 1],
            'min_imag': imag_data[0, 0]
        }

    return ranges

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
    u_metric_total, v_metric_total, T_metric_total, sample_total = 0,0,0,0
    res_dict = {
        'RMSE': {},
        'nRMSE': {},
        'MaxError': {},
        'fRMSE': {},
        'bRMSE': {},
    }
    for metric in res_dict:
        for var in model.variables:
            if metric != "fRMSE":
                res_dict[metric][f'{var}_real'] = 0
                res_dict[metric][f'{var}_imag'] = 0
            else:
                for band in ["low", "middle", "high"]:
                    res_dict['fRMSE'][f'{var}_{band}_imag'] = 0
                    res_dict['fRMSE'][f'{var}_{band}_real'] = 0

    def get_nRMSE():
        for var_idx, var in enumerate(model.variables):
            metric_value_real = torch.norm(pred_outputs[:,:,:,var_idx*2] - outputs[:,var_idx*2,:,:], 2, dim=(1, 2)) / torch.norm(outputs[:,var_idx*2,:,:], 2, dim=(1, 2))
            metric_value_imag = torch.norm(pred_outputs[:,:,:,var_idx*2+1] - outputs[:,var_idx*2+1,:,:], 2, dim=(1, 2)) / torch.norm(outputs[:,var_idx*2+1,:,:], 2, dim=(1, 2))
            res_dict['nRMSE'][f'{var}_real'] += metric_value_real.sum()
            res_dict['nRMSE'][f'{var}_imag'] += metric_value_imag.sum()


    def get_RMSE():
        for var_idx, var in enumerate(model.variables):
            metric_value_real = torch.sqrt(torch.mean((pred_outputs[:,:,:,var_idx*2] - outputs[:,var_idx*2,:,:]) ** 2, dim=(1, 2)))
            metric_value_imag = torch.sqrt(torch.mean((pred_outputs[:,:,:,var_idx*2+1] - outputs[:,var_idx*2+1,:,:]) ** 2, dim=(1, 2)))
            res_dict['RMSE'][f'{var}_real'] += metric_value_real.sum()
            res_dict['RMSE'][f'{var}_imag'] += metric_value_imag.sum()

    def get_MaxError():
        for var_idx, var in enumerate(model.variables):
            metric_value_real = torch.abs(pred_outputs[:,:,:,var_idx*2] - outputs[:,var_idx*2,:,:]).flatten(1).max(dim=1)[0]
            metric_value_imag = torch.abs(pred_outputs[:,:,:,var_idx*2+1] - outputs[:,var_idx*2+1,:,:]).flatten(1).max(dim=1)[0]
            res_dict['MaxError'][f'{var}_real'] += metric_value_real.sum()
            res_dict['MaxError'][f'{var}_imag'] += metric_value_imag.sum()

    def get_bRMSE():
        # 提取边界像素（上下左右各1像素）
        boundary_mask = torch.zeros_like(outputs[:, 0, :, :], dtype=bool)
        boundary_mask[:, 0, :] = True  # 上边界
        boundary_mask[:, -1, :] = True  # 下边界
        boundary_mask[:, :, 0] = True  # 左边界
        boundary_mask[:, :, -1] = True  # 右边界

        for var_idx, var in enumerate(model.variables):
            boundary_pred_real = pred_outputs[:,:,:,var_idx*2][boundary_mask].view(pred_outputs.shape[0], -1)
            boundary_true_real = outputs[:,var_idx*2,:,:][boundary_mask].view(outputs.shape[0], -1)
            real_metric = torch.sqrt(torch.mean((boundary_pred_real - boundary_true_real) ** 2, dim=1))
            res_dict['bRMSE'][f'{var}_real'] += real_metric.sum()

            boundary_pred_imag = pred_outputs[:, :, :, var_idx * 2+1][boundary_mask].view(pred_outputs.shape[0], -1)
            boundary_true_imag = outputs[:, var_idx * 2+1, :, :][boundary_mask].view(outputs.shape[0], -1)
            imag_metric = torch.sqrt(torch.mean((boundary_pred_imag - boundary_true_imag) ** 2, dim=1))
            res_dict['bRMSE'][f'{var}_imag'] += imag_metric.sum()

    def get_fRMSE():
        # 初始化结果存储
        for var_idx, var in enumerate(model.variables):
            for freq_band in ['low', 'middle', 'high']:
                res_dict['fRMSE'][f'{var}_{freq_band}_real'] = 0.0
                res_dict['fRMSE'][f'{var}_{freq_band}_real'] = 0.0
                res_dict['fRMSE'][f'{var}_{freq_band}_real'] = 0.0

                res_dict['fRMSE'][f'{var}_{freq_band}_imag'] = 0.0
                res_dict['fRMSE'][f'{var}_{freq_band}_imag'] = 0.0
                res_dict['fRMSE'][f'{var}_{freq_band}_imag'] = 0.0

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

        for var_idx, var in enumerate(model.variables):
            #real
            pred_fft = torch.fft.fft2(pred_outputs[:,:,:,var_idx*2])
            true_fft = torch.fft.fft2(outputs[:,var_idx*2,:,:])
            H, W = outputs.shape[-2], outputs.shape[-1]
            for band, (k_min, k_max) in freq_bands.items():
                error = compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W)
                res_dict['fRMSE'][f'{var}_{band}_real'] += error.sum()

            #imag
            pred_fft = torch.fft.fft2(pred_outputs[:, :, :, var_idx * 2+1])
            true_fft = torch.fft.fft2(outputs[:, var_idx * 2+1, :, :])
            H, W = outputs.shape[-2], outputs.shape[-1]
            for band, (k_min, k_max) in freq_bands.items():
                error = compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W)
                res_dict['fRMSE'][f'{var}_{band}_imag'] += error.sum()

    for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            coords = coords.to(device)
            outputs = outputs.to(device)
            pred_outputs = model.forward(inputs, coords).permute(0,3,2,1)

            for var_idx, var in enumerate(model.variables):
                var_minmax = model.range_data[var]
                outputs[:,var_idx*2,:,:] = ((outputs[:,var_idx*2,:,:] + 0.9) / 1.8 * (
                            var_minmax['max_real'] - var_minmax['min_real']) + var_minmax[
                                      'min_real']).to(torch.float64)
                outputs[:, var_idx * 2+1, :, :] = ((outputs[:, var_idx * 2+1, :, :] + 0.9) / 1.8 * (
                        var_minmax['max_imag'] - var_minmax['min_imag']) + var_minmax[
                                                     'min_imag']).to(torch.float64)

                pred_outputs[:, :, :, var_idx * 2] = ((pred_outputs[:, :, :, var_idx * 2] + 0.9) / 1.8 * (
                        var_minmax['max_real'] - var_minmax['min_real']) + var_minmax[
                                                     'min_real']).to(torch.float64)
                pred_outputs[:, :, :, var_idx * 2 + 1] = ((pred_outputs[:, :, :, var_idx * 2 + 1] + 0.9) / 1.8 * (
                        var_minmax['max_imag'] - var_minmax['min_imag']) + var_minmax[
                                                         'min_imag']).to(torch.float64)

        get_RMSE()
        get_nRMSE()
        get_MaxError()
        get_bRMSE()
        get_fRMSE()
        sample_total += outputs.shape[0]

    for metric in res_dict:
        for var in res_dict[metric]:
            res_dict[metric][var] /= sample_total
            # res_dict[metric][var] = res_dict[metric][var]

    return res_dict


class MultiPhysicsPINN(nn.Module):
    """处理电热耦合的PINN模型"""
    def __init__(self):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Linear(3, 128),
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
        self.output_layers = nn.ModuleList([nn.Linear(128, 1) for i in range(12)])

        # inv_normalization
        self.variables = ['p_t', 'Sxx', 'Sxy', 'Syy', 'x_u', 'x_v']
        self.range_data = load_ranges("/data/yangchangfan/DiffusionPDE/data/training/VA", self.variables)

        range_allrho_water_paths = "/data/yangchangfan/DiffusionPDE/data/training/VA/rho_water/range_allrho_water.mat"
        range_allrho_water = sio.loadmat(range_allrho_water_paths)['range_allrho_water']
        range_allrho_water = torch.tensor(range_allrho_water, device=device)
        self.max_rho_water = range_allrho_water[0, 1]
        self.min_rho_water = range_allrho_water[0, 0]

    def forward(self, inputs, coords):
        batch_size = inputs.shape[0]
        inputs = inputs.reshape(-1, 1)

        coords = coords.permute(0, 2, 3, 1)
        coords = coords.reshape(-1, 2)

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

        output = output.view(batch_size, 128, 128, 12).permute(0, 3, 1, 2)
        return output

    def compute_loss(self, inputs, coords, true_outputs):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        pred_outputs = self.forward(inputs, coords)
        data_loss = torch.nn.MSELoss()(pred_outputs,true_outputs)

        rho_water_N = ((inputs + 0.9) / 1.8 * (self.max_rho_water - self.min_rho_water) + self.min_rho_water).to(
            torch.float64)
        real_p_t_N = ((pred_outputs[:, 0] + 0.9) / 1.8 * (
                self.range_data['p_t']['max_real'] - self.range_data['p_t']['min_real']) +
                      self.range_data['p_t']['min_real']).to(torch.float64)
        imag_p_t_N = ((pred_outputs[:, 1] + 0.9) / 1.8 * (
                self.range_data['p_t']['max_imag'] - self.range_data['p_t']['min_imag']) +
                      self.range_data['p_t']['min_imag']).to(torch.float64)
        real_Sxx_N = ((pred_outputs[:, 2] + 0.9) / 1.8 * (
                self.range_data['Sxx']['max_real'] - self.range_data['Sxx']['min_real']) +
                      self.range_data['Sxx']['min_real']).to(torch.float64)
        imag_Sxx_N = ((pred_outputs[:, 3] + 0.9) / 1.8 * (
                self.range_data['Sxx']['max_imag'] - self.range_data['Sxx']['min_imag']) +
                      self.range_data['Sxx']['min_imag']).to(torch.float64)
        real_Sxy_N = ((pred_outputs[:, 4] + 0.9) / 1.8 * (
                self.range_data['Sxy']['max_real'] - self.range_data['Sxy']['min_real']) +
                      self.range_data['Sxy']['min_real']).to(torch.float64)
        imag_Sxy_N = ((pred_outputs[:, 5] + 0.9) / 1.8 * (
                self.range_data['Sxy']['max_imag'] - self.range_data['Sxy']['min_imag']) +
                      self.range_data['Sxy']['min_imag']).to(torch.float64)
        real_Syy_N = ((pred_outputs[:, 6] + 0.9) / 1.8 * (
                self.range_data['Syy']['max_real'] - self.range_data['Syy']['min_real']) +
                      self.range_data['Syy']['min_real']).to(torch.float64)
        imag_Syy_N = ((pred_outputs[:, 7] + 0.9) / 1.8 * (
                self.range_data['Syy']['max_imag'] - self.range_data['Syy']['min_imag']) +
                      self.range_data['Syy']['min_imag']).to(torch.float64)
        real_x_u_N = ((pred_outputs[:, 8] + 0.9) / 1.8 * (
                self.range_data['x_u']['max_real'] - self.range_data['x_u']['min_real']) +
                      self.range_data['x_u']['min_real']).to(torch.float64)
        imag_x_u_N = ((pred_outputs[:, 9] + 0.9) / 1.8 * (
                self.range_data['x_u']['max_imag'] - self.range_data['x_u']['min_imag']) +
                      self.range_data['x_u']['min_imag']).to(torch.float64)
        real_x_v_N = ((pred_outputs[:, 10] + 0.9) / 1.8 * (
                self.range_data['x_v']['max_real'] - self.range_data['x_v']['min_real']) +
                      self.range_data['x_v']['min_real']).to(torch.float64)
        imag_x_v_N = ((pred_outputs[:, 11] + 0.9) / 1.8 * (
                self.range_data['x_v']['max_imag'] - self.range_data['x_v']['min_imag']) +
                      self.range_data['x_v']['min_imag']).to(torch.float64)

        pde_loss_AC_real, pde_loss_AC_imag, pde_loss_structure_real_x, pde_loss_structure_imag_x, pde_loss_structure_real_y, pde_loss_structure_imag_y = self.get_VA_loss(rho_water_N, real_p_t_N, imag_p_t_N, real_Sxx_N, imag_Sxx_N, real_Sxy_N, imag_Sxy_N, real_Syy_N, imag_Syy_N,
                                              real_x_u_N, imag_x_u_N, real_x_v_N, imag_x_v_N, device=device)

        return pde_loss_AC_real, pde_loss_AC_imag, pde_loss_structure_real_x, pde_loss_structure_imag_x, pde_loss_structure_real_y, pde_loss_structure_imag_y, data_loss

    def get_VA_loss(self, rho_water, p_t_real, p_t_imag, Sxx_real, Sxx_imag, Sxy_real, Sxy_imag, Syy_real, Syy_imag, x_u_real, x_u_imag, x_v_real,
                x_v_imag, device=torch.device('cuda')):
        omega = torch.tensor(np.pi * 1e5, dtype=torch.float64, device=device)
        c_ac = 1.48144e3

        delta_x = (40 / 128) * 1e-3  # 1mm
        delta_y = (40 / 128) * 1e-3  # 1mm

        deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
        deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

        # Continuity_acoustic real
        grad_x_next_x_p_t_real = F.conv2d(p_t_real.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_p_t_real = F.conv2d(p_t_real.unsqueeze(1), deriv_y, padding=(1, 0))
        Laplace_p_t_real = F.conv2d((grad_x_next_x_p_t_real.squeeze() / rho_water).unsqueeze(1), deriv_x, padding=(0, 1)) + F.conv2d(
            (grad_x_next_y_p_t_real.squeeze() / rho_water).unsqueeze(1), deriv_y, padding=(1, 0))
        result_AC_real = Laplace_p_t_real.squeeze() + omega ** 2 * p_t_real / (rho_water * c_ac ** 2)

        # Continuity_acoustic imag
        grad_x_next_x_p_t_imag = F.conv2d(p_t_imag.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_p_t_imag = F.conv2d(p_t_imag.unsqueeze(1), deriv_y, padding=(1, 0))
        Laplace_p_t_imag = F.conv2d((grad_x_next_x_p_t_imag.squeeze() / rho_water).unsqueeze(1), deriv_x, padding=(0, 1)) + F.conv2d(
            (grad_x_next_y_p_t_imag.squeeze() / rho_water).unsqueeze(1), deriv_y, padding=(1, 0))
        result_AC_imag = Laplace_p_t_imag.squeeze() + omega ** 2 * p_t_imag / (rho_water * c_ac ** 2)

        # Continuity_structure real_x imag_x
        grad_x_next_x_Sxx_real = F.conv2d(Sxx_real.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_Sxy_real = F.conv2d(Sxy_real.unsqueeze(1), deriv_y, padding=(1, 0))
        result_structure_real_x = grad_x_next_x_Sxx_real.squeeze() + grad_x_next_y_Sxy_real.squeeze() + x_u_real.squeeze()

        grad_x_next_x_Sxx_imag = F.conv2d(Sxx_imag.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_Sxy_imag = F.conv2d(Sxy_imag.unsqueeze(1), deriv_y, padding=(1, 0))
        result_structure_imag_x = grad_x_next_x_Sxx_imag.squeeze() + grad_x_next_y_Sxy_imag.squeeze() + x_u_imag.squeeze()

        # Continuity_structure real_y imag_y
        grad_x_next_x_Sxy_real = F.conv2d(Sxy_real.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_Syy_real = F.conv2d(Syy_real.unsqueeze(1), deriv_y, padding=(1, 0))
        result_structure_real_y = grad_x_next_x_Sxy_real.squeeze() + grad_x_next_y_Syy_real.squeeze() + x_v_real.squeeze()

        grad_x_next_x_Sxy_imag = F.conv2d(Sxy_imag.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_Syy_imag = F.conv2d(Syy_imag.unsqueeze(1), deriv_y, padding=(1, 0))
        result_structure_imag_y = grad_x_next_x_Sxy_imag.squeeze() + grad_x_next_y_Syy_imag.squeeze() + x_v_imag.squeeze()

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

        pde_loss_AC_real = pde_loss_AC_real / 1000000
        pde_loss_AC_imag = pde_loss_AC_imag / 1000000

        pde_loss_structure_real_x = pde_loss_structure_real_x / 1000
        pde_loss_structure_imag_x = pde_loss_structure_imag_x / 1000
        pde_loss_structure_real_y = pde_loss_structure_real_y / 1000
        pde_loss_structure_imag_y = pde_loss_structure_imag_y / 1000

        return pde_loss_AC_real, pde_loss_AC_imag, pde_loss_structure_real_x, pde_loss_structure_imag_x, pde_loss_structure_real_y, pde_loss_structure_imag_y


def train_model(model, dataloader, train_dataset, test_dataset, optimizer, scheduler, Epoch, clip_value, save_dir, device='cuda'):
    for epoch in range(Epoch):
        total_loss = 0.0
        for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            coords = coords.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()
            pde_loss_AC_real, pde_loss_AC_imag, pde_loss_structure_real_x, \
            pde_loss_structure_imag_x, pde_loss_structure_real_y, pde_loss_structure_imag_y, \
            data_loss = model.compute_loss(inputs, coords, outputs)

            L_pde_AC_real = torch.norm(pde_loss_AC_real, 2) / (128 * 128)
            L_pde_AC_imag = torch.norm(pde_loss_AC_imag, 2) / (128 * 128)
            L_pde_structure_real_x = torch.norm(pde_loss_structure_real_x, 2) / (128 * 128)
            L_pde_structure_imag_x = torch.norm(pde_loss_structure_imag_x, 2) / (128 * 128)
            L_pde_structure_real_y = torch.norm(pde_loss_structure_real_y, 2) / (128 * 128)
            L_pde_structure_imag_y = torch.norm(pde_loss_structure_imag_y, 2) / (128 * 128)

            L_pde_AC_real = L_pde_AC_real * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_AC_real.clone().detach()))
            L_pde_AC_imag = L_pde_AC_imag * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_AC_imag.clone().detach()))
            L_pde_structure_real_x = L_pde_structure_real_x * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_structure_real_x.clone().detach()))
            L_pde_structure_imag_x = L_pde_structure_imag_x * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_structure_imag_x.clone().detach()))
            L_pde_structure_real_y = L_pde_structure_real_y * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_structure_real_y.clone().detach()))
            L_pde_structure_imag_y = L_pde_structure_imag_y * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_structure_imag_y.clone().detach()))

            # print(f'L_pde_AC_real: {L_pde_AC_real.item()}, \t L_pde_structure_imag_y: {L_pde_structure_imag_y.item()} \t Data_Loss: {data_loss.item()}')
            loss = 0.001 * (L_pde_AC_real + L_pde_AC_imag + L_pde_structure_real_x + L_pde_structure_imag_x + L_pde_structure_real_y + L_pde_structure_imag_y) + data_loss
            # loss = data_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # 梯度裁剪

            optimizer.step()
            total_loss += loss.item()

        if True:
            plot_results(model, test_dataset, savepath=os.path.join(save_dir, 'fig', f'show_{epoch}.png'), sample_idx=0)
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
        pred_outputs = model.forward(inputs, coords).squeeze()

    data = []
    titles = []
    for var_idx, var in enumerate(model.variables):
        # var_minmax = model.range_data[var]
        #
        # outputs[var_idx * 2, :, :] = ((outputs[var_idx * 2, :, :] + 0.9) / 1.8 * (
        #         var_minmax['max_real'] - var_minmax['min_real']) + var_minmax[
        #                                   'min_real']).to(torch.float64)
        # outputs[var_idx * 2 + 1, :, :] = ((outputs[var_idx * 2 + 1, :, :] + 0.9) / 1.8 * (
        #         var_minmax['max_imag'] - var_minmax['min_imag']) + var_minmax[
        #                                       'min_imag']).to(torch.float64)
        #
        # pred_outputs[:, :, var_idx * 2] = ((pred_outputs[:, :, var_idx * 2] + 0.9) / 1.8 * (
        #         var_minmax['max_real'] - var_minmax['min_real']) + var_minmax[
        #                                        'min_real']).to(torch.float64)
        # pred_outputs[:, :, var_idx * 2 + 1] = ((pred_outputs[:, :, var_idx * 2 + 1] + 0.9) / 1.8 * (
        #         var_minmax['max_imag'] - var_minmax['min_imag']) + var_minmax[
        #                                            'min_imag']).to(torch.float64)

        data.extend([outputs[var_idx * 2, :, :].cpu().numpy(), pred_outputs[var_idx * 2, :, :].cpu().numpy(),
                     outputs[var_idx * 2 + 1, :, :].cpu().numpy(), pred_outputs[var_idx * 2 + 1, :, :].cpu().numpy()])
        titles.extend([f'{var}_real (True)', f'{var}_real (pred)', f'{var}_imag (True)', f'{var}_imag (Pred)'])
    # 绘制结果
    fig, axes = plt.subplots(12, 2, figsize=(6, 24))
    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(data[i], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(savepath)

if __name__ == '__main__':
    base_dir = './data/VA'
    if not os.path.exists(os.path.join(base_dir,'fig')):
        os.mkdir(os.path.join(base_dir,'fig'))
    if not os.path.exists(os.path.join(base_dir,'ckpt')):
        os.mkdir(os.path.join(base_dir,'ckpt'))
    if not os.path.exists(os.path.join(base_dir,'fig_loss')):
        os.mkdir(os.path.join(base_dir,'fig_loss'))

    train_data = torch.load('/data/bailichen/PDE/PDE/pinns/data/VA/data/VA_train_128.pt')
    train_dataset = FieldDataset(train_data['x'][:10000], train_data['y'][:10000],
                                 poly_csv_path=None,
                                 is_test=False)

    test_data = torch.load('/data/bailichen/PDE/PDE/pinns/data/VA/data/VA_test_128.pt')
    test_polycsv = None
    test_dataset = FieldDataset(test_data['x'], test_data['y'],
                                poly_csv_path=None,
                                is_test=True)

    batch_size = 32
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
    Epoch = 30
    model = MultiPhysicsPINN().type(torch.float64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max = Epoch)
    train_model(model,train_dataloader,train_dataset, test_dataset, optimizer,scheduler,Epoch, 1.0 , base_dir, device)