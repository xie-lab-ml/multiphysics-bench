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
import json
from scipy.io import savemat


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

def generate_separa_mater(mater_iden):
    rho_air = 1.24246
    rho_copper = 8960
    Crho_air = 1005.10779
    Crho_copper = 385
    kappa_air = 0.02505
    kappa_copper = 400

    rho = torch.where(mater_iden > 1e-5, rho_copper, rho_air)
    Crho = torch.where(mater_iden > 1e-5, Crho_copper, Crho_air)
    kappa = torch.where(mater_iden > 1e-5, kappa_copper, kappa_air)

    # rho = rho.t()
    rho = rho.permute(0, 2, 1)
    # Crho = Crho.t()
    Crho = Crho.permute(0, 2, 1)
    # kappa = kappa.t()
    kappa = kappa.permute(0, 2, 1)

    return rho, Crho, kappa


def identify_mater(circle_params, device=torch.device('cuda')):
    # 输入形状: [64, 1, 3] -> squeeze后 [64, 3]
    circle_params = circle_params.squeeze()  # [64, 3]

    # 生成坐标网格 [128, 128]
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

        x_coord = torch.linspace(-0.635, 0.635, 128).view(1, 128, 1).expand(1, 128, 128)
        y_coord = torch.linspace(-0.635, 0.635, 128).view(1, 1, 128).expand(1, 128, 128)
        coords = torch.cat([x_coord, y_coord], dim=0)
        return x, coords, y, idx

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
        self.output_layer_1 = nn.Linear(128, 1)  # 输出: E_real, E_imag, T
        self.output_layer_2 = nn.Linear(128, 1)  # 输出: E_real, E_imag, T
        self.output_layer_3 = nn.Linear(128, 1)  # 输出: E_real, E_imag, T

        range_allQ_heat_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/Q_heat/range_allQ_heat.mat"
        range_allQ_heat = sio.loadmat(range_allQ_heat_paths)['range_allQ_heat']
        range_allQ_heat = torch.tensor(range_allQ_heat, device=device)
        self.max_Q_heat = range_allQ_heat[0, 1]
        self.min_Q_heat = range_allQ_heat[0, 0]

        range_allu_u_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_u/range_allu_u.mat"
        range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']
        range_allu_u = torch.tensor(range_allu_u, device=device)
        self.max_u_u = range_allu_u[0, 1]
        self.min_u_u = range_allu_u[0, 0]

        range_allu_v_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_v/range_allu_v.mat"
        range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']
        range_allu_v = torch.tensor(range_allu_v, device=device)
        self.max_u_v = range_allu_v[0, 1]
        self.min_u_v = range_allu_v[0, 0]

        range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/T/range_allT.mat"
        range_allT = sio.loadmat(range_allT_paths)['range_allT']
        range_allT = torch.tensor(range_allT, device=device)
        self.max_T = range_allT[0, 1]
        self.min_T = range_allT[0, 0]

        self.num_fourier_features = 32

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
        output_1 = self.output_layer_1(x)
        output_2 = self.output_layer_2(x)
        output_3 = self.output_layer_3(x)
        output = torch.cat([output_1,output_2,output_3], dim=-1)

        output = output.view(batch_size, 128, 128, 3).permute(0, 3, 1, 2)
        return output[:, 0], output[:, 1], output[:, 2]  # u, v, T

    def get_boundary_points(self,grid_size, device=torch.device('cuda')):
        boundary_mask = torch.ones((grid_size, grid_size), device=device)

        boundary_mask[0, :] = 0  # top
        boundary_mask[1, :] = 0  # top
        boundary_mask[-1, :] = 0  # bottom
        boundary_mask[-2, :] = 0  # bottom
        boundary_mask[:, 0] = 0  # left
        boundary_mask[:, 1] = 0  # left
        boundary_mask[:, -1] = 0  # right
        boundary_mask[:, -2] = 0  # right

        return boundary_mask

    def get_NS_heat_loss(self, Q_heat, u_u, u_v, T, mater_iden):
        rho, Crho, kappa = generate_separa_mater(mater_iden)

        delta_x = 0.128/128  # 1m
        delta_y = 0.128/128   # 1m

        deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
        deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

        # Continuity_NS
        grad_x_next_x_NS = F.conv2d(u_u.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_NS = F.conv2d(u_v.unsqueeze(1), deriv_y, padding=(1, 0))
        result_NS = grad_x_next_x_NS + grad_x_next_y_NS
        result_NS = result_NS.squeeze()

        # T_filed
        grad_x_next_x_T = F.conv2d(T.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_T = F.conv2d(T.unsqueeze(1), deriv_y, padding=(1, 0))
        Laplac_T = F.conv2d(grad_x_next_x_T, deriv_x, padding=(0, 1)) + F.conv2d(grad_x_next_y_T, deriv_y,
                                                                                 padding=(1, 0))
        Laplac_T = Laplac_T.squeeze()

        result_heat = rho * Crho * (u_u * grad_x_next_x_T.squeeze() + u_v * grad_x_next_y_T.squeeze()) - kappa * Laplac_T - Q_heat

        pde_loss_NS = result_NS
        pde_loss_heat = result_heat

        pde_loss_NS = pde_loss_NS.squeeze() / 1000
        pde_loss_heat = pde_loss_heat.squeeze() / 1000000

        return pde_loss_NS,pde_loss_heat

    def compute_loss(self, inputs, coords, poly_GT,u_true,v_true,T_true):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        u_pred,v_pred,T_pred = self.forward(inputs, coords)

        #inv normalization
        Q_heat_N = ((inputs + 0.9) / 1.8 * (self.max_Q_heat - self.min_Q_heat) + self.min_Q_heat).to(torch.float64)
        u_u_N = ((u_pred + 0.9) / 1.8 * (self.max_u_u - self.min_u_u) + self.min_u_u).to(torch.float64)
        u_v_N = ((v_pred + 0.9) / 1.8 * (self.max_u_v - self.min_u_v) + self.min_u_v).to(torch.float64)
        T_N = ((T_pred + 0.9) / 1.8 * (self.max_T - self.min_T) + self.min_T).to(torch.float64)

        circle_iden = identify_mater(poly_GT)

        pde_loss_NS, pde_loss_heat = self.get_NS_heat_loss(Q_heat_N, u_u_N, u_v_N, T_N, circle_iden)
        #TODO   去除PDE Loss异常点
        quantile_20 = torch.quantile(
            pde_loss_NS.reshape(pde_loss_NS.shape[0], -1),  # 展平后计算
            q=0.05,
            dim=1,
            keepdim=True
        ).unsqueeze(-1)  # 恢复为 [batch_size, 1, 1]
        quantile_80 = torch.quantile(
            pde_loss_NS.reshape(pde_loss_NS.shape[0], -1),
            q=0.95,
            dim=1,
            keepdim=True
        ).unsqueeze(-1)  # [batch_size, 1, 1]
        pde_loss_NS[(pde_loss_NS>quantile_80) | (pde_loss_NS<quantile_20)] = 0

        quantile_20 = torch.quantile(
            pde_loss_heat.reshape(pde_loss_heat.shape[0], -1),  # 展平后计算
            q=0.01,
            dim=1,
            keepdim=True
        ).unsqueeze(-1)  # 恢复为 [batch_size, 1, 1]
        quantile_80 = torch.quantile(
            pde_loss_heat.reshape(pde_loss_heat.shape[0], -1),
            q=0.95,
            dim=1,
            keepdim=True
        ).unsqueeze(-1)  # [batch_size, 1, 1]
        pde_loss_heat[(pde_loss_heat>quantile_80) | (pde_loss_heat<quantile_20)] = 0

        data_loss = torch.nn.MSELoss()(u_pred,u_true) + torch.nn.MSELoss()(v_pred,v_true) + torch.nn.MSELoss()(T_pred,T_true)
        return pde_loss_NS , pde_loss_heat , data_loss

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
            u_u_N,u_v_N,T = model.forward(inputs, coords)

            # GT 反归一化
            outputs[:, 0, :, :] = (
                        (outputs[:, 0, :, :] + 0.9) / 1.8 * (model.max_u_u - model.min_u_u) + model.min_u_u).to(
                torch.float64)
            outputs[:, 1, :, :] = (
                        (outputs[:, 1, :, :] + 0.9) / 1.8 * (model.max_u_v - model.min_u_v) + model.min_u_v).to(
                torch.float64)
            outputs[:, 2, :, :] = ((outputs[:, 2, :, :] + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(
                torch.float64)

            # Pred 反归一化
            u_u_N = ((u_u_N + 0.9) / 1.8 * (model.max_u_u - model.min_u_u) + model.min_u_u).to(
                torch.float64)
            u_v_N = ((u_v_N + 0.9) / 1.8 * (model.max_u_v - model.min_u_v) + model.min_u_v).to(
                torch.float64)
            T = ((T + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(
                torch.float64)

            pred = (u_u_N, u_v_N, T)

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

def eval_model(model, save_dir, device='cuda'):
    # plot_results(model, test_dataset, savepath=f'/data/bailichen/PDE/PDE/paint/data/NS_heat/PINNs', sample_idx=10984)
    # exit()

    res_dict = evaluate(model, test_dataloader)
    print('-' * 20)
    print(f'metric:')
    for metric in res_dict:
        for var in res_dict[metric]:
            print(f'{metric}\t\t{var}:\t\t{res_dict[metric][var]}')
    with open(os.path.join(save_dir, f'log_final.json'), "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False)

    data = res_dict
    res = []
    for metric in data:
        for var in data[metric]:
            res.append(data[metric][var])

    output_file = os.path.join(save_dir, './exp.csv')
    frmse_df = pd.DataFrame(res)
    frmse_df.to_csv(output_file, index=False, encoding="utf-8", float_format="%.16f")

def plot_loss(pde_loss_NS, pde_loss_T, save_path):
    pde_loss_NS = pde_loss_NS.to(
        torch.float64).detach().cpu().numpy().squeeze()
    pde_loss_T = pde_loss_T.to(
        torch.float64).detach().cpu().numpy().squeeze()
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    titles = ['pde_loss_NS', 'pde_loss_T']
    data = [pde_loss_NS, pde_loss_T]
    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(data[i], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax)
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.savefig(save_path)

def plot_results(model, dataset, savepath, sample_idx=0):
    sample_idx = sample_idx - 10001
    if not os.path.exists(os.path.join(savepath, str(sample_idx + 10001))):
        os.mkdir(os.path.join(savepath, str(sample_idx + 10001)))

    inputs, coords, outputs, polygt_idx = dataset[sample_idx]
    inputs = inputs.unsqueeze(0).to(device)
    coords = coords.unsqueeze(0).to(device)
    outputs = outputs.to(device)

    with torch.no_grad():
        u, v, T = model.forward(inputs, coords)
    #Pred 归一化
    Q_heat_N = ((inputs + 0.9) / 1.8 * (model.max_Q_heat - model.min_Q_heat) + model.min_Q_heat).to(torch.float64).cpu().numpy().squeeze()
    u_u_N = ((u + 0.9) / 1.8 * (model.max_u_u - model.min_u_u) + model.min_u_u).to(torch.float64).cpu().numpy().squeeze()
    u_v_N = ((v + 0.9) / 1.8 * (model.max_u_v - model.min_u_v) + model.min_u_v).to(torch.float64).cpu().numpy().squeeze()
    T_N = ((T + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(torch.float64).cpu().numpy().squeeze()

    #Ground Truth 归一化
    u_true = outputs[0]
    u_true = ((u_true + 0.9) / 1.8 * (model.max_u_u - model.min_u_u) + model.min_u_u).to(torch.float64).cpu().numpy().squeeze()
    v_true = outputs[1]
    v_true = ((v_true + 0.9) / 1.8 * (model.max_u_v - model.min_u_v) + model.min_u_v).to(torch.float64).cpu().numpy().squeeze()
    T_true = outputs[2]
    T_true = ((T_true + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(torch.float64).cpu().numpy().squeeze()

    # 绘制结果
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    titles = ['u (True)', 'u (Pred)', 'v (True)', 'v (Pred)', 'T (True)', 'T (Pred)']
    data = [u_true, u_u_N, v_true, u_v_N, T_true, T_N]

    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(data[i], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, str(sample_idx + 10001), "sample.png"))
    savemat(os.path.join(savepath, str(sample_idx + 10001), 'pred.mat'), {
        'u': u_u_N,
        'v': u_v_N,
        'T': T_N
    })

    savemat(os.path.join(savepath, str(sample_idx + 10001), 'gt.mat'), {
        'u': u_true,
        'v': v_true,
        'T': T_true
    })

if __name__ == '__main__':
    base_dir = './data/ns_heat'
    if not os.path.exists(os.path.join(base_dir,'fig')):
        os.mkdir(os.path.join(base_dir,'fig'))
    if not os.path.exists(os.path.join(base_dir,'ckpt')):
        os.mkdir(os.path.join(base_dir,'ckpt'))
    if not os.path.exists(os.path.join(base_dir,'fig_loss')):
        os.mkdir(os.path.join(base_dir,'fig_loss'))

    test_data = torch.load('/data/bailichen/PDE/PDE/pinns/data/ns_heat/NS_heat_test_128.pt')
    test_dataset = FieldDataset(test_data['x'], test_data['y'],
                                poly_csv_path='/data/yangchangfan/DiffusionPDE/data/testing/NS_heat/circlecsv',
                                is_test=True)

    batch_size = 64
    num_workers = 16  # 多进程加载数
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )

    device = 'cuda'
    Epoch = 20
    # model = MultiPhysicsPINN().type(torch.float64).to(device)
    model = torch.load(os.path.join(base_dir,'ckpt','model_19.pth')).to(device)  # 或 'model.pth'
    eval_model(model,base_dir,device)