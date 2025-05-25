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
            outputs[:, 0, :, :] = ((outputs[:, 0, :, :] + 0.9) / 1.8 * (model.max_ec_V - model.min_ec_V) + model.min_ec_V).to(
                torch.float64)
            outputs[:, 1, :, :] = ((outputs[:, 1, :, :] + 0.9) / 1.8 * (model.max_u_flow - model.min_u_flow) + model.min_u_flow).to(
                torch.float64)
            outputs[:, 2, :, :] = ((outputs[:, 2, :, :] + 0.9) / 1.8 * (model.max_v_flow - model.min_v_flow) + model.min_v_flow).to(
                torch.float64)

            #Pred 反归一化
            E_real_pred, E_imag_pred, T_pred = pred_outputs
            u_u_N = ((pred_outputs[0] + 0.9) / 1.8 * (model.max_ec_V - model.min_ec_V) + model.min_ec_V).to(
                torch.float64)
            u_v_N = ((pred_outputs[1] + 0.9) / 1.8 * (model.max_u_flow - model.min_u_flow) + model.min_u_flow).to(
                torch.float64)
            T_N = ((pred_outputs[2] + 0.9) / 1.8 * (model.max_v_flow - model.min_v_flow) + model.min_v_flow).to(
                torch.float64)
            pred = (u_u_N,u_v_N,T_N)

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

        # inv_normalization
        range_allkappa_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/kappa/range_allkappa.mat"
        range_allkappa = sio.loadmat(range_allkappa_paths)['range_allkappa']
        range_allkappa = torch.tensor(range_allkappa, device=device)

        self.max_kappa = range_allkappa[0, 1]
        self.min_kappa = range_allkappa[0, 0]

        range_allec_V_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/ec_V/range_allec_V.mat"
        range_allec_V = sio.loadmat(range_allec_V_paths)['range_allec_V']
        range_allec_V = torch.tensor(range_allec_V, device=device)

        self.max_ec_V = range_allec_V[0, 1]
        self.min_ec_V = range_allec_V[0, 0]

        range_allu_flow_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/u_flow/range_allu_flow.mat"
        range_allu_flow = sio.loadmat(range_allu_flow_paths)['range_allu_flow']
        range_allu_flow = torch.tensor(range_allu_flow, device=device)

        self.max_u_flow = range_allu_flow[0, 1]
        self.min_u_flow = range_allu_flow[0, 0]

        range_allv_flow_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/v_flow/range_allv_flow.mat"
        range_allv_flow = sio.loadmat(range_allv_flow_paths)['range_allv_flow']
        range_allv_flow = torch.tensor(range_allv_flow, device=device)

        self.max_v_flow = range_allv_flow[0, 1]
        self.min_v_flow = range_allv_flow[0, 0]

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
        output = self.output_layer(x)

        output = output.view(batch_size, 128, 128, 3).permute(0, 3, 1, 2)
        return output[:, 0], output[:, 1], output[:, 2]  # E_real, E_imag, T

    def compute_loss(self, inputs, coords, ec_V_GT,u_flow_GT,v_flow_GT):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        ec_V_N, u_flow_N, v_flow_N = self.forward(inputs, coords)

        data_loss = torch.nn.MSELoss()(ec_V_N,ec_V_GT) + torch.nn.MSELoss()(u_flow_N,u_flow_GT) + torch.nn.MSELoss()(v_flow_N,v_flow_GT)

        #inv norm
        kappa_N = ((inputs + 0.9) / 1.8 * (model.max_kappa - model.min_kappa) + model.min_kappa).to(torch.float64)
        ec_V_N = ((ec_V_N + 0.9) / 1.8 * (model.max_ec_V - model.min_ec_V) + model.min_ec_V).to(torch.float64)
        u_flow_N = ((u_flow_N + 0.9) / 1.8 * (model.max_u_flow - model.min_u_flow) + model.min_u_flow).to(torch.float64)
        v_flow_N = ((v_flow_N + 0.9) / 1.8 * (model.max_v_flow - model.min_v_flow) + model.min_v_flow).to(torch.float64)

        pde_loss_NS, pde_loss_J = self.get_E_flow_loss(kappa_N, ec_V_N, u_flow_N, v_flow_N, device)

        return pde_loss_NS ,pde_loss_J, data_loss

    def get_E_flow_loss(self, kappa, ec_V, u_flow, v_flow, device=torch.device('cuda')):
        """Return the loss of the E_flow equation and the observation loss."""

        delta_x = 1.28e-3 / 128  # 1mm
        delta_y = 1.28e-3 / 128  # 1mm

        deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / (2 * delta_x)
        deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / (2 * delta_y)

        # Continuity_NS
        grad_x_next_x_NS = F.conv2d(u_flow.unsqueeze(1), deriv_x, padding=(0, 1))
        grad_x_next_y_NS = F.conv2d(v_flow.unsqueeze(1), deriv_y, padding=(1, 0))
        result_NS = grad_x_next_x_NS + grad_x_next_y_NS


        # Continuity_J
        grad_x_next_x_V = F.conv2d(ec_V.unsqueeze(1), deriv_x, padding=(0, 1)).squeeze()
        grad_x_next_y_V = F.conv2d(ec_V.unsqueeze(1), deriv_y, padding=(1, 0)).squeeze()


        grad_x_next_x_J = F.conv2d((kappa * grad_x_next_x_V).unsqueeze(1), deriv_x, padding=(0, 1)).squeeze()
        grad_x_next_y_J = F.conv2d((kappa * grad_x_next_y_V).unsqueeze(1), deriv_y, padding=(1, 0)).squeeze()

        result_J = grad_x_next_x_J + grad_x_next_y_J

        pde_loss_NS = result_NS
        pde_loss_J = result_J

        pde_loss_NS = pde_loss_NS.squeeze()
        pde_loss_J = pde_loss_J.squeeze()

        pde_loss_NS = pde_loss_NS / 1000
        pde_loss_J = pde_loss_J / 1000000

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

        return pde_loss_NS, pde_loss_J

def train_model(model, dataloader, train_dataset, test_dataset, optimizer, scheduler, Epoch, clip_value, save_dir, device='cuda'):
    for epoch in range(Epoch):
        total_loss = 0.0
        for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            coords = coords.to(device)

            optimizer.zero_grad()
            pde_loss_NS ,pde_loss_J, data_loss = model.compute_loss(inputs, coords, outputs[:, 0].to(device),outputs[:, 1].to(device),outputs[:, 2].to(device))
            L_pde_NS = torch.norm(pde_loss_NS, 2) / (128 * 128)
            L_pde_J = torch.norm(pde_loss_J, 2) / (128 * 128)

            print(f'L_pde_NS: {L_pde_NS.item()}, \t L_pde_J: {L_pde_J.item()} \t Data_Loss: {data_loss.item()}')
            L_pde_NS = L_pde_NS * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_NS.clone().detach()))
            L_pde_J = L_pde_J * (torch.norm(data_loss.clone().detach()) / torch.norm(L_pde_J.clone().detach()))
            loss = 0.001 * (L_pde_NS + L_pde_J) + data_loss

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
        u, v, T = model.forward(inputs, coords)

    #Pred 归一化
    u_u_N = ((u + 0.9) / 1.8 * (model.max_ec_V - model.min_ec_V) + model.min_ec_V).to(torch.float64).cpu().numpy().squeeze()
    u_v_N = ((v + 0.9) / 1.8 * (model.max_u_flow - model.min_u_flow) + model.min_u_flow).to(torch.float64).cpu().numpy().squeeze()
    T_N = ((T + 0.9) / 1.8 * (model.max_v_flow - model.min_v_flow) + model.min_v_flow).to(torch.float64).cpu().numpy().squeeze()

    #Ground Truth 归一化
    u_true = outputs[0]
    u_true = ((u_true + 0.9) / 1.8 * (model.max_ec_V - model.min_ec_V) + model.min_ec_V).to(torch.float64).cpu().numpy().squeeze()
    v_true = outputs[1]
    v_true = ((v_true + 0.9) / 1.8 * (model.max_u_flow - model.min_u_flow) + model.min_u_flow).to(torch.float64).cpu().numpy().squeeze()
    T_true = outputs[2]
    T_true = ((T_true + 0.9) / 1.8 * (model.max_v_flow - model.min_v_flow) + model.min_v_flow).to(torch.float64).cpu().numpy().squeeze()

    # 绘制结果
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    titles = ['V (True)', 'V (Pred)', 'u_flow (True)', 'u_flow (Pred)', 'v_flow (True)', 'v_flow (Pred)']
    data = [u_true, u_u_N, v_true, u_v_N, T_true, T_N]

    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(data[i], cmap='jet', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax)
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(savepath)

if __name__ == '__main__':
    base_dir = './data/e_flow'
    if not os.path.exists(os.path.join(base_dir,'fig')):
        os.mkdir(os.path.join(base_dir,'fig'))
    if not os.path.exists(os.path.join(base_dir,'ckpt')):
        os.mkdir(os.path.join(base_dir,'ckpt'))
    if not os.path.exists(os.path.join(base_dir,'fig_loss')):
        os.mkdir(os.path.join(base_dir,'fig_loss'))

    train_data = torch.load('/data/bailichen/PDE/PDE/pinns/data/e_flow/E_flow_train_128.pt')
    train_dataset = FieldDataset(train_data['x'], train_data['y'],
                                 poly_csv_path=None,
                                 is_test=False)

    test_data = torch.load('/data/bailichen/PDE/PDE/pinns/data/e_flow/E_flow_test_128.pt')
    test_polycsv = None
    test_dataset = FieldDataset(test_data['x'], test_data['y'],
                                poly_csv_path=None,
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