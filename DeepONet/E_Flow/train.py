import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import scipy.io as sio
import json
import pandas as pd
import sys

sys.path.append('../src/')
from E_Flow_model_Large import E_Flow_DeepONet

torch.manual_seed(42)
np.random.seed(42)

class FieldDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data  # (N, 1, H, W)
        self.output_data = output_data  # (N, 3, H, W)

    def __len__(self):
        return len(self.input_data)


    def __getitem__(self, idx):
        x = self.input_data[idx]  # (1, 128, 128)
        y = self.output_data[idx]  # (3, 128, 128)

        x_coord = torch.linspace(-0.635, 0.635, 128).view(1, 128, 1).expand(1, 128, 128).to(torch.float64)
        y_coord = torch.linspace(-0.635, 0.635, 128).view(1, 1, 128).expand(1, 128, 128).to(torch.float64)
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
            u_u_N = ((pred_outputs[:,:,:,0] + 0.9) / 1.8 * (model.max_ec_V - model.min_ec_V) + model.min_ec_V).to(
                torch.float64)
            u_v_N = ((pred_outputs[:,:,:,1] + 0.9) / 1.8 * (model.max_u_flow - model.min_u_flow) + model.min_u_flow).to(
                torch.float64)
            T_N = ((pred_outputs[:,:,:,2] + 0.9) / 1.8 * (model.max_v_flow - model.min_v_flow) + model.min_v_flow).to(
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

def train_model(model, dataloader, train_dataset, test_dataloader, optimizer, scheduler, Epoch, clip_value, save_dir, device='cuda'):
    for epoch in range(Epoch):
        total_loss = 0.0
        for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            coords = coords.to(device)

            optimizer.zero_grad()
            data_loss = model.compute_loss(inputs, coords, outputs[:, 0].to(device),outputs[:, 1].to(device),outputs[:, 2].to(device))
            loss = data_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch%5==0:
            res_dict = evaluate(model, test_dataloader)
            print('-' * 20)
            print(f'metric:')
            for metric in res_dict:
                for var in res_dict[metric]:
                    print(f'{metric}\t\t{var}:\t\t{res_dict[metric][var]}')

            #TODO 保存log
            with open(os.path.join(save_dir, 'log', f'log_{epoch}.log'), "w", encoding="utf-8") as f:
                json.dump(res_dict, f, ensure_ascii=False)
            #TODO 画图
            plot_results(model, train_dataset, savepath=os.path.join(save_dir, 'fig', f'show_{epoch}.png'), sample_idx=0)
            #TODO 保存ckpt
            torch.save(model, os.path.join(save_dir, 'ckpt', f'model_{epoch}.pth'))
        scheduler.step()
    return model

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
    inputs, coords, outputs, polygt_idx = dataset[sample_idx]
    inputs = inputs.unsqueeze(0).to(device)
    coords = coords.unsqueeze(0).to(device)
    outputs = outputs.to(device)

    with torch.no_grad():
        output = model.forward(inputs, coords)
        u, v, T = output[:, :, :, 0], output[:, :, :, 1], output[:, :, :, 2]

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
    base_dir = './data/E_Flow_10000_Large'
    if not os.path.exists(os.path.join(base_dir)):
        os.mkdir(os.path.join(base_dir))
    if not os.path.exists(os.path.join(base_dir,'fig')):
        os.mkdir(os.path.join(base_dir,'fig'))
    if not os.path.exists(os.path.join(base_dir,'ckpt')):
        os.mkdir(os.path.join(base_dir,'ckpt'))
    if not os.path.exists(os.path.join(base_dir,'log')):
        os.mkdir(os.path.join(base_dir,'log'))

    train_data = torch.load('/data/bailichen/PDE/PDE/DeepONet/E_Flow/data/E_flow_train_128.pt')
    train_data['x'] = train_data['x']
    train_data['y'] = train_data['y']
    train_dataset = FieldDataset(train_data['x'], train_data['y'])

    test_data = torch.load('/data/bailichen/PDE/PDE/DeepONet/E_Flow/data/E_flow_test_128.pt')
    test_dataset = FieldDataset(test_data['x'], test_data['y'])

    batch_size = 64
    num_workers = 8  # 多进程加载数
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
    Epoch = 200
    model = E_Flow_DeepONet().type(torch.float64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max = Epoch)
    train_model(model,train_dataloader,train_dataset, test_dataloader, optimizer,scheduler,Epoch, 1.0 ,base_dir,device)