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
from scipy.io import savemat,loadmat

sys.path.append('../src/')
# from TE_model import TEHeatDeepONet
# from VA_model import VADeepONet

torch.manual_seed(42)
np.random.seed(42)


def invnormalize(data, min_val, max_val):
    return (data + 0.9) / 1.8 * (max_val - min_val) + min_val

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
    res_dict = {
        'RMSE': {}, 'nRMSE': {}, 'MaxError': {}, 'fRMSE_low': {}, 'fRMSE_middle': {}, 'fRMSE_high': {},'bRMSE': {}
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
            pred_outputs = model.forward(inputs, coords).permute(0,3,1,2)

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
            # print(f'{var}, {metric}', avg/count)
            res_dict[metric][var]['avg'] = avg / count
    return res_dict

def eval_model(model, test_dataloader, device='cuda'):
    res_dict = evaluate(model, test_dataloader)
    print('-' * 20)
    print(f'metric:')
    for metric in res_dict:
        for var in res_dict[metric]:
            print(f'{metric}\t\t{var}:\t\t{res_dict[metric][var]}')

if __name__ == '__main__':
    dataset_path = '../../bench_data/'
    ckpt_path = './ckpt'
    test_data = torch.load(os.path.join(dataset_path, 'Elder_test_128.pt'))
    test_dataset = FieldDataset(test_data['x'], test_data['y'])

    batch_size = 100
    num_workers = 8  # 多进程加载数
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输
    )

    device = 'cuda'
    model = torch.load(os.path.join(ckpt_path, 'model_195.pth')).to(device)  # 或 'model.pth'
    eval_model(model, test_dataloader, device)