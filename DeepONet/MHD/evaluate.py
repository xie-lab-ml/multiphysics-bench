import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import json
import sys
from scipy.io import savemat


sys.path.append('../src/')
# 设置随机种子
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
        y = self.output_data[idx]  # (5, 128, 128)

        x_coord = torch.linspace(-0.635, 0.635, 128).view(1, 128, 1).expand(1, 128, 128).to(torch.float64)
        y_coord = torch.linspace(-0.635, 0.635, 128).view(1, 1, 128).expand(1, 128, 128).to(torch.float64)
        coords = torch.cat([x_coord, y_coord], dim=0)
        return x, coords, y, idx

def evaluate(model, dataloader):
    u_metric_total, v_metric_total, T_metric_total, sample_total = 0,0,0,0
    res_dict = {
        'RMSE': {'Jx':0,'Jy':0,'Jz':0, 'u':0, 'v':0},
        'nRMSE': {'Jx':0,'Jy':0,'Jz':0, 'u':0, 'v':0},
        'MaxError': {'Jx':0,'Jy':0,'Jz':0, 'u':0, 'v':0},
        'fRMSE': {},
        'bRMSE': {'Jx':0,'Jy':0,'Jz':0, 'u':0, 'v':0},
    }
    def get_nRMSE():
        Jx,Jy,Jz,u,v = pred
        Jx_metric = torch.norm(Jx - outputs[:, 0, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 0, :, :], 2, dim=(1, 2))
        Jy_metric = torch.norm(Jy - outputs[:, 1, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 1, :, :], 2, dim=(1, 2))
        Jz_metric = torch.norm(Jz - outputs[:, 2, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 2, :, :], 2, dim=(1, 2))
        u_metric = torch.norm(u - outputs[:, 3, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 3, :, :], 2, dim=(1, 2))
        v_metric = torch.norm(v - outputs[:, 4, :, :], 2, dim=(1, 2)) / torch.norm(outputs[:, 4, :, :], 2, dim=(1, 2))

        res_dict['nRMSE']['Jx'] += Jx_metric.sum()
        res_dict['nRMSE']['Jy'] += Jy_metric.sum()
        res_dict['nRMSE']['Jz'] += Jz_metric.sum()
        res_dict['nRMSE']['u'] += u_metric.sum()
        res_dict['nRMSE']['v'] += v_metric.sum()

    def get_RMSE():
        Jx,Jy,Jz,u,v = pred
        Jx_metric = torch.sqrt(torch.mean((Jx - outputs[:, 0, :, :]) ** 2, dim=(1, 2)))
        Jy_metric = torch.sqrt(torch.mean((Jy - outputs[:, 1, :, :]) ** 2, dim=(1, 2)))
        Jz_metric = torch.sqrt(torch.mean((Jz - outputs[:, 2, :, :]) ** 2, dim=(1, 2)))
        u_metric = torch.sqrt(torch.mean((u - outputs[:, 3, :, :]) ** 2, dim=(1, 2)))
        v_metric = torch.sqrt(torch.mean((v - outputs[:, 4, :, :]) ** 2, dim=(1, 2)))
        res_dict['RMSE']['Jx'] += Jx_metric.sum()
        res_dict['RMSE']['Jy'] += Jy_metric.sum()
        res_dict['RMSE']['Jz'] += Jz_metric.sum()
        res_dict['RMSE']['u'] += u_metric.sum()
        res_dict['RMSE']['v'] += v_metric.sum()

    def get_MaxError():
        Jx,Jy,Jz,u,v = pred
        Jx_metric = torch.abs(Jx - outputs[:, 0, :, :]).flatten(1).max(dim=1)[0]  # 先展平再求max
        Jy_metric = torch.abs(Jy - outputs[:, 1, :, :]).flatten(1).max(dim=1)[0]  # 先展平再求max
        Jz_metric = torch.abs(Jz - outputs[:, 2, :, :]).flatten(1).max(dim=1)[0]  # 先展平再求max
        u_metric = torch.abs(u - outputs[:, 3, :, :]).flatten(1).max(dim=1)[0]  # 先展平再求max
        v_metric = torch.abs(v - outputs[:, 4, :, :]).flatten(1).max(dim=1)[0]  # 先展平再求max
        res_dict['MaxError']['Jx'] += Jx_metric.sum()
        res_dict['MaxError']['Jy'] += Jy_metric.sum()
        res_dict['MaxError']['Jz'] += Jz_metric.sum()
        res_dict['MaxError']['u'] += u_metric.sum()
        res_dict['MaxError']['v'] += v_metric.sum()

    def get_bRMSE():
        Jx, Jy, Jz, u, v = pred
        boundary_mask = torch.zeros_like(outputs[:, 0, :, :], dtype=bool)
        boundary_mask[:, 0, :] = True
        boundary_mask[:, -1, :] = True
        boundary_mask[:, :, 0] = True
        boundary_mask[:, :, -1] = True

        Jx_boundary_pred = Jx[boundary_mask].view(Jx.shape[0], -1)
        Jx_boundary_true = outputs[:, 0, :, :][boundary_mask].view(Jx.shape[0], -1)
        Jx_metric = torch.sqrt(torch.mean((Jx_boundary_pred - Jx_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['Jx'] += Jx_metric.sum()

        Jy_boundary_pred = Jy[boundary_mask].view(Jy.shape[0], -1)
        Jy_boundary_true = outputs[:, 1, :, :][boundary_mask].view(Jy.shape[0], -1)
        Jy_metric = torch.sqrt(torch.mean((Jy_boundary_pred - Jy_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['Jy'] += Jy_metric.sum()

        Jz_boundary_pred = Jz[boundary_mask].view(Jz.shape[0], -1)
        Jz_boundary_true = outputs[:, 2, :, :][boundary_mask].view(Jz.shape[0], -1)
        Jz_metric = torch.sqrt(torch.mean((Jz_boundary_pred - Jz_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['Jz'] += Jz_metric.sum()

        u_boundary_pred = u[boundary_mask].view(u.shape[0], -1)
        u_boundary_true = outputs[:, 3, :, :][boundary_mask].view(u.shape[0], -1)
        u_metric = torch.sqrt(torch.mean((u_boundary_pred - u_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['u'] += u_metric.sum()

        v_boundary_pred = v[boundary_mask].view(v.shape[0], -1)
        v_boundary_true = outputs[:, 4, :, :][boundary_mask].view(v.shape[0], -1)
        v_metric = torch.sqrt(torch.mean((v_boundary_pred - v_boundary_true) ** 2, dim=1))
        res_dict['bRMSE']['v'] += v_metric.sum()

    def get_fRMSE():
        Jx, Jy, Jz, u, v = pred

        for freq_band in ['low', 'middle', 'high']:
            res_dict['fRMSE'][f'Jx_{freq_band}'] = 0.0
            res_dict['fRMSE'][f'Jy_{freq_band}'] = 0.0
            res_dict['fRMSE'][f'Jz_{freq_band}'] = 0.0
            res_dict['fRMSE'][f'u_{freq_band}'] = 0.0
            res_dict['fRMSE'][f'v_{freq_band}'] = 0.0

        freq_bands = {
            'low': (0, 4),  # k_min=0, k_max=4
            'middle': (5, 12),  # k_min=5, k_max=12
            'high': (13, None)  # k_min=13, k_max=∞
        }

        def compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W):
            kx = torch.arange(H, device=pred_fft.device)
            ky = torch.arange(W, device=pred_fft.device)
            kx, ky = torch.meshgrid(kx, ky, indexing='ij')

            r = torch.sqrt(kx ** 2 + ky ** 2)
            if k_max is None:
                mask = (r >= k_min)
                k_max = max(H // 2, W // 2) #nyquist
            else:
                mask = (r >= k_min) & (r <= k_max)


            diff_fft = torch.abs(pred_fft - true_fft) ** 2
            band_error = diff_fft[:, mask].sum(dim=1)
            band_error = torch.sqrt(band_error) / (k_max - k_min + 1)
            return band_error

        for channel_idx, (pred_ch, true_ch, name) in enumerate([
            (Jx, outputs[:, 0, :, :], 'Jx'),
            (Jy, outputs[:, 1, :, :], 'Jy'),
            (Jz, outputs[:, 2, :, :], 'Jz'),
            (u, outputs[:, 3, :, :], 'u'),
            (v, outputs[:, 4, :, :], 'v')
        ]):
            pred_fft = torch.fft.fft2(pred_ch)
            true_fft = torch.fft.fft2(true_ch)
            H, W = pred_ch.shape[-2], pred_ch.shape[-1]

            for band, (k_min, k_max) in freq_bands.items():
                error = compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W)
                res_dict['fRMSE'][f'{name}_{band}'] += error.sum()

    for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            coords = coords.to(device)
            outputs = outputs.to(device)
            pred_outputs = model.forward(inputs, coords)

            # GT inv_norm
            outputs[:, 0, :, :] = ((outputs[:, 0, :, :] + 0.9) / 1.8 * (model.max_Jx - model.min_Jx) + model.min_Jx).to(torch.float64)
            outputs[:, 1, :, :] = ((outputs[:, 1, :, :] + 0.9) / 1.8 * (model.max_Jy - model.min_Jy) + model.min_Jy).to(torch.float64)
            outputs[:, 2, :, :] = ((outputs[:, 2, :, :] + 0.9) / 1.8 * (model.max_Jz - model.min_Jz) + model.min_Jz).to(torch.float64)
            outputs[:, 3, :, :] = ((outputs[:, 3, :, :] + 0.9) / 1.8 * (model.max_u_u - model.min_u_u) + model.min_u_u).to(torch.float64)
            outputs[:, 4, :, :] = ((outputs[:, 4, :, :] + 0.9) / 1.8 * (model.max_u_v - model.min_u_v) + model.min_u_v).to(torch.float64)

            # pred inv_norm
            pred_outputs[:, :, :, 0] = ((pred_outputs[:, :, :, 0] + 0.9) / 1.8 * (model.max_Jx - model.min_Jx) + model.min_Jx).to(
                torch.float64)
            pred_outputs[:, :, :, 1] = ((pred_outputs[:, :, :, 1] + 0.9) / 1.8 * (model.max_Jy - model.min_Jy) + model.min_Jy).to(
                torch.float64)
            pred_outputs[:, :, :, 2] = ((pred_outputs[:, :, :, 2] + 0.9) / 1.8 * (model.max_Jz - model.min_Jz) + model.min_Jz).to(
                torch.float64)
            pred_outputs[:, :, :, 3] = (
                        (pred_outputs[:, :, :, 3] + 0.9) / 1.8 * (model.max_u_u - model.min_u_u) + model.min_u_u).to(
                torch.float64)
            pred_outputs[:, :, :, 4] = (
                        (pred_outputs[:, :, :, 4] + 0.9) / 1.8 * (model.max_u_v - model.min_u_v) + model.min_u_v).to(
                torch.float64)
            pred = (pred_outputs[:, :, :, 0],pred_outputs[:, :, :, 1],pred_outputs[:, :, :, 2],pred_outputs[:, :, :, 3],pred_outputs[:, :, :, 4])

            pred = (
            pred_outputs[:, :, :, 0], pred_outputs[:, :, :, 1], pred_outputs[:, :, :, 2], pred_outputs[:, :, :, 3],
            pred_outputs[:, :, :, 4])

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

def eval_model(model,test_dataloader,base_dir,device='cuda'):
    res_dict = evaluate(model, test_dataloader)
    print('-' * 20)
    print(f'metric:')
    for metric in res_dict:
        for var in res_dict[metric]:
            print(f'{metric}\t\t{var}:\t\t{res_dict[metric][var]}')

if __name__ == '__main__':
    dataset_path = '../../bench_data/'
    ckpt_path = './ckpt'
    test_data = torch.load(os.path.join(dataset_path, 'MHD_test_128.pt'))
    test_dataset = FieldDataset(test_data['x'], test_data['y'])

    batch_size = 100
    num_workers = 8
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    device = 'cuda'
    model = torch.load(os.path.join(ckpt_path, 'model_195.pth')).to(device)
    eval_model(model, test_dataloader, device)