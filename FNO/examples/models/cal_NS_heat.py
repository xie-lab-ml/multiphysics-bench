import torch
from neuralop.models import FNO
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_NS_heat
import os
from scipy import io as sio
import numpy as np
import tqdm
import json
import pandas as pd

device = 'cuda'

if __name__ == '__main__':
    save_dir = './eval_results/NS_heat'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    range_allQ_heat_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/Q_heat/range_allQ_heat.mat"
    range_allQ_heat = sio.loadmat(range_allQ_heat_paths)['range_allQ_heat']
    range_allQ_heat = torch.tensor(range_allQ_heat, device=device)
    max_Q_heat = range_allQ_heat[0, 1]
    min_Q_heat = range_allQ_heat[0, 0]

    range_allu_u_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_u/range_allu_u.mat"
    range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']
    range_allu_u = torch.tensor(range_allu_u, device=device)
    max_u_u = range_allu_u[0, 1]
    min_u_u = range_allu_u[0, 0]

    range_allu_v_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_v/range_allu_v.mat"
    range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']
    range_allu_v = torch.tensor(range_allu_v, device=device)
    max_u_v = range_allu_v[0, 1]
    min_u_v = range_allu_v[0, 0]

    range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/T/range_allT.mat"
    range_allT = sio.loadmat(range_allT_paths)['range_allT']
    range_allT = torch.tensor(range_allT, device=device)
    max_T = range_allT[0, 1]
    min_T = range_allT[0, 0]


    # Let's load the TE_heat dataset.
    train_loader, test_loaders, data_processor = load_NS_heat(
            n_train=10000, batch_size=16,
            test_resolutions=[128], n_tests=[1000],
            test_batch_sizes=[64],
    )
    data_processor = data_processor.to(device)
    data_processor.eval()

    model = FNO(n_modes=(12, 12),
                 in_channels=1,
                 out_channels=3,
                 hidden_channels=128,
                 projection_channel_ratio=2)
    model = model.to(device)

    model.max_Q_heat = max_Q_heat
    model.min_Q_heat = min_Q_heat
    model.max_u_u = max_u_u
    model.min_u_u = min_u_u
    model.max_u_v = max_u_v
    model.min_u_v = min_u_v
    model.max_T = max_T
    model.min_T = min_T

    model.load_state_dict(torch.load("./checkpoints/NS_heat_10000/1/model_epoch_49_state_dict.pt", weights_only=False))
    print("Model weights loaded from model_weights.pt")

    # 将模型设置为评估模式
    model.eval()

    u_metric_total, v_metric_total, T_metric_total, sample_total = 0, 0, 0, 0
    res_dict = {
        'RMSE': {'u': 0, 'v': 0, 'T': 0},
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

    #开始测试
    for idx, sample in enumerate(tqdm.tqdm(test_loaders[128])):
        with torch.no_grad():
            print(sample['y'].mean())
            sample = data_processor.preprocess(sample)
            print(sample['y'].mean())
            inputs,outputs = sample['x'].to(device),sample['y'].to(device).squeeze()
            pred_outputs = model(inputs)
            pred_outputs, _ = data_processor.postprocess(pred_outputs)
            pred_outputs = pred_outputs.squeeze()

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
            pred_outputs[:,0,:,:] = ((pred_outputs[:,0,:,:] + 0.9) / 1.8 * (model.max_u_u - model.min_u_u) + model.min_u_u).to(
                torch.float64)
            pred_outputs[:,1,:,:] = ((pred_outputs[:,1,:,:] + 0.9) / 1.8 * (model.max_u_v - model.min_u_v) + model.min_u_v).to(
                torch.float64)
            pred_outputs[:,2,:,:] = ((pred_outputs[:,2,:,:] + 0.9) / 1.8 * (model.max_T - model.min_T) + model.min_T).to(
                torch.float64)

            pred = (pred_outputs[:, 0, :, :], pred_outputs[:, 1, :, :], pred_outputs[:, 2, :, :])

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
            res.append(data[metric][var])

    output_file = os.path.join(save_dir, './exp.csv')
    frmse_df = pd.DataFrame(res)
    frmse_df.to_csv(output_file, index=False, encoding="utf-8", float_format="%.16f")
