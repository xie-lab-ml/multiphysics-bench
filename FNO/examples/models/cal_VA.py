import torch
from neuralop.models import FNO
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_VA
import os
from scipy import io as sio
import numpy as np
import tqdm
import json
import pandas as pd

device = 'cuda'

if __name__ == '__main__':
    save_dir = './eval_results/VA'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Let's load the TE_heat dataset.
    train_loader, test_loaders, data_processor = load_VA(
            n_train=10000, batch_size=16,
            test_resolutions=[128], n_tests=[1000],
            test_batch_sizes=[64],
    )
    data_processor = data_processor.to(device)
    data_processor.eval()

    model = FNO(n_modes=(12, 12),
                 in_channels=1,
                 out_channels=12,
                 hidden_channels=128,
                 projection_channel_ratio=2)
    model = model.to(device)

    model.variables = ['p_t', 'Sxx', 'Sxy', 'Syy', 'x_u', 'x_v']


    def load_ranges(base_path, variables):
        ranges = {}
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
    model.ranges = load_ranges(base_path="/data/yangchangfan/DiffusionPDE/data/training/VA", variables=model.variables)

    model.load_state_dict(torch.load("./checkpoints/VA_10000/4/model_epoch_49_state_dict.pt", weights_only=False))
    print("Model weights loaded from model_weights.pt")

    # 将模型设置为评估模式
    model.eval()

    u_metric_total, v_metric_total, T_metric_total, sample_total = 0, 0, 0, 0
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
            metric_value_real = torch.norm(pred_outputs[:,var_idx*2,:,:] - outputs[:,var_idx*2,:,:], 2, dim=(1, 2)) / torch.norm(outputs[:,var_idx*2,:,:], 2, dim=(1, 2))
            metric_value_imag = torch.norm(pred_outputs[:,var_idx*2+1,:,:] - outputs[:,var_idx*2+1,:,:], 2, dim=(1, 2)) / torch.norm(outputs[:,var_idx*2+1,:,:], 2, dim=(1, 2))
            res_dict['nRMSE'][f'{var}_real'] += metric_value_real.sum()
            res_dict['nRMSE'][f'{var}_imag'] += metric_value_imag.sum()


    def get_RMSE():
        for var_idx, var in enumerate(model.variables):
            metric_value_real = torch.sqrt(torch.mean((pred_outputs[:,var_idx*2,:,:] - outputs[:,var_idx*2,:,:]) ** 2, dim=(1, 2)))
            metric_value_imag = torch.sqrt(torch.mean((pred_outputs[:,var_idx*2+1,:,:] - outputs[:,var_idx*2+1,:,:]) ** 2, dim=(1, 2)))
            res_dict['RMSE'][f'{var}_real'] += metric_value_real.sum()
            res_dict['RMSE'][f'{var}_imag'] += metric_value_imag.sum()

    def get_MaxError():
        for var_idx, var in enumerate(model.variables):
            metric_value_real = torch.abs(pred_outputs[:,var_idx*2,:,:] - outputs[:,var_idx*2,:,:]).flatten(1).max(dim=1)[0]
            metric_value_imag = torch.abs(pred_outputs[:,var_idx*2+1,:,:] - outputs[:,var_idx*2+1,:,:]).flatten(1).max(dim=1)[0]
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
            boundary_pred_real = pred_outputs[:,var_idx*2,:,:][boundary_mask].view(pred_outputs.shape[0], -1)
            boundary_true_real = outputs[:,var_idx*2,:,:][boundary_mask].view(outputs.shape[0], -1)
            real_metric = torch.sqrt(torch.mean((boundary_pred_real - boundary_true_real) ** 2, dim=1))
            res_dict['bRMSE'][f'{var}_real'] += real_metric.sum()

            boundary_pred_imag = pred_outputs[:, var_idx * 2+1, :, :][boundary_mask].view(pred_outputs.shape[0], -1)
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
            pred_fft = torch.fft.fft2(pred_outputs[:,var_idx*2,:,:])
            true_fft = torch.fft.fft2(outputs[:,var_idx*2,:,:])
            H, W = outputs.shape[-2], outputs.shape[-1]
            for band, (k_min, k_max) in freq_bands.items():
                error = compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W)
                res_dict['fRMSE'][f'{var}_{band}_real'] += error.sum()

            #imag
            pred_fft = torch.fft.fft2(pred_outputs[:, var_idx * 2+1, :, :])
            true_fft = torch.fft.fft2(outputs[:, var_idx * 2+1, :, :])
            H, W = outputs.shape[-2], outputs.shape[-1]
            for band, (k_min, k_max) in freq_bands.items():
                error = compute_band_fft(pred_fft, true_fft, k_min, k_max, H, W)
                res_dict['fRMSE'][f'{var}_{band}_imag'] += error.sum()

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

            # for var_idx, var in enumerate(model.variables):
            #     var_minmax = model.ranges[var]
            #     outputs[:,var_idx*2,:,:] = ((outputs[:,var_idx*2,:,:] + 0.9) / 1.8 * (
            #                 var_minmax['max_real'] - var_minmax['min_real']) + var_minmax[
            #                           'min_real']).to(torch.float64)
            #     outputs[:, var_idx * 2+1, :, :] = ((outputs[:, var_idx * 2+1, :, :] + 0.9) / 1.8 * (
            #             var_minmax['max_imag'] - var_minmax['min_imag']) + var_minmax[
            #                                          'min_imag']).to(torch.float64)
            #
            #     pred_outputs[:, var_idx * 2, :, :] = ((pred_outputs[:, var_idx * 2, :, :] + 0.9) / 1.8 * (
            #             var_minmax['max_real'] - var_minmax['min_real']) + var_minmax[
            #                                          'min_real']).to(torch.float64)
            #     pred_outputs[:, var_idx * 2 + 1, :, :] = ((pred_outputs[:, var_idx * 2 + 1, :, :] + 0.9) / 1.8 * (
            #             var_minmax['max_imag'] - var_minmax['min_imag']) + var_minmax[
            #                                              'min_imag']).to(torch.float64)

            get_RMSE()
            get_nRMSE()
            get_MaxError()
            get_bRMSE()
            get_fRMSE()
            sample_total += outputs.shape[0]

    for metric in res_dict:
        for var in res_dict[metric]:
            res_dict[metric][var] /= sample_total
            if type(res_dict[metric][var]) is not float:
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
