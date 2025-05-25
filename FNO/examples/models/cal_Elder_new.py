import torch
from neuralop.models import FNO
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_Elder
import os
from scipy import io as sio
import numpy as np
import tqdm
import json
import pandas as pd

device = 'cuda'

def invnormalize(data, min_val, max_val):
    return (data + 0.9) / 1.8 * (max_val - min_val ) + min_val

if __name__ == '__main__':
    save_dir = './eval_results/Elder_new'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Let's load the TE_heat dataset.
    train_loader, test_loaders, data_processor = load_Elder(
            n_train=1000, batch_size=16,
            test_resolutions=[128], n_tests=[100],
            test_batch_sizes=[64],
    )
    data_processor = data_processor.to(device)
    data_processor.eval()

    model = FNO(n_modes=(12, 12),
                in_channels=4,
                out_channels=30,
                hidden_channels=128,
                projection_channel_ratio=2)
    model = model.to(device)

    train_data_base_path = "/data/yangchangfan/DiffusionPDE/data/training/Elder/"
    # -------------------- 加载归一化范围 --------------------
    range_allS_c = sio.loadmat(os.path.join(train_data_base_path, "S_c/range_S_c_t.mat"))['range_S_c_t'][0]
    range_allu_u = sio.loadmat(os.path.join(train_data_base_path, "u_u/range_u_u_t_999.mat"))['range_u_u_t_999'][1:]
    range_allu_v = sio.loadmat(os.path.join(train_data_base_path, "u_v/range_u_v_t_99.mat"))['range_u_v_t_99'][1:]
    range_allc_flow = sio.loadmat(os.path.join(train_data_base_path, "c_flow/range_c_flow_t_99.mat"))[
        'range_c_flow_t_99'][1:]
    model.S_c_ranges = range_allS_c
    model.ranges = {
        'u_u': range_allu_u,
        'u_v': range_allu_v,
        'c_flow': range_allc_flow,
    }

    model.load_state_dict(torch.load("./checkpoints/Elder_new/14/model_epoch_49_state_dict.pt", weights_only=False))
    print("Model weights loaded from model_weights.pt")

    # 将模型设置为评估模式
    model.eval()

    res_dict = {
        'RMSE': {}, 'nRMSE': {}, 'MaxError': {}, 'fRMSE_low': {}, 'fRMSE_middle': {}, 'fRMSE_high': {}, 'bRMSE': {}
    }

    for metric in res_dict:
        for key in ['u_u', 'u_v', 'c_flow']:
            res_dict[metric][key] = {}
            for t in range(11):
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

    #开始测试
    sample_total = 0
    for idx, sample in enumerate(tqdm.tqdm(test_loaders[128])):
        with torch.no_grad():
            sample = data_processor.preprocess(sample)
            inputs,outputs = sample['x'].to(device),sample['y'].to(device).squeeze()
            pred_outputs = model(inputs)
            pred_outputs, _ = data_processor.postprocess(pred_outputs)
            pred_outputs = pred_outputs.squeeze()

            u_u_N = pred_outputs[:, 0:10, :, :]
            u_v_N = pred_outputs[:, 10:20, :, :]
            c_flow_N = pred_outputs[:, 20:30, :, :]

            u_u_gt = outputs[:, 0:10, :, :]
            u_v_gt = outputs[:, 10:20, :, :]
            c_flow_gt = outputs[:, 20:30, :, :]

            for t in range(10):
                u_u_N[:, t, :, :] = invnormalize(u_u_N[:, t, :, :], *model.ranges['u_u'][t, :]).to(torch.float64)
                u_v_N[:, t, :, :] = invnormalize(u_v_N[:, t, :, :], *model.ranges['u_v'][t, :]).to(torch.float64)
                c_flow_N[:, t, :, :] = invnormalize(c_flow_N[:, t, :, :], *model.ranges['c_flow'][t, :]).to(
                    torch.float64)

                u_u_gt[:, t, :, :] = invnormalize(u_u_gt[:, t, :, :], *model.ranges['u_u'][t, :]).to(torch.float64)
                u_v_gt[:, t, :, :] = invnormalize(u_v_gt[:, t, :, :], *model.ranges['u_v'][t, :]).to(torch.float64)
                c_flow_gt[:, t, :, :] = invnormalize(c_flow_gt[:, t, :, :], *model.ranges['c_flow'][t, :]).to(
                    torch.float64)

            get_RMSE()
            get_nRMSE()
            get_MaxError()
            get_bRMSE()
            get_fRMSE()
            sample_total += outputs.shape[0]

    for metric in res_dict:
        for var in res_dict[metric]:
            avg,count = 0,0
            for t in res_dict[metric][var]:
                res_dict[metric][var][t] /= sample_total
                if type(res_dict[metric][var][t]) != float:
                    res_dict[metric][var][t] = res_dict[metric][var][t].item()
                avg += res_dict[metric][var][t]
                count += 1
            res_dict[metric][var]['avg'] = avg/count

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

    output_file = os.path.join(save_dir, './exp.csv')
    frmse_df = pd.DataFrame(res)
    frmse_df.to_csv(output_file, index=False, encoding="utf-8", float_format="%.16f")
