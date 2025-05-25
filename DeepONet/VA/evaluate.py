import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import sys
from VA_model_Large import VADeepONet

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
        y = self.output_data[idx]  # (3, 128, 128)

        x_coord = torch.linspace(-0.635, 0.635, 128).view(1, 128, 1).expand(1, 128, 128).to(torch.float64)
        y_coord = torch.linspace(-0.635, 0.635, 128).view(1, 1, 128).expand(1, 128, 128).to(torch.float64)
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
        for var_idx, var in enumerate(model.variables):
            for freq_band in ['low', 'middle', 'high']:
                res_dict['fRMSE'][f'{var}_{freq_band}_real'] = 0.0
                res_dict['fRMSE'][f'{var}_{freq_band}_real'] = 0.0
                res_dict['fRMSE'][f'{var}_{freq_band}_real'] = 0.0

                res_dict['fRMSE'][f'{var}_{freq_band}_imag'] = 0.0
                res_dict['fRMSE'][f'{var}_{freq_band}_imag'] = 0.0
                res_dict['fRMSE'][f'{var}_{freq_band}_imag'] = 0.0

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
        sample_total += outputs.shape[0]
        with torch.no_grad():
            inputs = inputs.to(device)
            coords = coords.to(device)
            outputs = outputs.to(device)
            pred_outputs = model.forward(inputs, coords)

            for var_idx, var in enumerate(model.variables):
                var_minmax = model.ranges[var]
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

    for metric in res_dict:
        for var in res_dict[metric]:
            res_dict[metric][var] /= sample_total
            if type(res_dict[metric][var]) is not float:
                res_dict[metric][var] = res_dict[metric][var].item()

    return res_dict

def eval_model(model,test_dataloader, device='cuda'):
    res_dict = evaluate(model, test_dataloader)
    print('-' * 20)
    print(f'metric:')
    for metric in res_dict:
        for var in res_dict[metric]:
            print(f'{metric}\t\t{var}:\t\t{res_dict[metric][var]}')

if __name__ == '__main__':
    dataset_path = '../../bench_data/'
    ckpt_path = './ckpt'
    test_data = torch.load(os.path.join(dataset_path, 'VA_test_128.pt'))
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
    model = torch.load(os.path.join(ckpt_path,'model_195.pth')).to(device)  # 或 'model.pth'
    eval_model(model,test_dataloader,device)