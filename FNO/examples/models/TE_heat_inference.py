import torch
from neuralop.models import FNO
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_TE_heat
import os
from scipy import io
import numpy as np
from scipy.io import savemat

device = 'cuda'

os.environ["CUDA_VISIBLE_DEVICES"] = "7" 

# Let's load the TE_heat dataset. 
train_loader, test_loaders, data_processor = load_TE_heat(
        n_train=10000, batch_size=16, 
        test_resolutions=[128], n_tests=[1000],
        test_batch_sizes=[128],
)
data_processor = data_processor.to(device)
data_processor.eval()


# 创建相同架构的模型实例
model = FNO(n_modes=(12, 12),
             in_channels=1,
             out_channels=3,
             hidden_channels=128,
             projection_channel_ratio=2)
model = model.to(device)

model.load_state_dict(torch.load("./checkpoints/TE_heat_10000/1/model_epoch_49_state_dict.pt", weights_only=False))
print("Model weights loaded from model_weights.pt")

# 将模型设置为评估模式
model.eval()
offset=10903

if not os.path.exists(os.path.join("/data/bailichen/PDE/PDE/paint/data/TE_heat/FNO",str(offset))):
    os.mkdir(os.path.join("/data/bailichen/PDE/PDE/paint/data/TE_heat/FNO",str(offset)))

test_samples = test_loaders[128].dataset

fig = plt.figure(figsize=(14, 7))
for index in range(1):
    # data = test_samples[index+offset]
    data = test_samples[offset-10001]
    data = data_processor.preprocess(data)

    # Input x
    x = data['x'].to(device)
    # Ground-truth
    y = data['y'].to(device)
    # Model prediction
    out = model(x.unsqueeze(0).to(device))
    out, _ = data_processor.postprocess(out)

    # 将输入和标签数据移至 CPU，以便用于绘图（matplotlib 不支持 GPU 张量）
    x_cpu = x.squeeze().cpu().numpy()
    y_cpu = y.squeeze().cpu().numpy()
    out_cpu = out.squeeze().cpu().detach().numpy()

    out_cpu_final = np.zeros((3, 128, 128))
    y_cpu_final = np.zeros((3, 128, 128))

    max_abs_Ez_path = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/Ez/max_abs_Ez.mat"
    max_abs_Ez = io.loadmat(max_abs_Ez_path)['max_abs_Ez']

    # load T
    range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/TE_heat/T/range_allT.mat"
    range_allT = io.loadmat(range_allT_paths)['range_allT']

    max_T = range_allT[0,1]
    min_T = range_allT[0,0]

    out_cpu_final[0] = (out_cpu[0] * max_abs_Ez / 0.9)
    out_cpu_final[1] = (out_cpu[1] * max_abs_Ez / 0.9)
    out_cpu_final[2] =  ((out_cpu[2]+0.9)/1.8 *(max_T - min_T) + min_T)

    y_cpu_final[0] = (y_cpu[0] * max_abs_Ez / 0.9)
    y_cpu_final[1] = (y_cpu[1] * max_abs_Ez / 0.9)
    y_cpu_final[2] =  ((y_cpu[2]+0.9)/1.8 *(max_T - min_T) + min_T)

    vmin_real_Ez = y_cpu_final[0].min()
    vmax_real_Ez = y_cpu_final[0].max()
    vmin_imag_Ez = y_cpu_final[1].min()
    vmax_imag_Ez = y_cpu_final[1].max()
    vmin_T = y_cpu_final[2].min()
    vmax_T = y_cpu_final[2].max()

    # 绘制输入
    ax = fig.add_subplot(2, 4, index*4 + 1)
    im = ax.imshow(x_cpu)
    if index == 0: 
        ax.set_title('Input mater GT')
    plt.xticks([], [])
    plt.yticks([], [])
    fig.colorbar(im, ax=ax)  # 添加 colorbar

    # 绘制真实值 (real Ez)
    ax = fig.add_subplot(2, 4, index*4 + 2)
    im = ax.imshow(y_cpu_final[0], vmin=vmin_real_Ez, vmax=vmax_real_Ez)
    if index == 0: 
        ax.set_title('real Ez GT')
    plt.xticks([], [])
    plt.yticks([], [])
    fig.colorbar(im, ax=ax)  # 添加 colorbar

    # 绘制真实值 (imag Ez)
    ax = fig.add_subplot(2, 4, index*4 + 3)
    im = ax.imshow(y_cpu_final[1], vmin=vmin_imag_Ez, vmax=vmax_imag_Ez)
    if index == 0: 
        ax.set_title('imag Ez GT')
    plt.xticks([], [])
    plt.yticks([], [])
    fig.colorbar(im, ax=ax)  # 添加 colorbar

    # 绘制真实值 (T)
    ax = fig.add_subplot(2, 4, index*4 + 4)
    im = ax.imshow(y_cpu_final[2], vmin=vmin_T, vmax=vmax_T)
    if index == 0: 
        ax.set_title('T GT')
    plt.xticks([], [])
    plt.yticks([], [])
    fig.colorbar(im, ax=ax)  # 添加 colorbar

    # 空白子图
    ax = fig.add_subplot(2, 4, index*4 + 5)
    ax.axis('off')  # 关闭轴

    # 绘制预测值 (Predicted real Ez)
    ax = fig.add_subplot(2, 4, index*4 + 6)
    im = ax.imshow(out_cpu_final[0], vmin=vmin_real_Ez, vmax=vmax_real_Ez)
    if index == 0:
        ax.set_title('Predicted real Ez')
    plt.xticks([], [])
    plt.yticks([], [])
    fig.colorbar(im, ax=ax)  # 添加 colorbar

    # 绘制预测值 (Predicted imag Ez)
    ax = fig.add_subplot(2, 4, index*4 + 7)
    im = ax.imshow(out_cpu_final[1], vmin=vmin_imag_Ez, vmax=vmax_imag_Ez)
    if index == 0:
        ax.set_title('Predicted imag Ez')
    plt.xticks([], [])
    plt.yticks([], [])
    fig.colorbar(im, ax=ax)  # 添加 colorbar

    # 绘制预测值 (Predicted T)
    ax = fig.add_subplot(2, 4, index*4 + 8)
    # im = ax.imshow(out_cpu_final[2], vmin=vmin_T, vmax=vmax_T)
    im = ax.imshow(out_cpu_final[2])
    if index == 0:
        ax.set_title('Predicted T')
    plt.xticks([], [])
    plt.yticks([], [])
    fig.colorbar(im, ax=ax)  # 添加 colorbar

fig.suptitle('Inputs, ground-truth output and prediction (32x32).', y=0.98)
plt.savefig(os.path.join("/data/bailichen/PDE/PDE/paint/data/TE_heat/FNO",str(offset), "sample.png"))  # 保存为 PNG 文件
savemat(os.path.join("/data/bailichen/PDE/PDE/paint/data/TE_heat/FNO", str(offset), 'pred.mat'), {
        'E_real': out_cpu_final[0],
        'E_imag': out_cpu_final[1],
        'T': out_cpu_final[2]
    })

savemat(os.path.join("/data/bailichen/PDE/PDE/paint/data/TE_heat/FNO", str(offset), 'gt.mat'), {
    'E_real': y_cpu_final[0],
    'E_imag': y_cpu_final[1],
    'T': y_cpu_final[2]
})