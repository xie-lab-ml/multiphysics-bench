import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
class BranchNet(nn.Module):
    """处理输入函数（初始温度场T0）的Branch Net，使用CNN结构"""
    def __init__(self, p=256):
        super().__init__()
        # 原始通道数: 1->32->64->128
        # 扩大4倍后: 4->128->256->512
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),  # [128,128] -> [128,128]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [64,64]
            nn.Conv2d(32, 128, kernel_size=3, padding=1),  # [128,128] -> [128,128]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [64,64]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # -> [64,64]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [32,32]
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # -> [32,32]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # -> [4,4]
        )
        # 全连接层也相应扩大
        # 原始: 128*4*4->512->p
        # 扩大4倍: 512*4*4->2048->p
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, p)  # 输出p维系数
        )

    def forward(self, T0):
        # T0: [batch, 1, 128, 128]
        x = self.conv_layers(T0)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # [batch, p]

class TrunkNet(nn.Module):
    """处理坐标的Trunk Net，输出p维基函数"""
    def __init__(self, p=256):
        super().__init__()
        # 原始结构: 2->64->128->p
        # 扩大4倍: 2->256->512->p (保持输出维度p不变)
        self.fc = nn.Sequential(
            nn.Linear(2, 256),  # 第一层扩大4倍 (64×4=256)
            nn.ReLU(),
            nn.Linear(256, 512),  # 第二层扩大4倍 (128×4=512)
            nn.ReLU(),
            nn.Linear(512, p)  # 输出层保持p维不变
        )

    def forward(self, coords):
        # coords: [batch, num_points, 2]
        return self.fc(coords)  # [batch, num_points, p]

class ElderDeepONet(nn.Module):
    """完整的DeepONet模型"""
    def __init__(self, p=256, device='cuda'):
        super().__init__()
        self.branch = BranchNet(p)
        self.trunk = TrunkNet(p)
        self.output_net = nn.Linear(p, 30)

        train_data_base_path = "/data/yangchangfan/DiffusionPDE/data/training/Elder/"
        # -------------------- 加载归一化范围 --------------------
        range_allS_c = sio.loadmat(os.path.join(train_data_base_path, "S_c/range_S_c_t.mat"))['range_S_c_t'][0]
        range_allu_u = sio.loadmat(os.path.join(train_data_base_path, "u_u/range_u_u_t_999.mat"))['range_u_u_t_999'][1:]
        range_allu_v = sio.loadmat(os.path.join(train_data_base_path, "u_v/range_u_v_t_99.mat"))['range_u_v_t_99'][1:]
        range_allc_flow = sio.loadmat(os.path.join(train_data_base_path, "c_flow/range_c_flow_t_99.mat"))['range_c_flow_t_99'][1:]

        # range_allu_u = sio.loadmat(os.path.join(train_data_base_path, "u_u/range_u_u_t_999.mat"))['range_u_u_t_999']
        # range_allu_v = sio.loadmat(os.path.join(train_data_base_path, "u_v/range_u_v_t_99.mat"))['range_u_v_t_99']
        # range_allc_flow = sio.loadmat(os.path.join(train_data_base_path, "c_flow/range_c_flow_t_99.mat"))[
        #                       'range_c_flow_t_99']
        self.S_c_ranges = range_allS_c
        self.ranges = {
            'u_u': range_allu_u,
            'u_v': range_allu_v,
            'c_flow': range_allc_flow,
        }

    def forward(self, T0, coords):
        # T0: [batch, 1, 128, 128]
        # coords: [batch, num_points, 2]
        batch_size, grid_size = T0.shape[0], T0.shape[2]
        coords = coords.permute(0,2,3,1)
        coords = coords.reshape(coords.shape[0], -1, coords.shape[-1])
        b = self.branch(T0)  # [batch, p]
        t = self.trunk(coords)  # [batch, num_points, p]
        output = self.output_net(b.unsqueeze(1) * t).reshape(batch_size, grid_size, grid_size, -1)
        return output  # [batch, 3]

    def compute_loss(self, inputs, coords, gt):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        output = self.forward(inputs, coords)
        data_loss = 0
        for idx in range(gt.shape[1]):
            data_loss += torch.nn.MSELoss()(output[:,:,:,idx], gt[:,idx,:,:])
        return data_loss