import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

class BranchNet(nn.Module):
    """处理输入函数（初始温度场T0）的Branch Net，使用CNN结构"""
    def __init__(self, p=256):
        super().__init__()
        # 原始通道数: 1->32->64->128
        # 扩大4倍后: 4->128->256->512
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [128,128] -> [128,128]
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

class E_Flow_DeepONet(nn.Module):
    """完整的DeepONet模型"""
    def __init__(self, p=256, device='cuda'):
        super().__init__()
        self.branch = BranchNet(p)
        self.trunk = TrunkNet(p)
        self.output_net = nn.Linear(p, 3)

        range_allkappa_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/kappa/range_allkappa.mat"
        range_allkappa = sio.loadmat(range_allkappa_paths)['range_allkappa']

        self.max_kappa = range_allkappa[0, 1]
        self.min_kappa = range_allkappa[0, 0]

        # output ec_V, u_flow, v_flow

        # load max_ec_V min_ec_V
        range_allec_V_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/ec_V/range_allec_V.mat"
        range_allec_V = sio.loadmat(range_allec_V_paths)['range_allec_V']

        self.max_ec_V = range_allec_V[0, 1]
        self.min_ec_V = range_allec_V[0, 0]

        # load max_u_flow min_u_flow
        range_allu_flow_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/u_flow/range_allu_flow.mat"
        range_allu_flow = sio.loadmat(range_allu_flow_paths)['range_allu_flow']

        self.max_u_flow = range_allu_flow[0, 1]
        self.min_u_flow = range_allu_flow[0, 0]

        # load max_v_flow min_v_flow
        range_allv_flow_paths = "/data/yangchangfan/DiffusionPDE/data/training/E_flow/v_flow/range_allv_flow.mat"
        range_allv_flow = sio.loadmat(range_allv_flow_paths)['range_allv_flow']

        self.max_v_flow = range_allv_flow[0, 1]
        self.min_v_flow = range_allv_flow[0, 0]

    def forward(self, T0, coords):
        # T0: [batch, 1, 128, 128]
        # coords: [batch, num_points, 2]
        batch_size, grid_size = T0.shape[0], T0.shape[1]
        T0 = T0.unsqueeze(1)
        coords = coords.permute(0,2,3,1)
        coords = coords.reshape(coords.shape[0], -1, coords.shape[-1])
        b = self.branch(T0)  # [batch, p]
        t = self.trunk(coords)  # [batch, num_points, p]
        output = self.output_net(b.unsqueeze(1) * t).reshape(batch_size, grid_size, grid_size, -1)
        return output  # [batch, 3]

    def compute_loss(self, inputs, coords, E_U_true,U_flow_true,V_flow_true):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        output = self.forward(inputs, coords)
        E_U_pred, U_flow_pred, V_flow_pred = output[:,:,:,0],output[:,:,:,1],output[:,:,:,2]
        data_loss = torch.nn.MSELoss()(E_U_true, E_U_pred) + torch.nn.MSELoss()(U_flow_true, U_flow_pred) + torch.nn.MSELoss()(V_flow_true,
                                                                                                                 V_flow_pred)
        return data_loss