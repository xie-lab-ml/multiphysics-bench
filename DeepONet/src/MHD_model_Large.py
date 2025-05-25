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

class MHDDeepONet(nn.Module):
    """完整的DeepONet模型"""
    def __init__(self, p=256, device='cuda'):
        super().__init__()
        self.branch = BranchNet(p)
        self.trunk = TrunkNet(p)
        self.output_net = nn.Linear(p, 5)

        range_allBr_paths = "/data/yangchangfan/DiffusionPDE/data/training/MHD/Br/range_allBr.mat"
        range_allBr = sio.loadmat(range_allBr_paths)['range_allBr']

        self.max_Br = range_allBr[0, 1]
        self.min_Br = range_allBr[0, 0]

        # load max_Jx min_Jx
        range_allJx_paths = "/data/yangchangfan/DiffusionPDE/data/training/MHD/Jx/range_allJx.mat"
        range_allJx = sio.loadmat(range_allJx_paths)['range_allJx']

        self.max_Jx = range_allJx[0, 1]
        self.min_Jx = range_allJx[0, 0]

        # load max_Jy min_Jy
        range_allJy_paths = "/data/yangchangfan/DiffusionPDE/data/training/MHD/Jy/range_allJy.mat"
        range_allJy = sio.loadmat(range_allJy_paths)['range_allJy']

        self.max_Jy = range_allJy[0, 1]
        self.min_Jy = range_allJy[0, 0]

        # load max_Jz min_Jz
        range_allJz_paths = "/data/yangchangfan/DiffusionPDE/data/training/MHD/Jz/range_allJz.mat"
        range_allJz = sio.loadmat(range_allJz_paths)['range_allJz']

        self.max_Jz = range_allJz[0, 1]
        self.min_Jz = range_allJz[0, 0]

        # load max_u_u min_u_u
        range_allu_u_paths = "/data/yangchangfan/DiffusionPDE/data/training/MHD/u_u/range_allu_u.mat"
        range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']

        self.max_u_u = range_allu_u[0, 1]
        self.min_u_u = range_allu_u[0, 0]

        # load max_u_v min_u_v
        range_allu_v_paths = "/data/yangchangfan/DiffusionPDE/data/training/MHD/u_v/range_allu_v.mat"
        range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']

        self.max_u_v = range_allu_v[0, 1]
        self.min_u_v = range_allu_v[0, 0]

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

    def compute_loss(self, inputs, coords, Jx_true, Jy_true, Jz_true, u_u_true, u_v_true):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        output = self.forward(inputs, coords)
        Jx_pred, Jy_pred, Jz_pred, u_u_pred, u_v_pred = output[:, :, :, 0], output[:, :, :, 1], output[:, :, :,
                                                                                                2], output[:, :, :,
                                                                                                    3], output[:, :, :,
                                                                                                        4]
        data_loss = torch.nn.MSELoss()(Jx_pred, Jx_true) + torch.nn.MSELoss()(Jy_pred, Jy_true) + \
                    torch.nn.MSELoss()(Jz_pred, Jz_true) + torch.nn.MSELoss()(u_u_pred, u_u_true) + torch.nn.MSELoss()(
            u_v_pred, u_v_true)
        return data_loss
