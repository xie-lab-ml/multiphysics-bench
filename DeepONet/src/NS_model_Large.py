import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio


def generate_separa_mater(mater_iden):
    rho_air = 1.24246
    rho_copper = 8960
    Crho_air = 1005.10779
    Crho_copper = 385
    kappa_air = 0.02505
    kappa_copper = 400

    rho = torch.where(mater_iden > 1e-5, rho_copper, rho_air)
    Crho = torch.where(mater_iden > 1e-5, Crho_copper, Crho_air)
    kappa = torch.where(mater_iden > 1e-5, kappa_copper, kappa_air)

    # rho = rho.t()
    rho = rho.permute(0, 2, 1)
    # Crho = Crho.t()
    Crho = Crho.permute(0, 2, 1)
    # kappa = kappa.t()
    kappa = kappa.permute(0, 2, 1)

    return rho, Crho, kappa


def identify_mater(circle_params, device=torch.device('cuda')):
    # 输入形状: [64, 1, 3] -> squeeze后 [64, 3]
    circle_params = circle_params.squeeze()  # [64, 3]

    # 生成坐标网格 [128, 128]
    coords = (torch.arange(128, device=device) - 63.5) * 0.001
    xx, yy = torch.meshgrid(coords, coords, indexing='ij')  # [128, 128]
    xx = xx.to(device)
    yy = yy.to(device)

    # 提取 cx, cy, r 并扩展维度以支持广播 [64, 1, 1]
    cx = circle_params[:, 0].view(-1, 1, 1)  # [64, 1, 1]
    cy = circle_params[:, 1].view(-1, 1, 1)  # [64, 1, 1]
    r = circle_params[:, 2].view(-1, 1, 1)  # [64, 1, 1]

    distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2  # [64, 128, 128]
    mater_iden = torch.where(distance_sq <= r ** 2, 1, -1)  # [64, 128, 128]
    return mater_iden


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

class NSHeatDeepONet(nn.Module):
    """完整的DeepONet模型"""
    def __init__(self, p=256, device='cuda'):
        super().__init__()
        self.branch = BranchNet(p)
        self.trunk = TrunkNet(p)
        self.output_net = nn.Linear(p, 3)

        range_allQ_heat_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/Q_heat/range_allQ_heat.mat"
        range_allQ_heat = sio.loadmat(range_allQ_heat_paths)['range_allQ_heat']
        range_allQ_heat = torch.tensor(range_allQ_heat, device=device)
        self.max_Q_heat = range_allQ_heat[0, 1]
        self.min_Q_heat = range_allQ_heat[0, 0]

        range_allu_u_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_u/range_allu_u.mat"
        range_allu_u = sio.loadmat(range_allu_u_paths)['range_allu_u']
        range_allu_u = torch.tensor(range_allu_u, device=device)
        self.max_u_u = range_allu_u[0, 1]
        self.min_u_u = range_allu_u[0, 0]

        range_allu_v_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/u_v/range_allu_v.mat"
        range_allu_v = sio.loadmat(range_allu_v_paths)['range_allu_v']
        range_allu_v = torch.tensor(range_allu_v, device=device)
        self.max_u_v = range_allu_v[0, 1]
        self.min_u_v = range_allu_v[0, 0]

        range_allT_paths = "/data/yangchangfan/DiffusionPDE/data/training/NS_heat/T/range_allT.mat"
        range_allT = sio.loadmat(range_allT_paths)['range_allT']
        range_allT = torch.tensor(range_allT, device=device)
        self.max_T = range_allT[0, 1]
        self.min_T = range_allT[0, 0]

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

    def get_NS_heat_loss_grad(self, Q_heat, u_u, u_v, T, mater_iden, coords):
        rho, Crho, kappa = generate_separa_mater(mater_iden)
        du_dX = torch.autograd.grad(
            inputs=coords,
            outputs=u_u,
            grad_outputs=torch.ones_like(u_u),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )

        dv_dY = torch.autograd.grad(
            inputs=coords,
            outputs=u_v,
            grad_outputs=torch.ones_like(u_v),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )

        result_NS = du_dX[0][:,0] + dv_dY[0][:,1]
        result_NS = result_NS.squeeze()

        dT_dX = torch.autograd.grad(
            outputs=T,
            inputs=coords,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        # 温度二阶导数 (Laplacian)
        dT_dx = dT_dX[:, 0]  # ∂T/∂x
        dT_dy = dT_dX[:, 1]  # ∂T/∂y

        # 计算 ∂²T/∂x²
        d2T_dx2 = torch.autograd.grad(
            outputs=dT_dx,
            inputs=coords,
            grad_outputs=torch.ones_like(dT_dx),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0][:, 0]

        d2T_dy2 = torch.autograd.grad(
            outputs=dT_dy,
            inputs=coords,
            grad_outputs=torch.ones_like(dT_dy),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0][:, 1]

        Laplac_T = d2T_dx2 + d2T_dy2  # ∇²T = ∂²T/∂x² + ∂²T/∂y²
        u_dot_gradT = u_u * dT_dx + u_v * dT_dy  # 对流项 (u·∇T)
        result_heat = rho * Crho * u_dot_gradT - kappa * Laplac_T - Q_heat
        return result_NS , result_heat

    def compute_loss(self, inputs, coords, poly_GT, u_true, v_true, T_true):
        inputs = inputs.to(torch.float64).clone().detach().requires_grad_(True)
        coords = coords.to(torch.float64).clone().detach().requires_grad_(True)
        output = self.forward(inputs, coords)
        u_pred, v_pred, T_pred = output[:,:,:,0],output[:,:,:,1],output[:,:,:,2]
        data_loss = torch.nn.MSELoss()(u_pred, u_true) + torch.nn.MSELoss()(v_pred, v_true) + torch.nn.MSELoss()(T_pred,
                                                                                                                 T_true)
        return data_loss
