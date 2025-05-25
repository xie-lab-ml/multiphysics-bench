import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import tqdm
import scipy.io as sio
import json
import pandas as pd
import sys

sys.path.append('../src/')
from TE_model_Large import TEHeatDeepONet

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

def train_model(model, dataloader, train_dataset, optimizer, scheduler, Epoch, clip_value, save_dir, device='cuda'):
    for epoch in range(Epoch):
        total_loss = 0.0
        for inputs, coords, outputs, polygt_idx in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            coords = coords.to(device)

            E_real_true = outputs[:, 0].to(device)
            E_imag_true = outputs[:, 1].to(device)
            T_true = outputs[:, 2].to(device)

            optimizer.zero_grad()
            data_loss = model.compute_loss(inputs, coords, E_real_true,E_imag_true,T_true)
            loss = data_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch%5==0:
            torch.save(model, os.path.join(save_dir, 'ckpt', f'model_{epoch}.pth'))
        scheduler.step()
    return model

if __name__ == '__main__':
    dataset_path = '../../bench_data/'
    base_dir = './'
    if not os.path.exists(os.path.join(base_dir, 'fig')):
        os.mkdir(os.path.join(base_dir, 'fig'))
    if not os.path.exists(os.path.join(base_dir, 'ckpt')):
        os.mkdir(os.path.join(base_dir, 'ckpt'))
    if not os.path.exists(os.path.join(base_dir, 'log')):
        os.mkdir(os.path.join(base_dir, 'log'))

    train_data = torch.load(os.path.join(dataset_path, 'TE_heat_train_128.pt'))
    train_data['x'] = train_data['x']
    train_data['y'] = train_data['y']
    train_dataset = FieldDataset(train_data['x'], train_data['y'])

    batch_size = 48
    num_workers = 8
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    device = 'cuda'
    Epoch = 200
    model = TEHeatDeepONet().type(torch.float64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max = Epoch)
    train_model(model,train_dataloader,train_dataset, optimizer,scheduler,Epoch, 1.0 ,base_dir,device)