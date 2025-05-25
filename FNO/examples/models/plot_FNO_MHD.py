"""
Training an FNO on Darcy-Flow
=============================

We train a Fourier Neural Operator on our small `Darcy-Flow example <../auto_examples/plot_darcy_flow.html>`_ .

Note that this dataset is much smaller than one we would use in practice. The small Darcy-flow is an example built to
be trained on a CPU in a few seconds, whereas normally we would train on one or multiple GPUs.

"""

# %%
#

import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_MHD
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss,S3IMLoss
import os

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_loader, test_loaders, data_processor = load_MHD(
        n_train=10000, batch_size=48,
        test_resolutions=[128], n_tests=[1000],
        test_batch_sizes=[128],
)
data_processor = data_processor.to(device)

# %%
# We create a simple FNO model
model = FNO(n_modes=(12, 12),
             in_channels=1,
             out_channels=5,
             hidden_channels=128,
             projection_channel_ratio=2)
model = model.to(device)
n_params = count_model_params(model)
print(f'\nOur model has {n_params/(1024*1024)} M.')
sys.stdout.flush()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
s3imLoss = S3IMLoss()

train_loss = h1loss
# train_loss = l2loss
eval_losses={'h1': h1loss, 'l2': l2loss}


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# %%
# Create the trainer:
trainer = Trainer(model=model, n_epochs=50,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  eval_interval=1,
                  use_distributed=False,
                  verbose=True,
                  save_dir="./checkpoints/MHD")

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses,
              save_every=1)