# Multiphysics Bench

Dataset: [huggingface.co/datasets/Indulge-Bai/Multiphysics_Bench ](https://huggingface.co/datasets/Indulge-Bai/Multiphysics_Bench)

Paper: [Multiphysics Bench: Benchmarking and Investigating Scientific Machine Learning for Multiphysics PDEs](https://arxiv.org/abs/2505.17575)

We propose the first general multiphysics benchmark dataset that encompasses six canonical coupled scenarios across domains such as electromagnetics, heat transfer, fluid flow, solid mechanics, pressure acoustics, and mass transport. This benchmark features the most comprehensive coupling types, the most diverse PDEs, and the largest data scale.

![image](https://anonymous.4open.science/r/MultiphysicsBench/assets/intro.jpg)

---

# How to Use

## 1. Dataset Download

Please visit the link [huggingface.co/datasets/Indulge-Bai/Multiphysics_Bench ](https://huggingface.co/datasets/Indulge-Bai/Multiphysics_Bench) to download the dataset (you will obtain `training.tar.gz` and `testing.tar.gz` separately). Extract the files to get the following directory structure:

Run the preprocessing code to convert the training and testing datasets into normalized tensor format. The final directory structure will look like this:

```
Multiphysics_Bench/
├── training/
│ ├── problem_1/
│ ├── problem_2/
│ └── ...
├── testing/
│ ├── problem_1/
│ ├── problem_2/
│ └── ...
└── README.md
```

or you can run the command below to download our datasets:

```
from huggingface_hub import snapshot_download

folder_path = snapshot_download(
    repo_id="Indulge-Bai/Multiphysics_Bench",
    repo_type="dataset"
)
```


## 2. Multiphysics Problems

### (1). Electro-Thermal Coupling

This dataset contains simulations of the Wave Equation and Heat Conduction Equation. This coupling mechanism underpins various applications, including thermal management in electronic components (e.g., microprocessors), inductive heating (e.g., welding and metal processing), biomedical technologies (e.g., photothermal therapy), and aerospace engineering (e.g., radiative heating of satellite surfaces).

<img src="https://anonymous.4open.science/r/MultiphysicsBench/assets/problem_pic/TEheat_Data.png" width="50%" />


**How to use**

- **Input (1 channel):** `mater`
- **Output (3 channels):** `Re{Ez}`, `Im{Ez}`, `T`

**Dataset Format**

- `TE_heat/mater/1.csv ... 10000.csv`
- `TE_heat/mater/1.mat ... 10000.mat`
- `TE_heat/Ez/1.mat ... 10000.mat` *(a complex data file containing real and imaginary parts)*

**Dataset Generation**

The dataset generation code is located at:  

```
DataProcessing/data_generate/TE_heat/main_TE_heat.m
DataProcessing/data_generate/TE_heat/parm2matrix_TE_heat.m
```
---

### (2). Thermo-Fluid Coupling

This dataset contains simulations of the Navier–Stokes Equations and Heat Balance Equation. Thermo-Fluid coupling is essential in the design and optimization of systems such as electronic cooling (e.g., chip heat dissipation), energy systems (e.g., nuclear reactor cooling), and precision thermal control in manufacturing.

<img src="https://anonymous.4open.science/r/MultiphysicsBench/assets/problem_pic/NSheat_Data.png" width="40%" />


**How to use**

- **Input (1 channel):** `Q_heat`
- **Output (3 channels):** `u`, `v`, `T`

**Dataset Format**

- `NS_heat/circlecsv/1.csv ... 10000.csv`
- `NS_heat/Q_heat/1.csv ... 10000.csv`
- `NS_heat/u_u/1.csv ... 10000.csv`
- `NS_heat/u_v/1.csv ... 10000.csv`

**Dataset Generation**

The dataset generation code is located at:  

```
DataProcessing/data_generate/NS_heat/NS_heat.m
DataProcessing/data_generate/NS_heat/parm2matrix_NS_heat.m
```

---

### (3). Electro-Fluid Coupling

This dataset simulates the Navier–Stokes Equations and Current Continuity Equation. This coupling is foundational to applications such as micropumps and micromixers in microfluidic systems.

<img src="https://anonymous.4open.science/r/MultiphysicsBench/assets/problem_pic/Eflow_Data.png" width="40%" />

**How to use**

- **Input (1 channel):** `kappa`
- **Output (3 channels):** `ec_V`, `u_flow`, `v_flow`

**Dataset Format**

- `E_flow/kappa/1.mat ... 10000.mat`
- `E_flow/ec_V/1.mat ... 10000.mat`
- `E_flow/u_flow/1.mat ... 10000.mat`
- `E_flow/v_flow/1.mat ... 10000.mat`

**Dataset Generation**

The dataset generation code is located at:  

```
DataProcessing/data_generate/E_flow/main_E_flow.m
DataProcessing/data_generate/E_flow/parm2matrix_E_flow.m
```

---

### (4). Magneto-Hydrodynamic (MHD) Coupling

This dataset simulates Ampère’s Law, Continuity Equation, Navier–Stokes Equations, and Lorentz Force. This model finds extensive application in electromagnetic pumps, plasma confinement devices (e.g., tokamaks), astrophysical phenomena, and pollutant transport modeling.

<img src="https://anonymous.4open.science/r/MultiphysicsBench/assets/problem_pic/MHD_Data.png" width="40%" />


**How to use**

- **Input (1 channel):** `Br`
- **Output (5 channels):** `Jx`, `Jy`, `Jz`, `u_u`, `u_v`

**Dataset Format**

- `MHD/Br/1.mat ... 10000.mat`
- `MHD/Jx/1.mat ... 10000.mat`
- `MHD/Jy/1.mat ... 10000.mat`
- `MHD/Jz/1.mat ... 10000.mat`
- `MHD/u_u/1.mat ... 10000.mat`
- `MHD/u_v/1.mat ... 10000.mat`

**Dataset Generation**

The dataset generation code is located at:  

```DataProcessing/data_generate/MHD/main_MHD.m
DataProcessing/data_generate/MHD/parm2matrix_MHD.m
```

---

### (5). Acoustic–Structure Coupling

This dataset simulates the Acoustic Wave Equation and Structural Vibration Equation. The input is the spatial material density (1 channel). The outputs comprise the acoustic pressure field (2 channels), stress components (6 channels), and structural displacements (4 channels), for a total of **12 output channels**.

<img src="https://anonymous.4open.science/r/MultiphysicsBench/assets/problem_pic/VA_Data.png" width="50%" />


**How to use**

- **Input (1 channel):** `rho_water`
- **Output (12 channels):**
  - `Re{p_t}`, `Im{p_t}`
  - `Re{Sxx}`, `Im{Sxx}`
  - `Re{Sxy}`, `Im{Sxy}`
  - `Re{Syy}`, `Im{Syy}`
  - `Re{x_u}`, `Im{x_u}`
  - `Re{x_v}`, `Im{x_v}`

**Dataset Format**

- `VA/p_t/1.mat ... 10000.mat`
- `VA/Sxx/1.mat ... 10000.mat`
- `VA/Sxy/1.mat ... 10000.mat`
- `VA/Syy/1.mat ... 10000.mat`
- `VA/x_u/1.mat ... 10000.mat`
- `VA/x_v/1.mat ... 10000.mat`

> Note: `p_t`, `Sxx`, `Sxy`, `Syy`, `x_u`, and `x_v` are complex data files containing real and imaginary parts.

**Dataset Generation**

The dataset generation code is located at:  

```
DataProcessing/data_generate/VA/main_VA.m
DataProcessing/data_generate/VA/parm2matrix_VA.m
```

---

### (6). Mass Transport–Fluid Coupling

This dataset contains simulations based on Darcy’s Law and the Convection–Diffusion Equation. The input includes the source term `Sc` and the initial state of the system at time `t0` (concentration and velocity), totaling **4 channels**. The output consists of the predicted concentration and velocity fields across **10 future time steps**, resulting in **30 channels in total**.

<img src="https://anonymous.4open.science/r/MultiphysicsBench/assets/problem_pic/Elder_Data.gif" width="40%" />


**How to use**

- **Input (4 channels):** `S_c`, `u_u(t0)`, `u_v(t0)`, `c_flow(t0)`
- **Output (30 channels):** `u_u(t1-t10)`, `u_v(t1-t10)`, `c_flow(t1-t10)`

**Dataset Format**

- `Elder/S_c/1...1000/0.mat ... 10.mat`
- `Elder/u_u/1...1000/0.mat ... 10.mat`
- `Elder/u_v/1...1000/0.mat ... 10.mat`
- `Elder/c_flow/1...1000/0.mat ... 10.mat`

**Dataset Generation**

The dataset generation code is located at:  

```
DataProcessing/data_generate/Elder/main_Elder.m
DataProcessing/data_generate/Elder/parm2matrix_Elder.m
```

## 3. Training and Evaluation

### (1) Merging Dataset

In the 'DataProcessing' folder, we provide the data generation and normalization processing code for each type of problem. Specifically:  
- **xxx.m**: MATLAB code for generating dataset
- **xxx.py**: Python code for post-processing (facilitating execution across different baselines)

### Key Notes:
1. **Data Generation** → Use `.m` files (requires COMSOL Multiphysics 6.2 with MATLAB environment)  
2. **Data Processing** → Use `.py` files (Python ensures better compatibility with various baseline models)  

This design ensures reproducibility of data while simplifying adaptation for different frameworks.

### (2) Training & Evaluation

For **DeepONet** and **PINNs**, navigate to the corresponding problem directory and run `train.py` to start training (Note: you need to set the dataset path in advance, e.g., `Elder_train_128.pt`). To evaluate the model, run `evaluate.py`.

For **FNO**, please refer to the official documentation at [FNO]([http://www.xxx.com](https://github.com/NeuralOperator/neuraloperator)).

For **DiffusionPDE**, please refer to the official documentation at [DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE).


### Citing
If you find our dataset or code useful for your research, please cite our paper.

```
@article{yang2025multiphysics,
      title={Multiphysics Bench: Benchmarking and Investigating Scientific Machine Learning for Multiphysics PDEs}, 
      author={Changfan Yang and Lichen Bai and Yinpeng Wang and Shufei Zhang and Zeke Xie},
      year={2025},
      journal={arXiv preprint arXiv:2505.17575},
}
```
