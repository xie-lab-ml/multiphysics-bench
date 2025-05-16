# Multiphysics Bench

We propose the first general multiphysics benchmark dataset that encompasses six canonical coupled scenarios across domains such as electromagnetics, heat transfer, fluid flow, solid mechanics, pressure acoustics, and mass transport. This benchmark features the most comprehensive coupling types, the most diverse PDEs, and the largest data scale.

---

## How to Use

### 1. Dataset Download

Please visit the link [https://huggingface.co/datasets/Indulge-Bai/Multiphysics_Bench ](https://huggingface.co/datasets/Indulge-Bai/Multiphysics_Bench ) to download the dataset (you will obtain `training.tar.gz` and `testing.tar.gz` separately). Extract the files to get the following directory structure:

Run the preprocessing code to convert the training and testing datasets into normalized tensor format. The final directory structure will look like this:

Multiphysics_Bench/
├── training/
│ ├── problem_1/
│ ├── problem_2/
│ └── ...
├── testing/
│ ├── problem_1/
│ ├── problem_2/
│ └── ...
├── preprocess.py
└── README.md


### 2. Dataset Overview


   ![image](https://github.com/user-attachments/assets/039d2048-95fc-4784-8cfd-90914f97477b)


Each problem corresponds to a specific multiphysics coupling scenario, with input-output pairs, physical parameters, and boundary conditions provided.

### 3. Training and Evaluation

For **DeepONet** and **PINNs**, navigate to the corresponding problem directory and run `train.py` to start training (Note: you need to set the dataset path in advance). To evaluate the model, run `evaluate.py`.

For **FNO**, please refer to the official documentation at [FNO]([http://www.xxx.com](https://github.com/NeuralOperator/neuraloperator)).

For **DiffusionPDE**, please refer to the official documentation at [DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE).

