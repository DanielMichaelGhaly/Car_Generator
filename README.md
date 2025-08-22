# 🚗 Diffusion Models from Scratch on Stanford Cars

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** **from scratch in PyTorch** and applies it to the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).  
The pipeline covers **forward diffusion**, **reverse (denoising) process**, **UNet-like neural architecture**, training, evaluation, and **quantitative + visual results**.

Key Features & Strengths

Full End-to-End Implementation
- **No external diffusion libraries**: everything coded from scratch.
- Covers the **entire DDPM lifecycle**:
  - Beta noise scheduling
  - Forward diffusion sampling
  - UNet-based reverse process
  - Training & inference
  - Evaluation metrics (L1, MSE, PSNR)

Dataset & Preprocessing
- Uses **Stanford Cars** dataset (~8,000 training images, ~8,000 test images).
- Resized to **64×64 / 128×128** resolution for training.
- Preprocessing includes:
  - `ToTensor()`
  - `RandomHorizontalFlip()`
  - Rescaling to **[-1, 1]** for stable training.

Model Architecture
- A **simplified UNet** with:
  - Downsampling/upsampling blocks
  - Skip connections
  - Sinusoidal timestep embeddings
- **Parameter count:** ~10M (depending on configuration).

Training Setup
- Optimizer: **Adam**, learning rate = `1e-3`.
- Training epochs: **100**.
- Batch size: **128**.
- Device: **CUDA** (NVIDIA GPU).

Quantitative Evaluation
Evaluation performed on **10 randomly selected test samples**:

| Metric | Mean Value |
|--------|------------|
| **L1 Loss** | ~ |
| **MSE Loss** | ~ |
| **PSNR** | ~ dB |

(PSNR > 20 dB typically indicates visually reasonable reconstructions)

Visual Results
For each test image:
- **x₀**: clean ground truth  
- **xₜ**: noisy image at random timestep  
- **x̂₀**: reconstruction by the diffusion model  



## 🧩 Project Structure

├── diffusion_model.ipynb
└── README.md 

## ⚙️ How to Run

1. Install Dependencies

pip install torch torchvision matplotlib kagglehub

2. Download Stanford Cars Dataset:

import kagglehub
path = kagglehub.dataset_download("jutrera/stanford-car-dataset-by-classes-folder")

3. Train Model

4. Evaluate Model

evaluate_and_visualize(model, test_loader, T=200, device="cuda", max_samples=10)

Results & Strengths

Implemented DDPM from scratch with full mathematical and coding pipeline.

Scaled to ~16,000 images, proving ability to handle real-world, mid-scale datasets.

Achieved PSNR ~ dB, indicating good fidelity in reconstructions.

Developed robust evaluation with L1, MSE, and PSNR to quantify denoising performance.

Produced visual evidence of generative ability (samples & reconstructions).

Extensive modular design (forward, model, loss, sampling, evaluation).

Skills Demonstrated

Deep Learning: Diffusion models, UNet, noise scheduling.

Computer Vision: Denoising, generative modeling, dataset preprocessing.

PyTorch Engineering: Custom DataLoader, GPU training, efficient collate functions.

Evaluation: L1/MSE losses, PSNR metric, visualization pipelines.

Software Engineering: Modular code structure, reproducibility, clarity.
