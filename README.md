# Implicit Neural Representations for 3D Medical Point Clouds

This repository contains the code for the master thesis *"Implicit Neural Representations of 3D Medical Point Clouds"* (Technical University of Munich, 2024). It investigates how Implicit Neural Representations (INRs) can encode, reconstruct, and register 3D medical point clouds using signed distance functions learned by multi-layer perceptrons.

The work is structured into three experiments:

1. **SIREN-based INR** — Trains a SIREN network (sine-activated MLP) to learn the signed distance function of a single medical 3D shape, with analysis of hyperparameter sensitivity (activation function, network depth, frequency scaling).
2. **Cohort-based Sparse Shape Reconstruction** — Extends INRs with per-shape latent codes (auto-decoder) to reconstruct complete meshes from sparse point clouds across a cohort of medical shapes.
3. **INR-based 3D Registration** — Predicts displacement fields to align moving point clouds to fixed targets using a self-supervised loss (Chamfer + smoothness + Laplacian regularization), with both pairwise and cohort-based (latent space) variants.

## Repository Structure

```
├── siren/                                  # Experiment 1: SIREN-INR for SDF encoding
│   ├── experiment_scripts/
│   │   └── train_sdf.py                    # Train SIREN on mesh data
│   ├── modules.py                          # SIREN network architecture
│   ├── loss_functions.py                   # Eikonal, surface, and off-surface losses
│   ├── dataio.py                           # Data loading and point sampling
│   └── sdf_meshing.py                      # Mesh extraction from learned SDF
│
├── inr-implicit-shape-reconstruction-mesh/ # Experiment 2: Sparse shape reconstruction
│   ├── src/
│   │   └── impl_recon/train.py             # Training script
│   └── src/*.yml                           # Training, evaluation, and path configs
│
├── registration-pipeline/                  # Experiment 3a: Pairwise registration
│   ├── src/train.py                        # Training script
│   ├── src/config.yml                      # Config for DeformingThings4D
│   └── src/config_lung.yml                 # Config for Lung250M-4B
│
├── registration-pipeline-latent/           # Experiment 3b: Cohort-based registration
│   ├── src/train.py                        # Training script with latent vectors
│   ├── src/config.yml                      # Config for DeformingThings4D
│   └── src/config_lung.yml                 # Config for Lung250M-4B
│
├── data/                                   # Data preparation utilities
│   ├── download_txt.py                     # Download MedShapeNet samples
│   ├── data_transformation.py              # Mesh-to-point-cloud conversion
│   ├── make_pkl.py                         # Create pickle datasets
│   └── make_casenamefiles.py               # Generate case name lists
│
├── siren/environment.yml                   # Conda env for SIREN experiments
├── shape-reconstruction.yml                # Conda env for shape reconstruction
└── registration-pipeline.yml               # Conda env for registration experiments
```

## Datasets

| Dataset | Used for | Link |
|---------|----------|------|
| **MedShapeNet** | SIREN INR and sparse shape reconstruction | [medshapenet-ikim.streamlit.app](https://medshapenet-ikim.streamlit.app) |
| **DeformingThings4D** | Registration experiments (non-rigid animal sequences) | [github.com/rabbityl/DeformingThings4D](https://github.com/rabbityl/DeformingThings4D) |
| **Lung250M-4B** | Registration experiments (inspiratory/expiratory lung pairs) | [cloud.imi.uni-luebeck.de](https://cloud.imi.uni-luebeck.de/s/s64fqbPpXNexBPP) |

## Setup

Each experiment uses a separate Conda environment. Create and activate the appropriate one:

```bash
# SIREN experiments
conda env create -f siren/environment.yml
conda activate siren

# Shape reconstruction
conda env create -f shape-reconstruction.yml
conda activate shape-reconstruction
pip install -e inr-implicit-shape-reconstruction-mesh/

# Registration (pairwise and cohort-based)
conda env create -f registration-pipeline.yml
conda activate registration-pipeline-2
pip install -e registration-pipeline/        # pairwise
pip install -e registration-pipeline-latent/ # cohort-based
```

**Requirements:** Python >= 3.8, CUDA-compatible GPU, PyTorch. Key dependencies include `open3d`, `wandb`, `scipy`, `torch_geometric`, and `random-fourier-features-pytorch`.

## Usage

### 1. SIREN-INR (SDF encoding of a single shape)

```bash
cd siren
python experiment_scripts/train_sdf.py
```

Trains a 5-layer SIREN (256 neurons/layer) to learn the SDF of a mesh. The loss combines Eikonal, surface consistency, and off-surface penalty terms.

### 2. Sparse Shape Reconstruction

```bash
cd inr-implicit-shape-reconstruction-mesh
python src/impl_recon/train.py
```

Trains an auto-decoder with per-shape latent vectors (dim=128) and an 8-layer MLP to predict occupancy from sparse input. Meshes are extracted via Marching Cubes. Configure training parameters in `src/train_config_default.yml` and data paths in `src/paths_config_default.yml`.

### 3. Pairwise Registration

```bash
cd registration-pipeline
python src/train.py
```

Trains an INR to predict displacement fields aligning a moving point cloud to a fixed target. Uses Chamfer distance, smoothness, and Laplacian regularization losses. Configure via `src/config.yml` (DeformingThings4D) or `src/config_lung.yml` (Lung250M-4B).

### 4. Cohort-based Registration (with latent space)

```bash
cd registration-pipeline-latent
python src/train.py
```

Extends pairwise registration with per-pair latent vectors. During training, both network weights and latent vectors are optimized. During inference, only the latent vector is optimized for unseen pairs. Configure via `src/config.yml` or `src/config_lung.yml`.

All experiments log metrics to [Weights & Biases](https://wandb.ai).

## Method Overview

All three experiments are built on the same core idea: using an MLP to approximate a continuous function over 3D space.

- **INR/SIREN:** The network learns a signed distance function f(x) where the zero level set defines the surface. Sine activations with frequency scaling (omega_0=30) capture high-frequency geometric detail.
- **Shape Reconstruction:** An auto-decoder concatenates a learnable latent vector z_i with 3D coordinates and predicts occupancy probability. Positional encoding (random Fourier features) mitigates spectral bias. The loss combines BCE, Soft Dice, and latent regularization.
- **Registration:** The MLP takes source coordinates (with Fourier positional encoding) and outputs 3D displacement vectors. A self-supervised loss avoids the need for ground-truth flow fields.

## Citation

If you use this code, please cite:

```
@mastersthesis{gourdin2024inr3d,
  title={Implicit Neural Representations of 3D Medical Point Clouds},
  author={Gourdin, Benoit},
  school={Technical University of Munich},
  year={2024}
}
```

## License

See the individual subdirectories for licensing information. The `siren/` directory includes code from the [SIREN repository](https://github.com/vsitzmann/siren) (MIT License).
