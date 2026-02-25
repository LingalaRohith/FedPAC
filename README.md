# FedPAC – Personalized Federated Learning with Feature Alignment and Classifier Collaboration

## Overview

This repository contains a simplified implementation of **FedPAC** (Personalized Federated Learning with Feature Alignment and Classifier Collaboration) and several baseline federated learning methods.  
The project was developed as a **CSCI 8000 – Advanced Special Topics** final project (Fall 2024) under the guidance of **Dr. Jin Lu**.

The main goal is to study **personalized federated learning (PFL)** under non‑IID client data, compare FedPAC to common FL baselines, and reproduce key behaviors reported in the original FedPAC paper:

> Xu, Jian, Xinyi Tong, and Shao-Lun Huang. “Personalized Federated Learning with Feature Alignment and Classifier Collaboration.” ICLR 2023.

## Methods Implemented

We implemented and compared the following algorithms:

- **Local Training**  
  Each client trains its model independently with no communication or aggregation. This serves as a non‑federated baseline.

- **FedAvg**  
  Standard Federated Averaging. In each communication round, a subset of clients trains locally for a few epochs, and the server aggregates updates via a weighted average based on local data sizes.

- **LG-FedAvg** (Layered‑Global FedAvg)  
  Extends FedAvg by allowing clients to fine‑tune specific layers locally for personalization while still sharing a global backbone through aggregation.

- **FedPer** (Federated Personalization)  
  Splits the model into shared global layers and client‑specific personalized layers. Only the shared layers are aggregated at the server; personalized layers remain local.

- **FedPAC**  
  Combines **global–local feature alignment** with **personalized classifier collaboration**:
  - A shared encoder aligns local and global feature spaces.  
  - Client‑specific classifier heads collaborate through optimized combination weights, balancing global generalization and local personalization.

All core hyperparameters and settings are configured via the argument parser in `options.py`.

## Datasets

The project focuses on image classification benchmarks widely used in federated learning:

- **EMNIST**  
  - 62 classes: digits (0–9), uppercase letters (A–Z), lowercase letters (a–z).  
  - Used to evaluate performance on diverse handwritten character recognition tasks.

- **Fashion-MNIST**  
  - 10 classes of grayscale clothing items (e.g., T‑shirt, trouser, coat, sneaker).  
  - A more challenging, modern alternative to MNIST.

- **CIFAR-10**  
  - 10 classes of color images (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).  
  - Standard benchmark for image classification and widely used in FL research.

- **CINIC-10**  
  - 10 classes, constructed from CIFAR‑10 and downsampled ImageNet.  
  - Larger and more diverse than CIFAR‑10, used to test scalability and robustness under more heterogeneous data.

### Non-IID Client Partitioning

To simulate realistic heterogeneous data distributions, we use controlled non‑IID partitions:

- **Fashion-MNIST, CIFAR-10, CINIC-10 (10-class datasets)**  
  - Clients are divided into **5 groups**.  
  - Each group focuses on **3 consecutive dominant classes**, while the remaining classes are sampled more uniformly.

- **EMNIST (62-class dataset)**  
  - Clients are divided into **3 groups**:  
    - Digits (0–9)  
    - Uppercase letters (A–Z)  
    - Lowercase letters (a–z)

Experiments are run with **20** and **100** total clients to study scalability and the effect of client count on personalization.

## Model Architectures

Two convolutional neural network (CNN) backbones are used:

- **Shallow CNN (for EMNIST and Fashion-MNIST)**  
  - 2 convolutional layers with **16** and **32** channels, each followed by max pooling.  
  - 2 fully connected layers with **128** units and **10** output units before softmax.  
  - **LeakyReLU** activations.

- **Deeper CNN (for CIFAR-10 and CINIC-10)**  
  - Same structure as above, plus an additional convolutional layer with **64** channels for more complex color images.  
  - Also uses **LeakyReLU** activations.

## Training Setup

Shared training and optimization settings (unless otherwise noted):

- **Optimizer**: Mini‑batch **SGD**.  
- **Learning rate η**:  
  - EMNIST, Fashion-MNIST: **0.01**  
  - CIFAR-10, CINIC-10: **0.02**  
- **Classifier training**:  
  - Trained for **1 epoch** with learning rate **ηg = 0.1**.  
- **Feature extractor training**:  
  - Trained for multiple epochs with **ηf = η** (same as baselines).  
- **Weight decay**: **5 × 10⁻⁴**  
- **Momentum**: **0.5**  
- **Batch size**:  
  - EMNIST: **100**  
  - Fashion-MNIST, CIFAR-10, CINIC-10: **50**  
- **Local epochs per communication round**: **E = 5**  
- **Total communication rounds**: **200**  
- **Metric reported**: Average **test accuracy of local models** across all clients.

## Command-Line Arguments

The main configurable options are defined in `options.py`:

- `--epochs` (`int`): Number of training epochs / global rounds (default: `5`, experiments use up to `200`).  
- `--num_users` (`int`): Number of clients (e.g., `20` or `100`).  
- `--frac` (`float`): Fraction of clients participating per round.  
- `--local_epoch` (`int`): Number of local training epochs per round (default: `5`).  
- `--local_iter` (`int`): Number of local iterations per epoch (default: `1`).  
- `--local_bs` (`int`): Local batch size (default: `50`).  
- `--lr` (`float`): Base learning rate η.  
- `--momentum` (`float`): SGD momentum (default: `0.5`).  
- `--train_rule` (`str`): Training rule for PFL (e.g., `FedPAC`, `FedAvg`, etc.).  
- `--agg_g` (`int`): Flag for weighted aggregation behavior in personalized FL.  
- `--lam` (`float`): Coefficient for regularization term.  
- `--local_size` (`int`): Number of samples per client.  
- `--dataset` (`str`): Dataset name (`emnist`, `fashion-mnist`, `cifar`, `cinic`, etc.).  
- `--num_classes` (`int`): Number of classes (10 for most experiments).  
- `--device` (`str`): Device identifier (e.g., `cuda:0` or `cpu`).  
- `--optimizer` (`str`): Optimizer type (default: `sgd`).  
- `--iid` (`int`): Set `1` for IID and `0` for non‑IID splits.  
- `--noniid_s` (`int`): Controls strength of non‑IID partitioning.

## Example Usage

Example commands (adjust paths and script names to your setup):

```bash
# FedPAC on CIFAR-10 with 20 clients, non-IID
python main.py \
  --dataset cifar \
  --num_users 20 \
  --epochs 200 \
  --train_rule FedPAC \
  --iid 0 \
  --local_epoch 5 \
  --local_bs 50 \
  --lr 0.02

# FedAvg on Fashion-MNIST with 100 clients, non-IID
python main.py \
  --dataset fashion-mnist \
  --num_users 100 \
  --epochs 200 \
  --train_rule FedAvg \
  --iid 0
```

If your main training script is named differently (e.g., `train_fedpac.py`), replace `main.py` accordingly.

## Results Summary

We evaluated all five methods (**Local**, **FedAvg**, **LG-FedAvg**, **FedPer**, **FedPAC**) on **EMNIST**, **Fashion-MNIST**, **CIFAR-10**, and **CINIC-10** with **20** and **100** clients.  
Values below are **average test accuracies (%) of local models**.

| Dataset       | Method   | EMNIST 20c | EMNIST 100c | Fashion-MNIST 20c | Fashion-MNIST 100c | CIFAR-10 20c | CIFAR-10 100c | CINIC-10 20c | CINIC-10 100c |
|--------------|----------|-----------:|------------:|------------------:|-------------------:|-------------:|--------------:|-------------:|--------------:|
|              | **FedPAC**  | **86.46** | **89.88** | **91.83** | **92.72** | **81.13** | **83.36** | **74.96** | **77.36** |
|              | FedAvg   | 71.85 | 75.97 | 85.28 | 86.89 | 70.05 | 73.82 | 58.38 | 62.52 |
|              | LG-FedAvg| 74.76 | 75.81 | 85.66 | 86.28 | 65.19 | 65.38 | 63.75 | 63.93 |
|              | FedPer   | 75.91 | 77.06 | 87.43 | 87.88 | 68.37 | 72.26 | 63.47 | 66.26 |
|              | Local    | 73.85 | 74.11 | 85.68 | 86.37 | 65.43 | 64.68 | 63.19 | 63.33 |

Across all datasets and client configurations, **FedPAC consistently outperforms** the baselines, especially on the more challenging **CIFAR-10** and **CINIC-10** datasets, highlighting the benefit of combining global feature alignment with personalized classifier collaboration in heterogeneous federated settings.

## Project Contributors

- **Rohith Lingala** – EMNIST experiments (all methods)  
- **Manish Valeti** – Fashion-MNIST experiments (all methods)  
- **Sathwik Busireddy** – CIFAR-10 experiments (all methods)  
- **Navyanth Bollareddy** – CINIC-10 experiments (all methods)
