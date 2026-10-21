# IIT Kanpur IDEAL Lab

## Federated Learning Tutorial

This repository contains implementations of centralized and federated learning algorithms for MNIST digit classification using Python and PyTorch. It is designed to help students understand federated optimization, client data heterogeneity, and the communication mechanisms involved in distributed learning.

The repository provides two complementary implementations:

- **Basic implementations (`src/basic/`):** Simulate centralized and federated learning within a single Python process. These implementations focus on the learning algorithms without explicit network communication.
- **TCP-based implementations (`src/tcp/`):** Run the server and clients as separate Python processes that communicate through TCP sockets. They provide a foundation for studying client-server communication and experimenting with network effects.

## Implemented Algorithms

The repository includes the following algorithms:

- **Centralized Learning:** Trains a model using the complete training dataset at a single location.
- **Federated Averaging (FedAvg):** Aggregates locally trained client models to update a global model.
- **Federated Proximal (FedProx):** Adds a proximal regularization term to local training to help address heterogeneity across clients.
- **SCAFFOLD:** Uses control variates to mitigate client drift during federated training.

The implementations use a multilayer perceptron (MLP) for MNIST digit classification.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ideal-lab-server/fl-tutorial
cd fl-tutorial
```

### 2. Create a virtual environment

Using a virtual environment keeps the project's Python dependencies separate from other Python installations and projects.

**Linux / macOS**

```bash
python3 -m venv .venv
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
```

### 3. Activate the virtual environment

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

After activation, your terminal should indicate that the virtual environment is active.

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy
```

For GPU training, install a PyTorch build compatible with your machine's GPU and CUDA environment. CPU training does not require a CUDA-enabled GPU.

## Documentation

Refer to the README files in the respective directories for detailed instructions, configurations, implementation details, and suggested experiments.

- [`src/basic/README.md`](src/basic/README.md) — Single-process implementations.
- [`src/tcp/README.md`](src/tcp/README.md) — TCP-based client-server implementations.

---

**IIT Kanpur — IDEAL Lab**  
*Intelligent Decision and Algorithmic Learning*