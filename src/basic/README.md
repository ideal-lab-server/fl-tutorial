# IIT Kanpur IDEAL Lab

## Federated Learning Tutorial: Basic Implementations

This directory contains basic implementations of four learning algorithms for MNIST digit classification using a multilayer perceptron (MLP) implemented in PyTorch:

- Centralized Learning
- Federated Averaging (FedAvg)
- Federated Proximal (FedProx)
- SCAFFOLD

## Overview

- **Dataset preparation:** The MNIST dataset is downloaded to `fl-tutorial/data/` and randomly split into training and testing subsets using an 80:20 ratio. This ratio can be changed by modifying the appropriate configuration parameter. Instead of using MNIST's predefined training and test sets, we create a held-out test set from the downloaded dataset to facilitate experiments investigating the effects of biased train-test splits.

- **Centralized Learning:** The MLP is trained using the entire training dataset available at a single location.

- **Federated Learning:** FedAvg, FedProx, and SCAFFOLD partition the training dataset into equal-sized, non-overlapping subsets and assign them to different clients. Client data segregation is simulated logically; the subsets are not stored at separate physical locations.

- **Client simulation:** All clients are simulated using a `for` loop within a single Python process.

- **Local training:** In each federated round, every client starts with a copy of the current global model and performs local training using its allocated dataset.

- **Server aggregation:** After local training, the clients make their updated model parameters available to the central server. The server updates the global model according to the aggregation rule of the respective algorithm and makes it available to clients for the next round.

- **Communication:** Client-server communication is simulated logically within a single process. No explicit message transmission protocols or communication channel models are used.

- **SCAFFOLD control variates:** In addition to model parameters, SCAFFOLD maintains and updates control variates to mitigate client drift during local training.

- **Evaluation:** At the end of training, the global model is evaluated on the held-out test dataset, and its classification accuracy is reported.

## Directory Structure

```text
basic/
├── README.md
├── centralized.py
├── fedavg.py
├── fedprox.py
└── scaffold.py
```

## Requirements

The implementations require Python and the following packages:

- PyTorch
- Torchvision
- NumPy

Install the required packages in your Python environment:

```bash
pip install torch torchvision numpy
```

## Running the Code

Activate your Python environment and run the scripts from the root directory of the repository:

```bash
python src/basic/centralized.py
python src/basic/fedavg.py
python src/basic/fedprox.py
python src/basic/scaffold.py
```

## Suggested Experiments

1. **Establish a baseline:** Run all four algorithms with their default configurations and compare their classification accuracy.

2. **Biased train-test split:** Investigate the performance of centralized learning when the training and testing subsets have different class distributions.

3. **Biased client data partitioning:** Investigate the performance of the federated learning algorithms when the class distributions vary across clients. Compare their performance and interpret the results in terms of client drift and the mechanisms used by each algorithm.

4. **Effect of the optimizer:** Compare the performance of the algorithms under different optimizers. Investigate why SCAFFOLD's performance may deteriorate substantially with an optimizer such as Adam compared with SGD, and examine whether the behavior depends on the implementation and control-variate update rules.

5. **Effect of the number of clients:** Investigate how increasing the number of clients affects the final classification accuracy.

6. **Effect of local training and communication rounds:** Investigate how the number of federated rounds and the number of local epochs affect the performance.

All these experiments can be conducted by modifying the appropriate parameters in the configuration section of each script.

---

**IIT Kanpur — IDEAL Lab**  
*Intelligent Decision and Algorithmic Learning*
