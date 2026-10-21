# ==================================================================================
#                             IIT Kanpur IDEAL Lab
#                          Federated Learning Tutorial
#                     Centralized Classification of MNIST Data 
# ==================================================================================

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, random_split

# ==================================================================================
#                                  CONFIGURATION
# ==================================================================================

TRAIN_RATIO = 0.8       # train-test split ratio    
EPOCHS = 50             # epoch --> one round of training over entire training data
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 256
LR = 1e-3               # learning rate
BIASED_SPLIT = False    # train-test data distribution bias    
SEED = 42

DATA_DIR = "./data"

# ==================================================================================
#                                 REPRODUCIBILITY
# ==================================================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================================================================================
#                           CLASS OF MULTI-LAYER PERCEPTRON
# ==================================================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ==================================================================================
#                                   MAIN FUNCTION
# ==================================================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DATA_DIR, exist_ok=True)

    # ===============================================================================
    #                                DATA PREPARATION
    # ===============================================================================

    dataset = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    n_train = int(TRAIN_RATIO * len(dataset)) 

    if BIASED_SPLIT:
        
        sorted_indices = torch.argsort(dataset.targets)
        train_indices = sorted_indices[:n_train].tolist()
        test_indices = sorted_indices[n_train:].tolist()

        train_data = Subset(dataset, train_indices)
        test_data = Subset(dataset, test_indices)
    
    else:

        train_data, test_data = random_split(
            dataset,
            [n_train, len(dataset) - n_train],
            generator=torch.Generator().manual_seed(SEED)
        )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TRAIN)

    model = MLP().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # ================================================================================
    #                                MODEL TRAINING
    # ================================================================================

    for epoch in range(EPOCHS):

        model.train()

        for x, y in train_loader:

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()           # set gradients to zero
            loss = criterion(model(x), y)   # compute loss
            loss.backward()                 # backpropagation
            optimizer.step()                # update MLP parameters

        print(
            f"Epoch {epoch + 1}: "
            f"loss = {loss.item():.4f}"
        )

    # ================================================================================
    #                                 EVALUATION
    # ================================================================================ 
    
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:

            x, y = x.to(device), y.to(device)
            prediction = model(x).argmax(1)
            correct += (prediction == y).sum().item()

    accuracy = 100 * correct / len(test_data)
    print(f"Test accuracy: {accuracy:.2f}%")

# ==================================================================================
#                                    ENTRY POINT
# ==================================================================================

if __name__ == "__main__":
    main()