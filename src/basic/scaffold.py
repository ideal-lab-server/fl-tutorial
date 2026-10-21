# ==================================================================================
#                             IIT Kanpur IDEAL Lab
#                          Federated Learning Tutorial
#                 Classification of MNIST Data via SCAFFOLD Algorithm 
# ==================================================================================
import os
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# ==================================================================================
#                                 CONFIGURATION
# ==================================================================================

TRAIN_RATIO = 0.8         # train-test split ratio
NUM_CLIENTS = 3
ROUNDS = 5                # round --> fetch + local training + send
LOCAL_EPOCHS = 10         # epoch --> one round of training over entire local data
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 256
LR = 1e-3                 # learning rate
IID_DATA = True           # data distribution across clients
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
#                                   DATA PREPARATION
# ==================================================================================

def data_prep():

    dataset = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    n_train = int(TRAIN_RATIO * len(dataset))
    n_test = len(dataset) - n_train

    train_data, test_data = random_split(
            dataset,
            [n_train, n_test],
            generator=torch.Generator().manual_seed(SEED)
        )

    # ==============================================================================
    #                      DIVIDING TRAINING DATA AMONG CLIENTS
    # ==============================================================================
    
    client_sizes = [n_train // NUM_CLIENTS] * NUM_CLIENTS
    
    for i in range(n_train % NUM_CLIENTS):
        client_sizes[i] += 1
    
    if IID_DATA:

        client_datasets = random_split(
            train_data,
            client_sizes,
            generator = torch.Generator().manual_seed(SEED)
        )
    
    else:

        labels = dataset.targets[train_data.indices]
        sorted_indices = torch.argsort(labels).tolist()
        
        client_datasets = []
        start = 0

        for size in client_sizes:
            indices = sorted_indices[start:start + size]
            client_datasets.append(
                Subset(train_data, indices)
            )
            start += size

    # ==============================================================================
    #                            CREATING DATA LOADERS
    # ==============================================================================

    client_loaders = [
        DataLoader(
            client_dataset,
            batch_size=BATCH_SIZE_TRAIN,
            shuffle=True
        )
        for client_dataset in client_datasets
    ]

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False
    )
    
    return client_loaders, test_loader

# ==================================================================================
#                                   EVALUATION
# ==================================================================================

def evaluate(model, test_loader, device):
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy

# ==================================================================================
#                                   MAIN FUNCTION
# ==================================================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DATA_DIR, exist_ok=True)

    client_loaders, test_loader = data_prep()

    print("\nTraining samples per client:")
    for client_id, loader in enumerate(client_loaders):
        print(
            f"Client {client_id}: "
            f"{len(loader.dataset)} samples"
        )

    global_model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()

    server_control = {
        name: torch.zeros_like(param, device=device)
        for name, param in global_model.named_parameters()   
    }

    client_controls = [
        {
            name: torch.zeros_like(param, device=device)
            for name, param in global_model.named_parameters()
        }
    for _ in range(NUM_CLIENTS)
    ]

    # ==============================================================================
    #                                 TRAINING LOOP
    # ==============================================================================

    for round_num in range(ROUNDS):

        print(f"\nFederated round {round_num + 1}/{ROUNDS}")

        global_parameters = {
            name: param.detach().clone()
            for name, param in global_model.named_parameters()
        }

        client_states = []
        client_sizes = []

        # ==========================================================================
        #                             CLIENT TRAINING
        # ==========================================================================

        for client_id, loader in enumerate(client_loaders):
            
            local_model = copy.deepcopy(global_model)         # fetch a global copy
            optimizer = torch.optim.SGD(local_model.parameters(),lr=LR)

            old_client_control = {
                name: value.clone()
                for name, value in client_controls[client_id].items()
            }

            local_model.train()
            total_loss = 0.0
            local_steps = 0

            for epoch in range(LOCAL_EPOCHS):                 # local training

                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()                # setting gradients to zero
                    outputs = local_model(images)        # predict outputs
                    loss = criterion(outputs, labels)    # compute loss

                    loss.backward()                      # backpropagation

                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            param.grad.add_(             # gradient correction
                                server_control[name]
                                - old_client_control[name]
                            )

                    optimizer.step()                     # update MLP parameters
                    local_steps += 1
                    total_loss += loss.item() * labels.size(0)

            average_loss = total_loss / (LOCAL_EPOCHS * len(loader.dataset))

            print(
                f"  Client {client_id}: "
                f"training loss = {average_loss:.4f}"
            )

            local_parameters = {
                name: param.detach().clone()
                for name, param in local_model.named_parameters()
            }
            
            new_client_control = {}

            for name in old_client_control:              # client control update
                new_client_control[name] = (
                    old_client_control[name]
                    - server_control[name]
                    + (
                        global_parameters[name]
                        - local_parameters[name]
                    ) / (local_steps * LR)
                )
            
            client_controls[client_id] = new_client_control
 
            client_states.append({                       # cpu copy of local models
                key: value.detach().cpu().clone()
                for key, value
                in local_model.state_dict().items()
            })

            client_sizes.append(len(loader.dataset))

            del local_model, optimizer
        
        # ==========================================================================
        #                             SERVER AGGREGATION
        # ==========================================================================

        total_samples = sum(client_sizes)
        global_state = global_model.state_dict()

        for key in global_state:                         # global state update
            global_state[key] = sum(
                client_states[i][key] * client_sizes[i]
                for i in range(NUM_CLIENTS)
            ) / total_samples

        global_model.load_state_dict(global_state)

        for name in server_control:                      # global control update
            server_control[name] = torch.stack([
                element[name] for element in client_controls
            ]).mean(dim=0)

        print("  Server: global model updated.")

        test_accuracy = evaluate(
            global_model,
            test_loader,
            device
        )

        print(f"  Test accuracy = {test_accuracy:.2f}%")

    # ==============================================================================
    #                                 FINAL EVALUATION
    # ==============================================================================

    print("\nFinal evaluation on MNIST test dataset")

    test_accuracy = evaluate(
        global_model,
        test_loader,
        device
    )

    print(f"Test accuracy: {test_accuracy:.2f}%")

# ==================================================================================
#                                   ENTRY POINT
# ==================================================================================

if __name__ == "__main__":
    main()