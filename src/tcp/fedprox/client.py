import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..model import MNISTModel
from ..data import load_mnist
from ..serialization import serialize_state_dict

from ..common import (
    PROTOCOL_VERSION,
    MessageType,
)
from ..protocol import send_message
from .config import FedProxConfig

from ..fedavg.client import (
    connect_and_register,
    receive_partition,
    receive_global_model
)


def train_local(
    model,
    global_state_dict,
    train_dataset,
    local_epochs,
    batch_size,
    learning_rate,
    mu,
    device,
):
    """
    Train a local MNIST model initialized from global parameters.

    Returns:
        local_state_dict: Updated model parameters.
        num_samples: Number of local training examples.
        average_loss: Sample-weighted mean training loss over all epochs.
    """

    if len(train_dataset) == 0:
        raise ValueError("Client training dataset cannot be empty")

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1")
    
    model.load_state_dict(global_state_dict)
    model.to(device)
    model.train()

    global_params = [
        param.detach().clone()
        for param in model.parameters()
    ]

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    total_loss = 0.0
    total_examples = 0

    for _ in range(local_epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            data_loss = criterion(outputs, labels)

            proximal_loss = sum(
                torch.sum((param - global_param) ** 2)
                for param, global_param in zip(
                    model.parameters(), global_params
                )
            )

            loss = data_loss + (mu/2) * proximal_loss
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual

    average_loss = total_loss / total_examples

    local_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    return local_state_dict, len(train_dataset), average_loss



def train_and_send_update(
    sock,
    client_id,
    model,
    global_state_dict,
    client_dataset,
    config,
):
    """Train locally and send the updated model to the server."""
    
    print(f"Client {client_id} is training its local model with {config.device}.")

    local_state_dict, num_samples, average_loss = train_local(
        model=model,
        global_state_dict=global_state_dict,
        train_dataset=client_dataset,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size_train,
        learning_rate=config.learning_rate,
        mu=config.mu,
        device=config.device,
    )

    payload = serialize_state_dict(local_state_dict)

    send_message(
        sock,
        {
            "type": MessageType.LOCAL_UPDATE,
            "protocol_version": PROTOCOL_VERSION,
            "client_id": client_id,
            "num_samples": num_samples,
        },
        payload,
    )

    print(
        f"Client {client_id} sent its local model update."
    )


def main():
    sock, client_id = connect_and_register()

    try:
        train_dataset, _ = load_mnist()

        client_dataset = receive_partition(
            sock, client_id, train_dataset
        )

        print(
            f"Client {client_id} is ready with "
            f"{len(client_dataset)} training samples."
        )

        config = FedProxConfig()
        model = MNISTModel()
        round = 0
        
        # reduce cpu congestion 
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        while True:

            if round < config.num_rounds:
                round += 1
                print(f"\n{'=' * 50}")
                print(f"Federated round {round}/{config.num_rounds}")
                print(f"{'=' * 50}\n")
            
            global_state_dict = receive_global_model(
                sock, client_id
            )

            if global_state_dict is None:
                break

            train_and_send_update(
                sock=sock,
                client_id=client_id,
                model=model,
                global_state_dict=global_state_dict,
                client_dataset=client_dataset,
                config=config,
            )

    finally:
        sock.close()


if __name__ == "__main__":
    main()
