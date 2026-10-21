import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..model import MNISTModel
from ..data import load_mnist
from ..serialization import serialize_state_dict, deserialize_state_dict

from ..common import (
    PROTOCOL_VERSION,
    MessageType,
)
from ..protocol import send_message, recv_message
from .config import ScaffoldConfig
from ..fedavg.client import (
    connect_and_register,
    receive_partition,
)


def train_local(
    model,
    global_state_dict,
    global_control,
    client_control,
    train_dataset,
    local_epochs,
    batch_size,
    learning_rate,
    device,
):
    """
    SCAFFOLD local training using Option II control updates.

    Returns:
        local_state_dict: Updated local model.
        new_client_control: Updated local control variate.
        num_samples: Number of local training examples.
        average_loss: Sample-weighted mean training loss.
    """

    if len(train_dataset) == 0:
        raise ValueError("Client training dataset cannot be empty")

    if local_epochs < 1:
        raise ValueError("local_epochs must be at least 1")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    model.load_state_dict(global_state_dict)
    model.to(device)
    model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Control variates are kept on the same device as the model.
    global_control = {
        name: tensor.to(device)
        for name, tensor in global_control.items()
    }
    client_control = {
        name: tensor.to(device)
        for name, tensor in client_control.items()
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    total_loss = 0.0
    total_examples = 0
    num_steps = 0

    for _ in range(local_epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # SCAFFOLD correction:
            # gradient <- gradient - client_control + global_control
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        parameter.grad.add_(
                            global_control[name]
                            - client_control[name]
                        )

            optimizer.step()

            batch_size_actual = labels.size(0)
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual
            num_steps += 1

    # update local model 
    local_state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    # update control variate 
    if num_steps == 0:
        raise RuntimeError("No local training steps were performed")

    new_client_control = {}

    for name, old_control in client_control.items():
        new_client_control[name] = (
            old_control
            - global_control[name]
            + (
                global_state_dict[name].to(device)
                - local_state_dict[name].to(device)
            ) / (num_steps * learning_rate)
        ).detach().cpu().clone()

    average_loss = total_loss / total_examples

    return (
        local_state_dict,
        new_client_control,
        len(train_dataset),
        average_loss,
    )



def receive_global_state(sock, client_id):
    header, payload = recv_message(sock)

    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol version")

    if header.get("client_id") != client_id:
        raise ValueError("Incorrect client ID")

    if header.get("type") == MessageType.STOP:
        if payload:
            raise ValueError("STOP must not contain a payload")
        print(f"Server stopped client {client_id}.")
        return None, None

    if header.get("type") != MessageType.GLOBAL_MODEL:
        raise ValueError("Expected GLOBAL_MODEL")

    if not payload:
        raise ValueError("Global model payload is empty")

    global_state_dict = deserialize_state_dict(payload)

    header, payload = recv_message(sock)

    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol version")

    if header.get("type") != MessageType.GLOBAL_CONTROL:
        raise ValueError("Expected GLOBAL_CONTROL")

    if header.get("client_id") != client_id:
        raise ValueError("Incorrect control message client ID")

    if not payload:
        raise ValueError("Global control payload is empty")

    global_control = deserialize_state_dict(payload)

    print(f"Client {client_id} received the global state.")

    return global_state_dict, global_control


def send_scaffold_update(
    sock,
    client_id,
    model_delta,
    control_delta,
    num_samples,
):
    # send the model delta
    send_message(
        sock,
        {
            "type": MessageType.LOCAL_UPDATE,
            "protocol_version": PROTOCOL_VERSION,
            "client_id": client_id,
            "num_samples": num_samples,
        },
        serialize_state_dict(model_delta),
    )

    # send the local control-variate delta
    send_message(
        sock,
        {
            "type": MessageType.CONTROL_UPDATE,
            "protocol_version": PROTOCOL_VERSION,
            "client_id": client_id,
        },
        serialize_state_dict(control_delta),
    )

    print(
        f"Client {client_id} sent its local state delta."
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

        config = ScaffoldConfig()
        model = MNISTModel()

        # initialize a zero local control variate
        client_control = {
            name: torch.zeros_like(parameter, device="cpu")
            for name, parameter in model.named_parameters()
        }

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
            
            global_state_dict, global_control = receive_global_state(
                sock, client_id
            )

            if global_state_dict is None:
                break
            
            print(f"Client {client_id} is training its local model with {config.device}.")

            (
                local_state_dict,
                new_client_control,
                num_samples,
                average_loss,
            ) = train_local(
                model=model,
                global_state_dict=global_state_dict,
                global_control=global_control,
                client_control=client_control,
                train_dataset=client_dataset,
                local_epochs=config.local_epochs,
                batch_size=config.batch_size_train,
                learning_rate=config.learning_rate,
                device=config.device,
            )

            model_delta = {
                name: local_state_dict[name] - global_state_dict[name]
                for name, _ in model.named_parameters()
            }

            control_delta = {
                name: new_client_control[name] - client_control[name]
                for name in client_control
            }

            client_control = new_client_control

            send_scaffold_update(
                sock=sock,
                client_id=client_id,
                model_delta=model_delta,
                control_delta=control_delta,
                num_samples=num_samples,
            )

    finally:
        sock.close()


if __name__ == "__main__":
    main()
