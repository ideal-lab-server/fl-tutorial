import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..model import MNISTModel
from ..data import load_mnist, create_client_datasets
from ..serialization import serialize_state_dict, deserialize_state_dict

import time
import socket
from ..common import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    CLIENT_RETRY,
    MessageType,
)
from ..protocol import send_message, recv_message
from .config import FedAvgConfig


def train_local(
    model,
    global_state_dict,
    train_dataset,
    local_epochs,
    batch_size,
    learning_rate,
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
            loss = criterion(outputs, labels)
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


def connect_and_register(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Connect to the server and receive an automatically assigned ID."""

    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            sock.connect((host, port))
            print(f"Successfully connected to the server at {host}:{port}.")
            break

        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            sock.close()
            print(
                f"Server unavailable at {host}:{port}. "
                f"Retrying in {CLIENT_RETRY} seconds..."
            )
            time.sleep(CLIENT_RETRY)

    try:
        # Request registration without specifying a client ID.
        send_message(
            sock,
            {
                "type": MessageType.REGISTER,
                "protocol_version": PROTOCOL_VERSION,
            },
        )

        # Receive the server's registration acknowledgment.
        header, payload = recv_message(sock)

        if payload:
            raise ValueError(
                "REGISTER_ACK must not contain a payload"
            )

        if header.get("type") != MessageType.REGISTER_ACK:
            raise ValueError("Expected REGISTER_ACK message")

        if header.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported protocol version")

        client_id = header.get("client_id")

        if (
            isinstance(client_id, bool)
            or not isinstance(client_id, int)
            or client_id < 0
        ):
            raise ValueError("Invalid client ID received from server")

        print(f"\nRegistered successfully. Client ID: {client_id}")

        # Return the open socket for subsequent communication.
        return sock, client_id

    except Exception:
        sock.close()
        raise


def receive_partition(sock, client_id, train_dataset):
    """Receive and acknowledge this client's MNIST partition."""

    header, payload = recv_message(sock)

    if payload:
        raise ValueError(
            "PARTITION_ASSIGNMENT must not contain a payload"
        )

    if header.get("type") != MessageType.PARTITION_ASSIGNMENT:
        raise ValueError("Expected PARTITION_ASSIGNMENT message")

    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol version")

    if header.get("client_id") != client_id:
        raise ValueError("Partition assignment has the wrong client ID")

    indices = header.get("indices")

    if not isinstance(indices, list):
        raise ValueError("Partition indices must be a list")

    if not indices:
        raise ValueError("Received an empty partition")

    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(train_dataset)
        for index in indices
    ):
        raise ValueError("Partition contains invalid sample indices")

    if len(set(indices)) != len(indices):
        raise ValueError("Partition contains duplicate indices")

    # Build this client's local view of the MNIST training dataset.
    client_dataset = create_client_datasets(
        train_dataset,
        [indices],
    )[0]

    return client_dataset


def receive_global_model(sock, client_id):
    header, payload = recv_message(sock)

    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol version")

    message_type = header.get("type")

    if message_type == MessageType.STOP:
        if header.get("client_id") != client_id:
            raise ValueError("STOP message has incorrect client ID")
        if payload:
            raise ValueError("STOP message must not contain a payload")

        print(f"\nServer stopped client {client_id}.")
        return None

    if message_type != MessageType.GLOBAL_MODEL:
        raise ValueError(
            f"Expected GLOBAL_MODEL or STOP, received {message_type}"
        )

    if header.get("client_id") != client_id:
        raise ValueError("GLOBAL_MODEL message has incorrect client ID")

    if not payload:
        raise ValueError("GLOBAL_MODEL payload is empty")

    global_state_dict = deserialize_state_dict(payload)

    print(f"Client {client_id} received the global model.")

    return global_state_dict


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

        config = FedAvgConfig()
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
