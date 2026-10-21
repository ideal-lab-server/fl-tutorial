import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..model import MNISTModel
from ..serialization import serialize_state_dict, deserialize_state_dict

import socket
from ..common import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    MessageType,
)
from ..protocol import send_message, recv_message
from ..data import load_mnist, create_partition_indices
from .config import FedAvgConfig
from .strategy import aggregate_state_dicts


def prepare_experiment(config: FedAvgConfig):
    """Load MNIST and generate client sample assignments."""

    print("\nPreparing FedAvg experiment...")
    print(f"Number of clients: {config.num_clients}")
    print(f"Partition strategy: {config.partition}")
    print(f"Random seed: {config.seed}")

    if config.partition == "dirichlet":
        print(f"Dirichlet alpha: {config.alpha}")

    # Load the training dataset on the server.
    train_dataset, _ = load_mnist()

    print(f"Training dataset size: {len(train_dataset)}")

    # Generate the partition using the experiment configuration.
    client_indices = create_partition_indices(
        train_dataset,
        num_clients=config.num_clients,
        partition=config.partition,
        alpha=config.alpha,
        seed=config.seed,
    )

    # Validate the generated assignments.
    if len(client_indices) != config.num_clients:
        raise RuntimeError(
            "The number of partitions does not match num_clients"
        )

    all_indices = [
        index
        for indices in client_indices
        for index in indices
    ]

    if len(all_indices) != len(train_dataset):
        raise RuntimeError(
            "The partition does not cover the entire training dataset"
        )

    if len(set(all_indices)) != len(train_dataset):
        raise RuntimeError(
            "Some training samples are assigned more than once"
        )

    if set(all_indices) != set(range(len(train_dataset))):
        raise RuntimeError(
            "The partition contains invalid sample indices"
        )

    print("\nPartition summary:")

    for client_id, indices in enumerate(client_indices):
        print(
            f"Client {client_id}: {len(indices)} training samples"
        )

    print("\nExperiment preparation completed.")

    return train_dataset, client_indices


def register_clients(num_clients, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Accept client connections and assign IDs in registration order."""

    clients = []
    
    # AF_INET: IPv4 Address, SOCK_STREAM: streaming protocol such as TCP

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        server_socket.bind((host, port))
        server_socket.listen(num_clients)

        print(f"Server listening on {host}:{port}")
        print(f"Waiting for {num_clients} clients...")

        try:
            while len(clients) < num_clients:
                conn, address = server_socket.accept()

                try:
                    header, payload = recv_message(conn)

                    if payload:
                        raise ValueError(
                            "REGISTER must not contain a payload"
                        )

                    if header.get("type") != MessageType.REGISTER:
                        raise ValueError("Expected REGISTER message")

                    if (
                        header.get("protocol_version")
                        != PROTOCOL_VERSION
                    ):
                        raise ValueError("Unsupported protocol version")

                    # Assign the next available ID automatically.
                    client_id = len(clients)

                    send_message(
                        conn,
                        {
                            "type": MessageType.REGISTER_ACK,
                            "protocol_version": PROTOCOL_VERSION,
                            "client_id": client_id,
                        },
                    )

                    clients.append((conn, client_id))

                    print(
                        f"Registered client {client_id} "
                        f"from {address}"
                    )

                except Exception:
                    conn.close()
                    raise

        except Exception:
            for conn, _ in clients:
                conn.close()
            raise

    print(f"Successfully registered {len(clients)} clients.")

    # Return open client connections for subsequent communication.
    return clients


def send_partitions(clients, client_indices):
    """Send each client its partition and await acknowledgment."""

    for conn, client_id in clients:
        indices = client_indices[client_id]

        send_message(
            conn,
            {
                "type": MessageType.PARTITION_ASSIGNMENT,
                "protocol_version": PROTOCOL_VERSION,
                "client_id": client_id,
                "indices": indices,
            },
        )

        print(
            f"Sent {len(indices)} sample indices "
            f"to client {client_id}."
        )


def send_global_model(clients, global_state_dict):
    """Send the initial global model to every client."""

    payload = serialize_state_dict(global_state_dict)

    for conn, client_id in clients:
        send_message(
            conn,
            {
                "type": MessageType.GLOBAL_MODEL,
                "protocol_version": PROTOCOL_VERSION,
                "client_id": client_id,
            },
            payload,
        )

        print(f"Sent initial global model to client {client_id}.")


def receive_local_updates(clients):
    """Receive and validate one local model update from each client."""

    updates = {}

    for conn, client_id in clients:
        header, payload = recv_message(conn)

        if header.get("type") != MessageType.LOCAL_UPDATE:
            raise ValueError("Expected LOCAL_UPDATE message")

        if header.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported protocol version")

        if header.get("client_id") != client_id:
            raise ValueError("Update has the wrong client ID")

        num_samples = header.get("num_samples")

        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples <= 0
        ):
            raise ValueError("Invalid number of training samples")

        if not payload:
            raise ValueError("Local model update payload is empty")

        local_state_dict = deserialize_state_dict(payload)

        updates[client_id] = {
            "state_dict": local_state_dict,
            "num_samples": num_samples,
        }

        print(
            f"Received model update from client {client_id}."
        )

    return updates


def stop_clients(clients):
    for conn, client_id in clients:
        send_message(
            conn,
            {
                "type": MessageType.STOP,
                "protocol_version": PROTOCOL_VERSION,
                "client_id": client_id,
            },
        )


def evaluate_global_model(
    global_state_dict,
    test_dataset,
    batch_size,
    device="cpu",
):
    model = MNISTModel().to(device)
    model.load_state_dict(global_state_dict)
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(reduction="sum")
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

    accuracy = correct / total

    return accuracy


def main():
    config = FedAvgConfig()

    _, client_indices = prepare_experiment(config)
    _, test_dataset = load_mnist()
    clients = register_clients(num_clients=config.num_clients)

    try:
        send_partitions(clients, client_indices)

        global_model = MNISTModel()
        global_state_dict = global_model.state_dict()

        for round_num in range(1, config.num_rounds + 1):
            print(f"\n{'=' * 50}")
            print(f"Federated round {round_num}/{config.num_rounds}")
            print(f"{'=' * 50}\n")

            # Send the current global model to all clients.
            send_global_model(clients, global_state_dict)

            # Receive local model updates from all clients.
            updates = receive_local_updates(clients)

            # Arrange updates in client registration order.
            ordered_updates = [
                updates[client_id]
                for _, client_id in clients
            ]

            client_state_dicts = [
                update["state_dict"]
                for update in ordered_updates
            ]

            client_sizes = [
                update["num_samples"]
                for update in ordered_updates
            ]

            # Compute the next global model using FedAvg.
            global_state_dict = aggregate_state_dicts(
                client_state_dicts,
                client_sizes,
            )

            print("\nFedAvg aggregation completed.")
            print(f"Total training samples: {sum(client_sizes)}")
            print(f"Number of participating clients: {len(client_sizes)}")

            test_accuracy = evaluate_global_model(
                global_state_dict=global_state_dict,
                test_dataset=test_dataset,
                batch_size=config.batch_size_test
            )

            print(f"Test accuracy: {100 * test_accuracy:.2f}%")

        print("\nAll communication rounds completed.")

        stop_clients(clients)

    finally:
        for conn, _ in clients:
            conn.close()

        print("All client connections closed.")


if __name__ == "__main__":
    main()
