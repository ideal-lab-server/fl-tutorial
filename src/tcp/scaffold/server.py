import torch
from ..model import MNISTModel
from ..serialization import serialize_state_dict, deserialize_state_dict

from ..common import (
    PROTOCOL_VERSION,
    MessageType,
)
from ..protocol import send_message, recv_message
from ..data import load_mnist
from .config import ScaffoldConfig
from .strategy import average_state_dicts
from ..fedavg.server import (
    prepare_experiment,
    register_clients,
    send_partitions,
    stop_clients,
    evaluate_global_model,
)


def send_global_state(clients, global_state_dict, global_control):
    
    model_payload = serialize_state_dict(global_state_dict)
    control_payload = serialize_state_dict(global_control)

    for conn, client_id in clients:
        send_message(
            conn,
            {
                "type": MessageType.GLOBAL_MODEL,
                "protocol_version": PROTOCOL_VERSION,
                "client_id": client_id,
            },
            model_payload,
        )

        send_message(
            conn,
            {
                "type": MessageType.GLOBAL_CONTROL,
                "protocol_version": PROTOCOL_VERSION,
                "client_id": client_id,
            },
            control_payload,
        )

        print(f"Sent the global state to client {client_id}.")


def receive_scaffold_updates(clients):
    updates = {}

    for conn, client_id in clients:
        # receive model delta
        header, payload = recv_message(conn)

        if header.get("type") != MessageType.LOCAL_UPDATE:
            raise ValueError("Expected LOCAL_UPDATE")

        if header.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported protocol version")

        if header.get("client_id") != client_id:
            raise ValueError("Incorrect client ID")

        num_samples = header.get("num_samples")

        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples <= 0
        ):
            raise ValueError("Invalid sample count")

        if not payload:
            raise ValueError("Empty model delta")

        model_delta = deserialize_state_dict(payload)

        # receive control-variate delta
        header, payload = recv_message(conn)

        if header.get("type") != MessageType.CONTROL_UPDATE:
            raise ValueError("Expected CONTROL_UPDATE")

        if header.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported protocol version")

        if header.get("client_id") != client_id:
            raise ValueError("Incorrect control update client ID")

        if not payload:
            raise ValueError("Empty control delta")

        control_delta = deserialize_state_dict(payload)

        updates[client_id] = {
            "model_delta": model_delta,
            "control_delta": control_delta,
            "num_samples": num_samples,
        }

        print(f"Received SCAFFOLD updates from client {client_id}.")

    return updates


def main():
    config = ScaffoldConfig()

    _, client_indices = prepare_experiment(config)
    _, test_dataset = load_mnist()
    clients = register_clients(num_clients=config.num_clients)

    try:
        send_partitions(clients, client_indices)

        model = MNISTModel()
        global_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        # initialize global control variate
        global_control = {
            name: torch.zeros_like(parameter, device="cpu")
            for name, parameter in model.named_parameters()
        }

        for round_num in range(1, config.num_rounds + 1):
            print(
                f"\n{'=' * 50}\n"
                f"SCAFFOLD round {round_num}/{config.num_rounds}\n"
                f"{'=' * 50}\n"
            )

            send_global_state(
                clients,
                global_state_dict,
                global_control,
            )

            updates = receive_scaffold_updates(clients)

            ordered_updates = [
                updates[client_id]
                for _, client_id in clients
            ]

            # full participation: average client model deltas equally
            mean_model_delta = average_state_dicts(
                [u["model_delta"] for u in ordered_updates]
            )

            # average control deltas equally
            mean_control_delta = average_state_dicts(
                [u["control_delta"] for u in ordered_updates]
            )

            global_state_dict = {
                name: (
                    # update only trainable parameters in model
                    tensor + mean_model_delta[name]
                    if name in mean_model_delta  
                    else tensor
                )
                for name, tensor in global_state_dict.items()
            }

            global_control = {
                name: tensor + mean_control_delta[name]
                for name, tensor in global_control.items()
            }

            print("SCAFFOLD aggregation completed.")

            test_accuracy = evaluate_global_model(
                global_state_dict=global_state_dict,
                test_dataset=test_dataset,
                batch_size=config.batch_size_test,
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
