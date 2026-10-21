from ..model import MNISTModel
from ..data import load_mnist
from .config import FedProxConfig
from ..fedavg.strategy import aggregate_state_dicts
from ..fedavg.server import (
    prepare_experiment,
    register_clients,
    send_partitions,
    send_global_model,
    receive_local_updates,
    stop_clients,
    evaluate_global_model,
)


def main():
    config = FedProxConfig()

    # prepare_experiment(config) works because of config=FedProxConfig() inherits 
    # the shared configuration fields of FedAvgConfig()
    
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

            # compute the next global model
            global_state_dict = aggregate_state_dicts(
                client_state_dicts,
                client_sizes,
            )

            print("\nFedProx aggregation completed.")
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
