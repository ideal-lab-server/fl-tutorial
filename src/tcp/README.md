# IIT Kanpur IDEAL Lab

## Federated Learning Tutorial: TCP-Based Implementations

This directory contains implementations of federated learning algorithms for MNIST digit classification using PyTorch and Python TCP sockets. Unlike the implementations in `src/basic/`, which simulate all clients within a single Python process, these implementations run the server and clients as separate processes and communicate through an explicit client-server protocol.

The tutorial is designed to help students understand both the optimization algorithms underlying federated learning and the communication mechanisms required to implement a federated learning system from scratch, without relying on a high-level federated learning framework such as Flower.

## Overview

- **Client-server architecture:** A central server coordinates multiple clients. The server and clients run as independent Python processes and can execute on the same machine or on different machines connected through a network.
- **Dataset preparation:** MNIST is used for digit classification. The server supports `iid`, `dirichlet`, and `sorted` strategies for partitioning data among clients. The server does not send MNIST images over the network. Instead, it sends partition indices, allowing clients to select their assigned subsets from their own downloaded copies of the dataset.
- **TCP communication:** Clients and the server exchange messages using TCP sockets. Shared communication utilities handle message framing, serialization, and deserialization.
- **Local training:** In each round, clients receive the current global model and perform local training using their assigned portions of the dataset. Local training can use a CPU or a compatible CUDA-enabled GPU, depending on the configuration and available hardware. Clients send their updated model parameters to the server.
- **Server aggregation:** After receiving updates from all clients, the server computes the next global model according to the selected algorithm. FedAvg and FedProx use the same server-side aggregation strategy. SCAFFOLD additionally maintains and updates control variates to mitigate client drift.
- **Evaluation:** At the end of each federated round, the global model is evaluated on the test dataset, and its classification accuracy is reported. 

## Directory Structure

```text
tcp/
├── README.md
├── config.py
├── common.py
├── protocol.py
├── serialization.py
├── model.py
├── data.py
├── fedavg/
│   ├── config.py
│   ├── client.py
│   ├── server.py
│   └── strategy.py
├── fedprox/
│   ├── config.py
│   ├── client.py
│   └── server.py
└── scaffold/
    ├── config.py
    ├── client.py
    ├── server.py
    └── strategy.py
```

The main responsibilities of the shared modules are:

- **`config.py`:** Common training parameters.
- **`common.py`:** Common network parameters.
- **`protocol.py`:** Definitions and handling of messages exchanged during federated training.
- **`serialization.py`:** Serialization and deserialization of transmitted data, including model parameters.
- **`model.py`:** Definition of the MNIST classification MLP.
- **`data.py`:** Dataset preparation, client data partitioning, and related data utilities.

Each algorithm-specific directory contains its client and server implementations, along with the strategy module. FedProx reutilizes the strategy module of FedAvg. 

## Requirements

The implementations require Python and the following packages:

- PyTorch
- Torchvision
- NumPy

Install the required packages in your Python environment:

```bash
pip install torch torchvision numpy
```

For GPU training, install a PyTorch build compatible with the CUDA environment on the machine. CPU training does not require a CUDA-enabled GPU.

## Running the Code

Activate a Python virtual environment and run the following commands from the root directory of the `fl-tutorial` repository.

### 1. Start the server

For FedAvg, run:

```bash
python -m src.tcp.fedavg.server
```

### 2. Start the clients

Open a separate terminal for each client and run:

```bash
python -m src.tcp.fedavg.client
```

The number of clients is specified by the `num_clients` variable in `src/tcp/config.py`. Ensure that the number of client processes matches this value. The server waits until the configured number of clients has registered before starting training.

### 3. Run FedProx

Start the FedProx server:

```bash
python -m src.tcp.fedprox.server
```

Then launch the clients in separate terminals:

```bash
python -m src.tcp.fedprox.client
```

### 4. Run SCAFFOLD

Start the SCAFFOLD server:

```bash
python -m src.tcp.scaffold.server
```

Then launch the clients in separate terminals:

```bash
python -m src.tcp.scaffold.client
```

Run only one algorithm's server for a given experiment, and launch the corresponding clients. The server and clients must use compatible configurations, including network settings and training parameters. By default, the server address is configured as the loopback address `127.0.0.1`, which is suitable when all processes run on the same machine.

## Running on Multiple Machines

The server and clients can run on separate computers connected through the same local area network (LAN). 

### Server Machine

1. Identify the server machine's LAN IP address. On Linux, use:

   ```bash
   ip addr
   ```

   On macOS, use:

   ```bash
   ifconfig
   ```

   On Windows, use:

   ```powershell
   ipconfig
   ```

2. Change `DEFAULT_HOST` in `src/tcp/common.py` to the LAN IP address of the server. If necessary, change `DEFAULT_PORT` in `src/tcp/common.py`.
3. Ensure that the server's firewall permits incoming TCP connections on the configured port.
4. Start the server.

### Client Machines

1. Ensure that each client machine can reach the server over the network.
2. Configure the client's `DEFAULT_HOST` and `DEFAULT_PORT` to point to the server's reachable LAN IP address and listening port. 
3. Start the required number of client processes using the appropriate algorithm-specific entry point.

For example, if the server is reachable at `192.168.1.100` and listens on port `5000`, clients should connect to `192.168.1.100:5000`. Replace this example address with the actual server address and configured port.

## Communication Protocol

The TCP implementation separates the federated learning algorithm from the mechanics of transmitting data. The communication and training workflow for FedAvg and FedProx consists broadly of the following steps:

1. **Client registration:** Clients send a `REGISTER` request to the server. The server responds with a `REGISTER_ACK` message containing the unique client ID assigned to that client. The server waits for the configured number of clients to register before proceeding.
2. **Data partitioning:** After registration, the server sends a `PARTITION_ASSIGNMENT` message to each client containing its assigned data indices. These indices define logical, non-overlapping MNIST partitions among clients.
3. **Global model distribution:** The server sends a `GLOBAL_MODEL` message containing the global model parameters to each client.
4. **Local training and update transmission:** Each client trains its model using its assigned portion of the dataset and sends its updated parameters to the server in a `LOCAL_UPDATE` message.
5. **Server aggregation:** The server collects updates from all required clients and computes the next global model. The performance of the computed model is evaluated by obtaining its prediction accuracy in the test dataset. The model is then ready to be distributed in the next federated round. 
6. **Termination:** After the configured number of federated rounds, the server sends a `STOP` message to the clients to terminate their local processes. The server process is then terminated.

The SCAFFOLD workflow additionally exchanges control variates:

- In addition to `GLOBAL_MODEL`, the server sends a `GLOBAL_CONTROL` message containing the global control variates.
- The clients send their local model-deltas via `LOCAL_UPDATE`, followed by the transmission of their local control-deltas via `CONTROL_UPDATE` message.

## From Message to Bytes

Since TCP provides a reliable byte stream but does not preserve application-level message boundaries, application-level framing is required to delimit each message. Each application message consists of three parts:

- **Prefix:** A fixed-length 8-byte field describing the lengths of the header and payload.
- **Header:** A variable-length dictionary containing metadata such as message type, protocol version, and client ID.
- **Payload:** Typically used to transmit model state dictionaries and control variates.

A model's or control variate's state dictionary is converted into a byte string using the `serialize_state_dict()` function in the serialization module and included in the payload. The header is also serialized before the complete message is transmitted through the TCP socket. At the receiving end, the reverse operations reconstruct the application message.

## Current Limitations

1. **No timeouts:** There are currently no timeouts for client registration or for receiving client updates. If a configured client fails to connect, register, or return its update, the server may remain blocked while waiting for that client.
2. **Full client participation:** The implementation uses synchronous federated training. The server waits for all configured clients to register before training and for all required client updates before aggregation in each round. Partial client participation and asynchronous training are not currently implemented.
3. **Logical data partitioning:** Clients use assigned indices to select subsets of their own downloaded copies of MNIST. This arrangement simulates data locality but does not itself provide privacy guarantees.
4. **No built-in wireless channel simulation:** Experiments involving imposed latency, bandwidth limits, or wireless-channel effects require a channel simulator, which is not included in the current framework.

## Suggested Experiments

### Federated Optimization

1. **Compare algorithms:** Compare FedAvg, FedProx, and SCAFFOLD under comparable experimental settings.
2. **Investigate client data heterogeneity:** Compare IID and non-IID client partitions, including `dirichlet` and `sorted` distributions.
3. **Study client drift:** Examine how local training and heterogeneous client data affect performance, and investigate how FedProx and SCAFFOLD address client drift.
4. **Vary the number of clients:** Investigate the effect of client count on final accuracy and training time.
5. **Vary local training and communication rounds:** Change the number of local epochs and federated rounds to study their effects on performance.
6. **Compare optimizers:** Investigate the effect of SGD and Adam on convergence, paying particular attention to SCAFFOLD's control-variate updates and the implementation-dependent effects of changing optimizers.
7. **Compare centralized and federated learning:** Compare the performance of centralized training with that of federated algorithms. 

### Communication and Networking

8. **Measure communication overhead:** Record the size of transmitted model updates, the number of messages exchanged, and the total communication volume over training rounds.
9. **Study network effects:** Develop a channel simulator to investigate how latency, bandwidth limitations, and connection failures affect training. Distinguish simulated conditions from measurements made over an actual network.
10. **Investigate synchronization failures:** Examine what happens when a client fails to register or stops responding during a round. Extend the implementation with timeouts or failure-handling mechanisms and evaluate their effects.

### Computational Resources

11. **Study hardware effects:** Compare CPU and GPU execution and run equivalent experiments with all processes on one machine and with clients distributed across multiple machines. Record wall-clock training time and document how available CPU cores and GPUs affect performance.

Experiments can be conducted by modifying the appropriate configuration parameters and extending the existing client, server, or strategy implementations.

---

**IIT Kanpur — IDEAL Lab**  
*Intelligent Decision and Algorithmic Learning*
