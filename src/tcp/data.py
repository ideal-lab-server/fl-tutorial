
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_mnist(data_dir="./data"):
    """Load the standard MNIST training and test datasets."""

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def partition_iid(dataset, num_clients, seed=42):
    """Randomly partition training examples approximately equally."""
    
    if isinstance(num_clients, bool) or not isinstance(num_clients, int):
        raise TypeError("num_clients must be an integer")
    
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    rng = np.random.default_rng(seed)                       # random number generator
    indices = rng.permutation(len(dataset))
    client_indices = np.array_split(indices, num_clients)

    return [indices.tolist() for indices in client_indices]


def partition_dirichlet(dataset, num_clients, alpha=0.5, seed=42):
    """
    Partition examples by label using a Dirichlet distribution.
    Each example is assigned to exactly one client. Client sizes may differ.
    Smaller alpha generally produces greater label heterogeneity.
    """
    
    if isinstance(num_clients, bool) or not isinstance(num_clients, int):
        raise TypeError("num_clients must be an integer")

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    rng = np.random.default_rng(seed)

    labels = np.asarray(dataset.targets)
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for label in range(num_classes):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)

        # Draw the proportion of this label assigned to each client.
        proportions = rng.dirichlet(
            np.full(num_clients, alpha)
        )

        # convert proportions into integer split sizes
        split_points = (
            np.cumsum(proportions)[:-1] * len(label_indices)
        ).astype(int)

        splits = np.split(label_indices, split_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # shuffle each client's examples
    for indices in client_indices:
        rng.shuffle(indices)

    return client_indices


def partition_sorted(dataset, num_clients):
    """Sort examples by label, then distribute equal-sized chunks."""
    
    if isinstance(num_clients, bool) or not isinstance(num_clients, int):
        raise TypeError("num_clients must be an integer")

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    labels = np.asarray(dataset.targets)
    sorted_indices = np.argsort(labels, kind="stable")
    client_indices = np.array_split(sorted_indices, num_clients)

    return [indices.tolist() for indices in client_indices]


def create_partition_indices(
    dataset,
    num_clients=3,
    partition="iid",
    alpha=0.5,
    seed=42,
):
    """Generate sample indices assigned to each client."""

    if partition == "iid":
        return partition_iid(dataset, num_clients, seed)

    elif partition == "dirichlet":
        return partition_dirichlet(
            dataset, num_clients, alpha, seed
        )

    elif partition == "sorted":
        return partition_sorted(dataset, num_clients)

    else:
        raise ValueError(
            "partition must be 'iid', 'dirichlet', or 'sorted'"
        )


def create_client_datasets(train_dataset, client_indices):
    """Create client datasets from explicitly assigned indices."""

    return [
        Subset(train_dataset, indices)
        for indices in client_indices
    ]


def create_test_loader(test_dataset, batch_size=256):
    """Create a common test loader for global-model evaluation."""

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
