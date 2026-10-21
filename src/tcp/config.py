from dataclasses import dataclass
import torch

@dataclass
class CommonConfig:
    num_clients: int = 3

    partition: str = "iid"
    alpha: float = 0.5
    seed: int = 42

    num_rounds: int = 5
    local_epochs: int = 10
    batch_size_train: int = 64
    batch_size_test: int = 256
    learning_rate: float = 0.01

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        
        if isinstance(self.num_clients, bool) or not isinstance(self.num_clients, int):
            raise TypeError("num_clients must be an integer")
        
        if self.num_clients < 1:
            raise ValueError("num_clients must be at least 1")

        if self.partition not in {"iid", "dirichlet", "sorted"}:
            raise ValueError("partition must be 'iid', 'dirichlet', or 'sorted'")

        if self.partition == "dirichlet" and self.alpha <= 0:
            raise ValueError("alpha must be positive for Dirichlet partitioning")
        
        if isinstance(self.num_rounds, bool) or not isinstance(self.num_rounds, int):
            raise TypeError("num_rounds must be an integer")

        if self.num_rounds < 1:
            raise ValueError("num_rounds must be at least 1")

        if isinstance(self.local_epochs, bool) or not isinstance(self.local_epochs, int):
            raise TypeError("local_epochs must be an integer")

        if self.local_epochs < 1:
            raise ValueError("local_epochs must be at least 1")
        
        if isinstance(self.batch_size_train, bool) or not isinstance(self.batch_size_train, int):
            raise TypeError("batch_size_train must be an integer")

        if self.batch_size_train < 1:
            raise ValueError("batch_size_train must be at least 1")
        
        if isinstance(self.batch_size_test, bool) or not isinstance(self.batch_size_test, int):
            raise TypeError("batch_size_test must be an integer")

        if self.batch_size_test < 1:
            raise ValueError("batch_size_test must be at least 1")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")