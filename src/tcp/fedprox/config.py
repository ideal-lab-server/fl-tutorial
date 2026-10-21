from dataclasses import dataclass
from ..config import CommonConfig

@dataclass
class FedProxConfig(CommonConfig):
    mu: float = 0.1

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.mu, bool) or not isinstance(
            self.mu, (int, float)
        ):
            raise TypeError("mu must be a number")

        if self.mu < 0:
            raise ValueError("mu must be non-negative")
