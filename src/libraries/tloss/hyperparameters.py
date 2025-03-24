from dataclasses import dataclass
from typing import Optional
from torch.cuda import is_available
import json

@dataclass
class Hyperparameters:
    load_model: bool = False # activate to load the model
    cuda: bool = is_available() # activate to use CUDA
    gpu: int = 0  # index of GPU used for computations (default: 0)
    batch_size: int = 20  # batch size for training
    channels: int = 40 # number of channels in the network
    compared_length: Optional[int] = None  # length to which the time series are compared
    depth: int = 10 # depth of the network
    nb_steps: int = 150  # number of training steps
    in_channels: Optional[int] = None  # number of input channels
    out_channels: int = 320 # embedding size
    kernel_size: int = 3
    penalty: Optional[float] = None
    lr: float = 0.001 # learning rate
    nb_random_samples: int = 10  # number of random samples
    negative_penalty: int = 1
    reduced_size: int = 160
    early_stopping: int = 1 # number of epochs without improvement before stopping

    # Cast the parameters to the correct types
    def __post_init__(self):
        self.gpu = int(self.gpu)
        self.batch_size = int(self.batch_size)
        self.channels = int(self.channels)
        self.depth = int(self.depth)
        self.nb_steps = int(self.nb_steps)
        self.out_channels = int(self.out_channels)
        self.kernel_size = int(self.kernel_size)
        self.lr = float(self.lr)
        self.nb_random_samples = int(self.nb_random_samples)
        self.negative_penalty = int(self.negative_penalty)
        self.reduced_size = int(self.reduced_size)
        self.early_stopping = int(self.early_stopping)

    def to_json(self) -> dict:
        return self.__dict__

    @classmethod
    def from_json(cls, json_dict: dict):
        return cls(**json_dict)

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r') as f:
            json_dict = json.load(f)
        return cls.from_json(json_dict)
    