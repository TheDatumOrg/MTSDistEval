from dataclasses import dataclass
from typing import Optional

@dataclass
class Parameters:
    gpu: int = 0  # The GPU number used for training and inference (defaults to 0)
    batch_size: int = 8  # The batch size (defaults to 8)
    lr: float = 0.001  # The learning rate (defaults to 0.001)
    repr_dims: int = 320  # The representation dimension (defaults to 320)

    max_train_length: int = 3000  # For sequences longer than max_train_length, they will be cropped (defaults to 3000)
    iters: Optional[int] = None  # The number of iterations
    epochs: Optional[int] = None  # The number of epochs
    save_every: Optional[int] = None  # Save the checkpoint every save_every iterations/epochs
    seed: Optional[int] = None  # The random seed
    max_threads: Optional[int] = None  # The maximum allowed number of threads used by this process
    eval: bool = False  # Whether to perform evaluation after training
    irregular: float = 0.0  # The ratio of missing observations (defaults to 0)

    # Cast the parameters to the correct types
    def __post_init__(self):
        self.gpu = int(self.gpu)
        self.batch_size = int(self.batch_size)
        self.lr = float(self.lr)
        self.repr_dims = int(self.repr_dims)
        self.max_train_length = int(self.max_train_length)
        if self.iters is not None:
            self.iters = int(self.iters)
        if self.epochs is not None:
            self.epochs = int(self.epochs)
        if self.save_every is not None:
            self.save_every = int(self.save_every)
        if self.seed is not None:
            self.seed = int(self.seed)
        if self.max_threads is not None:
            self.max_threads = int(self.max_threads)
        self.irregular = float(self.irregular)