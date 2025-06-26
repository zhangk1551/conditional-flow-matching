from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DataType(str, Enum):
    GAUSSIANS_MOONS = 'gaussians_moons'
    MNIST = 'mnist'


class ModelType(str, Enum):
    MLP = 'mlp'
    U_NET = 'u_net'


@dataclass
class DataConfig:
    batch_size: int = 256
    data_type: DataType = DataType.GAUSSIANS_MOONS


@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.MLP


@dataclass
class MeanFlowConfig:
    use_optimal_transport: bool = False
    unequal_ratio: float = 0.25


@dataclass
class TrainerConfig:
    monitor: str = "train_loss"
    monitor_mode: str = "min"
    max_epochs: int = 20
    fast_dev_run: bool = True
    overfit_batches: float = 0.0
    devices: int = 1
    num_nodes: int = 1
    lr: float = 0.001
    optim_method: str = "Adam"


@dataclass
class EvaluationConfig:
    num_train_epoch_per_evaluation: int = 1


@dataclass
class MainConfig:
    data: DataConfig
    model: ModelConfig
    mean_flow: MeanFlowConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    seed: Optional[int] = None
