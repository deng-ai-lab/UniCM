from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import math
from config.cfg_base import Config

@dataclass()
class ConfigImage(Config):
    # Reproducibility
    seed: int = 0
    devices: List[int] = field(default_factory=lambda: [0])

    # Data Config
    image_size: Tuple[int, int] = (64, 64)
    #data_dir: str = "data/cifar"
    data_dir: str = "data/celeba"
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    minibatch: int = 64

    # Consistency Model Config
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    initial_timesteps: int = 2
    final_timesteps: int = 150
    initial_ema_decay_rate: float = 0.95
    cm_type: str = 'DCM'

    # Lightning Model Config
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_start_factor: float = 1 / 3
    lr_scheduler_iters: int = 500
    sample_every_n_steps: int = 1_000
    num_samples: int = 8
    num_sampling_steps: List[int] = field(default_factory=lambda: [1, 2, 5])

    # Tensorboard Logger
    name: str = "consistency_models"
    #version: str = "cifar10"
    version: str = "Celeba"

    # Checkpoint Callback
    every_n_train_steps: int = 10_000

    # Trainer
    accelerator: str = "auto"
    max_steps: int = 200_001
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 20
    precision: Union[int, str] = 16
    detect_anomaly: bool = False

    # Training Loop
    skip_training: bool = False

    # Model checkpoint
    ckpt_path: str = "checkpoints"
    resume_ckpt_path: Optional[str] = None

    # Test FID/IS
    test_NFE: int = 2
    eval_num_samples: int = 64
    ckpt_name: str = "epoch=78-step=200000.ckpt"
    image_dir: str = "eval"
    ref_dir: str = "assets/stats/celeba_test"



@dataclass()
class ConfigImage2(Config):
    # Reproducibility
    seed: int = 0
    devices: List[int] = field(default_factory=lambda: [0])

    # Data Config
    image_size: Tuple[int, int] = (64, 64)
    # data_dir: str = "data/cifar"
    data_dir: str = "data/celeba"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    minibatch: int = 32

    # Consistency Model Config
    sigma_min: float = 0.0001
    sigma_max: float = 0.9999
    rho: float = 7.0
    sigma_data: float = 0.5
    initial_timesteps: int = 2
    final_timesteps: int = 150
    initial_ema_decay_rate: float = 0.95
    cm_type: str = 'CCM-OT'

    # Lightning Model Config
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_start_factor: float = 1 / 3
    lr_scheduler_iters: int = 500
    sample_every_n_steps: int = 1_000
    num_samples: int = 8
    num_sampling_steps: List[int] = field(default_factory=lambda: [1, 2, 5])

    # Tensorboard Logger
    name: str = "consistency_models"
    #version: str = "cifar10"
    version: str = "Celeba"

    # Checkpoint Callback
    every_n_train_steps: int = 10_000

    # Trainer
    accelerator: str = "auto"
    max_steps: int = 200_001
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 20
    precision: Union[int, str] = 16
    detect_anomaly: bool = False

    # Training Loop
    skip_training: bool = False

    # Model checkpoint
    ckpt_path: str = "checkpoints"
    resume_ckpt_path: Optional[str] = None

    # Test FID/IS
    test_NFE: int = 2
    eval_num_samples: int = 64
    #ckpt_name: str = "epoch=15384-step=200000-v1.ckpt"
    ckpt_name: str = "epoch=78-step=200000.ckpt"
    image_dir: str = "eval"
    ref_dir: str = "assets/stats/celeba_test"



@dataclass()
class ConfigImage3(Config):
    # Reproducibility
    seed: int = 0

    # Data Config
    image_size: Tuple[int, int] = (64, 64)
    #data_dir: str = "data/cifar"
    data_dir: str = "data/celeba"
    batch_size: int = 1024
    num_workers: int = 2
    pin_memory: bool = True
    minibatch: int = 32

    # Consistency Model Config
    sigma_min: float = 0.002 * math.sqrt(128)
    sigma_max: float = 80.0 * math.sqrt(128)
    rho: float = 7.0
    sigma_data: float = 0.5
    initial_timesteps: int = 2
    final_timesteps: int = 150
    initial_ema_decay_rate: float = 0.95
    cm_type: str = 'PCM'

    # Lightning Model Config
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_start_factor: float = 1 / 3
    lr_scheduler_iters: int = 500
    sample_every_n_steps: int = 1_000
    num_samples: int = 8
    num_sampling_steps: List[int] = field(default_factory=lambda: [1, 2, 5])

    # Tensorboard Logger
    name: str = "consistency_models"
    #version: str = "cifar10"
    version: str = "Celeba"

    # Checkpoint Callback
    every_n_train_steps: int = 10_000

    # Trainer
    accelerator: str = "auto"
    max_steps: int = 200_001
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 20
    precision: Union[int, str] = 16
    detect_anomaly: bool = False

    # Training Loop
    skip_training: bool = False

    # Model checkpoint
    ckpt_path: str = "checkpoints"
    resume_ckpt_path: Optional[str] = None





@dataclass()
class ConfigIm2Im2(Config):
    # Reproducibility
    seed: int = 0

    # Data Config
    image_size: Tuple[int, int] = (256, 256)
    data_dir: str = "data/afhq/afhq/"
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True
    source_type: str = 'cat'
    target_type: str = 'dog'

    # Consistency Model Config
    sigma_min: float = 0.0001
    sigma_max: float = 0.9999
    rho: float = 7.0
    sigma_data: float = 0.5
    initial_timesteps: int = 2
    final_timesteps: int = 150
    initial_ema_decay_rate: float = 0.95
    cm_type: str = 'CCM-OT'
    on_latent = True

    # Lightning Model Config
    lr: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_start_factor: float = 1 / 3
    lr_scheduler_iters: int = 500
    sample_every_n_steps: int = 1_000
    num_samples: int = 8
    num_sampling_steps: List[int] = field(default_factory=lambda: [1, 2, 5])

    # Tensorboard Logger
    name: str = "consistency_models"
    version: str = "AFHQ"

    # Checkpoint Callback
    every_n_train_steps: int = 10_000

    # Trainer
    accelerator: str = "auto"
    max_steps: int = 200000
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 20
    precision: Union[int, str] = 16
    detect_anomaly: bool = False

    # Training Loop
    skip_training: bool = False

    # Model checkpoint
    ckpt_path: str = "checkpoints"
    resume_ckpt_path: Optional[str] = None