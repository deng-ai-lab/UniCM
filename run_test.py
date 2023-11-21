import lightning as L
import matplotlib
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from utils import save_model_ckpt
from config.cfg_syndata import ConfigSyndata
from config.cfg_image import ConfigImage, ConfigIm2Im
from dataset.image import CifarDataModule, transform_fn, afhqDataModule, NoiseDataset, NoiseDataModule, transform_fn_eval
from dataset.syn_data import SynDataModule
from models.unet import UNet
from models.mlp import TimedepnedSELUMLP, TimedepnedReLUMLP, TimedepnedSiLUMLP
from consistency_models.consistency_models import ConsistencyTraining, ConsistencySamplingAndEditing
from LM import LitConsistencyModelImage, LitConsistencyModelSyndata
from cm_vfe import ConsistencyTrainingVFE, ConsistencySamplingVFE
import torchvision
from pytorch_lightning.strategies.ddp import DDPStrategy


# ----------------------------------------------------------------------------------
# Run image data, e.g. cifar10
# ----------------------------------------------------------------------------------
def run_testing_image(config: ConfigImage) -> None:
    # -------------------------------------------
    # Reproducibility
    # -------------------------------------------
    #L.seed_everything(config.seed)

    # -------------------------------------------
    # Configure Matplotlib
    # -------------------------------------------
    # Prevents pixelated fonts on figures
    matplotlib.use("webagg")
    matplotlib.style.use(["ggplot", "fast"])

    # -------------------------------------------
    # Data
    # -------------------------------------------
    # Calculate length of dataset
    dataset_length = config.eval_num_samples // len(config.devices)
    # Define the noise shape
    noise_shape = (3, *config.image_size)

    datamodule = NoiseDataModule(
        length=dataset_length,
        noise_shape=noise_shape,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    datamodule.setup()

    # -----------------------------------------
    # Models
    # ------------------------------------------
    consistency_training = ConsistencyTrainingVFE(
        cm_type=config.cm_type,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
        sigma_data=config.sigma_data,
        initial_timesteps=config.initial_timesteps,
        final_timesteps=config.final_timesteps,
    )
    consistency_sampling = ConsistencySamplingVFE(cm_type=config.cm_type,
                                                  sigma_min=config.sigma_min, sigma_data=config.sigma_data
                                                  )
    unet = UNet(config.image_size)
    ema_unet = UNet(config.image_size)
    ema_unet.load_state_dict(unet.state_dict())

    # -----------------------------------------
    # Lit Model
    # ------------------------------------------
    lit_consistency_model = LitConsistencyModelImage(
        consistency_training,
        consistency_sampling,
        unet,
        ema_unet,
        initial_ema_decay_rate=config.initial_ema_decay_rate,
        lr=config.lr,
        betas=config.betas,
        lr_scheduler_start_factor=config.lr_scheduler_start_factor,
        lr_scheduler_iters=config.lr_scheduler_iters,
        sample_every_n_steps=config.sample_every_n_steps,
        num_samples=config.num_samples,
        num_sampling_steps=config.num_sampling_steps,
        test_NFE=config.test_NFE,
        eval_num_samples=config.eval_num_samples,
        eval_dir=config.ckpt_path + f'/{config.cm_type}/{config.version}/{config.image_dir}',
        ref_dir=config.ref_dir,
        source_loader=datamodule.test_dataloader()
    )
    # -----------------------------------------
    # Trainer
    # ------------------------------------------
    logger = TensorBoardLogger(f'logs/{config.cm_type}', name=f'{config.version}')

    trainer = Trainer(
        logger=logger,
        devices=config.devices,
        accelerator=config.accelerator,
        max_epochs=1,
        log_every_n_steps=config.log_every_n_steps,
        precision=32,
        benchmark=True,
    )

    # -----------------------------------------
    # Run Testing
    # ------------------------------------------
    # test_loader =
    trainer.test(
        lit_consistency_model,
        # dataloaders=test_loader,
        ckpt_path=config.ckpt_path + f'/{config.cm_type}/{config.version}/{config.ckpt_name}',
    )





def run_testing_im2im(config: ConfigIm2Im) -> None:
    # -------------------------------------------
    # Reproducibility
    # -------------------------------------------
    L.seed_everything(config.seed)

    # -------------------------------------------
    # Configure Matplotlib
    # -------------------------------------------
    # Prevents pixelated fonts on figures
    matplotlib.use("webagg")
    matplotlib.style.use(["ggplot", "fast"])

    # -------------------------------------------
    # Data
    # -------------------------------------------

    transform = transform_fn_eval(config.image_size)
    datamodule_source = afhqDataModule(
        config.data_dir,
        target=config.source_type,
        transform=transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    datamodule_source.setup()

    # -----------------------------------------
    # Models
    # ------------------------------------------
    consistency_training = ConsistencyTrainingVFE(
        cm_type=config.cm_type,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
        sigma_data=config.sigma_data,
        initial_timesteps=config.initial_timesteps,
        final_timesteps=config.final_timesteps,
    )
    consistency_sampling = ConsistencySamplingVFE(cm_type=config.cm_type,
                                                  sigma_min=config.sigma_min, sigma_data=config.sigma_data
                                                  )
    unet = UNet(config.image_size)
    ema_unet = UNet(config.image_size)

    # -----------------------------------------
    # Lit Model
    # ------------------------------------------
    is_latent = 'latent' if getattr(config, 'on_latent', False) else 'nolatent'
    lit_consistency_model = LitConsistencyModelImage(
        consistency_training,
        consistency_sampling,
        unet,
        ema_unet,
        initial_ema_decay_rate=config.initial_ema_decay_rate,
        lr=config.lr,
        betas=config.betas,
        lr_scheduler_start_factor=config.lr_scheduler_start_factor,
        lr_scheduler_iters=config.lr_scheduler_iters,
        sample_every_n_steps=config.sample_every_n_steps,
        num_samples=config.num_samples,
        num_sampling_steps=config.num_sampling_steps,
        test_NFE=config.test_NFE,
        eval_dir=config.ckpt_path + f'/{config.cm_type}/{config.source_type}-{config.target_type}'
                                    f'/{config.image_dir}',
        #global_seed=config.seed,
        source_loader=datamodule_source.val_dataloader(),
        #metric_compute=False
    )
    # -----------------------------------------
    # Trainer  
    # ------------------------------------------
    logger = TensorBoardLogger(f'logs/{config.cm_type}', name=f'{config.source_type}-{config.target_type}-'
                                                              f'{is_latent}-{config.image_size[0]}')

    trainer = Trainer(
        logger=logger,
        devices=1,
        accelerator=config.accelerator,
        max_epochs=1,
        log_every_n_steps=config.log_every_n_steps,
        precision=config.precision,
        benchmark=True,
        strategy='ddp'
    )

    # -----------------------------------------
    # Run Testing
    # ------------------------------------------
    trainer.test(
        lit_consistency_model,
        ckpt_path=config.ckpt_path + f'/{config.cm_type}/{config.source_type}-{config.target_type}'
                                     f'/{config.ckpt_name}',
    )
