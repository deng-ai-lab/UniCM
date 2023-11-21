import lightning as L
import matplotlib
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from utils import save_model_ckpt
from config.cfg_syndata import ConfigSyndata
from config.cfg_image import ConfigImage
from dataset.image import CifarDataModule, transform_fn, afhqDataModule, CelebaDataModule
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
def run_training_image(config: ConfigImage) -> None:
    # -------------------------------------------
    # Reproducibility
    # -------------------------------------------
    # L.seed_everything(config.seed)

    # -------------------------------------------
    # Configure Matplotlib
    # -------------------------------------------
    # Prevents pixelated fonts on figures
    matplotlib.use("webagg")
    matplotlib.style.use(["ggplot", "fast"])

    # -------------------------------------------
    # Data 
    # -------------------------------------------
    transform = transform_fn(config.image_size)

    if config.version == 'cifar10':

        datamodule = CifarDataModule(
            config.data_dir,
            transform=transform,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    elif config.version == 'Celeba':

        datamodule = CelebaDataModule(
            config.data_dir,
            transform=transform,
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
        on_latnet=False,
        minibatch=config.minibatch
    )
    consistency_sampling = ConsistencySamplingVFE(cm_type=config.cm_type,
        sigma_min=config.sigma_min, sigma_data=config.sigma_data
    )
    unet = UNet(config.image_size)
    ema_unet = UNet(config.image_size)
    ema_unet.load_state_dict(unet.state_dict())
    for param in ema_unet.parameters():
        param.requires_grad = False

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
        #max_iters=2041
    )
    # -----------------------------------------
    # Trainer
    # ------------------------------------------
    logger = TensorBoardLogger(f'logs/{config.cm_type}',name=f'{config.version}')
    checkpoint_callback = ModelCheckpoint(config.ckpt_path+f'/{config.cm_type}/{config.version}',save_top_k = -1,
        every_n_train_steps=config.every_n_train_steps
    )
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        devices=4,
        accelerator=config.accelerator,
        max_steps=config.max_steps,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
        precision=config.precision,
        detect_anomaly=config.detect_anomaly,
        benchmark=True,
        strategy='ddp'
    )

    # -----------------------------------------
    # Run Training
    # ------------------------------------------
    if not config.skip_training:
        train_loader=datamodule.train_dataloader()
        trainer.fit(
            lit_consistency_model,
            train_dataloaders=train_loader,
            ckpt_path=config.resume_ckpt_path,
        )

    # -------------------------------------------
    # Save Checkpoint
    # -------------------------------------------
    save_model_ckpt(lit_consistency_model.unet, config.ckpt_path+f'/{config.cm_type}/{config.version}/'+'net.pth')
    save_model_ckpt(lit_consistency_model.ema_unet,  config.ckpt_path+f'/{config.cm_type}/{config.version}/'+'ema_net.pth')







# ----------------------------------------------------------------------------------
# Run image to image on afhq, e.g. cat->dog
# ----------------------------------------------------------------------------------
def run_training_im2im(config: ConfigImage) -> None:
    # -------------------------------------------
    # Reproducibility
    # -------------------------------------------
    #L.seed_everything(config.seed)

    # -------------------------------------------
    # Configure Matplotlib
    # -------------------------------------------
    # Prevents pixelated fonts on figures
    # matplotlib.use("webagg")
    # matplotlib.style.use(["ggplot", "fast"])

    # -------------------------------------------
    # Data 
    # -------------------------------------------
    transform = transform_fn(config.image_size)

    datamodule_source = afhqDataModule(
        config.data_dir,
        target=config.source_type,
        transform=transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    datamodule_target = afhqDataModule(
        config.data_dir,
        target=config.target_type,
        transform=transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    datamodule_source.setup()
    datamodule_target.setup()
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
        on_latnet=config.on_latent
    )
    consistency_sampling = ConsistencySamplingVFE(cm_type=config.cm_type,
        sigma_min=config.sigma_min, sigma_data=config.sigma_data
    )
    unet = UNet(config.image_size)
    ema_unet = UNet(config.image_size)
    ema_unet.load_state_dict(unet.state_dict())
    for param in ema_unet.parameters():
        param.requires_grad = False

    # -----------------------------------------
    # Lit Model
    # ------------------------------------------
    train_source_loader=datamodule_source.train_dataloader()
    train_target_loader=datamodule_target.train_dataloader()

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
        source_loader=train_source_loader,
        target_loader=train_target_loader
    )
    # -----------------------------------------
    # Trainer
    # ------------------------------------------
    logger = TensorBoardLogger(f'logs/{config.cm_type}',name=f'{config.source_type}-{config.target_type}')
    checkpoint_callback = ModelCheckpoint(config.ckpt_path+f'/{config.cm_type}/{config.source_type}-{config.target_type}', save_top_k = -1,
        every_n_train_steps=config.every_n_train_steps,
    )
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        devices=4,
        accelerator=config.accelerator,
        max_steps=config.max_steps,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
        precision=config.precision,
        detect_anomaly=config.detect_anomaly,
        benchmark=True,
        strategy='ddp'
    )

    # -----------------------------------------
    # Run Training
    # ------------------------------------------
    if not config.skip_training:

        trainer.fit(
            lit_consistency_model,
            ckpt_path=config.resume_ckpt_path,
        )

    # -------------------------------------------
    # Save Checkpoint
    # -------------------------------------------
    save_model_ckpt(lit_consistency_model.unet, config.ckpt_path+f'/{config.cm_type}/{config.source_type}-{config.target_type}/'+'net.pth')
    save_model_ckpt(lit_consistency_model.ema_unet,  config.ckpt_path+f'/{config.cm_type}/{config.source_type}-{config.target_type}/'+'ema_net.pth')









