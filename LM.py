from typing import Any, List, Optional, Union

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torch.nn import functional as F
#from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from piq import LPIPS
from consistency_models.consistency_models import (
    ConsistencySamplingAndEditing,
    ConsistencyTraining,
    ema_decay_rate_schedule,
    karras_schedule,
    timesteps_schedule,
)
from cm_vfe import ConsistencyTrainingVFE
from consistency_models.utils import update_ema_model

from diffusers import UNet2DModel
from typing import Tuple
from torch import Tensor
import torch
from utils import log_samples, wasserstein, plot_pred_true
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only
# from torch.distributed import all_gather_into_tensor, scatter, get_rank, get_world_size
from torchmetrics.image.inception import InceptionScore
import numpy as np
import os
from pytorch_fid import fid_score
import imageio
from torchvision.utils import save_image
from utils import rand_hyperplane

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    

class LitConsistencyModelImage(LightningModule):
    def __init__(
        self,
        consistency_training: ConsistencyTrainingVFE,
        consistency_sampling: ConsistencySamplingAndEditing,
        unet: UNet2DModel,
        ema_unet: UNet2DModel,
        initial_ema_decay_rate: float = 0.95,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.5, 0.999),
        lr_scheduler_start_factor: float = 1 / 3,
        lr_scheduler_iters: int = 500,
        sample_every_n_steps: int = 500,
        num_samples: int = 8,
        num_sampling_steps: List[int] = [1, 2, 5],
        max_iters=2041,
        test_NFE: int = 1,
        eval_num_samples: int = 30000,
        eval_dir: str = '',
        ref_dir: str = '',
        target_loader=None,
        source_loader=None
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=["consistency_training",
                    "consistency_sampling", "unet", "ema_unet"]
        )

        self.consistency_training = consistency_training
        self.consistency_sampling = consistency_sampling
        self.unet = unet
        self.ema_unet = ema_unet
        self.initial_ema_decay_rate = initial_ema_decay_rate
        self.lr = lr
        self.betas = betas
        self.lr_scheduler_start_factor = lr_scheduler_start_factor
        self.lr_scheduler_iters = lr_scheduler_iters
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.num_sampling_steps = num_sampling_steps
        self.max_iter = max_iters
        self.test_NFE = test_NFE
        self.eval_num_samples = eval_num_samples
        self.eval_dir = eval_dir
        self.ref_dir = ref_dir

        self.source_loader=source_loader
        self.target_loader = target_loader

        self.lpips_loss = LPIPS(replace_pooling=True)


        
    def training_step(
        self, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> Tensor:
        # # Drop labels if present
        if isinstance(batch, list):
            img, _ = batch[0], batch[1]
            batch = {}
            batch['target'] = img
            if self.consistency_training.cm_type == 'DCM-MC':
                minibatch = self.consistency_training.minibatch
                batch['source'] = torch.randn(torch.Size(
                    [minibatch] + list(img.shape[1:])), device=img.device)

            elif self.consistency_training.cm_type == 'PCM':
                minibatch = self.consistency_training.minibatch
                batch['source'] = rand_hyperplane(torch.Size(
                    [minibatch] + list(img.shape[1:])), device=img.device)

            else:
                batch['source'] = torch.randn_like(img, device=img.device)

        predicted, target = self.consistency_training(
            self.unet, self.ema_unet, batch, self.global_step, self.trainer.max_steps, is_cross_node=True
        )



        if predicted.shape[-1] < 256:
            denoised_x = F.interpolate(predicted, size=224, mode="bilinear")
            target_x = F.interpolate(target, size=224, mode="bilinear")
        else:
            denoised_x = predicted
            target_x = target
        loss = self.lpips_loss(
                (denoised_x + 1) / 2.0,
                (target_x + 1) / 2.0,)


        self.log_dict(
            {
                "lpips_loss": loss,
            }
        )

        # Sample and log samples
        if self.global_step % self.sample_every_n_steps == 0:
            self.__sample_and_log_samples(batch)

        return loss



    def on_train_batch_end(self, *args) -> None:
        # Update the ema model
        num_timesteps = timesteps_schedule(
            self.global_step,
            self.trainer.max_steps,
            initial_timesteps=self.consistency_training.initial_timesteps,
            final_timesteps=self.consistency_training.final_timesteps,
        )
        ema_decay_rate = ema_decay_rate_schedule(
            num_timesteps,
            initial_ema_decay_rate=self.initial_ema_decay_rate,
            initial_timesteps=self.consistency_training.initial_timesteps,
        )
        self.ema_unet = update_ema_model(
            self.ema_unet, self.unet, ema_decay_rate)
        self.log_dict(
            {"num_timesteps": num_timesteps, "ema_decay_rate": ema_decay_rate}
        )

    def configure_optimizers(self):
        opt = optim.RAdam(self.unet.parameters(), lr=self.lr, betas=self.betas, eps=1e-8)
        # sched = optim.lr_scheduler.LinearLR(
        #     opt,
        #     start_factor=self.lr_scheduler_start_factor,
        #     total_iters=self.lr_scheduler_iters,
        # )
        # sched = {"scheduler": sched, "interval": "step"}
        # sched = CosineWarmupScheduler(
        #     optimizer=opt, warmup=1, max_iters=self.max_iter)

        # return [opt], [sched]
        return [opt]

    @torch.no_grad()
    def __sample(self, batch: Tensor) -> Tensor:
        # self.eval_num_samples
        #noise = batch['source']
        noise = batch

        # Sample an extra step and reverse the schedule as the last step (sigma=sigma_min)
        # is useless as the model returns identity
        sigmas = karras_schedule(
            self.test_NFE + 1,
            sigma_min=self.consistency_training.sigma_min,
            sigma_max=self.consistency_training.sigma_max,
            rho=self.consistency_training.rho,
            device=self.device,
            sigma_scale=self.consistency_training.sigma_scale,
        )
        sigmas = sigmas.flipud()[:-1]
        samples = self.consistency_sampling(
            self.ema_unet, noise, sigmas, clip_denoised=True, verbose=True
        )
        samples = samples.clamp(min=-1.0, max=1.0)

        return samples

    @torch.no_grad()
    def __sample_and_log_samples(self, batch: Tensor) -> None:
        # Ensure the number of samples does not exceed the batch size
        
        if len(batch['target']) >= 100:
            noise = batch['source'][:self.num_samples]
            target = batch['target'][:self.num_samples]
        else:
            noise = batch['source']
            target = batch['target']

        # Log ground truth samples
        log_samples(
            self.logger,
            target,
            f"ground_truth",
            self.global_step,
        )

        # Log ground truth samples
        log_samples(
            self.logger,
            noise,
            f"source",
            self.global_step,
        )

        for steps in self.num_sampling_steps:
            # Sample an extra step and reverse the schedule as the last step (sigma=sigma_min)
            # is useless as the model returns identity
            sigmas = karras_schedule(
                steps + 1,
                sigma_min=self.consistency_training.sigma_min,
                sigma_max=self.consistency_training.sigma_max,
                rho=self.consistency_training.rho,
                device=self.device,
                sigma_scale=self.consistency_training.sigma_scale
            )

            sigmas = sigmas.flipud()[:-1]

            samples = self.consistency_sampling(
                self.ema_unet, noise, sigmas, clip_denoised=True, verbose=True
            )
            samples = samples.clamp(min=-1.0, max=1.0)

            # Generated samples
            log_samples(
                self.logger,
                samples,
                f"generated_samples-steps={steps}",
                self.global_step,
            )
    def train_dataloader(self):

        if self.source_loader!=None:
            return {"source": self.source_loader, "target": self.target_loader}
        else:
            return self.target_loader
        
    def test_dataloader(self):
        return self.source_loader
        #return {"source": self.source_loader}

    def on_test_start(self):
        torch.manual_seed(666)
        # Check if the directory exists, create it if not
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        # Calculate IS score
        self.inception_model = InceptionScore().to(self.device)
    
    def test_step(self, batch, batch_idx):
        # Generate samples and scale to range [0, 255]
        batch_size = batch.shape[0]


        if self.consistency_training.cm_type == 'PCM':
            noise = rand_hyperplane(
            batch.shape, device=batch.device)
        else:
            noise = torch.randn_like(batch)


        samples = (self.__sample(noise) * 127.5 + 127.5).to(torch.uint8)
        self.inception_model.update(samples)
        # Convert samples to numpy
        samples_np = samples.permute(0, 2, 3, 1).cpu().numpy()

        # Save samples
        for i, sample in enumerate(samples_np):
            # Define the filename. Include the rank (GPU ID) and sample index.
            filename = f"sample_{batch_idx * batch_size + i}_rank{self.local_rank}.png"

            # Save the image
            imageio.imwrite(os.path.join(self.eval_dir, filename), sample.astype(np.uint8))


        return {}
    
    def on_test_end(self):
        # Calculate FID score
        fid = fid_score.calculate_fid_given_paths([self.eval_dir, self.ref_dir],
                                                  batch_size=128,
                                                  device=self.device,
                                                  dims=2048,
                                                  num_workers=4)

        IS_mean, IS_std = self.inception_model.compute()
        print(f"FID: {fid}, IS: {IS_mean}, IS_std: {IS_std}")

        self.logger.experiment.add_scalar("FID", fid)
        self.logger.experiment.add_scalar("IS", IS_mean)
        self.logger.experiment.add_scalar("IS_std", IS_std)








