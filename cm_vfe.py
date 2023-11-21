from consistency_models.consistency_models import timesteps_schedule, karras_schedule, pad_dims_like, model_forward_wrapper
import torch
from torch import Tensor, nn
import torch.nn as nn
from typing import Callable, Iterable, Optional, Tuple, Any, Union
from OT import OTPlanSampler
from tqdm import tqdm
from torch.distributed import all_gather_into_tensor, scatter, get_rank, get_world_size, gather
import numpy as np
from utils import rand_hyperplane

class ConsistencyTrainingVFE():
    def __init__(self, cm_type, sigma_min: float = 0.001, sigma_max: float = 0.999, rho: float = 7, sigma_data: float = 0.5,
                 initial_timesteps: int = 2, final_timesteps: int = 150, on_latnet: bool = False, minibatch: int = 100,) -> None:
        super().__init__(sigma_min, sigma_max, rho,
                         sigma_data, initial_timesteps, final_timesteps)
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.cm_type = cm_type

        if self.cm_type == 'CCM-OT':
            self.ot_sampler = OTPlanSampler(
                method='exact', on_latent=on_latnet)
        else:
            self.ot_sampler = None

        if self.cm_type != 'DCM' and self.cm_type != 'DCM-MS' and self.cm_type != 'PCM':
            self.sigma_scale = 80.0
        else:
            self.sigma_scale = 1.0
        self.minibatch = minibatch

    def __call__(
        self,
        online_model: nn.Module,
        ema_model: nn.Module,
        batch: Tensor,
        current_training_step: int,
        total_training_steps: int,
        sigma_fix: float = 0.00,
        is_cross_node: bool = False,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Runs one step of the consistency training algorithm.

        Parameters
        ----------
        online_model : nn.Module
            Model that is being trained.
        ema_model : nn.Module
            An EMA of the online model.
        x : Tensor
            Clean data.
        current_training_step : int
            Current step in the training loop.
        total_training_steps : int
            Total number of steps in the training loop.
        sigmafix : float
            THe hyperparamter sigma for nogaussion methods.
        **kwargs : Any
            Additional keyword arguments to be passed to the models.

        Returns
        -------
        (Tensor, Tensor)
            The predicted and target values for computing the loss.
        """
        num_timesteps = timesteps_schedule(
            current_training_step,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, batch[
                'source'].device, sigma_scale=self.sigma_scale
        )

        timesteps = torch.randint(
            0, num_timesteps - 1, (batch['source'].shape[0],), device=batch['source'].device)
        current_sigmas = sigmas[timesteps]
        next_sigmas = sigmas[timesteps + 1]

        if self.cm_type == 'DCM':

            noise = batch['source']
            x = batch['target']
            ut = noise
            next_x = x + pad_dims_like(next_sigmas, x) * noise
            # current_x = x + pad_dims_like(current_sigmas, x) * noise
            current_x = next_x + \
                pad_dims_like(current_sigmas-next_sigmas, x) * ut

        elif self.cm_type == 'DCM-MS':

            ref_image = batch['target']
            x = ref_image[:self.minibatch]
            noise = batch['source']
            next_x = x + pad_dims_like(next_sigmas, x) * noise
            ut = self.cal_batch_ut(next_x, next_sigmas, ref_image)
            current_x = next_x + \
                pad_dims_like(current_sigmas-next_sigmas, x) * ut

        elif self.cm_type == 'PCM':
            ref_image = batch['target']
            x = ref_image[:self.minibatch]
            #noise = rand_hyperplane(x.shape,device=x.device)
            noise = batch['source']
            next_x = x + pad_dims_like(next_sigmas, x) * noise

            ut = self.cal_poission_ut(
                next_x, next_sigmas, ref_image)

            current_x = next_x + \
                pad_dims_like(current_sigmas-next_sigmas, x) * ut

        elif self.cm_type == 'CCM':
            noise = batch['source']
            x = batch['target']
            sigma_scale = 80.0
            ut = noise-x
            next_x = (1-pad_dims_like(next_sigmas/sigma_scale, x)) * x + pad_dims_like(next_sigmas/sigma_scale,
                                                                                       x) * noise + sigma_fix * torch.randn_like(x, device=x.device)
            current_x = next_x + \
                pad_dims_like((current_sigmas-next_sigmas)/sigma_scale, x) * ut

        elif self.cm_type == 'CCM-OT':
            noise = batch['source']
            x = batch['target']
            sigma_scale = 80.0

            if is_cross_node and get_world_size() > 1:
                world_size = get_world_size()
                with torch.no_grad():
                    if get_rank() == 0:

                        x_gather = [torch.zeros(*x.shape, device=x.device)
                                    for _ in range(world_size)]
                        noise_gather = [torch.zeros(
                            *noise.shape, device=noise.device) for _ in range(world_size)]
                    else:
                        x_gather = []
                        noise_gather = []

                    gather(x, x_gather,  dst=0)
                    gather(noise, noise_gather, dst=0)
                    if get_rank() == 0:
                        x_gather = torch.cat(x_gather, dim=0)
                        noise_gather = torch.cat(noise_gather, dim=0)

                        x_gather, noise_gather = self.ot_sampler.sample_plan(
                            x_gather, noise_gather)
                        x_gather = x_gather.view(world_size, *x.shape)
                        noise_gather = noise_gather.view(
                            world_size, *noise.shape)
                        x_gather = [x_i for x_i in x_gather]
                        noise_gather = [n_i for n_i in noise_gather]
                    else:
                        x_gather = None
                        noise_gather = None
                    # Synchronize all processes

                    scatter(x, x_gather, src=0)
                    scatter(noise, noise_gather, src=0)

                    del x_gather
                    del noise_gather
            else:
                x, noise = self.ot_sampler.sample_plan(x, noise)

            ut = noise-x
            next_x = (1-pad_dims_like(next_sigmas/sigma_scale, x)) * x + pad_dims_like(next_sigmas/sigma_scale,
                                                                                       x) * noise + sigma_fix * torch.randn_like(x, device=x.device)
            current_x = next_x + \
                pad_dims_like((current_sigmas-next_sigmas)/sigma_scale, x) * ut

        next_x = model_forward_wrapper(
            online_model,
            next_x,
            next_sigmas,
            self.sigma_data,
            self.sigma_min,
            **kwargs,
        )

        with torch.no_grad():
            current_x = model_forward_wrapper(
                ema_model,
                current_x,
                current_sigmas,
                self.sigma_data,
                self.sigma_min,
                **kwargs,
            )

        return (next_x, current_x)

    def cal_batch_ut(self, perturbed_samples, sigmas, ref):
        with torch.no_grad():
            # minibatch, N, H, W -> minibatch, N*H*W
            perturbed_samples_vec = perturbed_samples.reshape(
                (len(perturbed_samples), -1))
            # batch, N, H, W -> batch, N*H*W
            ref_vec = ref.reshape((len(ref), -1))

            # minibatch, batch
            gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - ref_vec) ** 2,
                                    dim=[-1])
            gt_distance = - gt_distance / (2 * sigmas.unsqueeze(1) ** 2)
            # adding a constant to the log-weights to prevent numerical issue
            distance = - torch.max(gt_distance, dim=1,
                                   keepdim=True)[0] + gt_distance
            distance = torch.exp(distance)[:, :, None]
            # self-normalize the per-sample weight of reference batch
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))

            # minibatch, 1(batch), N*H*W - batch, N*H*W ->minibatch, batch, N*H*W
            target_per_sample = perturbed_samples_vec.unsqueeze(1)-ref_vec

            target = (target_per_sample /
                      pad_dims_like(sigmas, target_per_sample))
            # calculate the  targets with reference batch
            stable_targets = torch.sum(weights * target, dim=1)
            return stable_targets.view_as(perturbed_samples)

    def cal_poission_ut(self, perturbed_samples, sigmas, ref):
        data_dim = torch.prod(torch.tensor(perturbed_samples.shape[1:]))

        with torch.no_grad():
            # minibatch, N, H, W -> minibatch, N*H*W
            perturbed_samples_vec = perturbed_samples.reshape(
                (len(perturbed_samples), -1))
            perturbed_samples_vec = torch.cat(
                (perturbed_samples_vec, sigmas[...,None]), dim=-1)
            # batch, N, H, W -> batch, N*H*W
            ref_vec = ref.reshape((len(ref), -1))
            augment_dim_zero = torch.zeros((len(ref_vec),1),device=ref_vec.device)
            ref_vec = torch.cat((ref_vec, augment_dim_zero), dim=-1)

            # minibatch, batch
            gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - ref_vec) ** 2,
                                    dim=[-1]).sqrt()

            # For numerical stability, timing each row by its minimum value
            distance = torch.min(gt_distance, dim=1, keepdim=True)[
                0] / (gt_distance + 1e-7)
            distance = distance ** (data_dim + 1)
            distance = distance[:, :, None]

            # self-normalize the per-sample weight of reference batch
            weights = distance / \
                (torch.sum(distance, dim=1, keepdim=True) + 1e-7)

            # minibatch, 1(batch), N*H*W - batch, N*H*W ->minibatch, batch, N*H*W
            target_per_sample = perturbed_samples_vec.unsqueeze(1)-ref_vec

            target = (target_per_sample /
                      pad_dims_like(sigmas, target_per_sample))
            # calculate the targets with reference batch
            stable_targets = torch.sum(weights * target, dim=1)

            stable_targets_wrt_r = stable_targets[..., :-1] / \
                (stable_targets[..., -1][..., None] + 1e-7)

            return stable_targets_wrt_r.view_as(perturbed_samples)


class ConsistencySamplingVFE():

    def __init__(self, cm_type, sigma_min: float = 0.002, sigma_data: float = 0.5) -> None:

        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.cm_type = cm_type

    def __call__(self,
                 model: nn.Module,
                 batch: Tensor,
                 sigmas: Iterable[Union[Tensor, float]],
                 clip_denoised: bool = True,
                 verbose: bool = False, sigma_fix=0.00, **kwargs: Any
                 ) -> Tensor:
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        y : Tensor
            Reference sample e.g: a masked image or noise.
        sigmas : Iterable[Union[Tensor, float]]
            Decreasing standard deviations of the noise.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        sigma_fix: float, default=0.01
            Standard deviation for pt(x_t|z)
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Edited/sampled sample.
        """

        #noise = batch
        x = batch
        if self.cm_type == 'DCM' or self.cm_type == 'DCM-MS' or self.cm_type == 'PCM':
            x = model_forward_wrapper(
                model, x*sigmas[0], torch.full((x.shape[0],), sigmas[0],
                                               dtype=x.dtype, device=x.device), self.sigma_data, self.sigma_min, **kwargs
            )
        else:
            x = model_forward_wrapper(
                model, x, torch.full((x.shape[0],), sigmas[0],
                                     dtype=x.dtype, device=x.device), self.sigma_data, self.sigma_min, **kwargs
            )
            sigma_scale = 80.0

        if clip_denoised:
            x = x.clamp(min=-1.0, max=1.0)

        pbar = tqdm(sigmas[1:], disable=(not verbose))
        for sigma in pbar:
            pbar.set_description(f"sampling (σ={sigma:.4f})")
            sigma = torch.full((x.shape[0],), sigma,
                               dtype=x.dtype, device=x.device)
            if self.cm_type == 'PCM':
                noise=rand_hyperplane(batch.shape,batch.device)
            else:
                noise=torch.randn_like(batch)

            if self.cm_type == 'DCM' or self.cm_type == 'DCM-MS'or self.cm_type == 'PCM':
    
                x = x + pad_dims_like(
                    (sigma**2 - self.sigma_min**2) ** 0.5, x
                ) * noise

            elif self.cm_type == 'CCM' or self.cm_type == 'CCM-OT':
                x = (1-pad_dims_like(
                    ((sigma/sigma_scale)**2 - self.sigma_min**2) ** 0.5, x
                )) * x + pad_dims_like(
                    ((sigma/sigma_scale)**2 - self.sigma_min**2) ** 0.5, x
                ) * noise + sigma_fix * torch.randn_like(x, device=x.device)


            x = model_forward_wrapper(
                model, x, sigma, self.sigma_data, self.sigma_min, **kwargs
            )
            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)

        return x
