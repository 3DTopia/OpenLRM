# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import math
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from accelerate.logging import get_logger

from .base_trainer import Trainer
from openlrm.utils.profiler import DummyProfiler
from openlrm.runners import REGISTRY_RUNNERS


logger = get_logger(__name__)


@REGISTRY_RUNNERS.register('train.lrm')
class LRMTrainer(Trainer):
    def __init__(self):
        super().__init__()

        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        self.train_loader, self.val_loader = self._build_dataloader(self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        self.pixel_loss_fn, self.perceptual_loss_fn, self.tv_loss_fn = self._build_loss_fn(self.cfg)

    def _build_model(self, cfg):
        assert cfg.experiment.type == 'lrm', \
            f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        from openlrm.models import ModelLRM
        model = ModelLRM(**cfg.model)
        return model

    def _build_optimizer(self, model: nn.Module, cfg):
        decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in model.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # monitor this to make sure we don't miss any parameters
        logger.info("======== Weight Decay Parameters ========")
        logger.info(f"Total: {len(decay_params)}")
        logger.info("======== No Weight Decay Parameters ========")
        logger.info(f"Total: {len(no_decay_params)}")

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=cfg.train.optim.lr,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )

        return optimizer

    def _build_scheduler(self, optimizer, cfg):
        local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
        total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters
        logger.debug(f"======== Scheduler effective max iters: {total_global_batches} ========")
        logger.debug(f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========")
        if cfg.train.scheduler.type == 'cosine':
            from openlrm.utils.scheduler import CosineWarmupScheduler
            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(f"Scheduler type {cfg.train.scheduler.type} not implemented")
        return scheduler

    def _build_dataloader(self, cfg):
        # dataset class
        from openlrm.datasets import MixerDataset

        # build dataset
        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            normalize_camera=cfg.dataset.normalize_camera,
            normed_dist_to_center=cfg.dataset.normed_dist_to_center,
        )
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            normalize_camera=cfg.dataset.normalize_camera,
            normed_dist_to_center=cfg.dataset.normed_dist_to_center,
        )

        # build data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.dataset.num_train_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.dataset.num_val_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=False,
        )

        return train_loader, val_loader

    def _build_loss_fn(self, cfg):
        from openlrm.losses import PixelLoss, LPIPSLoss, TVLoss
        pixel_loss_fn = PixelLoss()
        with self.accelerator.main_process_first():
            perceptual_loss_fn = LPIPSLoss(device=self.device, prefech=True)
        tv_loss_fn = TVLoss()
        return pixel_loss_fn, perceptual_loss_fn, tv_loss_fn

    def register_hooks(self):
        pass

    def forward_loss_local_step(self, data):

        source_camera = data['source_camera']
        render_camera = data['render_camera']
        source_image = data['source_image']
        render_image = data['render_image']
        render_anchors = data['render_anchors']
        render_full_resolutions = data['render_full_resolutions']
        render_bg_colors = data['render_bg_colors']

        N, M, C, H, W = render_image.shape

        # forward
        outputs = self.model(
            image=source_image,
            source_camera=source_camera,
            render_cameras=render_camera,
            render_anchors=render_anchors,
            render_resolutions=render_full_resolutions,
            render_bg_colors=render_bg_colors,
            render_region_size=self.cfg.dataset.render_image.region,
        )

        # loss calculation
        loss = 0.
        loss_pixel = None
        loss_perceptual = None
        loss_tv = None

        if self.cfg.train.loss.pixel_weight > 0.:
            loss_pixel = self.pixel_loss_fn(outputs['images_rgb'], render_image)
            loss += loss_pixel * self.cfg.train.loss.pixel_weight
        if self.cfg.train.loss.perceptual_weight > 0.:
            loss_perceptual = self.perceptual_loss_fn(outputs['images_rgb'], render_image)
            loss += loss_perceptual * self.cfg.train.loss.perceptual_weight
        if self.cfg.train.loss.tv_weight > 0.: 
            loss_tv = self.tv_loss_fn(outputs['planes'])
            loss += loss_tv * self.cfg.train.loss.tv_weight

        return outputs, loss, loss_pixel, loss_perceptual, loss_tv

    def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile):
        self.model.train()

        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        for data in loader:

            logger.debug(f"======== Starting global step {self.global_step} ========")
            with self.accelerator.accumulate(self.model):

                # forward to loss
                outs, loss, loss_pixel, loss_perceptual, loss_tv = self.forward_loss_local_step(data)
                
                # backward
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # track local losses
                local_step_losses.append(torch.stack([
                    _loss.detach() if _loss is not None else torch.tensor(float('nan'), device=self.device)
                    for _loss in [loss, loss_pixel, loss_perceptual, loss_tv]
                ]))

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                logger.debug(f"======== Scheduler step ========")
                self.global_step += 1
                global_step_loss = self.accelerator.gather(torch.stack(local_step_losses)).mean(dim=0).cpu()
                loss, loss_pixel, loss_perceptual, loss_tv = global_step_loss.unbind()
                loss_kwargs = {
                    'loss': loss.item(),
                    'loss_pixel': loss_pixel.item(),
                    'loss_perceptual': loss_perceptual.item(),
                    'loss_tv': loss_tv.item(),
                }
                self.log_scalar_kwargs(
                    step=self.global_step, split='train',
                    **loss_kwargs
                )
                self.log_optimizer(step=self.global_step, attrs=['lr'], group_ids=[0, 1])
                local_step_losses = []
                global_step_losses.append(global_step_loss)

                # manage display
                pbar.update(1)
                description = {
                    **loss_kwargs,
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                description = '[TRAIN STEP]' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v))
                pbar.set_description(description)

                # periodic actions
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.cfg.val.global_step_period == 0:
                    self.evaluate()
                    self.model.train()
                if self.global_step % self.cfg.logger.image_monitor.train_global_steps == 0:
                    self.log_image_monitor(
                        step=self.global_step, split='train',
                        renders=outs['images_rgb'].detach()[:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                        gts=data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                    )

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss, epoch_loss_pixel, epoch_loss_perceptual, epoch_loss_tv = epoch_losses.unbind()
        epoch_loss_dict = {
            'loss': epoch_loss.item(),
            'loss_pixel': epoch_loss_pixel.item(),
            'loss_perceptual': epoch_loss_perceptual.item(),
            'loss_tv': epoch_loss_tv.item(),
        }
        self.log_scalar_kwargs(
            epoch=self.current_epoch, split='train',
            **epoch_loss_dict,
        )
        logger.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )

    def train(self):
        
        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        with tqdm(
            range(0, self.N_max_global_steps),
            initial=self.global_step,
            disable=(not self.accelerator.is_main_process),
        ) as pbar:

            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=10, warmup=10, active=100,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
                    self.cfg.logger.tracker_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                )),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) if self.cfg.logger.enable_profiler else DummyProfiler()
            
            with profiler:

                self.optimizer.zero_grad()
                for _ in range(self.current_epoch, self.cfg.train.epochs):

                    loader = skipped_loader or self.train_loader
                    skipped_loader = None
                    self.train_epoch(pbar=pbar, loader=loader, profiler=profiler)
                    if self.accelerator.check_trigger():
                        break

            logger.info(f"======== Training finished at global step {self.global_step} ========")

            # final checkpoint and evaluation
            self.save_checkpoint()
            self.evaluate()

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self, epoch: int = None):
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
        running_losses = []
        sample_data, sample_outs = None, None

        for data in tqdm(self.val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):

            if len(running_losses) >= max_val_batches:
                logger.info(f"======== Early stop validation at {len(running_losses)} batches ========")
                break

            outs, loss, loss_pixel, loss_perceptual, loss_tv = self.forward_loss_local_step(data)
            sample_data, sample_outs = data, outs

            running_losses.append(torch.stack([
                _loss if _loss is not None else torch.tensor(float('nan'), device=self.device)
                for _loss in [loss, loss_pixel, loss_perceptual, loss_tv]
            ]))

        total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
        total_loss, total_loss_pixel, total_loss_perceptual, total_loss_tv = total_losses.unbind()
        total_loss_dict = {
            'loss': total_loss.item(),
            'loss_pixel': total_loss_pixel.item(),
            'loss_perceptual': total_loss_perceptual.item(),
            'loss_tv': total_loss_tv.item(),
        }

        if epoch is not None:
            self.log_scalar_kwargs(
                epoch=epoch, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self.log_image_monitor(
                epoch=epoch, split='val',
                renders=sample_outs['images_rgb'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )
        else:
            self.log_scalar_kwargs(
                step=self.global_step, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL STEP] {self.global_step}/{self.N_max_global_steps}: ' + \
                    ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
            self.log_image_monitor(
                step=self.global_step, split='val',
                renders=sample_outs['images_rgb'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
                gts=sample_data['render_image'][:self.cfg.logger.image_monitor.samples_per_log].cpu(),
            )

    @Trainer.control('on_main_process')
    def log_image_monitor(
        self, epoch: int = None, step: int = None, split: str = None,
        renders: torch.Tensor = None, gts: torch.Tensor = None,
        ):
        M = renders.shape[1]
        merged = torch.stack([renders, gts], dim=1)[0].view(-1, *renders.shape[2:])
        renders, gts = renders.view(-1, *renders.shape[2:]), gts.view(-1, *gts.shape[2:])
        renders, gts, merged = make_grid(renders, nrow=M), make_grid(gts, nrow=M), make_grid(merged, nrow=M)
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f'/{split}' if split else ''
        self.log_images({
            f'Images_split{split}/rendered': renders.unsqueeze(0),
            f'Images_split{split}/gt': gts.unsqueeze(0),
            f'Images_merged{split}': merged.unsqueeze(0),
        }, log_progress)
