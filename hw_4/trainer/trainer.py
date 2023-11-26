import os
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from hw_4.base import BaseTrainer
from hw_4.utils.configs import MelSpectrogram
from hw_4.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = 50
        self.melspec = MelSpectrogram()

        self.train_metrics = MetricTracker(
            "grad norm", "loss_d_all", "loss_g_all", "loss_mel", "loss_mpd", "loss_msd",  writer=self.writer
        )

        self.test_folder = ['test_data/audio_1.wav', 'test_data/audio_2.wav', 'test_data/audio_3.wav']
        self.test_wavs = [torchaudio.load(path)[0] for path in self.test_folder]
        self.test_mels = [self.melspec(wav.to(device)) for wav in self.test_wavs]

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        batch_idx = 0
        for batch in (
                tqdm(self.train_dataloader)
        ):
            self.move_batch_to_device(batch, self.device)
            loss_d_all, loss_mpd, loss_msd, loss_g_all, loss_mel = self.model.optim_g_d(batch,
                                                                                        self.optimizer['optimizer_g'],
                                                                                        self.optimizer['optimizer_d'])
            self.train_metrics.update("grad norm", self.get_grad_norm())
            self.train_metrics.update("loss_d_all", loss_d_all.item())
            self.train_metrics.update("loss_g_all", loss_g_all.item())
            self.train_metrics.update("loss_mel", loss_mel.item())
            self.train_metrics.update("loss_mpd", loss_mpd.item())
            self.train_metrics.update("loss_msd", loss_msd.item())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} loss_d: {:.6f} loss_g: {:.6f} loss_mel: {:.6f} loss_mpd: {:.6f} loss_msd: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss_d_all.item(), loss_g_all.item(), loss_mel.item(), loss_mpd.item(), loss_msd.item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )

                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                if batch_idx >= self.len_epoch:
                    break
            batch_idx += 1

        self.lr_scheduler['lr_scheduler_g'].step()
        self.lr_scheduler['lr_scheduler_d'].step()
        log = last_train_metrics
        self.evaluation()

        return log

    def evaluation(self):
        self.model.eval()
        for i, mel in enumerate(self.test_mels):
            gen_wav = self.model(mel).squeeze(0)
            self.writer.add_audio(f"audio_{i}", gen_wav, sample_rate=22050)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
