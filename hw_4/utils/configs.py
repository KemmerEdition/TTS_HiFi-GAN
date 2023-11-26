from dataclasses import dataclass
from typing import List

import torch
from torch import nn

import torchaudio

import librosa


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    segment_size: int = 8192

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class TrainConfig:
    upsample_channel: int = 128
    upsample_rates: List[int] = [8, 8, 2, 2]
    upsample_kernel_sizes: List[int] = [16, 16, 4, 4]
    resblock_kernel_sizes: List[int] = [3, 7, 11]
    resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig = MelSpectrogramConfig()):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=False,
            pad=(config.n_fft - config.hop_length) // 2
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio.squeeze(1)) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


train_config = TrainConfig()
mel_spec_config = MelSpectrogramConfig()
