import torch.nn as nn
import torch.nn.functional as F
from hw_4.utils.configs import TrainConfig, MelSpectrogram, train_config
from hw_4.model.HifiGan.full_generator import Generator
from hw_4.model.HifiGan.discriminator_blocks import MultiScaleDiscriminator, MultiPeriodDiscriminator
from hw_4.loss.gen_dis_feat_loss import feature_loss, discriminator_loss, generator_loss


class HiFiGan(nn.Module):
    def __init__(self, train_config=train_config):
        super().__init__()

        self.generator = Generator(train_config)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.mel_spec = MelSpectrogram()

    def forward(self, target_mel):
        return self.generator(target_mel)

    def optim_g_d(self, batch, optimizer_g, optimizer_d):
        y_wav = batch['audio']
        y_mel = batch['spectrogram']
        y_hat_wav = self.generator(y_mel)
        loss_d_all, loss_mpd, loss_msd = self.optim_d(optimizer_d, y_mel, y_wav, y_hat_wav)
        loss_g_all, loss_mel = self.optim_g(optimizer_g, y_mel, y_wav, y_hat_wav)
        return loss_d_all, loss_mpd, loss_msd, loss_g_all, loss_mel

    def optim_d(self, optimizer_d, y_mel, y_wav, y_hat_wav):
        optimizer_d.zero_grad()
        y_outs, y_hat_outs, y_feats, y_hat_feats = self.mpd(y_wav, y_hat_wav.detach())
        loss_disc_f = discriminator_loss(y_outs, y_hat_outs)

        y_outs, y_hat_outs, y_feats, y_hat_feats = self.msd(y_wav, y_hat_wav.detach())
        loss_disc_s = discriminator_loss(y_outs, y_hat_outs)

        loss_d_all = loss_disc_f + loss_disc_s
        loss_d_all.backward()
        optimizer_d.step()

        return loss_d_all, loss_disc_f, loss_disc_s

    def optim_g(self, optimizer_g, y_mel, y_wav, y_hat_wav):
        optimizer_g.zero_grad()
        pred_mel = self.mel_spec(y_hat_wav)
        loss_mel = F.l1_loss(y_mel, pred_mel) * 45

        y_outs_p, y_hat_outs_p, y_feats_p, y_hat_feats_p = self.mpd(y_wav, y_hat_wav)
        y_outs_s, y_hat_outs_s, y_feats_s, y_hat_feats_s = self.msd(y_wav, y_hat_wav)
        loss_mpd_feat = feature_loss(y_feats_p, y_hat_feats_p)
        loss_msd_feat = feature_loss(y_feats_s, y_hat_feats_s)
        loss_mpd_g = generator_loss(y_hat_outs_p)
        loss_msd_g = generator_loss(y_hat_outs_s)
        loss_g_all = loss_mpd_feat + loss_msd_feat + loss_mpd_g + loss_msd_g + loss_mel
        loss_g_all.backward()
        optimizer_g.step()

        return loss_g_all, loss_mel
