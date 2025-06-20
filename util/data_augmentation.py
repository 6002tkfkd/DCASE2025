import os
import random
import librosa
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.autograd import Variable


class _DataAugmentation(nn.Module):
    """ Base Module for data augmentation techniques. """

class MixUp(_DataAugmentation):
    def __init__(self, alpha=0.3):
        super(MixUp, self).__init__()
        self.alpha = alpha
        self.lam = 1

    def forward(self, x, y, y_soft=None):
        indices = torch.randperm(x.size(0)).to(x.device)
        x2 = x[indices]
        y2 = y[indices]
        x_mix = self.lam * x + (1 - self.lam) * x2

        if isinstance(y, tuple):  # 이미 mix 된 상태인 경우
            y_mix = y
        else:
            y_mix = (y, y2)

        if y_soft is not None:
            y_soft2 = y_soft[indices]
            y_soft_mix = (y_soft, y_soft2)
            return x_mix, y_mix, y_soft_mix
        else:
            return x_mix, y_mix

class SoftMixUp(_DataAugmentation):
    def __init__(self, alpha=0.3):
        super(SoftMixUp, self).__init__()
        self.alpha = alpha
        self.lam = 1

    def forward(self, x, y, s):
        self.lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        x = self.lam * x + (1 - self.lam) * x[index, :]
        y_a, y_b = y, y[index]
        s_a, s_b = s, s[index]
        x, y_a, y_b, s_a, s_b = map(Variable, (x, y_a, y_b, s_a, s_b))
        return x, (y_a, y_b), (s_a, s_b)

class FreqMixStyle(_DataAugmentation):
    def __init__(self, alpha=0.3, p=0.7, eps=1e-6):
        super(FreqMixStyle, self).__init__()
        self.alpha = alpha
        self.p = p
        self.eps = eps

    def forward(self, x):
        if np.random.rand() > self.p:
            return x
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        batch_size = x.size(0)
        # Changed from dim=[2,3] to dim=[1,3] from channel-wise statistics to frequency-wise statistics
        f_mu = x.mean(dim=[1, 3], keepdim=True)
        f_var = x.var(dim=[1, 3], keepdim=True)
        # Compute instance standard deviation
        f_sig = (f_var + self.eps).sqrt()
        # Block gradients
        f_mu, f_sig = f_mu.detach(), f_sig.detach()
        # Normalize x
        x_normed = (x - f_mu) / f_sig
        # Generate shuffling indices
        perm = torch.randperm(batch_size).to(x.device)
        # Shuffling
        f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]
        # Generate mixed mean
        mu_mix = f_mu * lam + f_mu_perm * (1 - lam)
        # Generate mixed standard deviation
        sig_mix = f_sig * lam + f_sig_perm * (1 - lam)
        # Denormalize x using the mixed statistics
        return x_normed * sig_mix + mu_mix

class SpecAugmentation(_DataAugmentation):
    def __init__(self, mask_size=0.1, p=0.8):
        super().__init__()
        self.mask_size = mask_size
        self.p = p

    def forward(self, x):
        _, _, f, t = x.size()
        # Create frequency mask and time mask
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=round(f * self.mask_size), iid_masks=True)
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param=round(t * self.mask_size), iid_masks=True)
        # Apply mask according to random probability
        x = freq_masking(x) if np.random.uniform(0, 1) < self.p else x
        x = time_masking(x) if np.random.uniform(0, 1) < self.p else x
        return x

class DeviceImpulseResponseAugmentation(_DataAugmentation):
    def __init__(self, path_ir, p=0.4, mode="full"):
        super().__init__()
        self.path_ir = path_ir
        self.ir_files = os.listdir(path_ir)
        self.p = p
        self.mode = mode

    def forward(self, x, d):
        batch_size = x.size(0)
        for i in range(batch_size):
            # Only apply for data from device A
            if d[i] == torch.tensor([0]).to(x.device) and np.random.uniform(0, 1) < self.p:
                # Randomly select an impulse response
                random_file = random.choice(self.ir_files)
                ir, _ = librosa.load(f"{self.path_ir}/{random_file}", sr=32000)
                ir = torch.from_numpy(ir).to(x.device)
                y = torchaudio.functional.fftconvolve(x[i], ir, mode=self.mode)
                x[i] = y[:len(x[i])]
        return x

class FilterAugmentation(_DataAugmentation):
    def __init__(self, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
        super().__init__()
        self.db_range = db_range
        self.n_band = n_band
        self.min_bw = min_bw
        self.filter_type = filter_type

    def forward(self, x):
        # x: (B, C, F, T) → reshape to (B, F, T) for filt_aug
        x = x.squeeze(1)
        x = filt_aug(x, self.db_range, self.n_band, self.min_bw, self.filter_type)
        return x.unsqueeze(1)

class AdditiveNoiseAugmentation(_DataAugmentation):
    def __init__(self, snrs=(15, 30)):
        super().__init__()
        self.snrs = snrs

    def forward(self, x):
        x = x.squeeze(1)
        x = add_noise(x, snrs=self.snrs)
        return x.unsqueeze(1)

class FrequencyMaskAugmentation(_DataAugmentation):
    def __init__(self, mask_ratio=16):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        x = x.squeeze(1)
        x = freq_mask(x, self.mask_ratio)
        return x.unsqueeze(1)

class TimeMaskAugmentation(_DataAugmentation):
    def __init__(self, mask_ratios=(10, 20), net_pooling=None):
        super().__init__()
        self.mask_ratios = mask_ratios
        self.net_pooling = net_pooling

    def forward(self, x, labels=None):
        x = x.squeeze(1)  # (B, F, T)

        if labels is not None:
            x, labels = time_mask(x, labels, self.net_pooling, self.mask_ratios)
            x = x.unsqueeze(1)
            return x, labels
        else:
            x = time_mask(x, mask_ratios=self.mask_ratios)
            return x.unsqueeze(1)

class FrameShiftAugmentation(_DataAugmentation):
    def __init__(self, net_pooling=None):
        super().__init__()
        self.net_pooling = net_pooling

    def forward(self, x, labels=None):
        x = x.squeeze(1)  # (B, F, T)

        if labels is not None:
            x, labels = frame_shift(x, label=labels, net_pooling=self.net_pooling)
            x = x.unsqueeze(1)
            return x, labels
        else:
            x = frame_shift(x)
            return x.unsqueeze(1)


def filt_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
    # this is updated FilterAugment algorithm used for ICASSP 2022
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6

    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                       band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
        return features * freq_filt

    else:
        return features

def filt_aug_prototype(features, db_range=(-7.5, 6), n_bands=(2, 5)):
    # this is FilterAugment algorithm used for DCASE 2021 Challeng Task 4
    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_bands[0], high=n_bands[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        band_bndry_freqs = torch.cat((torch.tensor([0]),
                                      torch.sort(torch.randint(1, n_freq_bin-1, (n_freq_band - 1, )))[0],
                                      torch.tensor([n_freq_bin])))
        band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
        band_factors = 10 ** (band_factors / 20)

        freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
        for i in range(n_freq_band):
            freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)
        return features * freq_filt
    else:
        return features

def frame_shift(features, label=None, net_pooling=None):
    if label is not None:
        batch_size, _, _ = features.shape
        shifted_feature = []
        shifted_label = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, 90))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
            shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
            shifted_label.append(torch.roll(label[idx], shift, dims=-1))
        return torch.stack(shifted_feature), torch.stack(shifted_label)
    else:
        batch_size, _, _ = features.shape
        shifted_feature = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, 90))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
        return torch.stack(shifted_feature)

def time_mask(features, labels=None, net_pooling=None, mask_ratios=(10, 20)):
    if labels is not None:
        _, _, n_frame = labels.shape
        t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
        t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
        features[:, :, t_low * net_pooling:(t_low+t_width)*net_pooling] = 0
        labels[:, :, t_low:t_low+t_width] = 0
        return features, labels
    else:
        _, _, n_frame = features.shape
        t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
        t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
        features[:, :, t_low:(t_low + t_width)] = 0
        return features

def freq_mask(features, mask_ratio=16):
    batch_size, n_freq_bin, _ = features.shape
    max_mask = int(n_freq_bin/mask_ratio)
    if max_mask == 1:
        f_widths = torch.ones(batch_size)
    else:
        f_widths = torch.randint(low=1, high=max_mask, size=(batch_size,))   # [low, high)

    for i in range(batch_size):
        f_width = f_widths[i]
        f_low = torch.randint(low=0, high=n_freq_bin-f_width, size=(1,))

        features[i, f_low:f_low+f_width, :] = 0
    return features

def add_noise(features, snrs=(15, 30), dims=(1, 2)):
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand((features.shape[0],), device=features.device).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)
    sigma = torch.std(features, dim=dims, keepdim=True) / snr
    return features + torch.randn(features.shape, device=features.device) * sigma