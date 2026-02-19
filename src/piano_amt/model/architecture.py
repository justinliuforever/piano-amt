"""Self-contained PyTorch model architecture for Onsets & Velocities.

Extracted from the upstream iamusica_training repository
(github.com/andres-fr/iamusica_training) to avoid transitive dependency issues.
Only used for loading the pretrained checkpoint and exporting to ONNX.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def init_weights(module, init_fn=torch.nn.init.kaiming_normal_,
                 bias_val=0.0, verbose=False):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if init_fn is not None:
            init_fn(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(bias_val)


def get_relu(leaky_slope=None):
    if leaky_slope is None:
        return nn.ReLU(inplace=True)
    return nn.LeakyReLU(leaky_slope, inplace=True)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Permuter(nn.Module):
    def __init__(self, *permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


class SubSpectralNorm(nn.Module):
    def __init__(self, C, F, S, momentum=0.1, eps=1e-5):
        super().__init__()
        self.S = S
        self.bn = nn.BatchNorm2d(C * S, momentum=momentum)
        assert F % S == 0

    def forward(self, x):
        N, C, F, T = x.size()
        x = x.reshape(N, C * self.S, F // self.S, T)
        x = self.bn(x)
        return x.reshape(N, C, F, T)


class SELayer(nn.Module):
    def __init__(self, in_chans, hidden_chans=None, out_chans=None, bn_momentum=0.1):
        super().__init__()
        if hidden_chans is None:
            hidden_chans = in_chans // 4
        if out_chans is None:
            out_chans = in_chans
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_chans, hidden_chans, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_chans, out_chans, bias=True),
            nn.Sigmoid(),
        )

    def set_biases(self, val=0):
        self.apply(lambda module: init_weights(module, init_fn=None, bias_val=val))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)[:, :, 0, 0]
        y = self.fc(y)[:, :, None, None]
        return y


class ContextAwareModule(nn.Module):
    def __init__(self, in_chans, hdc_chans=None, se_bottleneck=None,
                 ksizes=((3, 5), (3, 5), (3, 5), (3, 5)),
                 dilations=((1, 1), (1, 2), (1, 3), (1, 4)),
                 paddings=((1, 2), (1, 4), (1, 6), (1, 8)),
                 bn_momentum=0.1):
        super().__init__()
        num_convs = len(ksizes)
        if hdc_chans is None:
            hdc_chans = in_chans // num_convs
        hdc_out_chans = hdc_chans * num_convs
        if se_bottleneck is None:
            se_bottleneck = in_chans // 4
        self.se = SELayer(in_chans, se_bottleneck, hdc_out_chans, bn_momentum)
        self.hdcs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_chans, hdc_chans, stride=1,
                          kernel_size=ks, dilation=dil, padding=pad, bias=False),
                nn.BatchNorm2d(hdc_chans, momentum=bn_momentum),
                nn.ReLU(inplace=True),
            )
            for ks, dil, pad in zip(ksizes, dilations, paddings)
        ])
        self.skip = nn.Sequential(
            nn.Conv2d(in_chans, hdc_out_chans, kernel_size=1, bias=False),
            nn.BatchNorm2d(hdc_out_chans, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        se_att = self.se(x)
        skip = self.skip(x)
        hdc = torch.cat([h(x) for h in self.hdcs], dim=1)
        return skip + (hdc * se_att)


class DepthwiseConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, h_in, h_out, kernel_width=1, bias=True):
        super().__init__()
        self.ch_out = ch_out
        self.h_out = h_out
        assert kernel_width % 2 == 1
        self.conv = nn.Conv2d(
            ch_in, ch_out * h_out,
            (h_in, kernel_width), padding=(0, kernel_width // 2),
            groups=ch_in, bias=bias,
        )

    def forward(self, x):
        b = x.shape[0]
        x = self.conv(x)
        x = x.squeeze(2).reshape(b, self.ch_out, self.h_out, -1)
        return x


def conv1x1net(hid_chans, bn_momentum=0.1, last_layer_bn_relu=False,
               dropout_drop_p=None, leaky_relu_slope=None, kernel_width=1):
    assert kernel_width % 2 == 1
    wpad = kernel_width // 2
    result = nn.Sequential()
    n_layers = len(hid_chans) - 1
    for i, (h_in, h_out) in enumerate(zip(hid_chans[:-1], hid_chans[1:]), 1):
        if (i < n_layers) or ((i == n_layers) and last_layer_bn_relu):
            result.append(nn.Conv2d(h_in, h_out, (1, kernel_width),
                                    padding=(0, wpad), bias=False))
            result.append(nn.BatchNorm2d(h_out, momentum=bn_momentum))
            result.append(get_relu(leaky_relu_slope))
            if dropout_drop_p is not None:
                result.append(nn.Dropout(dropout_drop_p, inplace=False))
        else:
            result.append(nn.Conv2d(h_in, h_out, (1, kernel_width),
                                    padding=(0, wpad), bias=True))
    return result


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class OnsetsAndVelocities(nn.Module):
    """Onsets & Velocities piano transcription model."""

    STEM_NUM_CAMS = 3
    STEM_CAM_HDC_CHANS = 4
    STEM_CAM_SE_BOTTLENECK = 8
    STEM_CAM_KSIZES = ((3, 5), (3, 5), (3, 5), (3, 5))
    STEM_CAM_DILATIONS = ((1, 1), (1, 2), (1, 3), (1, 4))
    STEM_CAM_PADDINGS = ((1, 2), (1, 4), (1, 6), (1, 8))

    NUM_ONSET_STAGES = 3

    OSTAGE_NUM_CAMS = 3
    OSTAGE_CAM_HDC_CHANS = 4
    OSTAGE_CAM_SE_BOTTLENECK = 8
    OSTAGE_CAM_KSIZES = ((1, 11), (1, 11), (1, 11))
    OSTAGE_CAM_DILATIONS = ((1, 1), (1, 2), (1, 3))
    OSTAGE_CAM_PADDINGS = ((0, 5), (0, 10), (0, 15))

    VSTAGE_NUM_CAMS = 1
    VSTAGE_CAM_HDC_CHANS = 4
    VSTAGE_CAM_SE_BOTTLENECK = 8
    VSTAGE_CAM_KSIZES = ((1, 11), (1, 11), (1, 11))
    VSTAGE_CAM_DILATIONS = ((1, 1), (1, 2), (1, 3))
    VSTAGE_CAM_PADDINGS = ((0, 5), (0, 10), (0, 15))

    @staticmethod
    def get_cam_stage(in_chans, out_bins, conv1x1head=(200, 200),
                      num_cam_bottlenecks=3, cam_hdc_chans=4,
                      cam_se_bottleneck=8,
                      cam_ksizes=((1, 10), (1, 10), (1, 10)),
                      cam_dilations=((1, 1), (1, 2), (1, 3)),
                      cam_paddings=((0, 4), (0, 8), (0, 12)),
                      bn_momentum=0.1, leaky_relu_slope=0.1, dropout_p=0.1,
                      summary_width=3, conv1x1_kw=1):
        cam_out_chans = cam_hdc_chans * len(cam_ksizes)
        cam = nn.Sequential(
            nn.Conv2d(in_chans, cam_out_chans, (1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(cam_out_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            *[nn.Sequential(
                ContextAwareModule(
                    cam_out_chans, cam_hdc_chans, cam_se_bottleneck,
                    cam_ksizes, cam_dilations, cam_paddings, bn_momentum),
                nn.BatchNorm2d(cam_out_chans, momentum=bn_momentum),
                get_relu(leaky_relu_slope))
              for _ in range(num_cam_bottlenecks)],
            nn.Conv2d(cam_out_chans, conv1x1head[0], (out_bins, summary_width),
                      padding=(0, 1), bias=False),
            nn.BatchNorm2d(conv1x1head[0], momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            conv1x1net((*conv1x1head, out_bins), bn_momentum,
                       last_layer_bn_relu=False,
                       dropout_drop_p=dropout_p,
                       leaky_relu_slope=leaky_relu_slope,
                       kernel_width=conv1x1_kw),
            Permuter(0, 2, 1, 3),
        )
        return cam

    def __init__(self, in_chans, in_height, out_height, conv1x1head=(200, 200),
                 bn_momentum=0.1, leaky_relu_slope=0.1, dropout_drop_p=0.1,
                 init_fn=torch.nn.init.kaiming_normal_, se_init_bias=1.0):
        super().__init__()
        stem_chans = self.STEM_CAM_HDC_CHANS * len(self.STEM_CAM_KSIZES)
        vel_in_chans = stem_chans + 1

        self.specnorm = SubSpectralNorm(in_chans, in_height, in_height, bn_momentum)

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chans, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(stem_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            *[nn.Sequential(
                ContextAwareModule(
                    stem_chans, self.STEM_CAM_HDC_CHANS,
                    self.STEM_CAM_SE_BOTTLENECK, self.STEM_CAM_KSIZES,
                    self.STEM_CAM_DILATIONS, self.STEM_CAM_PADDINGS,
                    bn_momentum),
                nn.BatchNorm2d(stem_chans, momentum=bn_momentum),
                get_relu(leaky_relu_slope))
              for _ in range(self.STEM_NUM_CAMS)],
            DepthwiseConv2d(stem_chans, stem_chans, in_height, out_height,
                            kernel_width=1, bias=False),
            nn.BatchNorm2d(stem_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
        )

        self.onset_stages = nn.ModuleList([
            nn.Sequential(
                self.get_cam_stage(
                    stem_chans, out_height, conv1x1head,
                    self.OSTAGE_NUM_CAMS, self.OSTAGE_CAM_HDC_CHANS,
                    self.OSTAGE_CAM_SE_BOTTLENECK, self.OSTAGE_CAM_KSIZES,
                    self.OSTAGE_CAM_DILATIONS, self.OSTAGE_CAM_PADDINGS,
                    bn_momentum, leaky_relu_slope, dropout_drop_p),
                SubSpectralNorm(1, out_height, out_height, bn_momentum))
            for _ in range(self.NUM_ONSET_STAGES)
        ])

        self.velocity_stage = nn.Sequential(
            self.get_cam_stage(
                vel_in_chans, out_height, conv1x1head,
                self.VSTAGE_NUM_CAMS, self.VSTAGE_CAM_HDC_CHANS,
                self.VSTAGE_CAM_SE_BOTTLENECK, self.VSTAGE_CAM_KSIZES,
                self.VSTAGE_CAM_DILATIONS, self.VSTAGE_CAM_PADDINGS,
                bn_momentum, leaky_relu_slope, dropout_drop_p),
            SubSpectralNorm(1, out_height, out_height, bn_momentum),
        )

        if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, bias_val=0.0))
        self.apply(lambda module: self._set_se_biases(module, se_init_bias))

    @staticmethod
    def _set_se_biases(module, bias_val):
        try:
            module.se.set_biases(bias_val)
        except AttributeError:
            pass

    def forward_onsets(self, x):
        xdiff = x.diff(dim=-1)
        x = torch.stack([x[:, :, 1:], xdiff]).permute(1, 0, 2, 3)
        x = self.specnorm(x)
        stem_out = self.stem(x)
        x = self.onset_stages[0](stem_out)
        x_stages = [x]
        for stg in self.onset_stages[1:]:
            x = stg(stem_out) + x_stages[-1]
            x_stages.append(x)
        for st in x_stages:
            st.squeeze_(1)
        return x_stages, stem_out

    def forward(self, x, trainable_onsets=True):
        if trainable_onsets:
            x_stages, stem_out = self.forward_onsets(x)
            stem_out = torch.cat([stem_out, x_stages[-1].unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                x_stages, stem_out = self.forward_onsets(x)
                stem_out = torch.cat([stem_out, x_stages[-1].unsqueeze(1)], dim=1)
        velocities = self.velocity_stage(stem_out).squeeze(1)
        return x_stages, velocities


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    """Load the O&V model from a PyTorch checkpoint. Returns model in eval mode."""
    model = OnsetsAndVelocities(
        in_chans=2,
        in_height=229,
        out_height=88,
        conv1x1head=(200, 200),
        bn_momentum=0,
        leaky_relu_slope=0.1,
        dropout_drop_p=0,
    )

    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model
