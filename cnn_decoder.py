import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or out_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


class SNNFeatureCNNDecoder(nn.Module):
    def __init__(
        self,
        in_channels=(64, 128, 256),
        embedding_dim=128,
        max_velocity=2.0,
        init_velocity=1.1,
        log_tau_center=-4.0,
        log_tau_scale=2.0,
    ):
        super().__init__()
        c1, c2, c3 = in_channels
        self.max_velocity = float(max_velocity)
        self.log_tau_center = float(log_tau_center)
        self.log_tau_scale = float(log_tau_scale)

        self.project3 = nn.Sequential(
            nn.Conv2d(c3, 128, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.block3 = ResidualConvBlock(128, 128)

        self.project2 = nn.Sequential(
            nn.Conv2d(c2, 64, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.block2 = ResidualConvBlock(64 + 128, 96)

        self.project1 = nn.Sequential(
            nn.Conv2d(c1, 32, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualConvBlock(32 + 96, 64)
        self.fused_block = ResidualConvBlock(64, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Sequential(
            nn.Linear(64 * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.velocity_head = nn.Linear(embedding_dim, 1)
        self.log_tau_head = nn.Linear(embedding_dim, 1)
        self._init_velocity_bias(init_velocity)

    def _init_velocity_bias(self, init_velocity):
        init_ratio = min(max(float(init_velocity) / self.max_velocity, 1e-4), 1.0 - 1e-4)
        bias = math.log(init_ratio / (1.0 - init_ratio))
        nn.init.normal_(self.velocity_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.velocity_head.bias, bias)

    def forward(self, features, patches_per_sample):
        feat_1, feat_2, feat_3 = features

        x3 = self.block3(self.project3(feat_3))
        x3_up = F.interpolate(x3, size=feat_2.shape[-2:], mode="bilinear", align_corners=False)

        x2 = self.block2(torch.cat((self.project2(feat_2), x3_up), dim=1))
        x2_up = F.interpolate(x2, size=feat_1.shape[-2:], mode="bilinear", align_corners=False)

        x1 = self.block1(torch.cat((self.project1(feat_1), x2_up), dim=1))
        fused_map = self.fused_block(x1)

        patch_embedding = self.pool(fused_map).flatten(1)
        patch_count, channels = patch_embedding.shape
        if patch_count % patches_per_sample != 0:
            raise ValueError("Patch count is not divisible by patches_per_sample.")
        batch_size = patch_count // patches_per_sample
        patch_embedding = patch_embedding.view(batch_size, patches_per_sample, channels)
        sample_embedding = torch.cat(
            (
                patch_embedding.mean(dim=1),
                patch_embedding.std(dim=1, unbiased=False),
            ),
            dim=1,
        )

        cnn_embedding = self.embedding(sample_embedding)
        v_pred_raw = self.velocity_head(cnn_embedding).view(-1)
        v_pred = self.max_velocity * torch.sigmoid(v_pred_raw)
        log_tau_delta = self.log_tau_head(cnn_embedding).view(-1)
        log_tau_pred = self.log_tau_center + self.log_tau_scale * torch.tanh(log_tau_delta)
        tau_pred = torch.exp(log_tau_pred)

        return {
            "v_pred": v_pred,
            "v_pred_raw": v_pred_raw,
            "log_tau_pred": log_tau_pred,
            "log_tau_delta": log_tau_delta,
            "tau_pred": tau_pred,
            "cnn_embedding": cnn_embedding,
            "fused_map": fused_map,
        }
