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
        use_beta_conditioning=False,
        use_bounded_scatter=False,
        scatter_scale=0.3,
        beta_eps=1e-8,
    ):
        super().__init__()
        c1, c2, c3 = in_channels
        self.max_velocity = float(max_velocity)
        self.log_tau_center = float(log_tau_center)
        self.log_tau_scale = float(log_tau_scale)
        self.use_beta_conditioning = bool(use_beta_conditioning)
        self.use_bounded_scatter = bool(use_bounded_scatter)
        self.scatter_scale = float(scatter_scale)
        self.beta_eps = float(beta_eps)

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
        self.calibration_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        fused_dim = embedding_dim + 32
        self.beta_tau_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.scatter_gamma_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.scatter_delta_head = nn.Sequential(
            nn.Linear(fused_dim + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self._init_velocity_bias(init_velocity)

    def _init_velocity_bias(self, init_velocity):
        init_ratio = min(max(float(init_velocity) / self.max_velocity, 1e-4), 1.0 - 1e-4)
        bias = math.log(init_ratio / (1.0 - init_ratio))
        nn.init.normal_(self.velocity_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.velocity_head.bias, bias)

    def forward(self, features, patches_per_sample, log_beta_max=None, beta_max=None):
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
        if log_beta_max is None:
            log_beta_max = cnn_embedding.new_zeros(batch_size)
        else:
            log_beta_max = log_beta_max.to(device=cnn_embedding.device, dtype=cnn_embedding.dtype).view(-1)
        if beta_max is None:
            beta_max = cnn_embedding.new_ones(batch_size)
        else:
            beta_max = beta_max.to(device=cnn_embedding.device, dtype=cnn_embedding.dtype).view(-1)
        if log_beta_max.shape[0] != batch_size:
            raise ValueError(f"log_beta_max shape {tuple(log_beta_max.shape)} does not match batch_size={batch_size}.")
        if beta_max.shape[0] != batch_size:
            raise ValueError(f"beta_max shape {tuple(beta_max.shape)} does not match batch_size={batch_size}.")
        beta_max = torch.clamp(beta_max, min=self.beta_eps)

        calib_input = log_beta_max.unsqueeze(-1) if self.use_beta_conditioning else cnn_embedding.new_zeros(batch_size, 1)
        calib_embedding = self.calibration_encoder(calib_input)
        fused_embedding = torch.cat((cnn_embedding, calib_embedding), dim=-1)

        v_pred_raw = self.velocity_head(cnn_embedding).view(-1)
        v_pred = self.max_velocity * torch.sigmoid(v_pred_raw)
        if self.use_beta_conditioning:
            log_tau_delta = self.beta_tau_head(fused_embedding).view(-1)
        else:
            log_tau_delta = self.log_tau_head(cnn_embedding).view(-1)
        log_tau_base = self.log_tau_center + self.log_tau_scale * torch.tanh(log_tau_delta)
        tau_base = torch.exp(log_tau_base)

        if self.use_bounded_scatter:
            raw_gamma = self.scatter_gamma_head(fused_embedding).view(-1)
            gamma = torch.sigmoid(raw_gamma).clamp(min=self.beta_eps, max=1.0 - self.beta_eps)
            beta_eff = gamma * beta_max
            scatter_input = torch.cat((fused_embedding, gamma.unsqueeze(-1)), dim=-1)
            scatter_delta_raw = self.scatter_delta_head(scatter_input).view(-1)
            scatter_delta = self.scatter_scale * torch.tanh(scatter_delta_raw)
        else:
            raw_gamma = cnn_embedding.new_zeros(batch_size)
            gamma = torch.sigmoid(raw_gamma).clamp(min=self.beta_eps, max=1.0 - self.beta_eps)
            beta_eff = gamma * beta_max
            scatter_delta = cnn_embedding.new_zeros(batch_size)

        beta_eff_ratio = beta_eff / (beta_max + self.beta_eps)
        log_tau_eff = log_tau_base + scatter_delta
        tau_eff = torch.exp(log_tau_eff)

        return {
            "v_pred": v_pred,
            "v_pred_raw": v_pred_raw,
            "log_tau_pred": log_tau_eff,
            "log_tau_delta": log_tau_delta,
            "tau_pred": tau_eff,
            "cnn_embedding": cnn_embedding,
            "calib_embedding": calib_embedding,
            "fused_embedding": fused_embedding,
            "fused_map": fused_map,
            "log_tau_base": log_tau_base,
            "tau_base": tau_base,
            "raw_gamma": raw_gamma,
            "gamma": gamma,
            "beta_max": beta_max,
            "beta_eff": beta_eff,
            "beta_eff_ratio": beta_eff_ratio,
            "scatter_delta": scatter_delta,
            "log_tau_eff": log_tau_eff,
            "tau_eff": tau_eff,
        }
