import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from cnn_decoder import SNNFeatureCNNDecoder
from snn_encoder import SNNEncoder


class SNN_CNN_Hybrid(nn.Module):
    def __init__(
        self,
        in_channels=1,
        max_velocity=2.0,
        accumulation_mode="total",
        decoder_norm=None,
        use_beta_conditioning=False,
        use_bounded_scatter=False,
        scatter_scale=0.3,
    ):
        super().__init__()
        if accumulation_mode != "total":
            raise ValueError("Only accumulation_mode='total' is enabled in the clean SNN-CNN backbone.")
        self.accumulation_mode = accumulation_mode
        self.decoder_norm = decoder_norm
        self.use_beta_conditioning = bool(use_beta_conditioning)
        self.use_bounded_scatter = bool(use_bounded_scatter)
        self.scatter_scale = float(scatter_scale)
        self.snn_encoder = SNNEncoder(in_channels=in_channels)
        self.decoder = SNNFeatureCNNDecoder(
            in_channels=(
                self.snn_encoder.enc1.conv.out_channels,
                self.snn_encoder.enc2.conv.out_channels,
                self.snn_encoder.enc3.conv.out_channels,
            ),
            max_velocity=max_velocity,
            init_velocity=1.1,
            log_tau_center=-4.0,
            log_tau_scale=2.0,
            use_beta_conditioning=self.use_beta_conditioning,
            use_bounded_scatter=self.use_bounded_scatter,
            scatter_scale=self.scatter_scale,
        )

    @staticmethod
    def _zeros_like_states(mems):
        return tuple(torch.zeros_like(mem) for mem in mems)

    @staticmethod
    def _aggregate_base_to_snn(x_base_block, snn_bin_size, scale_mode):
        if x_base_block.shape[0] % snn_bin_size != 0:
            raise ValueError(
                f"base block length {x_base_block.shape[0]} is not divisible by snn_bin_size={snn_bin_size}."
            )
        num_snn_steps = x_base_block.shape[0] // snn_bin_size
        x_snn = x_base_block.reshape(
            num_snn_steps,
            snn_bin_size,
            x_base_block.shape[1],
            x_base_block.shape[2],
            x_base_block.shape[3],
            x_base_block.shape[4],
        ).sum(dim=1)
        if scale_mode == "sqrt":
            return x_snn / (float(snn_bin_size) ** 0.5)
        if scale_mode == "mean":
            return x_snn / float(snn_bin_size)
        if scale_mode == "none":
            return x_snn
        raise ValueError(f"Unsupported snn_input_scale_mode={scale_mode!r}.")

    def _block_forward(self, x_block, mem1, mem2, mem3):
        local_mems = (mem1, mem2, mem3)
        accum = self._zeros_like_states(local_mems)

        for t in range(x_block.shape[0]):
            spikes, local_mems = self.snn_encoder.forward_step(x_block[t], local_mems)
            accum = tuple(acc + spk for acc, spk in zip(accum, spikes))

        return local_mems, accum

    def forward(
        self,
        dataloader_or_generator,
        base_total_steps=10000,
        base_block_size=400,
        snn_bin_size=40,
        snn_input_scale_mode="sqrt",
        base_dt_us=20,
        env_maps=None,
        log_beta_max=None,
        beta_max=None,
    ):
        if base_total_steps % snn_bin_size != 0:
            raise ValueError("base_total_steps must be divisible by snn_bin_size.")
        if base_total_steps % base_block_size != 0:
            raise ValueError("base_total_steps must be divisible by base_block_size.")
        if base_block_size % snn_bin_size != 0:
            raise ValueError("base_block_size must be divisible by snn_bin_size.")

        device = next(self.parameters()).device
        patches_per_sample = dataloader_or_generator.patches_per_sample
        patch_batch_size = dataloader_or_generator.batch_size * patches_per_sample
        patch_shape = dataloader_or_generator.patch_shape

        mems = self.snn_encoder.init_state(patch_batch_size, patch_shape, device)
        accum = self._zeros_like_states(mems)
        num_blocks = base_total_steps // base_block_size
        expected_snn_steps = base_total_steps // snn_bin_size
        processed_snn_steps = 0
        spike_sums = [0.0, 0.0, 0.0]
        spike_numels = [0, 0, 0]

        for block_idx in range(num_blocks):
            x_base_block = dataloader_or_generator.get_block_dense(block_idx, base_block_size).to(device)
            if x_base_block.shape[0] != base_block_size:
                raise RuntimeError(
                    f"Expected base block size {base_block_size}, got {x_base_block.shape[0]} at block {block_idx}."
                )
            x_block = self._aggregate_base_to_snn(
                x_base_block,
                snn_bin_size=snn_bin_size,
                scale_mode=snn_input_scale_mode,
            )
            processed_snn_steps += x_block.shape[0]
            if self.training:
                x_block = x_block.requires_grad_(True)
                mems, block_accum = checkpoint(
                    self._block_forward,
                    x_block,
                    *mems,
                    use_reentrant=False,
                )
            else:
                mems, block_accum = self._block_forward(x_block, *mems)
            accum = tuple(acc + block for acc, block in zip(accum, block_accum))
            for layer_idx, block in enumerate(block_accum):
                spike_sums[layer_idx] += float(block.detach().sum().cpu())
                spike_numels[layer_idx] += block.numel() * x_block.shape[0]

        if processed_snn_steps <= 0:
            raise RuntimeError("No SNN frames were processed.")
        if processed_snn_steps != expected_snn_steps:
            raise RuntimeError(f"Expected {expected_snn_steps} SNN frames, processed {processed_snn_steps}.")

        feat_1, feat_2, feat_3 = (acc / float(processed_snn_steps) for acc in accum)
        batch_size = dataloader_or_generator.batch_size
        if log_beta_max is None:
            log_beta_max = feat_1.new_zeros(batch_size)
        if beta_max is None:
            beta_max = feat_1.new_ones(batch_size)

        decoder_output = self.decoder(
            (feat_1, feat_2, feat_3),
            patches_per_sample=patches_per_sample,
            log_beta_max=log_beta_max,
            beta_max=beta_max,
        )

        return {
            **decoder_output,
            "snn_feat_1": feat_1,
            "snn_feat_2": feat_2,
            "snn_feat_3": feat_3,
            "patches_per_sample": patches_per_sample,
            "patch_grid": (
                dataloader_or_generator.grid_rows,
                dataloader_or_generator.grid_cols,
            ),
            "patch_shape": patch_shape,
            "accumulation_mode": self.accumulation_mode,
            "layer1_spike_rate": spike_sums[0] / max(spike_numels[0], 1),
            "layer2_spike_rate": spike_sums[1] / max(spike_numels[1], 1),
            "layer3_spike_rate": spike_sums[2] / max(spike_numels[2], 1),
            "base_total_steps": base_total_steps,
            "base_block_size": base_block_size,
            "snn_bin_size": snn_bin_size,
            "snn_steps": processed_snn_steps,
            "snn_step_us": int(base_dt_us * snn_bin_size),
            "window_ms": int(base_total_steps * base_dt_us / 1000),
            "snn_input_scale_mode": snn_input_scale_mode,
            "use_beta_conditioning": self.use_beta_conditioning,
            "use_bounded_scatter": self.use_bounded_scatter,
            "scatter_scale": self.scatter_scale,
        }
