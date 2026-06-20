import torch
import torch.nn as nn


class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scale=3.0):
        ctx.scale = scale
        ctx.save_for_backward(input_tensor)
        output = torch.zeros_like(input_tensor, dtype=input_tensor.dtype)
        output[input_tensor > 0] = 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input_tensor) + 1.0) ** 2
        return grad, None


class DenseMSFConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, max_spikes=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.max_spikes = max_spikes
        self.reset_step = 1.0
        self.eps = 1e-4

        self.raw_beta = nn.Parameter(torch.zeros(out_channels))
        self.raw_threshold = nn.Parameter(torch.zeros(out_channels))
        self.spike_fn = SurrogateHeaviside.apply

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.normal_(self.raw_beta, mean=1.5, std=0.05)
        nn.init.normal_(self.raw_threshold, mean=-2.0, std=0.05)

    def _broadcast_beta(self):
        beta = self.eps + (1.0 - 2.0 * self.eps) * torch.sigmoid(self.raw_beta)
        return beta.view(1, -1, 1, 1)

    def _broadcast_threshold(self):
        threshold = torch.nn.functional.softplus(self.raw_threshold) + self.eps
        return threshold.view(1, -1, 1, 1)

    def forward(self, x, mem):
        conv_out = self.conv(x)
        if mem is None:
            mem = torch.zeros_like(conv_out)

        beta = self._broadcast_beta()
        threshold = self._broadcast_threshold()

        new_mem = mem * beta + conv_out * (1.0 - beta)

        d_vals = torch.arange(
            self.max_spikes,
            device=new_mem.device,
            dtype=new_mem.dtype,
        ).view(self.max_spikes, 1, 1, 1, 1)
        mthr_all = new_mem.unsqueeze(0) - threshold.unsqueeze(0) - d_vals * self.reset_step
        spk = self.spike_fn(mthr_all).sum(dim=0)

        final_mem = torch.clamp(new_mem - spk * self.reset_step, min=0.0)
        return spk, final_mem


class DenseLegacySpikingConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        D=4,
        h=1.0,
        beta_init=0.8,
        b_init=0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.D = int(D)
        self.h = float(h)
        self.eps = float(eps)
        self.beta = nn.Parameter(torch.empty(1))
        self.b = nn.Parameter(torch.empty(out_channels))
        self.spike_fn = SurrogateHeaviside.apply
        self.last_diagnostics = {}
        self._beta_init = beta_init
        self._b_init = b_init
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain("sigmoid"))
        nn.init.normal_(self.beta, mean=self._beta_init, std=0.01)
        nn.init.normal_(self.b, mean=self._b_init, std=0.01)
        self.clamp()

    def clamp(self, min_beta=0.0, max_beta=1.0, min_b=0.0):
        self.beta.data.clamp_(min_beta, max_beta)
        self.b.data.clamp_(min=min_b)

    def _kernel_norm(self):
        return self.conv.weight.pow(2).sum(dim=(1, 2, 3)).view(1, -1, 1, 1).clamp_min(self.eps)

    def forward(self, x, mem):
        conv_out = self.conv(x)
        if mem is None:
            mem = torch.zeros_like(conv_out)

        beta = torch.clamp(self.beta, 0.0, 1.0).view(1, 1, 1, 1)
        b = torch.clamp(self.b, min=0.0).view(1, -1, 1, 1)
        kernel_norm = self._kernel_norm()

        new_mem = mem * beta + conv_out * (1.0 - beta)
        mthr = new_mem / kernel_norm - b

        spk = torch.zeros_like(mthr)
        for d in range(self.D):
            spk = spk + self.spike_fn(mthr - d * self.h)

        final_mem = torch.clamp(new_mem - spk * self.h * kernel_norm, min=0.0)
        self.last_diagnostics = {
            "conv_out_mean": float(conv_out.detach().mean().cpu()),
            "conv_out_max": float(conv_out.detach().max().cpu()),
            "mem_max": float(new_mem.detach().max().cpu()),
            "mthr_max": float(mthr.detach().max().cpu()),
            "spike_rate": float(spk.detach().mean().cpu()),
            "beta_mean": float(beta.detach().mean().cpu()),
            "b_mean": float(b.detach().mean().cpu()),
            "kernel_norm_mean": float(kernel_norm.detach().mean().cpu()),
            "kernel_norm_min": float(kernel_norm.detach().min().cpu()),
            "kernel_norm_max": float(kernel_norm.detach().max().cpu()),
        }
        return spk, final_mem


class SNNEncoder(nn.Module):
    def __init__(self, in_channels=1, layer_cls=DenseLegacySpikingConv2D):
        super().__init__()
        self.enc1 = layer_cls(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.enc2 = layer_cls(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = layer_cls(128, 256, kernel_size=3, stride=2, padding=1)
        self.layers = (self.enc1, self.enc2, self.enc3)

    @staticmethod
    def _conv_out_size(size, kernel_size, stride, padding, dilation=1):
        return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    def init_state(self, batch_size, spatial_shape, device):
        height, width = spatial_shape
        states = []
        in_height, in_width = height, width

        for layer in self.layers:
            out_height = self._conv_out_size(
                in_height,
                layer.conv.kernel_size[0],
                layer.conv.stride[0],
                layer.conv.padding[0],
                layer.conv.dilation[0],
            )
            out_width = self._conv_out_size(
                in_width,
                layer.conv.kernel_size[1],
                layer.conv.stride[1],
                layer.conv.padding[1],
                layer.conv.dilation[1],
            )
            states.append(
                torch.zeros(
                    batch_size,
                    layer.conv.out_channels,
                    out_height,
                    out_width,
                    device=device,
                )
            )
            in_height, in_width = out_height, out_width

        return tuple(states)

    def forward_step(self, x_t, mems):
        mem1, mem2, mem3 = mems

        spk1, mem1 = self.enc1(x_t, mem1)
        spk2, mem2 = self.enc2(spk1, mem2)
        spk3, mem3 = self.enc3(spk2, mem3)

        return (spk1, spk2, spk3), (mem1, mem2, mem3)

    def get_layer_diagnostics(self):
        return [getattr(layer, "last_diagnostics", {}) for layer in self.layers]
