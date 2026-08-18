"""
losses.py

Differentiable loss terms for Experiments 2–6:

    L = lambda_pixel * L_pixel + lambda_struct * L_struct
      + lambda_edge * L_edge + lambda_freq * L_freq (Exp 6+)

New in Exp 6:
- L_freq : L1 loss in the Fourier frequency domain (rfft2). Penalizes
           errors in high-frequency content (edges, textures, fine detail)
           that Charbonnier / SSIM losses tend to smooth away because they
           operate purely in the spatial domain. Used in NAFNet, Restormer,
           and recent SOTA denoisers. Weight: lambda_freq=0.05 (small but
           complementary to the spatial terms).

- L_pixel   : Charbonnier (same as the Phase 2 baseline -- keeps the basic
              pixel-fidelity signal so this experiment is additive, not a
              replacement).
- L_struct  : 1 - SSIM. Differentiable Gaussian-window SSIM (same window
              settings as the metric in train.py, but usable in backward()).
- L_edge    : |Sobel(pred) - Sobel(gt)|, L1 on gradient-magnitude maps.
              Directly penalizes the "regression to the mean" blur we saw
              in the baseline's face/texture predictions -- an edge that
              gets smoothed away costs loss here even if the pixel values
              happen to average out close to correct.

All three operate on already-clipped-to-[0,1] predictions is NOT assumed;
clipping is left to the caller (train_exp2.py clips before computing loss,
consistent with train.py's evaluation-time-only clipping policy -- see
model.py docstring).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


class SSIMLoss(nn.Module):
    """Differentiable 1-SSIM, single-scale, fixed 11x11 Gaussian window."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # (1,1,K,K)
        self.register_buffer("window", window_2d)
        self.pad = window_size // 2
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self.window.to(pred.device)

        mu_x = F.conv2d(pred, window, padding=self.pad)
        mu_y = F.conv2d(target, window, padding=self.pad)
        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sigma_x_sq = F.conv2d(pred * pred, window, padding=self.pad) - mu_x_sq
        sigma_y_sq = F.conv2d(target * target, window, padding=self.pad) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, window, padding=self.pad) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / (
            (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        )
        return 1.0 - ssim_map.mean()


class EdgeLoss(nn.Module):
    """
    L1 distance between Sobel gradient-magnitude maps of pred and target.
    Sobel kernels are fixed (not learned) -- this is a classical edge
    detector, not a trainable module, so it can't "cheat" by learning to
    ignore hard cases.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred = self._gradient_magnitude(pred)
        grad_target = self._gradient_magnitude(target)
        return torch.abs(grad_pred - grad_target).mean()


class FFTLoss(nn.Module):
    """
    L1 loss in the 2-D frequency domain (torch.fft.rfft2).

    Penalizes errors in amplitude and phase of each frequency component,
    which directly targets high-frequency detail that spatial losses miss.
    This is the same loss used in NAFNet and Restormer.

    Works on single-channel (B,1,H,W) tensors; generalises trivially
    to multi-channel inputs.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft   = torch.fft.rfft2(pred,   norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        # L1 on complex magnitudes avoids phase-wrapping instability
        loss_real  = torch.abs(pred_fft.real - target_fft.real).mean()
        loss_imag  = torch.abs(pred_fft.imag - target_fft.imag).mean()
        return loss_real + loss_imag


class CombinedLoss(nn.Module):
    """
    L = lambda_pixel * Charbonnier
      + lambda_struct * (1-SSIM)
      + lambda_edge  * EdgeL1
      + lambda_freq  * FFT_L1      (optional, 0.0 disables it)

    Returns both the total loss and a dict of the individual (unweighted)
    components, so training scripts can log each term separately -- useful
    for checking whether one term is dominating/starving the others.
    """

    def __init__(
        self,
        lambda_pixel: float = 1.0,
        lambda_struct: float = 0.2,
        lambda_edge: float = 0.3,
        lambda_freq: float = 0.0,   # set to 0.05 for Exp 6+
    ):
        super().__init__()
        self.lambda_pixel  = lambda_pixel
        self.lambda_struct = lambda_struct
        self.lambda_edge   = lambda_edge
        self.lambda_freq   = lambda_freq
        self.ssim_loss     = SSIMLoss()
        self.edge_loss     = EdgeLoss()
        self.fft_loss      = FFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_clipped = pred.clamp(0.0, 1.0)

        l_pixel  = charbonnier_loss(pred, target)
        l_struct = self.ssim_loss(pred_clipped, target)
        l_edge   = self.edge_loss(pred_clipped, target)
        l_freq   = self.fft_loss(pred_clipped, target)

        total = (
            self.lambda_pixel  * l_pixel
            + self.lambda_struct * l_struct
            + self.lambda_edge   * l_edge
            + self.lambda_freq   * l_freq
        )

        components = {
            "pixel":  l_pixel.item(),
            "struct": l_struct.item(),
            "edge":   l_edge.item(),
            "freq":   l_freq.item(),
        }
        return total, components


def _self_test():
    print("=" * 60)
    print("losses.py self-test")
    print("=" * 60)

    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)

    loss_fn = CombinedLoss()
    total, components = loss_fn(pred, target)
    print(f"Total loss: {total.item():.5f}")
    print(f"Components: {components}")

    total.backward()
    has_grad = pred.grad is not None and torch.isfinite(pred.grad).all()
    print(f"Gradient finite and present: {has_grad}")
    assert has_grad, "Gradient did not flow through CombinedLoss!"

    perfect_pred = target.clone().requires_grad_(True)
    perfect_total, perfect_components = loss_fn(perfect_pred, target)
    print(f"Loss on perfect prediction (should be near 0): {perfect_total.item():.6f}")

    print()
    print("Self-test passed.")


if __name__ == "__main__":
    _self_test()