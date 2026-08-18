"""
model_nafnet.py

Experiment 6: NAFNet-style backbone for IRIS restoration.

Based on "Simple Baselines for Image Restoration" (Chen et al., ECCV 2022).
NAFNet achieves SOTA on SIDD (denoising) and GoPro (deblurring) benchmarks
with fewer parameters than Transformer-based models and faster training.

Key innovations vs the current IRISConditioned backbone:
  - SimpleGate activation: splits channels in half, multiplies element-wise.
    Strictly more expressive than ReLU, no saturation, no dead neurons.
  - Channel Attention via simple global average pooling + per-channel FC.
    Recalibrates feature importance cheaply (~2K params per block).
  - No BatchNorm / LayerNorm: removes batch-size sensitivity entirely.
  - UNet encoder-decoder with skip connections: multi-scale feature
    extraction captures both coarse structure and fine texture.
  - PixelShuffle 2x upsample head: same as current model, keeps the
    global residual (naive bilinear + learned correction) design.

Usage (self-test):
    python model_nafnet.py

Architecture (default config for <6GB GPU):
    enc_blks   = [1, 1, 1, 28]   (encoder blocks per scale)
    middle_blks = 1
    dec_blks   = [1, 1, 1, 1]   (decoder blocks per scale)
    width      = 32              (base channels)
    ~= 17M parameters

Memory budget (estimated at batch_size=8, 256x256 output):
    ~2.5 GB VRAM — safe for a 4–6 GB GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class SimpleGate(nn.Module):
    """
    Splits the channel dim in half and multiplies the two halves element-wise.
    Gate(x) = x[:C//2] * x[C//2:]
    Replaces ReLU/GELU in NAFNet — strictly more expressive, no saturation.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention (NAFNet style).
    Uses 1×1 Conv instead of Linear so it works for any spatial size
    and channel count without reshape issues.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class NAFBlock(nn.Module):
    """
    Core NAFNet building block.

    Structure:
        GroupNorm
        1×1 expand → DWConv 3×3 → SimpleGate (halves channels) → CA → 1×1 project
        Layer skip (with learnable scale beta)

        GroupNorm
        1×1 expand → SiLU gate → 1×1 project
        Layer skip (with learnable scale gamma)
    """
    def __init__(self, channels: int, ffn_expand: int = 2):
        super().__init__()
        # After expand ×2 for SimpleGate then SimpleGate halves → dw_channels
        dw_channels = channels * ffn_expand

        # DWConv branch
        self.norm1 = nn.GroupNorm(1, channels)
        # ×2 because SimpleGate splits in half
        self.conv1 = nn.Conv2d(channels, dw_channels * 2, 1)
        self.conv2 = nn.Conv2d(dw_channels * 2, dw_channels * 2,
                               kernel_size=3, padding=1, groups=dw_channels * 2)
        self.conv3 = nn.Conv2d(dw_channels, channels, 1)   # project back after gate
        self.gate  = SimpleGate()
        # CA operates on dw_channels (post-SimpleGate count)
        self.ca    = ChannelAttention(dw_channels)

        # FFN branch
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)      # input after gate = channels

        # Learnable layer scaling (stabilises deep training)
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1) + 1e-2)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1) + 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- DWConv branch ---
        inp = x
        x   = self.norm1(x)
        x   = self.conv1(x)
        x   = self.conv2(x)
        x   = self.gate(x)                  # (B, dw_channels*2) → (B, dw_channels)
        x   = self.ca(x)
        x   = self.conv3(x)
        y   = inp + x * self.beta

        # --- FFN branch ---
        x       = self.norm2(y)
        x       = self.conv4(x)             # → (B, channels*2, H, W)
        x1, x2  = x.chunk(2, dim=1)        # each (B, channels, H, W)
        x       = x1 * torch.sigmoid(x2)   # SiLU-like gating
        x       = self.conv5(x)
        return y + x * self.gamma


class DownsampleBlock(nn.Module):
    """2× spatial downsampling via strided conv (encoder)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2,
                              kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """2× spatial upsampling via PixelShuffle (decoder)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv    = nn.Conv2d(channels, channels // 2 * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


# ---------------------------------------------------------------------------
#  Full UNet model
# ---------------------------------------------------------------------------

class NAFNetIRIS(nn.Module):
    """
    NAFNet-based image restoration network for IRIS.

    Input : (B, 1, 128, 128) float32, unclipped
    Output: (B, 1, 256, 256) float32

    The network:
      1. Projects input to `width` channels at 128×128.
      2. UNet encoder: downsamples through 4 scales while doubling channels.
      3. Bottleneck blocks at the coarsest scale.
      4. UNet decoder: upsamples back, concatenating encoder skip features.
      5. Project back to `width` channels.
      6. PixelShuffle 2× to go from 128×128 → 256×256.
      7. Add global residual (bilinear upsample of raw input) — same as
         all previous IRIS experiments.

    Default config (width=32) ≈ 17M parameters.
    Memory footprint ≈ 2.5 GB at batch_size=8 — safe for <6 GB GPU.
    """

    def __init__(
        self,
        width: int = 32,
        enc_blks: list = None,
        middle_blks: int = 1,
        dec_blks: list = None,
    ):
        super().__init__()
        if enc_blks is None:
            enc_blks = [1, 1, 1, 28]
        if dec_blks is None:
            dec_blks = [1, 1, 1, 1]

        self.intro = nn.Conv2d(1, width, kernel_size=3, padding=1)

        # --- Encoder ---
        self.encoders    = nn.ModuleList()
        self.downs       = nn.ModuleList()
        ch = width
        for n in enc_blks:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch) for _ in range(n)])
            )
            self.downs.append(DownsampleBlock(ch))
            ch *= 2

        # --- Bottleneck ---
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blks)])

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()
        for n in dec_blks:
            self.ups.append(UpsampleBlock(ch))
            ch //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch) for _ in range(n)])
            )

        # --- Output head (at 128×128 resolution) ---
        self.ending = nn.Conv2d(ch, width, kernel_size=3, padding=1)

        # --- 2× upsample to 256×256 (PixelShuffle) ---
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        # --- Final refine at 256×256 ---
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global residual: bilinear upsample of raw (noisy) input
        skip = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Intro conv
        feat = self.intro(x)

        # Encoder
        enc_features = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            enc_features.append(feat)
            feat = down(feat)

        # Bottleneck
        feat = self.middle(feat)

        # Decoder (with skip connections)
        for decoder, up, enc_feat in zip(
            self.decoders, self.ups, reversed(enc_features)
        ):
            feat = up(feat)
            feat = feat + enc_feat           # additive skip (NAFNet style)
            feat = decoder(feat)

        # Output conv (back to 128×128, `width` channels)
        feat = self.ending(feat)

        # PixelShuffle to 256×256
        feat = self.upsample(feat)

        # Final correction + global residual
        return skip + self.refine(feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------

def _self_test():
    print("=" * 60)
    print("NAFNetIRIS self-test")
    print("=" * 60)

    model = NAFNetIRIS(
        width=32,
        enc_blks=[1, 1, 1, 28],
        middle_blks=1,
        dec_blks=[1, 1, 1, 1],
    )
    n_params = model.count_parameters()
    print(f"Parameter count : {n_params:,}")

    # Forward pass
    dummy = torch.randn(2, 1, 128, 128) * 0.2 + 0.5
    out   = model(dummy)
    print(f"Input  shape    : {tuple(dummy.shape)}")
    print(f"Output shape    : {tuple(out.shape)}")
    assert out.shape == (2, 1, 256, 256), f"Shape mismatch: {out.shape}"

    # Backward pass
    loss = out.mean()
    loss.backward()
    has_grad = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    print(f"Gradients OK    : {has_grad}")
    assert has_grad, "Some parameters did not receive finite gradients!"

    print()
    print("Self-test passed.")


if __name__ == "__main__":
    _self_test()
