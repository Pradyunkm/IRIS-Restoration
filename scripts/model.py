"""
model.py

Phase 2 baseline model for IRIS: a lightweight residual CNN that maps
128x128 NoisyLR -> 256x256 restored output, using PixelShuffle for the
2x upsampling stage.

Deliberately simple, per the project roadmap:
    Phase 1: dataset audit           DONE (dataset_audit.py, visualize_pairs.py)
    Phase 2: simple baseline + L1    <- this file
    Phase 3: stronger backbone
    Phase 4: degradation encoder
    Phase 5: adaptive routing
    ...

No degradation encoder, no expert routing, no multi-term loss here on
purpose -- this exists to give every later addition (Phase 4+) a
quantitative baseline to beat. Do not add complexity to this file;
create new model classes for later phases instead, so the baseline
numbers stay reproducible and comparable.

Architecture:
    Conv stem (1 -> C channels)
    N residual blocks (Conv-ReLU-Conv + skip), operating at 128x128
    PixelShuffle upsampling head (C -> 4C -> shuffle -> C, at 256x256)
    Final conv (C -> 1) with a global residual: output = upsampled_input + refine(features)

The global residual (adding an upsampled copy of the input) means the
network only needs to learn the *correction* on top of a naive
upsample, which is standard practice for SR/restoration baselines and
trains faster than learning the mapping from scratch.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class PixelShuffleUpsample(nn.Module):
    """2x spatial upsampling via sub-pixel convolution (PixelShuffle)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)  # (B, C, H, W) -> (B, C, 2H, 2W)
        return self.relu(x)


class IRISBaseline(nn.Module):
    """
    Baseline restoration + 2x super-resolution network.

    Input : (B, 1, 128, 128), float32, UNCLIPPED (may be outside [0,1])
    Output: (B, 1, 256, 256), float32 (not clipped internally -- clip
            at evaluation/visualization time, not inside the model,
            so gradients near 0/1 aren't killed during training)
    """

    def __init__(self, channels: int = 64, num_res_blocks: int = 8):
        super().__init__()

        self.stem = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.stem_relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.pre_upsample_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = PixelShuffleUpsample(channels)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Global residual path: naive bilinear upsample of the raw input.
        # The network only has to learn the correction on top of this.
        naive_upsample = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )

        feat = self.stem_relu(self.stem(x))
        feat = self.res_blocks(feat)
        feat = self.pre_upsample_conv(feat)
        feat = self.upsample(feat)  # now at 256x256

        correction = self.refine(feat)

        return naive_upsample + correction

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class IRISStronger(nn.Module):
    """
    Phase 3: stronger backbone. Same building blocks as IRISBaseline
    (residual blocks + PixelShuffle + global residual) -- ONLY the
    capacity changes (more channels, more blocks, one extra refine
    layer). This isolates capacity as the variable being tested,
    matching the ablation discipline used for the loss experiment:
    same architecture family, same everything else, one change at a time.

    Motivation (from visual inspection of Experiment 1 and Experiment 2
    predictions): dense, high-frequency content (e.g. a crowd of faces)
    was still under-resolved even after adding structural/edge loss,
    suggesting the 813K-parameter baseline lacks the representational
    capacity to reconstruct that much detail, not just the wrong loss
    signal. This variant tests that hypothesis directly.

    Default config (channels=112, num_res_blocks=16) is roughly ~3x the
    baseline's parameter count -- large enough to meaningfully test the
    capacity hypothesis, still comfortably small for a 6GB GPU at
    batch_size=16, 256x256 output.
    """

    def __init__(self, channels: int = 112, num_res_blocks: int = 16):
        super().__init__()

        self.stem = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.stem_relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.pre_upsample_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = PixelShuffleUpsample(channels)

        # One extra post-upsample residual block at full 256x256
        # resolution, before the final refine head -- gives the network
        # capacity to fix detail specifically at the target resolution,
        # not just at 128x128 before upsampling.
        self.post_upsample_block = ResidualBlock(channels)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        naive_upsample = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )

        feat = self.stem_relu(self.stem(x))
        feat = self.res_blocks(feat)
        feat = self.pre_upsample_conv(feat)
        feat = self.upsample(feat)  # now at 256x256
        feat = self.post_upsample_block(feat)

        correction = self.refine(feat)

        return naive_upsample + correction

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _self_test():
    print("=" * 60)
    print("IRISBaseline self-test")
    print("=" * 60)

    model = IRISBaseline(channels=64, num_res_blocks=8)
    n_params = model.count_parameters()
    print(f"Parameter count: {n_params:,}")

    dummy_input = torch.randn(2, 1, 128, 128) * 0.2 + 0.5  # roughly [0,1]-ish, unclipped
    output = model(dummy_input)

    print(f"Input shape : {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (2, 1, 256, 256), "Output shape mismatch!"

    # Backward pass sanity check -- confirms gradients flow through the
    # whole network (residual blocks + pixel shuffle head + global skip).
    loss = output.mean()
    loss.backward()
    has_grad = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    print(f"Gradients finite and present for all params: {has_grad}")
    assert has_grad, "Some parameters did not receive finite gradients!"

    print()
    print("Self-test passed.")


def _self_test_stronger():
    print()
    print("=" * 60)
    print("IRISStronger self-test")
    print("=" * 60)

    model = IRISStronger(channels=112, num_res_blocks=16)
    n_params = model.count_parameters()
    print(f"Parameter count: {n_params:,}")

    dummy_input = torch.randn(2, 1, 128, 128) * 0.2 + 0.5
    output = model(dummy_input)

    print(f"Input shape : {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (2, 1, 256, 256), "Output shape mismatch!"

    loss = output.mean()
    loss.backward()
    has_grad = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
    print(f"Gradients finite and present for all params: {has_grad}")
    assert has_grad, "Some parameters did not receive finite gradients!"

    print()
    print("Self-test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRIS model self-tests")
    parser.parse_args()
    _self_test()
    _self_test_stronger()