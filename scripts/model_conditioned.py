"""
model_conditioned.py

Experiment 5: degradation-aware conditioning, built on top of the
IRISStronger backbone (Experiment 3's architecture), the final
future-work item from the original project plan.

DegradationEncoder: a small CNN that looks at the raw NoisyLR input and
produces a compact embedding meant to capture "what kind of degradation
is present" (noise level, speckle severity, etc.) -- WITHOUT being told
explicitly what those are; it learns whatever representation helps
minimize the downstream restoration loss.

FiLM (Feature-wise Linear Modulation): the embedding is projected, per
residual block, into a per-channel scale (gamma) and shift (beta) that
modulate that block's features: out = gamma * features + beta. This lets
the network adapt its internal processing based on the detected
degradation characteristics of each specific input.

Usage (self-test):
    python model_conditioned.py
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ResidualBlock, PixelShuffleUpsample


class DegradationEncoder(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):
        feat = self.net(x).flatten(1)
        return self.fc(feat)


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.film = nn.Linear(embed_dim, channels * 2)

    def forward(self, x, embedding):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        gamma_beta = self.film(embedding)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        out = out * (1.0 + gamma) + beta
        return identity + out


class IRISConditioned(nn.Module):
    def __init__(self, channels: int = 112, num_res_blocks: int = 16, embed_dim: int = 32):
        super().__init__()

        self.degradation_encoder = DegradationEncoder(embed_dim=embed_dim)

        self.stem = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.stem_relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.ModuleList(
            [FiLMResidualBlock(channels, embed_dim) for _ in range(num_res_blocks)]
        )

        self.pre_upsample_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = PixelShuffleUpsample(channels)
        self.post_upsample_block = ResidualBlock(channels)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        naive_upsample = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        embedding = self.degradation_encoder(x)

        feat = self.stem_relu(self.stem(x))
        for block in self.res_blocks:
            feat = block(feat, embedding)

        feat = self.pre_upsample_conv(feat)
        feat = self.upsample(feat)
        feat = self.post_upsample_block(feat)

        correction = self.refine(feat)
        return naive_upsample + correction

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _self_test():
    print("=" * 60)
    print("IRISConditioned self-test")
    print("=" * 60)

    model = IRISConditioned(channels=112, num_res_blocks=16, embed_dim=32)
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
    assert has_grad, "Some parameters did not receive finite gradients (encoder or FiLM likely disconnected)!"

    with torch.no_grad():
        emb1 = model.degradation_encoder(torch.randn(1, 1, 128, 128))
        emb2 = model.degradation_encoder(torch.randn(1, 1, 128, 128))
        diff = (emb1 - emb2).abs().mean().item()
    print(f"Mean embedding difference for two random inputs: {diff:.4f} (should be > 0)")
    assert diff > 1e-6, "Encoder produced identical embeddings for different inputs!"

    print()
    print("Self-test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRISConditioned model self-test")
    parser.parse_args()
    _self_test()