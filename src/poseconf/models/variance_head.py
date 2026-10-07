"""Heteroscedastic keypoint variance head and `VarianceKeypointNet` over a frozen P1 KeypointNet.

IMPLEMENTATION_PLAN §1.6. The head reads P1's decoder features (the input to the final 1x1 conv,
(B, 256, 64, 64)) and predicts, per heatmap pixel and keypoint, `(log σx, log σy, atanh ρ)`. Each
keypoint's parameters are the average of those maps under the exact distribution P1's soft-argmax
refines its coordinate over (`relu(h)^2 · window / Σ`), so the covariance describes the point P1
returned. Only multiply, exp and reduce ops: no ArgMax, TopK or NonZero in an exported graph.

The head never changes the pose (CLAUDE.md invariant 10). `VarianceKeypointNet.forward` calls the
P1 module's own `forward` and observes its decoder features through a pre-hook, so the coordinates,
confidences and heatmaps it returns *are* P1's tensors. The trunk is frozen (no gradients, always in
eval mode so BatchNorm statistics never move); `p1_state_hash` lets a test or a training run prove it.

Units: σ in crop px (the network-input crop, 256 px), Σ in crop px². `cov_chol (B, K, 3)` holds the
lower Cholesky factor `L = [[l0, 0], [l1, l2]]`, `Σ = L Lᵀ`, with `l0 = σx`, `l1 = ρσy`,
`l2 = σy·sqrt(1 − ρ²)`. Clamps (`σ ∈ [σ_min, σ_max]`, `|ρ| ≤ ρ_max < 1`) make Σ positive definite
for every input.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "HeadConfig",
    "VarianceHead",
    "VarianceKeypointNet",
    "chol_to_cov",
    "p1_state_hash",
    "refine_weights",
]

#: Parameters predicted per keypoint: log σx, log σy, atanh ρ.
PARAMS_PER_KEYPOINT = 3


@dataclass(frozen=True)
class HeadConfig:
    """Variance-head architecture and output clamps (from `configs/variance_head.yaml`).

    Attributes:
        in_channels: Decoder feature channels (P1: 256).
        hidden_channels: Width of the 3x3 conv.
        num_keypoints: K (P1: 11).
        sigma_min: Lower σ clamp, crop px.
        sigma_max: Upper σ clamp, crop px.
        rho_max: Largest |ρ|; < 1 keeps Σ positive definite.
        sigma_init: σ every keypoint starts at (final conv zero weights, bias log σ_init).
    """

    in_channels: int
    hidden_channels: int
    num_keypoints: int
    sigma_min: float
    sigma_max: float
    rho_max: float
    sigma_init: float

    def __post_init__(self) -> None:
        """Validate the clamps."""
        if not 0.0 < self.sigma_min < self.sigma_init < self.sigma_max:
            raise ValueError("need 0 < sigma_min < sigma_init < sigma_max")
        if not 0.0 < self.rho_max < 1.0:
            raise ValueError("rho_max must lie in (0, 1)")

    @classmethod
    def from_dict(cls, block: dict[str, Any]) -> HeadConfig:
        """Build from the config's `head` block."""
        return cls(
            in_channels=int(block["in_channels"]),
            hidden_channels=int(block["hidden_channels"]),
            num_keypoints=int(block["num_keypoints"]),
            sigma_min=float(block["sigma_min"]),
            sigma_max=float(block["sigma_max"]),
            rho_max=float(block["rho_max"]),
            sigma_init=float(block["sigma_init"]),
        )


def refine_weights(heatmaps: Tensor, head: Any) -> tuple[Tensor, Tensor]:
    """The normalised distribution P1's soft-argmax refine pass takes its coordinate from.

    Built as `SoftArgmax2d.forward` builds it (fp32, locate centroid of `relu(h)^locate`, Gaussian
    window, `relu(h)^refine · window`), through the head's own `_centroid`, so its mean is P1's
    coordinate. Same distribution as `poseconf.data.dump.heatmap_moments` (tested).

    Args:
        heatmaps: (B, K, H, W) raw heatmaps.
        head: P1's `SoftArgmax2d` (`locate_exponent`, `refine_exponent`, `window_sigma`,
            `_centroid`).

    Returns:
        `(probability (B, K, H, W) fp32, empty (B, K) bool)`. An empty channel (no positive
        activation inside the window) has all-zero probability.
    """
    maps = heatmaps.detach().float()
    height, width = maps.shape[-2:]
    xs = torch.arange(width, dtype=maps.dtype, device=maps.device)
    ys = torch.arange(height, dtype=maps.dtype, device=maps.device)
    rectified = torch.relu(maps)
    locate = head._centroid(rectified.pow(head.locate_exponent), xs, ys)
    two_sigma_squared = 2.0 * head.window_sigma * head.window_sigma
    offset_x = xs.view(1, 1, 1, width) - locate[..., 0].unsqueeze(-1).unsqueeze(-1)
    offset_y = ys.view(1, 1, height, 1) - locate[..., 1].unsqueeze(-1).unsqueeze(-1)
    window = torch.exp(-(offset_x * offset_x + offset_y * offset_y) / two_sigma_squared)
    weights = rectified.pow(head.refine_exponent) * window
    mass = weights.sum(dim=(-2, -1), keepdim=True)
    probability = weights / mass.clamp_min(torch.finfo(weights.dtype).tiny)
    return probability, mass.squeeze(-1).squeeze(-1) == 0


def chol_to_cov(chol: Tensor) -> Tensor:
    """(…, 3) lower Cholesky factors → (…, 2, 2) covariances `L Lᵀ`."""
    l0, l1, l2 = chol.unbind(-1)
    cxx = l0 * l0
    cxy = l0 * l1
    cyy = l1 * l1 + l2 * l2
    return torch.stack((torch.stack((cxx, cxy), -1), torch.stack((cxy, cyy), -1)), -2)


class VarianceHead(nn.Module):
    """3x3 conv → ReLU → 1x1 conv to per-pixel `(log σx, log σy, atanh ρ)`, heatmap-weighted readout."""

    def __init__(self, config: HeadConfig) -> None:
        """Build the head; the final conv starts at zero weight so every σ starts at `sigma_init`.

        Args:
            config: Architecture and clamps.
        """
        super().__init__()
        self.config = config
        k = config.num_keypoints
        self.conv = nn.Conv2d(config.in_channels, config.hidden_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.out = nn.Conv2d(config.hidden_channels, PARAMS_PER_KEYPOINT * k, 1)
        nn.init.zeros_(self.out.weight)
        bias = torch.zeros(k, PARAMS_PER_KEYPOINT)
        bias[:, 0] = bias[:, 1] = math.log(config.sigma_init)
        with torch.no_grad():
            self.out.bias.copy_(bias.reshape(-1))
        self.log_sigma_min = math.log(config.sigma_min)
        self.log_sigma_max = math.log(config.sigma_max)

    def forward(self, features: Tensor, heatmaps: Tensor, soft_argmax: Any) -> Tensor:
        """Predict each keypoint's Cholesky factor.

        Args:
            features: (B, C, H, W) decoder features (detached).
            heatmaps: (B, K, H, W) P1 heatmaps (detached).
            soft_argmax: P1's `SoftArgmax2d`, for the readout weights.

        Returns:
            (B, K, 3) fp32 `cov_chol`, crop px.
        """
        batch, _, height, width = features.shape
        k = self.config.num_keypoints
        raw = self.out(self.relu(self.conv(features))).float()
        raw = raw.view(batch, k, PARAMS_PER_KEYPOINT, height, width)
        probability, _ = refine_weights(heatmaps, soft_argmax)
        params = (raw * probability.unsqueeze(2)).sum(dim=(-2, -1))
        log_sx = params[..., 0].clamp(self.log_sigma_min, self.log_sigma_max)
        log_sy = params[..., 1].clamp(self.log_sigma_min, self.log_sigma_max)
        rho = torch.tanh(params[..., 2]).clamp(-self.config.rho_max, self.config.rho_max)
        sx, sy = torch.exp(log_sx), torch.exp(log_sy)
        return torch.stack((sx, rho * sy, sy * torch.sqrt(1.0 - rho * rho)), dim=-1)


class VarianceKeypointNet(nn.Module):
    """A frozen P1 `KeypointNet` plus a `VarianceHead`; P1's outputs pass through untouched."""

    def __init__(self, p1_model: nn.Module, head: VarianceHead) -> None:
        """Wrap and freeze.

        Args:
            p1_model: A P1 `KeypointNet` (EMA weights, from `p1_adapter.load_keypoint_model`).
            head: The variance head.

        Raises:
            TypeError: If the network's decoder does not end in the expected 1x1 conv.
        """
        super().__init__()
        final = p1_model.decoder[-1]
        if not isinstance(final, nn.Conv2d) or final.kernel_size != (1, 1):
            raise TypeError(f"expected KeypointNet.decoder to end in a 1x1 Conv2d, got {final}")
        if final.out_channels != head.config.num_keypoints:
            raise ValueError("head and P1 network disagree on the number of keypoints")
        self.p1 = p1_model
        self.head = head
        for parameter in self.p1.parameters():
            parameter.requires_grad_(False)
        self.p1.eval()

    def train(self, mode: bool = True) -> VarianceKeypointNet:
        """Set the head's mode; the P1 trunk stays in eval mode (BatchNorm statistics frozen)."""
        super().train(mode)
        self.p1.eval()
        return self

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run P1 unchanged and predict a covariance per keypoint.

        Args:
            images: (B, 1, S, S) normalised crops.

        Returns:
            `(coords (B, K, 2) crop px, confidence (B, K), heatmaps (B, K, H, W), cov_chol (B, K, 3))`;
            the first three are exactly what `p1_model(images)` returns.
        """
        captured: list[Tensor] = []

        def on_final_conv(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            captured.append(inputs[0])

        handle = self.p1.decoder[-1].register_forward_pre_hook(on_final_conv)
        try:
            with torch.no_grad():
                coords, confidence, heatmaps = self.p1(images)
        finally:
            handle.remove()
        cov_chol = self.head(captured[0].detach(), heatmaps.detach(), self.p1.head)
        return coords, confidence, heatmaps, cov_chol


def p1_state_hash(model: nn.Module) -> str:
    """SHA-256 over every tensor of `model.state_dict()` (parameters and buffers), name-sorted."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
