"""Export `VarianceKeypointNet` to ONNX (opset 17, fp32/fp16) through P1's `export_model`. Phase 7.

The graph signature is `images -> coords, confidence, cov_chol, keypoint_empty`:

* `coords`, `confidence` are P1's tensors, untouched (CLAUDE.md invariant 10).
* `cov_chol (B, K, 3)` is the variance head's lower Cholesky factor, crop px.
* `keypoint_empty (B, K)` bool marks channels with no positive activation inside P1's refine window.
  C1/C2 treat such a keypoint as unconstrained (its set is the whole image), exactly as Levels B/C
  and the shift matrix did with the dump's `heatmap_empty`. Without it a deployed head would read
  the readout's value on an empty channel (σ = 1 px, a fabricated tight ellipse) as a real
  covariance. It is computed by the same `refine_weights` the head reads out with.

P1's `heatmaps` output is dropped: no deployed score needs it (B2 is not deployable).

fp16 follows P1's policy: the *module* is halved, so the heatmap decode and the head's readout keep
their explicit fp32 casts and the graph is mixed precision by construction.
"""

from __future__ import annotations

from torch import Tensor, nn

from poseconf.models.variance_head import VarianceKeypointNet, refine_weights

__all__ = ["VARIANCE_OUTPUTS", "VarianceExportNet"]

#: Output names of the exported variance graph, in forward order.
VARIANCE_OUTPUTS = ("coords", "confidence", "cov_chol", "keypoint_empty")


class VarianceExportNet(nn.Module):
    """`VarianceKeypointNet` with the export signature: heatmaps dropped, empty mask added."""

    def __init__(self, net: VarianceKeypointNet) -> None:
        """Wrap a loaded variance network.

        Args:
            net: The P1 trunk plus variance head (`train_variance.load_variance_net`).
        """
        super().__init__()
        self.net = net

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run P1 and the head.

        Args:
            images: (B, 1, S, S) normalised crops.

        Returns:
            `(coords (B, K, 2), confidence (B, K), cov_chol (B, K, 3), keypoint_empty (B, K))`.
        """
        coords, confidence, heatmaps, cov_chol = self.net(images)
        _, empty = refine_weights(heatmaps, self.net.p1.head)
        return coords, confidence, cov_chol, empty
