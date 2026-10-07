"""Phase 5 variance head: P1 is frozen and bit-identical, covariances are PD, export-safe ops.

CLAUDE.md invariant 10: the head never changes the pose. These tests must stay green.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch

from poseconf import p1_adapter
from poseconf.data import dump as dump_module
from poseconf.engine import train_variance
from poseconf.models.variance_head import (
    HeadConfig,
    VarianceHead,
    VarianceKeypointNet,
    chol_to_cov,
    p1_state_hash,
    refine_weights,
)

RUN = "keypoint_a2"
HEAD = HeadConfig(
    in_channels=256,
    hidden_channels=16,
    num_keypoints=11,
    sigma_min=0.05,
    sigma_max=64.0,
    rho_max=0.99,
    sigma_init=2.0,
)
FORBIDDEN_OPS = {"ArgMax", "ArgMin", "TopK", "NonZero"}


@pytest.fixture(scope="module")
def random_p1(tmp_path_factory):
    """A random-init P1 keypoint network, loaded through the adapter like the real one."""
    root = tmp_path_factory.mktemp("random_p1")
    p1_adapter.write_random_init_checkpoints(root, run=RUN)
    paths = p1_adapter.P1Paths(root, root, root, root)
    return paths


def _p1_model(paths):
    model, _ = p1_adapter.load_keypoint_model(RUN, paths=paths)
    return model


def _net(paths, seed: int = 0) -> VarianceKeypointNet:
    torch.manual_seed(seed)
    return VarianceKeypointNet(_p1_model(paths), VarianceHead(HEAD))


def _batch(n: int = 4, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(n, 11, generator=generator) > 0.2
    return {
        "image": torch.randn(n, 1, 256, 256, generator=generator),
        "keypoints": torch.rand(n, 11, 2, generator=generator) * 255,
        "mask": mask,
    }


def test_outputs_are_p1_outputs_bit_for_bit(random_p1):
    net = _net(random_p1)
    bare = _p1_model(random_p1)
    images = _batch()["image"]
    with torch.no_grad():
        expected = bare(images)
    coords, confidence, heatmaps, chol = net(images)
    for got, want in zip((coords, confidence, heatmaps), expected, strict=True):
        assert torch.equal(got, want)
    assert chol.shape == (4, 11, 3)
    assert not net.p1.decoder[-1]._forward_pre_hooks  # hook removed after every forward


def test_frozen_parameter_hash_unchanged_after_a_training_step(random_p1):
    net = _net(random_p1)
    before = p1_state_hash(net.p1)
    head_before = {k: v.clone() for k, v in net.head.state_dict().items()}
    net.train()
    optimiser = torch.optim.AdamW(net.head.parameters(), lr=1e-2)
    for seed in range(2):  # the second step moves the 3x3 conv too (the 1x1 starts at zero)
        train_variance._step(
            net,
            _batch(seed=seed),
            optimiser,
            device=torch.device("cpu"),
            amp=torch.bfloat16,
            beta=0.5,
            grad_clip=1.0,
        )
    assert p1_state_hash(net.p1) == before
    assert all(p.grad is None and not p.requires_grad for p in net.p1.parameters())
    after = net.head.state_dict()
    assert any(not torch.equal(head_before[k], after[k]) for k in head_before)


def test_trunk_stays_in_eval_mode(random_p1):
    net = _net(random_p1).train()
    assert net.training and net.head.training
    assert not net.p1.training
    assert all(not module.training for module in net.p1.modules())


def test_state_hash_sees_buffers(random_p1):
    model = _p1_model(random_p1)
    before = p1_state_hash(model)
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    bn.running_mean.add_(1e-6)
    assert p1_state_hash(model) != before


def test_initial_sigma_is_sigma_init(random_p1):
    net = _net(random_p1)
    _, _, heatmaps, chol = net(_batch()["image"])
    _, empty = refine_weights(heatmaps, net.p1.head)
    expected = torch.tensor([HEAD.sigma_init, 0.0, HEAD.sigma_init])
    assert torch.allclose(chol[~empty], expected.expand_as(chol[~empty]))
    # An empty channel has zero readout weight: σ = exp(0) = 1, still PD; scores mark it unconstrained.
    assert torch.allclose(chol[empty], torch.tensor([1.0, 0.0, 1.0]).expand_as(chol[empty]))


@pytest.mark.parametrize("feature_scale", [1e-3, 1.0, 1e3])
def test_covariances_positive_definite_and_clamped(random_p1, feature_scale):
    head = VarianceHead(HEAD)
    torch.manual_seed(3)
    for module in (head.conv, head.out):
        torch.nn.init.normal_(module.weight, std=1.0)
    features = torch.randn(3, 256, 64, 64) * feature_scale
    heatmaps = torch.randn(3, 11, 64, 64)
    heatmaps[0, 0] = -1.0  # an empty channel: no positive activation
    chol = head(features, heatmaps, _p1_model(random_p1).head)
    assert torch.isfinite(chol).all()
    cov = chol_to_cov(chol.double())
    assert (torch.linalg.eigvalsh(cov) > 0).all()
    torch.linalg.cholesky(cov)
    sx = chol[..., 0]
    sy = torch.hypot(chol[..., 1], chol[..., 2])
    for sigma in (sx, sy):
        assert (sigma >= HEAD.sigma_min * (1 - 1e-6)).all()
        assert (sigma <= HEAD.sigma_max * (1 + 1e-6)).all()
    assert (chol[..., 1].abs() / sy <= HEAD.rho_max + 1e-6).all()


def test_chol_to_cov_is_l_lt():
    chol = torch.tensor([[1.5, -0.4, 0.7]], dtype=torch.float64)
    lower = torch.tensor([[[1.5, 0.0], [-0.4, 0.7]]], dtype=torch.float64)
    assert torch.allclose(chol_to_cov(chol), lower @ lower.transpose(-1, -2))


def test_refine_weights_are_the_dump_moment_distribution(random_p1):
    model = _p1_model(random_p1)
    images = _batch(n=3, seed=5)["image"]
    with torch.no_grad():
        coords, _, heatmaps = model(images)
    probability, empty = refine_weights(heatmaps, model.head)
    moments = dump_module.heatmap_moments(heatmaps, model.head, float(model.stride))
    assert torch.equal(empty, moments.empty)
    xs = torch.arange(64, dtype=torch.float32)
    mean_x = (probability.sum(-2) * xs).sum(-1)
    mean_y = (probability.sum(-1) * xs).sum(-1)
    mean_crop = (torch.stack((mean_x, mean_y), -1) + 0.5) * model.stride - 0.5
    hit = ~empty
    assert torch.allclose(mean_crop[hit], coords[hit], atol=1e-4)
    assert torch.allclose(mean_crop[hit], moments.mean_crop[hit], atol=1e-4)


def test_exported_graph_has_no_forbidden_ops(random_p1):
    onnx = pytest.importorskip("onnx")
    net = _net(random_p1).eval()
    buffer = io.BytesIO()
    torch.onnx.export(
        net,
        (torch.randn(1, 1, 256, 256),),
        buffer,
        opset_version=17,
        dynamo=False,
        input_names=["images"],
        output_names=["coords", "confidence", "heatmaps", "cov_chol"],
    )
    graph = onnx.load_from_string(buffer.getvalue()).graph
    ops = {node.op_type for node in graph.node}
    assert not ops & FORBIDDEN_OPS
    assert [o.name for o in graph.output] == ["coords", "confidence", "heatmaps", "cov_chol"]


def test_rejects_a_head_with_the_wrong_keypoint_count(random_p1):
    wrong = HeadConfig(256, 16, 7, 0.05, 64.0, 0.99, 2.0)
    with pytest.raises(ValueError, match="number of keypoints"):
        VarianceKeypointNet(_p1_model(random_p1), VarianceHead(wrong))


def test_head_config_validates_clamps():
    with pytest.raises(ValueError):
        HeadConfig(256, 16, 11, 0.05, 64.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        HeadConfig(256, 16, 11, 3.0, 64.0, 0.9, 2.0)


def test_load_variance_net_refuses_another_trunk(random_p1, tmp_path):
    net = _net(random_p1)
    state = {
        "head": net.head.state_dict(),
        "head_config": HEAD.__dict__,
        "p1_run": RUN,
        "p1_checkpoint_sha256": "0" * 64,
        "p1_state_hash": p1_state_hash(net.p1),
    }
    path = tmp_path / "best.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="sha256"):
        train_variance.load_variance_net(path, paths=random_p1, device=torch.device("cpu"))
    from poseconf.provenance import sha256_file

    state["p1_checkpoint_sha256"] = sha256_file(Path(random_p1.p1_runs) / RUN / "best.pt")
    torch.save(state, path)
    loaded, meta = train_variance.load_variance_net(
        path, paths=random_p1, device=torch.device("cpu")
    )
    images = _batch(n=2)["image"]
    for got, want in zip(loaded(images), net(images), strict=True):
        assert torch.equal(got, want)
    assert "head" not in meta


# --- the real checkpoint -------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.dataset
@pytest.mark.slow
def test_real_checkpoint_bit_identical_under_p1_autocast(local_paths):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    bare, _ = p1_adapter.load_keypoint_model(RUN, paths=local_paths, device=device)
    net = VarianceKeypointNet(
        p1_adapter.load_keypoint_model(RUN, paths=local_paths, device=device)[0],
        VarianceHead(HEAD),
    ).to(device)
    torch.manual_seed(0)
    images = torch.randn(8, 1, 256, 256, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = bare(images)
        got = net(images)
    for a, b in zip(got[:3], expected, strict=True):
        assert torch.equal(a, b)
    assert p1_state_hash(net.p1) == p1_state_hash(bare)
    cov = chol_to_cov(got[3].double())
    assert (torch.linalg.eigvalsh(cov) > 0).all()
    np.testing.assert_array_equal(got[0].float().cpu().numpy(), expected[0].float().cpu().numpy())
