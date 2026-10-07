"""Phase 7 export: the variance graph through P1's `export_model`, ORT round trip, parity helpers.

CPU only, random-init P1 trunk (no checkpoint, no dataset): the graph-structure checks are the
checklist G1 items; the numeric check is ORT-CPU vs torch-CPU on the same graph, which is what the
committed `cpu_reference` block measures on real crops. fp16 export needs CUDA (P1 refuses it on
CPU) and is marked `gpu`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poseconf import p1_adapter
from poseconf.conformal.scores import set_mahalanobis
from poseconf.export import parity
from poseconf.export.to_onnx import VARIANCE_OUTPUTS, VarianceExportNet
from poseconf.models.variance_head import HeadConfig, VarianceHead, VarianceKeypointNet

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

RUN = "keypoint_a2"
HEAD = HeadConfig(256, 16, 11, 0.05, 64.0, 0.99, 2.0)


@pytest.fixture(scope="module")
def export_net(tmp_path_factory):
    """A random-init P1 trunk plus a head with non-trivial (random) output weights."""
    root = tmp_path_factory.mktemp("random_p1")
    p1_adapter.write_random_init_checkpoints(root, run=RUN)
    model, _ = p1_adapter.load_keypoint_model(RUN, paths=p1_adapter.P1Paths(root, root, root, root))
    torch.manual_seed(0)
    head = VarianceHead(HEAD)
    with torch.no_grad():
        head.out.weight.normal_(0.0, 0.05)
    return VarianceExportNet(VarianceKeypointNet(model, head)).eval()


@pytest.fixture(scope="module")
def exported(export_net, tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "vhead.onnx"
    return p1_adapter.onnx_export(
        export_net,
        output_path=path,
        input_shape=(1, 1, 256, 256),
        output_names=VARIANCE_OUTPUTS,
        half_precision=False,
    )


def test_graph_meets_the_export_checklist(exported):
    summary = p1_adapter.onnx_graph_summary(exported)
    assert summary["opset"] == 17
    assert summary["forbidden_ops_present"] == []
    assert [o["name"] for o in summary["outputs"]] == list(VARIANCE_OUTPUTS)
    assert [i["name"] for i in summary["inputs"]] == ["images"]
    shapes = {o["name"]: o["shape"] for o in summary["outputs"]}
    assert shapes["cov_chol"] == ["batch", 11, 3]
    assert shapes["keypoint_empty"] == ["batch", 11]
    onnx.checker.check_model(onnx.load(str(exported)))


def test_ort_cpu_reproduces_torch_cpu(export_net, exported):
    images = torch.randn(3, 1, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = {k: v.numpy() for k, v in zip(VARIANCE_OUTPUTS, export_net(images), strict=True)}
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    outputs = session.run(list(VARIANCE_OUTPUTS), {"images": images.numpy()})
    got = dict(zip(VARIANCE_OUTPUTS, outputs, strict=True))
    assert got["keypoint_empty"].dtype == np.bool_
    np.testing.assert_array_equal(got["keypoint_empty"], ref["keypoint_empty"])
    record = parity.parity_record({"torch_fp32": ref, "ort": got}, c1_quantile=3.0)
    stable = record["decode_stable"]["ort"]
    assert stable["keypoints_stable"] > 0
    assert record["tensor_parity"]["ort"]["sigma_max_relative"]["p99"] < 1e-3
    assert record["tensor_parity"]["ort"]["confidence"]["max"] < 1e-4


def test_export_outputs_are_p1_outputs_plus_covariance(export_net):
    images = torch.randn(2, 1, 256, 256, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        coords, confidence, heatmaps, chol = export_net.net(images)
        out = export_net(images)
    assert torch.equal(out[0], coords) and torch.equal(out[1], confidence)
    assert torch.equal(out[2], chol)
    assert out[3].dtype == torch.bool and out[3].shape == (2, 11)


@pytest.mark.gpu
def test_fp16_export_keeps_fp32_outputs(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("fp16 export needs CUDA")
    root = tmp_path / "p1"
    p1_adapter.write_random_init_checkpoints(root, run=RUN)
    model, _ = p1_adapter.load_keypoint_model(RUN, paths=p1_adapter.P1Paths(root, root, root, root))
    net = VarianceExportNet(VarianceKeypointNet(model, VarianceHead(HEAD)))
    path = p1_adapter.onnx_export(
        net,
        output_path=tmp_path / "vhead_fp16.onnx",
        input_shape=(1, 1, 256, 256),
        output_names=VARIANCE_OUTPUTS,
        half_precision=True,
    )
    summary = p1_adapter.onnx_graph_summary(path)
    assert summary["inputs"][0]["elem_type"] == "FLOAT16"
    types = {o["name"]: o["elem_type"] for o in summary["outputs"]}
    assert types["cov_chol"] == "FLOAT" and types["coords"] == "FLOAT"


# --------------------------------------------------------------------------------------------------
# parity helpers
# --------------------------------------------------------------------------------------------------


def test_c1_radius_matches_the_keypoint_set():
    rng = np.random.default_rng(3)
    chol = np.stack(
        [rng.uniform(0.5, 4, (20, 11)), rng.normal(size=(20, 11)), rng.uniform(0.5, 4, (20, 11))],
        -1,
    )
    empty = np.zeros((20, 11), dtype=bool)
    empty[0, 3] = True
    q = 2.7
    want = set_mahalanobis(
        rng.uniform(0, 256, (20, 11, 2)),
        np.ones(20, bool),
        q,
        cov=parity.chol_to_cov(chol),
        unconstrained=empty,
    ).radius_px
    got = parity.c1_radius(chol, empty, q)
    assert np.isinf(got[0]) and np.isinf(want[0])
    np.testing.assert_allclose(got[1:], want[1:], rtol=1e-12)


def test_gate_status_scores_variance_outputs_only():
    def block(cov_max: float, mismatches: int = 0) -> dict:
        return {
            "cov_chol": {"max": cov_max},
            "cov": {"max": cov_max},
            "keypoint_empty": {"mismatches": mismatches},
        }

    record = {"tensor_parity": {"ort_fp32": block(5e-5), "ort_fp16": block(0.2)}}
    status = parity.gate_status(record, fp16_tensor_max_abs=0.05)
    assert status["variance_outputs_fp32_met"]
    assert status["failed"] == ["ort_fp16.tensor.cov", "ort_fp16.tensor.cov_chol"]
    assert status["inherited"]["p1_keypoint_parity"] == "unmet"
    record["tensor_parity"]["ort_fp32"] = block(5e-5, mismatches=1)
    assert not parity.gate_status(record, 0.05)["variance_outputs_fp32_met"]
