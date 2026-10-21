# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Unit tests for the LayerTestRunner and kbench config parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn import RMSNorm
from testbed.correctness import print_correctness_report
from testbed.harness import dict_to_dataclass
from testbed.harnesses.rms_norm import (
    RMSNormDynamicParams,
    RMSNormHarness,
    RMSNormStaticParams,
)
from testbed.ir_dump import dump_mo_ir
from testbed.runner import LayerTestRunner, create_session

# -- IR dump -----------------------------------------------------------------


def test_ir_dump(tmp_path: Path) -> None:
    """dump_mo_ir writes non-empty MO IR containing expected ops."""
    dim = 4096
    layer = RMSNorm(dim=dim, dtype=DType.bfloat16, eps=1e-6)
    layer.load_state_dict({"weight": torch.randn(dim, dtype=torch.bfloat16)})

    input_type = TensorType(
        dtype=DType.bfloat16,
        shape=["total_tokens", dim],
        device=DeviceRef.GPU(),
    )
    with Graph("RMSNorm", input_types=(input_type,)) as graph:
        (x,) = graph.inputs
        graph.output(layer(x.tensor))

    mo_path = dump_mo_ir(graph, tmp_path / "rms_norm.mo.mlir")
    assert mo_path.exists()
    content = mo_path.read_text()
    assert len(content) > 100
    assert "rms_norm" in content.lower() or "custom" in content.lower()


def test_dump_ir_via_runner_rms_norm(tmp_path: Path) -> None:
    """LayerTestRunner.dump_ir works for RMSNorm harness."""
    session, device = create_session()
    harness = RMSNormHarness(
        RMSNormStaticParams(dim=4096, eps=1e-6), session, device
    )
    runner = LayerTestRunner(harness)
    mo_path = runner.dump_ir(tmp_path / "rms_norm.mo.mlir")
    assert mo_path.exists()
    content = mo_path.read_text()
    assert len(content) > 100
    assert "rms_norm" in content.lower() or "custom" in content.lower()


# -- Correctness -------------------------------------------------------------


def test_correctness_rms_norm() -> None:
    """RMSNorm MAX vs torch outputs match within tolerance."""
    session, device = create_session()
    harness = RMSNormHarness(
        RMSNormStaticParams(dim=4096, eps=1e-6), session, device
    )
    runner = LayerTestRunner(harness)
    shapes = [
        RMSNormDynamicParams(batch_size=1, seq_len=128),
        RMSNormDynamicParams(batch_size=4, seq_len=256),
        RMSNormDynamicParams(batch_size=1, seq_len=1),
    ]
    results = runner.correctness(
        shapes, atol=1e-2, rtol=1e-2, cos_threshold=0.001
    )
    print_correctness_report(results)
    for r in results:
        assert r.passed, f"Correctness failed for {r.label}: {r}"


# -- dict_to_dataclass -------------------------------------------------------


@dataclass
class _SampleParams:
    required_a: int
    required_b: str
    optional_c: float = 1.0


def test_dict_to_dataclass_valid() -> None:
    result = dict_to_dataclass(
        _SampleParams, {"required_a": 42, "required_b": "hello"}
    )
    assert isinstance(result, _SampleParams)
    assert result.required_a == 42
    assert result.required_b == "hello"
    assert result.optional_c == 1.0


def test_dict_to_dataclass_with_optional() -> None:
    result = dict_to_dataclass(
        _SampleParams,
        {"required_a": 1, "required_b": "x", "optional_c": 9.9},
    )
    assert isinstance(result, _SampleParams)
    assert result.optional_c == 9.9


def test_dict_to_dataclass_missing_required() -> None:
    with pytest.raises(TypeError, match=r"missing required fields.*required_b"):
        dict_to_dataclass(_SampleParams, {"required_a": 42})


def test_dict_to_dataclass_extra_keys() -> None:
    with pytest.raises(TypeError, match=r"unexpected fields.*unknown"):
        dict_to_dataclass(
            _SampleParams,
            {"required_a": 1, "required_b": "x", "unknown": 99},
        )
