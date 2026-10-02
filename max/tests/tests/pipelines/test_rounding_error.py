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
"""Tests the offline dequant-vs-source rounding-error checker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from max.pipelines.weights.fp4_quantization import FP4Format
from max.pipelines.weights.fp6_quantization import FP6Format
from max.pipelines.weights.quantize_checkpoint import quantize_checkpoint
from max.pipelines.weights.rounding_error import (
    compare_arrays,
    compare_checkpoints,
    compare_written,
    quantized_weight_names,
)
from safetensors.torch import save_file

_EXPERT = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
_N, _K = 64, 128


def _write_source(src: Path) -> None:
    """Writes a one-shard bf16 checkpoint holding a single expert weight."""
    src.mkdir(parents=True, exist_ok=True)
    save_file(
        {_EXPERT: torch.randn(_N, _K, dtype=torch.float32).to(torch.bfloat16)},
        src / "model-00001-of-00001.safetensors",
    )
    (src / "config.json").write_text(json.dumps({"model_type": "test"}))


def test_compare_arrays_is_zero_for_identical_values() -> None:
    """A matching pair must not invent error."""
    values = np.ones((4, 8), dtype=np.float32)
    error = compare_arrays("w", values, values)
    assert error.rel_l2 == 0.0
    assert error.max_abs == 0.0


def test_export_records_in_memory_rounding_error(tmp_path: Path) -> None:
    """Export dequants the written tensors and records relative L2."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_source(src)

    stats = quantize_checkpoint(src, dst, FP4Format.NVFP4)

    assert stats["quantized"] == 1
    assert 0 < float(stats["mean_rel_l2"]) < 0.2
    assert float(stats["max_rel_l2"]) == float(stats["mean_rel_l2"])


def test_compare_checkpoints_matches_export_for_nvfp4(tmp_path: Path) -> None:
    """The CLI path, reading dest off disk, agrees with the in-export check."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_source(src)

    stats = quantize_checkpoint(src, dst, FP4Format.NVFP4)
    report = compare_checkpoints(src, dst)

    assert len(report.tensors) == 1
    assert report.tensors[0].name == _EXPERT
    assert report.mean_rel_l2 == pytest.approx(
        float(stats["mean_rel_l2"]), rel=1e-5, abs=1e-6
    )


def test_compare_checkpoints_infers_mxfp6_format(tmp_path: Path) -> None:
    """Dest config.json is enough; the caller need not pass --format."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_source(src)

    quantize_checkpoint(src, dst, FP6Format.E2M3)
    report = compare_checkpoints(src, dst)

    assert len(report.tensors) == 1
    assert 0 < report.mean_rel_l2 < 0.15


def test_quantized_weight_names_require_nvfp4_global_scale() -> None:
    """A lone ``_scale`` key is MXFP6; NVFP4 also needs ``_scale_2``."""
    keys = [_EXPERT, f"{_EXPERT}_scale"]
    assert quantized_weight_names(keys, FP4Format.NVFP4) == []
    assert quantized_weight_names(keys, FP6Format.E2M3) == [_EXPERT]
    keys.append(f"{_EXPERT}_scale_2")
    assert quantized_weight_names(keys, FP4Format.NVFP4) == [_EXPERT]


def test_target_regex_selects_fp4_names_only(tmp_path: Path) -> None:
    """--target keeps a mixed dump from scoring MXFP8 / other weights."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_source(src)
    quantize_checkpoint(src, dst, FP4Format.NVFP4)

    skipped = compare_checkpoints(
        src, dst, targets=[r"\.mlp\.(gate|up|down)_proj\.weight$"]
    )
    kept = compare_checkpoints(
        src, dst, targets=[r"block_sparse_moe\.experts\.\d+\.w[123]\.weight$"]
    )
    assert skipped.tensors == ()
    assert [item.name for item in kept.tensors] == [_EXPERT]


def test_modelopt_float8_e4m3_scales_are_viewed_as_uint8(
    tmp_path: Path,
) -> None:
    """ModelOpt writes E4M3 scales as float8_e4m3fn; numpy cannot convert that."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_source(src)
    stats = quantize_checkpoint(src, dst, FP4Format.NVFP4)

    from safetensors.torch import safe_open

    with safe_open(
        dst / "model-00001-of-00001.safetensors", framework="pt"
    ) as handle:
        written = {
            _EXPERT: handle.get_tensor(_EXPERT),
            f"{_EXPERT}_scale": handle.get_tensor(f"{_EXPERT}_scale").view(
                torch.float8_e4m3fn
            ),
            f"{_EXPERT}_scale_2": handle.get_tensor(f"{_EXPERT}_scale_2"),
        }
    with safe_open(
        src / "model-00001-of-00001.safetensors", framework="pt"
    ) as handle:
        reference = handle.get_tensor(_EXPERT).to(torch.float32).numpy()

    error = compare_written(_EXPERT, reference, written, FP4Format.NVFP4)
    assert error.rel_l2 == pytest.approx(
        float(stats["mean_rel_l2"]), rel=1e-5, abs=1e-6
    )
