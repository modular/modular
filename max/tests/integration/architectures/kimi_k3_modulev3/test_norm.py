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
"""Numerical verification for :func:`kimi_rms_norm_gated`.

Compares against an independent numpy reference built directly from the real
fla Triton kernel's arithmetic (``fused_norm_gate.py``'s
``layer_norm_gated_fwd_kernel``), not from this port's own implementation --
so a shared mistake between the port and the reference would not hide a bug.
"""

from __future__ import annotations

import numpy as np
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.norm import (
    KimiRMSNormGated,
    kimi_rms_norm_gated,
)
from max.tensor import Tensor, default_device


def _numpy_reference(
    x: np.ndarray, weight: np.ndarray, gate: np.ndarray, eps: float
) -> np.ndarray:
    """Independent reference: normalize, *weight, *sigmoid(gate), in fp64.

    Mirrors the real kernel's ``b_rstd = 1/sqrt(mean(x**2) + eps)``,
    ``b_y = (x * b_rstd) * weight``, ``b_y *= sigmoid(gate)`` -- weight
    multiply strictly before the gate, not after or fused into one step.
    Computed in float64 so the ``out_dtype`` test below can measure fp32-path
    precision against it; the kernel's own fp32 arithmetic sits ~1e-6 away,
    far inside every tolerance used here.
    """
    x64 = x.astype(np.float64)
    w64 = weight.astype(np.float64)
    g64 = gate.astype(np.float64)
    rstd = 1.0 / np.sqrt(np.mean(x64**2, axis=-1, keepdims=True) + eps)
    normed = x64 * rstd
    weighted = normed * w64
    sigmoid_gate = 1.0 / (1.0 + np.exp(-g64))
    return weighted * sigmoid_gate


def test_matches_independent_numpy_reference_flat() -> None:
    """Flat ``[T, D]`` shape, asymmetric weight, non-trivial gate."""
    rng = np.random.default_rng(0)
    eps = 1e-5
    x_np = rng.normal(size=(5, 32)).astype(np.float32)
    weight_np = rng.normal(size=(32,)).astype(np.float32)
    gate_np = rng.normal(size=(5, 32)).astype(np.float32)
    expected = _numpy_reference(x_np, weight_np, gate_np, eps)

    with default_device(Accelerator()):
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        weight = Tensor(weight_np, dtype=DType.float32).cast(DType.bfloat16)
        gate = Tensor(gate_np, dtype=DType.float32).cast(DType.bfloat16)
        out = kimi_rms_norm_gated(x, weight, gate, eps)
        out_np = out.cast(DType.float32).to_numpy()

    assert out.dtype == DType.bfloat16
    # bf16 has ~3 decimal digits of precision; the reference is fp64.
    np.testing.assert_allclose(out_np, expected, rtol=0.05, atol=0.05)


def test_matches_independent_numpy_reference_per_head() -> None:
    """``[T, num_heads, head_dim]`` shape, the real KDA ``o_norm`` usage:

    weight is ``[head_dim]``, broadcasting across ``num_heads`` -- the
    normalization statistic must be computed per-head (over the LAST axis
    only), not over the flattened ``num_heads * head_dim`` width.
    """
    rng = np.random.default_rng(1)
    eps = 1e-5
    total_seq_len, num_heads, head_dim = 3, 4, 16
    x_np = rng.normal(size=(total_seq_len, num_heads, head_dim)).astype(
        np.float32
    )
    weight_np = rng.normal(size=(head_dim,)).astype(np.float32)
    gate_np = rng.normal(size=(total_seq_len, num_heads, head_dim)).astype(
        np.float32
    )
    expected = _numpy_reference(x_np, weight_np, gate_np, eps)

    with default_device(Accelerator()):
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        weight = Tensor(weight_np, dtype=DType.float32).cast(DType.bfloat16)
        gate = Tensor(gate_np, dtype=DType.float32).cast(DType.bfloat16)
        out = kimi_rms_norm_gated(x, weight, gate, eps)
        out_np = out.cast(DType.float32).to_numpy()

    assert out.shape == [total_seq_len, num_heads, head_dim]
    np.testing.assert_allclose(out_np, expected, rtol=0.05, atol=0.05)


def test_out_dtype_rounds_once_at_the_end() -> None:
    """fp32 in, bf16 out: ``out_dtype`` narrows the output only.

    The KDA layer relies on exactly this contract: ``kda_decode`` writes
    fp32 and ``o_norm`` rounds to the model dtype inside the same fused
    norm, so an implementation that rounds the input to ``out_dtype``
    before normalizing would silently lose precision.
    """
    rng = np.random.default_rng(4)
    eps = 1e-5
    x_np = rng.normal(size=(6, 32)).astype(np.float32)
    weight_np = rng.normal(size=(32,)).astype(np.float32)
    gate_np = rng.normal(size=(6, 32)).astype(np.float32)
    expected = _numpy_reference(x_np, weight_np, gate_np, eps)

    with default_device(Accelerator()):
        x = Tensor(x_np, dtype=DType.float32)
        weight = Tensor(weight_np, dtype=DType.float32)
        gate = Tensor(gate_np, dtype=DType.float32)
        unrounded = kimi_rms_norm_gated(x, weight, gate, eps)
        out = kimi_rms_norm_gated(
            x, weight, gate, eps, out_dtype=DType.bfloat16
        )
        rounded_ref = unrounded.cast(DType.bfloat16)
    assert unrounded.dtype == DType.float32
    assert out.dtype == DType.bfloat16
    unrounded_np = unrounded.cast(DType.float32).to_numpy()
    out_np = out.cast(DType.float32).to_numpy()
    rounded_ref_np = rounded_ref.cast(DType.float32).to_numpy()

    # The fp32 path must match the fp64 reference tightly; an early bf16
    # rounding of the input lands ~2e-3 away, outside this tolerance.
    np.testing.assert_allclose(unrounded_np, expected, rtol=1e-3, atol=1e-5)
    # `out_dtype` moves the single rounding only: the bf16 result is the
    # fp32 path's result rounded once, element for element.
    np.testing.assert_array_equal(out_np, rounded_ref_np)
    # That single rounding is the only bf16-sized gap left from the
    # reference.
    np.testing.assert_allclose(out_np, expected, rtol=0.01, atol=0.01)


def test_gate_applies_after_weight_not_before() -> None:
    """Distinguishes this op from ``gated_group_rmsnorm``, which gates the
    INPUT (with silu) before normalizing -- a different operation. Here the
    gate is the LAST multiply, applied after normalization has already
    rescaled ``x`` to O(1), so a gate that saturates sigmoid near 0 collapses
    the output near 0 regardless of ``x``'s or ``weight``'s scale.
    """
    rng = np.random.default_rng(2)
    eps = 1e-5
    x_np = rng.normal(size=(4, 16)).astype(np.float32) * 100.0
    weight_np = rng.normal(size=(16,)).astype(np.float32) * 100.0
    gate_np = np.full((4, 16), -50.0, dtype=np.float32)

    with default_device(Accelerator()):
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        weight = Tensor(weight_np, dtype=DType.float32).cast(DType.bfloat16)
        gate = Tensor(gate_np, dtype=DType.float32).cast(DType.bfloat16)
        out = kimi_rms_norm_gated(x, weight, gate, eps)
        out_np = out.cast(DType.float32).to_numpy()

    assert np.abs(out_np).max() < 1e-2, (
        "a saturating gate must collapse the output to ~0 regardless of x"
        f" or weight scale -- got max abs {np.abs(out_np).max()}"
    )


def test_module_wrapper_matches_the_function() -> None:
    """:class:`KimiRMSNormGated` must produce the same output as the plain
    function it wraps, given the same weight."""
    rng = np.random.default_rng(3)
    eps = 1e-5
    dim = 24
    x_np = rng.normal(size=(2, dim)).astype(np.float32)
    gate_np = rng.normal(size=(2, dim)).astype(np.float32)

    with default_device(Accelerator()):
        x = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)
        gate = Tensor(gate_np, dtype=DType.float32).cast(DType.bfloat16)

        module = KimiRMSNormGated(dim, eps=eps)
        via_module = module(x, gate).cast(DType.float32).to_numpy()
        via_function = (
            kimi_rms_norm_gated(x, module.weight, gate, eps)
            .cast(DType.float32)
            .to_numpy()
        )

    np.testing.assert_array_equal(via_module, via_function)
