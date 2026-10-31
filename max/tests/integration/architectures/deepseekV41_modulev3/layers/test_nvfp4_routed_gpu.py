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
"""GPU execution tests for the V4.1 NVFP4 routed-expert fallback.

Scope: the ``nvfp4_ops.routed`` wrapper's own logic -- ragged group
construction via ``moe_create_indices``, gather/restore order, the
nonlocal/invalid mask and the after-GEMM weight placement -- executed in
one compiled graph. This is routing/wrapper coverage, NOT an independent
validation of the NVFP4 quantizer: the reference is single-expert probing
through the same grouped kernels, and because the routing weights are
applied after the quantized GEMMs, the multi-expert result must be the
exact weighted sum of one-expert-at-a-time results.

The experts carry heterogeneous scales -- distinct per-expert
``weight_scale_2`` and ``input_scale``, distinct per-expert block scales,
and different scale sets for w1/w3/w2 -- so an expert/scale association
error in the wrapper changes a case result and fails the sum.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import Accelerator, Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.common_layers.quant_tensor import NVFP4Tensor
from max.nn.module import Module
from max.pipelines.architectures.deepseekV41_modulev3.layers.nvfp4_ops import (
    routed,
)
from max.tensor.tensor import Tensor, TensorType

_HIDDEN = 256
_MOE_DIM = 128
_EXPERTS = 4
_FIRST = 2  # global ids [2, 6) are local; 0, 1 and >= 6 are not
_TOPK = 6
_TOKENS = 6


@dataclass(frozen=True)
class _Stack:
    """One projection's stacked host arrays: codes, block scales, scalars."""

    codes: np.ndarray  # [experts, rows, cols // 2] uint8
    block_scales: np.ndarray  # [experts, rows, cols // 16] f32 powers of two
    weight_scale_2: np.ndarray  # [experts] f32, distinct per expert
    input_scale: np.ndarray  # [experts] f32, distinct per expert


def _stack(seed: int, rows: int, cols: int) -> _Stack:
    """Heterogeneous per-expert scales for one projection."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(_EXPERTS, rows, cols // 2)).astype(
        np.uint8
    )
    # Block scales: powers of two are exactly representable in E4M3, and
    # every expert's blocks get their own draws.
    exponents = rng.integers(-3, 4, size=(_EXPERTS, rows, cols // 16))
    block_scales = np.power(2.0, exponents).astype(np.float32)
    weight_scale_2 = np.array([0.5, 1.0, 2.0, 4.0][:_EXPERTS], np.float32)
    input_scale = np.array([1.0, 0.5, 2.0, 0.25][:_EXPERTS], np.float32)
    return _Stack(codes, block_scales, weight_scale_2, input_scale)


class _Routed(Module[..., Tensor]):
    """``nvfp4_ops.routed`` over compile-visible operand tensors.

    All quantized weights ride as graph inputs and the NVFP4Tensors are
    assembled inside the compiled body, so one compiled graph serves
    every routing pattern the tests exercise.
    """

    def forward(
        self,
        x: Tensor,
        ids: Tensor,
        weights: Tensor,
        w1_codes: Tensor,
        w1_scales: Tensor,
        w1_ws2: Tensor,
        w1_is: Tensor,
        w3_codes: Tensor,
        w3_scales: Tensor,
        w3_ws2: Tensor,
        w3_is: Tensor,
        w2_codes: Tensor,
        w2_scales: Tensor,
        w2_ws2: Tensor,
        w2_is: Tensor,
    ) -> Tensor:
        w1 = NVFP4Tensor(
            weight=w1_codes,
            weight_scale=w1_scales.cast(DType.float8_e4m3fn),
            weight_scale_2=w1_ws2,
            input_scale=w1_is,
            block_size=(1, 16),
        )
        w3 = NVFP4Tensor(
            weight=w3_codes,
            weight_scale=w3_scales.cast(DType.float8_e4m3fn),
            weight_scale_2=w3_ws2,
            input_scale=w3_is,
            block_size=(1, 16),
        )
        w2 = NVFP4Tensor(
            weight=w2_codes,
            weight_scale=w2_scales.cast(DType.float8_e4m3fn),
            weight_scale_2=w2_ws2,
            input_scale=w2_is,
            block_size=(1, 16),
        )
        return routed(x, ids, weights, w1, w3, w2, _FIRST, 10.0)


def _input_types() -> list[TensorType]:
    gpu = DeviceRef.GPU()
    n, h, m, t = _EXPERTS, _HIDDEN, _MOE_DIM, _TOKENS
    return [
        TensorType(DType.bfloat16, [t, h], gpu),  # x
        TensorType(DType.int32, [t, _TOPK], gpu),  # ids
        TensorType(DType.float32, [t, _TOPK], gpu),  # weights
        # w1: [moe_dim, hidden]; w3 same; w2: [hidden, moe_dim].
        TensorType(DType.uint8, [n, m, h // 2], gpu),
        TensorType(DType.float32, [n, m, h // 16], gpu),
        TensorType(DType.float32, [n], gpu),
        TensorType(DType.float32, [n], gpu),
        TensorType(DType.uint8, [n, m, h // 2], gpu),
        TensorType(DType.float32, [n, m, h // 16], gpu),
        TensorType(DType.float32, [n], gpu),
        TensorType(DType.float32, [n], gpu),
        TensorType(DType.uint8, [n, h, m // 2], gpu),
        TensorType(DType.float32, [n, h, m // 16], gpu),
        TensorType(DType.float32, [n], gpu),
        TensorType(DType.float32, [n], gpu),
    ]


def _stack_args(stack: _Stack, device: Device) -> list[Tensor]:
    return [
        Tensor.from_dlpack(stack.codes).to(device),
        Tensor.from_dlpack(stack.block_scales).to(device),
        Tensor.from_dlpack(stack.weight_scale_2).to(device),
        Tensor.from_dlpack(stack.input_scale).to(device),
    ]


@dataclass(frozen=True)
class _Fixture:
    run: Callable[..., Tensor]
    x: np.ndarray
    w1: _Stack
    w3: _Stack
    w2: _Stack
    single_expert: np.ndarray


@pytest.fixture(scope="module")
def compiled() -> _Fixture:
    """One compiled ``routed`` graph plus the single-expert reference.

    Everything the tests need executes against the same compiled body,
    so the suite pays one graph compile, not one per test.
    """
    rng = np.random.default_rng(30)
    x = (rng.standard_normal((_TOKENS, _HIDDEN)) * 0.5).astype(np.float32)
    w1, w3, w2 = (
        _stack(31, _MOE_DIM, _HIDDEN),
        _stack(71, _MOE_DIM, _HIDDEN),
        _stack(111, _HIDDEN, _MOE_DIM),
    )
    run = _Routed().compile(*_input_types())
    device = Accelerator()

    def call(
        ids: np.ndarray, weights: np.ndarray, x_arr: np.ndarray
    ) -> np.ndarray:
        out = run(
            Tensor.from_dlpack(x_arr).to(device).cast(DType.bfloat16),
            Tensor.from_dlpack(ids.astype(np.int32)).to(device),
            Tensor.from_dlpack(weights.astype(np.float32)).to(device),
            *_stack_args(w1, device),
            *_stack_args(w3, device),
            *_stack_args(w2, device),
        )
        return out.cast(DType.float32).to_numpy()

    # The single-expert reference: one local expert per row, weight 1.
    single_expert = np.zeros((_EXPERTS, _TOKENS, _HIDDEN), np.float32)
    for e in range(_EXPERTS):
        ids = np.full((_TOKENS, _TOPK), -1, np.int32)
        ids[:, 0] = _FIRST + e
        weights = np.zeros((_TOKENS, _TOPK), np.float32)
        weights[:, 0] = 1.0
        single_expert[e] = call(ids, weights, x)

    return _Fixture(
        run=run, x=x, w1=w1, w3=w3, w2=w2, single_expert=single_expert
    )


def _mixed_ids() -> tuple[np.ndarray, np.ndarray]:
    """Global top-6 ids mixing local, nonlocal and -1 slots."""
    ids = np.array(
        [
            [2, 3, 4, 5, 0, -1],  # four local, one nonlocal, one invalid
            [5, -1, -1, -1, 2, 3],  # sparse local, invalid padding
            [1, 0, 6, 7, 8, 9],  # every slot nonlocal
            [-1, -1, -1, -1, -1, -1],  # every slot invalid
            [3, 3, 5, 2, 4, 4],  # repeated local experts
            [2, 5, -1, 1, 3, 4],  # mixed
        ],
        np.int32,
    )
    weights = np.array(
        [
            [0.4, 0.3, 0.1, 0.05, 0.1, 0.05],
            [0.5, 0.0, 0.0, 0.0, 0.25, 0.25],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0.3, 0.2, 0.1, 0.1, 0.2, 0.1],
        ],
        np.float32,
    )
    return ids, weights


def test_routed_matches_single_expert_reference(compiled: _Fixture) -> None:
    """The multi-expert result is the weighted sum of per-expert results.

    With heterogeneous per-expert scales, this also pins the association
    of each expert's scalars: if the wrapper crossed any expert's
    ``weight_scale_2``/``input_scale``/block scales, its single-expert row would
    not reconstruct the mixed routing.
    """
    ids, weights = _mixed_ids()
    got = (
        compiled.run(
            Tensor.from_dlpack(compiled.x)
            .to(Accelerator())
            .cast(DType.bfloat16),
            Tensor.from_dlpack(ids).to(Accelerator()),
            Tensor.from_dlpack(weights).to(Accelerator()),
            *_stack_args(compiled.w1, Accelerator()),
            *_stack_args(compiled.w3, Accelerator()),
            *_stack_args(compiled.w2, Accelerator()),
        )
        .cast(DType.float32)
        .to_numpy()
    )

    want = np.zeros((_TOKENS, _HIDDEN), np.float32)
    for t in range(_TOKENS):
        for k in range(_TOPK):
            expert = ids[t, k] - _FIRST
            if 0 <= expert < _EXPERTS:
                want[t] += weights[t, k] * compiled.single_expert[expert, t]
    # The single-expert reference and the mixed routing run the same
    # quantized GEMMs but
    # over differently sized (padded) expert groups, so float32
    # accumulation reassociates: the residual is ~1e-5 relative
    # (measured 1.9e-4 absolute at magnitudes ~1e4), not a wrapper
    # error -- the exact checks (zeros, doubling, offset) pin those.
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-2)


def test_nonlocal_and_invalid_slots_contribute_zero(compiled: _Fixture) -> None:
    """Rows routing only nonlocal or only -1 experts return exact zeros."""
    ids, weights = _mixed_ids()
    got = (
        compiled.run(
            Tensor.from_dlpack(compiled.x)
            .to(Accelerator())
            .cast(DType.bfloat16),
            Tensor.from_dlpack(ids).to(Accelerator()),
            Tensor.from_dlpack(weights).to(Accelerator()),
            *_stack_args(compiled.w1, Accelerator()),
            *_stack_args(compiled.w3, Accelerator()),
            *_stack_args(compiled.w2, Accelerator()),
        )
        .cast(DType.float32)
        .to_numpy()
    )
    np.testing.assert_array_equal(got[2], 0.0)  # all nonlocal
    np.testing.assert_array_equal(got[3], 0.0)  # all -1


def test_weights_apply_after_the_quantized_gemms(compiled: _Fixture) -> None:
    """Doubling the routing weights doubles the output exactly.

    Weighting before the NVFP4 quantizer would rescale the activation
    blocks and change the codes; weighting after makes the combination
    exactly linear in the weights.
    """
    ids = np.array(
        [
            [2, 3, 4, 5, -1, -1],
            [3, 5, 2, -1, 4, -1],
            [2, 2, 3, 3, 5, 5],
            [4, 5, 2, 3, -1, -1],
            [5, 2, 4, -1, 3, -1],
            [2, 4, 3, 5, 4, 3],
        ],
        np.int32,
    )
    weights = np.array(
        [
            [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.2, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
            [0.15, 0.15, 0.2, 0.2, 0.2, 0.1],
            [0.1, 0.1, 0.1, 0.2, 0.2, 0.3],
        ],
        np.float32,
    )
    device = Accelerator()
    base = (
        compiled.run(
            Tensor.from_dlpack(compiled.x).to(device).cast(DType.bfloat16),
            Tensor.from_dlpack(ids).to(device),
            Tensor.from_dlpack(weights).to(device),
            *_stack_args(compiled.w1, device),
            *_stack_args(compiled.w3, device),
            *_stack_args(compiled.w2, device),
        )
        .cast(DType.float32)
        .to_numpy()
    )
    doubled = (
        compiled.run(
            Tensor.from_dlpack(compiled.x).to(device).cast(DType.bfloat16),
            Tensor.from_dlpack(ids).to(device),
            Tensor.from_dlpack(2.0 * weights).to(device),
            *_stack_args(compiled.w1, device),
            *_stack_args(compiled.w3, device),
            *_stack_args(compiled.w2, device),
        )
        .cast(DType.float32)
        .to_numpy()
    )
    np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-6, atol=1e-6)


def test_local_window_offset_masks_correctly(compiled: _Fixture) -> None:
    """Shifting ``first`` and the global ids together changes nothing."""
    ids = np.array(
        [
            [2, 3, 4, 5, -1, -1],
            [5, 4, 3, 2, -1, -1],
            [2, 5, -1, 3, 4, -1],
            [4, 4, 2, -1, 5, 3],
            [3, 2, 5, 4, -1, -1],
            [2, 5, 3, -1, 4, 2],
        ],
        np.int32,
    )
    weights = np.array(
        [
            [0.4, 0.3, 0.1, 0.1, 0.05, 0.05],
            [0.1, 0.2, 0.3, 0.3, 0.05, 0.05],
            [0.25, 0.25, 0.2, 0.15, 0.1, 0.05],
            [0.3, 0.3, 0.2, 0.1, 0.05, 0.05],
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0.15, 0.15, 0.2, 0.2, 0.2, 0.1],
        ],
        np.float32,
    )
    device = Accelerator()

    # ``routed`` bakes ``first`` into the compiled body, so the shifted
    # window runs a second compiled graph with the same operands.
    class _Shifted(_Routed):
        def forward(
            self,
            x: Tensor,
            ids: Tensor,
            weights: Tensor,
            w1_codes: Tensor,
            w1_scales: Tensor,
            w1_ws2: Tensor,
            w1_is: Tensor,
            w3_codes: Tensor,
            w3_scales: Tensor,
            w3_ws2: Tensor,
            w3_is: Tensor,
            w2_codes: Tensor,
            w2_scales: Tensor,
            w2_ws2: Tensor,
            w2_is: Tensor,
        ) -> Tensor:
            w1 = NVFP4Tensor(
                weight=w1_codes,
                weight_scale=w1_scales.cast(DType.float8_e4m3fn),
                weight_scale_2=w1_ws2,
                input_scale=w1_is,
                block_size=(1, 16),
            )
            w3 = NVFP4Tensor(
                weight=w3_codes,
                weight_scale=w3_scales.cast(DType.float8_e4m3fn),
                weight_scale_2=w3_ws2,
                input_scale=w3_is,
                block_size=(1, 16),
            )
            w2 = NVFP4Tensor(
                weight=w2_codes,
                weight_scale=w2_scales.cast(DType.float8_e4m3fn),
                weight_scale_2=w2_ws2,
                input_scale=w2_is,
                block_size=(1, 16),
            )
            return routed(x, ids, weights, w1, w3, w2, 0, 10.0)

    at_two = (
        compiled.run(
            Tensor.from_dlpack(compiled.x).to(device).cast(DType.bfloat16),
            Tensor.from_dlpack(ids).to(device),
            Tensor.from_dlpack(weights).to(device),
            *_stack_args(compiled.w1, device),
            *_stack_args(compiled.w3, device),
            *_stack_args(compiled.w2, device),
        )
        .cast(DType.float32)
        .to_numpy()
    )

    shifted = _Shifted().compile(*_input_types())
    at_zero = (
        shifted(
            Tensor.from_dlpack(compiled.x).to(device).cast(DType.bfloat16),
            Tensor.from_dlpack(ids - _FIRST).to(device),
            Tensor.from_dlpack(weights).to(device),
            *_stack_args(compiled.w1, device),
            *_stack_args(compiled.w3, device),
            *_stack_args(compiled.w2, device),
        )
        .cast(DType.float32)
        .to_numpy()
    )
    np.testing.assert_array_equal(at_two, at_zero)
