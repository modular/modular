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
"""The block-diagonal wo_a launch vs the per-group reference it replaces.

``_grouped_dense``'s production arm quantizes the ``wo_a`` activation
once, token-major, and runs one block-diagonal block-scaled matmul (main
#105486's mechanism); the reference arm keeps one quantize chain and one
dense launch per group. The quantize blocks of 32 never straddle two
groups, so the one full-width launch must produce exactly the bytes the
per-group chains produce -- this pins that at the production width
(``[rows, 8 x 4096]``) -- and the matmul receipt compares both arms'
bf16 outputs, within one ulp and byte-for-byte counted, since the
block-diagonal and dense launches pick their own tile configs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max import functional as F
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.module import Module
from max.pipelines.architectures.deepseekV41_modulev3.layers.mxfp8_linear import (
    _grouped_dense_reference,
    _quantize,
    grouped_dense_op,
)
from max.tensor import Tensor

# The production ``wo_a``: 8 groups of [1024 out, 4096 in].
_GROUPS = 8
_GROUP_ROWS = 1024
_K = 4096
_ROWS = [96, 2048]


class _WoAArm(Module[..., list[Tensor]]):
    """The production arm: ``grouped_dense_op``, no switch to flip."""

    def forward(self, x: Tensor, weight: Tensor, scale: Tensor) -> list[Tensor]:
        return [grouped_dense_op(x, weight, scale, _GROUP_ROWS)]


class _ReferenceArm(Module[..., list[Tensor]]):
    """The reference arm: per-group chains, one dense launch per group."""

    def forward(self, x: Tensor, weight: Tensor, scale: Tensor) -> list[Tensor]:
        return [
            F.functional(_grouped_dense_reference)(
                x, weight, scale, _GROUP_ROWS
            )
        ]


class _WoARopeArm(Module[..., list[Tensor]]):
    """The production arm with the inverse rope folded into its quantize."""

    def forward(
        self, x: Tensor, weight: Tensor, scale: Tensor, cos_sin: Tensor
    ) -> list[Tensor]:
        return [
            grouped_dense_op(
                x,
                weight,
                scale,
                _GROUP_ROWS,
                rope=(cos_sin, 512),
            )
        ]


def _woa_inputs(rows: int, seed: int) -> tuple[torch.Tensor, ...]:
    """bf16 activations, an E4M3 leaf and compact E8M0 scale bytes."""
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(
        (rng.standard_normal((rows, _GROUPS * _K)) * 2.0).astype(np.float32)
    ).to(torch.bfloat16)
    w = torch.from_numpy(
        (rng.standard_normal((_GROUPS * _GROUP_ROWS, _K)) * 0.5).astype(
            np.float32
        )
    ).to(torch.float8_e4m3fn)
    # Random exponents near 2^0 in every scale slot: the uint8 payload of
    # an E8M0 tensor is the biased exponent itself.
    scale_bytes = rng.integers(
        124, 131, (_GROUPS * _GROUP_ROWS // 32, _K // 32)
    ).astype(np.uint8)
    return x, w, torch.from_numpy(scale_bytes)


def _woa_types(
    inputs: tuple[torch.Tensor, ...], gpu: Accelerator
) -> tuple[TensorType, ...]:
    x, w, scale_bytes = inputs
    ref = DeviceRef.from_device(gpu)
    return (
        TensorType(DType.bfloat16, x.shape, ref),
        TensorType(DType.float8_e4m3fn, w.shape, ref),
        TensorType(DType.float8_e8m0fnu, scale_bytes.shape, ref),
    )


def _run_arm(
    arm: Module[..., list[Tensor]],
    inputs: tuple[torch.Tensor, ...],
    gpu: Accelerator,
) -> torch.Tensor:
    x, w, scale_bytes = inputs
    with F.lazy():
        module = arm
    run = module.compile(*_woa_types(inputs, gpu))
    scale_gpu = Tensor.from_dlpack(
        Buffer.from_dlpack(scale_bytes).to(gpu).view(DType.float8_e8m0fnu)
    )
    out = run(
        Tensor.from_dlpack(Buffer.from_dlpack(x)).to(gpu),
        Tensor.from_dlpack(Buffer.from_dlpack(w)).to(gpu),
        scale_gpu,
    )[0]
    return torch.from_dlpack(out.to(CPU()))


def test_woa_traces_the_block_diagonal_launch() -> None:
    """The wo_a site routes through the block-diagonal op, no env."""
    gpu = Accelerator()
    with F.lazy():
        module = _WoAArm()
    staged = str(module.trace(*_woa_types(_woa_inputs(_ROWS[0], seed=0), gpu)))
    assert "cluster_splitk.block_diagonal" in staged, (
        "wo_a did not trace the block-diagonal op on an SM100 GPU"
    )


def test_woa_folds_the_inverse_rope_into_the_quantize() -> None:
    """With a rope table the quantize is main's rope-quantize op, no chain."""
    gpu = Accelerator()
    inputs = _woa_inputs(_ROWS[0], seed=0)
    with F.lazy():
        module = _WoARopeArm()
    staged = str(
        module.trace(
            *_woa_types(inputs, gpu),
            TensorType(
                DType.float32, [_ROWS[0], 32, 2], DeviceRef.from_device(gpu)
            ),
        )
    )
    assert "rope.quantize.dynamic.block.scaled" in staged
    assert "cluster_splitk.block_diagonal" in staged


@pytest.mark.parametrize("rows", _ROWS)
def test_block_diagonal_woa_matches_the_grouped_reference(
    rows: int,
) -> None:
    """Both arms' outputs agree within an ulp; the byte count is printed."""
    inputs = _woa_inputs(rows, seed=rows)
    gpu = Accelerator()
    fused = _run_arm(_WoAArm(), inputs, gpu)
    grouped = _run_arm(_ReferenceArm(), inputs, gpu)
    assert grouped.shape == fused.shape

    f32, g32 = fused.float(), grouped.float()
    tol = 2.0**-7 * torch.maximum(f32.abs(), g32.abs()) + 2.0**-14
    bad = (f32 - g32).abs() > tol
    assert not bad.any(), f"{int(bad.sum())} outputs differ by more than an ulp"

    differing = int(
        (fused.view(torch.int16) != grouped.view(torch.int16)).sum()
    )
    print(
        f"[receipt] rows={rows}: {differing}/{fused.numel()} output values"
        " differ byte-for-byte between the arms"
    )


@pytest.mark.parametrize("rows", _ROWS)
def test_full_width_quantize_matches_the_group_chains(rows: int) -> None:
    """One full-width quantize equals the eight per-group quantizes.

    The per-32 amax blocks are independent of the group slicing, so the
    one full-width quantize's codes and scales are the bytes the per-group
    quantizes produce, sliced per group. This is the composition the
    block-diagonal arm rests on, pinned at the production width.
    """
    gpu = Accelerator()
    ref = DeviceRef.from_device(gpu)
    rng = np.random.default_rng(rows)
    x_np = (rng.standard_normal((rows, _GROUPS * _K)) * 2.0).astype(np.float32)
    x_t = torch.from_numpy(x_np).to(torch.bfloat16)

    with Graph(
        "quant_pin",
        input_types=[TensorType(DType.bfloat16, [rows, _GROUPS * _K], ref)],
    ) as graph:
        x = graph.inputs[0].tensor
        full_codes, full_scales = _quantize(x)
        outs = [full_codes, full_scales]
        for g in range(_GROUPS):
            group_codes, group_scales = _quantize(x[:, g * _K : (g + 1) * _K])
            outs += [group_codes, group_scales]
        graph.output(*outs)
    model = InferenceSession(devices=[gpu]).load(graph)
    results = model.execute(
        Buffer.from_dlpack(x_t).to(gpu),
    )
    host = [r.to(CPU()) for r in results]
    fc = torch.from_dlpack(host[0].view(DType.uint8))
    fs = torch.from_dlpack(host[1].view(DType.uint8))

    for g in range(_GROUPS):
        got_codes = host[2 + 2 * g].view(DType.uint8)
        got_scales = host[3 + 2 * g].view(DType.uint8)
        want_codes = fc[:, g * _K : (g + 1) * _K]
        # The interleaved scales' axis 1 counts groups of four 32-column
        # blocks, so group g's scale bytes are a contiguous slice of it.
        tiles = _K // 128
        want_scales = fs[:, g * tiles : (g + 1) * tiles]
        assert torch.equal(torch.from_dlpack(got_codes), want_codes), (
            f"group {g}: full-width codes differ from the chain's"
        )
        assert torch.equal(torch.from_dlpack(got_scales), want_scales), (
            f"group {g}: full-width scales differ from the chain's"
        )
