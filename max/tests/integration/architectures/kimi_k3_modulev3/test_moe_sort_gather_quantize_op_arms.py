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
"""Both arms of the sort-fused MoE routing-glue op, executed on gfx950.

``mo.moe.sort.gather.quantize.dynamic.block.scaled`` picks its arm at
execution, because the serving graph is built once over a dynamic token
count: a small launch runs the one-launch sort+gather+quantize kernel, and
anything larger runs ``moe_create_indices`` followed by the source-major
scatter+quantize *inside* the op. The kernel tests
(``max/kernels/test/gpu/nn/test_moe_gather_quantize.mojo``) cover each
arm's arithmetic by calling the kernels directly, so they all pass while
the op's pick, its create-indices staging, or its immutable re-view of
``restore_token_order`` is broken -- nothing else executes the op body.

This test compiles the op once over a symbolic slot count and runs it on
both sides of the boundary, ``P == 1024`` (the fused kernel's slot bound)
and ``P == 1040`` (the fallback arm), and checks the create-indices
contract on each: the CSR is the histogram prefix sum of the routed ids,
``ids`` is ``0..E-1``, the usage stats hold the maximum expert count and
``E``, ``order`` is a bijection whose ``restore`` is its inverse and whose
every slot lands inside its expert's CSR range, and each sorted row's
payload dequantizes to its source token's activations to e4m3 precision.
It is its own target because it is the only test that compiles and runs
the op.
"""

from __future__ import annotations

from dataclasses import dataclass

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_architecture_name
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import moe_sort_gather_quantize_dynamic_block_scaled

# K3's routing glue: top-16 of 896 experts over the 3584-wide latent, the
# shape the AMD W4A8 path runs.
NUM_EXPERTS = 896
TOPK = 16
K = 3584
SF = 32

# The fused kernel covers at most 1024 slots; one more than that takes the
# two-launch fallback arm. Both are multiples of topk, which the op needs.
FUSED_SLOTS = 1024
FALLBACK_SLOTS = 1040


def _is_gfx950_gpu() -> bool:
    try:
        return accelerator_architecture_name().startswith("gfx950")
    except Exception:
        return False


_requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950_gpu(), reason="the MoE routing glue is an AMD CDNA4 path"
)


@pytest.fixture(scope="module")
def model() -> Model:
    """The op, compiled once over a symbolic slot count."""
    session = InferenceSession(devices=[md.Accelerator()])
    gpu = DeviceRef.GPU()
    with Graph(
        "moe_sort_gather_quantize_op_arms",
        input_types=[
            TensorType(DType.bfloat16, ["m", K], device=gpu),
            TensorType(DType.int32, ["p"], device=gpu),
        ],
    ) as graph:
        latent, topk_ids = (i.tensor for i in graph.inputs)
        graph.output(
            *moe_sort_gather_quantize_dynamic_block_scaled(
                latent, topk_ids, topk=TOPK, num_experts=NUM_EXPERTS
            )
        )
    return session.load(graph)


def _topk_ids(num_tokens: int, seed: int) -> np.ndarray:
    """`TOPK` distinct experts per token, as the router emits them."""
    rng = np.random.default_rng(seed)
    ids = np.empty((num_tokens, TOPK), dtype=np.int32)
    for token in range(num_tokens):
        ids[token] = rng.choice(NUM_EXPERTS, size=TOPK, replace=False)
    return ids.reshape(-1)


def _params(
    hidden: md.Buffer, dtype: DType, shape: tuple[int, ...]
) -> np.ndarray:
    """The host copy of a float8 output, as raw bytes."""
    return hidden.view(dtype, shape).to_numpy()


@dataclass(frozen=True)
class _Run:
    """One execution's host outputs, as the checks read them."""

    latent: np.ndarray
    """The bf16 activations the op quantized, rounded back to float32."""

    ids: np.ndarray
    """The routed expert ids, one per (token, pick) pair."""

    order: np.ndarray
    start: np.ndarray
    restore: np.ndarray
    expert_ids: np.ndarray
    stats: np.ndarray
    quant: np.ndarray
    """The fp8 payload, as raw bytes."""

    scales: np.ndarray
    """The E8M0 block scales, as raw bytes."""


def _dequantize(quant: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """fp8 payload x its E8M0 block scales, as float32."""
    values = (
        torch.from_numpy(quant)
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
        .numpy()
    )
    block_scales = np.exp2(scales.astype(np.int32) - 127).astype(np.float32)
    return values * np.repeat(block_scales, SF, axis=1)


def _run(model: Model, num_slots: int, seed: int) -> _Run:
    """Executes the op at one slot count and returns its host outputs."""
    num_tokens = num_slots // TOPK
    rng = np.random.default_rng(seed)
    # A per-token magnitude, so the E8M0 scale is exercised away from 1.0.
    latent = rng.standard_normal((num_tokens, K)).astype(np.float32) * np.exp2(
        rng.integers(-3, 4, size=(num_tokens, 1))
    ).astype(np.float32)
    latent_t = torch.from_numpy(latent).to(torch.bfloat16)
    ids = _topk_ids(num_tokens, seed + 1)

    device = model.input_devices[0]
    outputs = model.execute(
        md.Buffer.from_dlpack(latent_t).to(device),
        md.Buffer.from_numpy(ids).to(device),
    )
    order, start, restore, expert_ids, stats, quant, scales = outputs
    return _Run(
        latent=latent_t.float().numpy(),
        ids=ids,
        order=order.to_numpy(),
        start=start.to_numpy(),
        restore=restore.to_numpy(),
        expert_ids=expert_ids.to_numpy(),
        stats=stats.to_numpy(),
        quant=_params(quant, DType.uint8, (num_slots, K)),
        scales=_params(scales, DType.uint8, (num_slots, K // SF)),
    )


def _check(num_slots: int, run: _Run) -> None:
    """The create-indices contract plus the payload's source mapping."""
    latent, ids = run.latent, run.ids
    order, start, restore = run.order, run.start, run.restore
    expert_ids, stats = run.expert_ids, run.stats
    counts = np.bincount(ids, minlength=NUM_EXPERTS)
    slots = np.arange(num_slots)

    np.testing.assert_array_equal(expert_ids, np.arange(NUM_EXPERTS))
    # The CSR is the histogram's exclusive prefix sum, and its last entry is
    # the slot count.
    np.testing.assert_array_equal(
        start[:NUM_EXPERTS], np.concatenate(([0], np.cumsum(counts)[:-1]))
    )
    assert start[NUM_EXPERTS] == num_slots, "start[E] is not the slot count"
    np.testing.assert_array_equal(stats, [counts.max(), NUM_EXPERTS])

    # The order is a bijection onto the slots that restore inverts, and each
    # slot sits inside the CSR range of the expert it was routed to.
    assert np.array_equal(np.sort(order), slots), "order is not a bijection"
    np.testing.assert_array_equal(restore[order], slots)
    experts = ids[order]
    assert np.all(start[experts] <= slots), "a slot precedes its expert's range"
    assert np.all(slots < start[experts + 1]), (
        "a slot passes its expert's range"
    )

    # Each sorted row holds the quantized activations of the token it was
    # routed from: dequantized to e4m3 precision, and byte-identical for two
    # slots of the same token.
    payload = _dequantize(run.quant, run.scales)
    sources = order // TOPK
    for source in np.unique(sources):
        rows = payload[sources == source]
        np.testing.assert_array_equal(
            rows[0], rows[1], err_msg="one token's slots disagree"
        )
        reference = latent[source]
        tolerance = 0.0625 * np.abs(reference) + 0.002 * np.abs(reference.max())
        error = np.abs(rows[0] - reference)
        assert np.all(error <= tolerance), (
            f"token {source} dequantizes {error.max():.3e} off"
        )


@_requires_gfx950
def test_fused_arm(model: Model) -> None:
    """`P == 1024` is the fused kernel's largest launch."""
    _check(FUSED_SLOTS, _run(model, FUSED_SLOTS, seed=7))


@_requires_gfx950
def test_fallback_arm(model: Model) -> None:
    """`P == 1040` is past the fused bound: create_indices plus the
    source-major scatter+quantize, both staged inside the op."""
    _check(FALLBACK_SLOTS, _run(model, FALLBACK_SLOTS, seed=11))
