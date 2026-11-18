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
"""One KDA row must compute the same whether or not another row rides along.

The recurrence never crosses a sequence boundary -- ``input_row_offsets`` is
what tells the kernel where each row starts -- so a row's output cannot depend
on what else is packed into the batch. That is a stronger and far cheaper
statement than reference parity: it compares one computation against itself,
so it needs no golden values and no tolerance for the model's own
approximations, only for the GEMM retiling that a different total token count
legitimately causes.

The lengths matter. The serving failure this pins down (whole responses of a
single repeated token) only appears once a row is a few hundred tokens long,
and every other KDA test runs rows of five tokens or fewer, which is why
nothing caught it.

Ported from the graph-API K3's ``test_kda_batch_invariance_gpu.py`` onto
:class:`KimiK3KDAAttention`: the layer under test is the ModuleV3 one, at the
same geometry, with its fused q/k/v/g/beta projection and its conv-raw
decode path -- both of which are ModuleV3's own and can carry the same class
of batch dependence the graph-API op could.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
)
from max.pipelines.architectures.kimi_k3_modulev3.weight_adapters import (
    interleave_for_tp,
)
from max.tensor import Tensor, default_device

# Same geometry as the parity tests: the kernel is compiled for
# (key, value) head dims of 32, with `hidden` cut down to keep the fixture
# small.
HIDDEN = 64
NUM_HEADS = 8
HEAD_DIM = 32
CONV_KERNEL = 4
INNER = NUM_HEADS * HEAD_DIM
CONV_DIM = 3 * INNER
MAX_SLOTS = 4

# Non-adjacent, non-zero slots, so a batch-index-for-slot confusion cannot
# coincidentally read the right row.
SLOT_A = 3
SLOT_B = 1

# A packed batch changes the GEMM's M dimension, so the projections retile and
# round differently. That is legitimate and lands near the dtype's epsilon; a
# row whose recurrence actually saw another row's tokens scores O(1).
REL_TOL = 2e-2

_WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "q_proj_weight": (INNER, HIDDEN),
    "k_proj_weight": (INNER, HIDDEN),
    "v_proj_weight": (INNER, HIDDEN),
    "conv_weight": (CONV_DIM, CONV_KERNEL),
    "f_a_proj_weight": (HEAD_DIM, HIDDEN),
    "f_b_proj_weight": (INNER, HEAD_DIM),
    "b_proj_weight": (NUM_HEADS, HIDDEN),
    "a_log": (NUM_HEADS, HEAD_DIM),
    "dt_bias": (NUM_HEADS, HEAD_DIM),
    "g_proj_weight": (INNER, HIDDEN),
    "o_norm_weight": (HEAD_DIM,),
    "o_proj_weight": (HIDDEN, INNER),
}


def _weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260913)
    weights = {
        name: (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(
            np.float32
        )
        for name, shape in _WEIGHT_SHAPES.items()
    }
    # Published K3's `A_log` is per-head, broadcast over keys. Varying it along
    # both axes exercises the bounded operator's per-key generality, which a
    # per-head fixture would leave untested.
    weights["a_log"] = np.linspace(
        -1.0, 0.5, NUM_HEADS * HEAD_DIM, dtype=np.float32
    ).reshape(NUM_HEADS, HEAD_DIM)
    weights["o_norm_weight"] = (
        1.0 + 0.1 * weights["o_norm_weight"] * np.sqrt(HEAD_DIM)
    ).astype(np.float32)
    return weights


def _layer(weights: dict[str, np.ndarray]) -> KimiK3KDAAttention:
    """The ModuleV3 KDA layer with ``weights`` on its parameters.

    The tuples assigned below are the checkpoint's own blocks, so the fused
    q/k/v/g/beta weight carries five of them and the layer keeps its separate
    ``f_a_proj`` (see ``_fuses_gate_down``).
    """
    layer = KimiK3KDAAttention(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
        gate_lower_bound=-5.0,
    )
    layer.qkvg_proj.weight = Tensor(
        interleave_for_tp(
            [
                weights["q_proj_weight"],
                weights["k_proj_weight"],
                weights["v_proj_weight"],
                weights["g_proj_weight"],
                weights["b_proj_weight"],
            ],
            1,
        ),
        dtype=DType.float32,
    )
    layer.conv_weight = Tensor(weights["conv_weight"], dtype=DType.float32)
    layer.f_a_proj.weight = Tensor(
        weights["f_a_proj_weight"], dtype=DType.float32
    )
    layer.f_b_proj.weight = Tensor(
        weights["f_b_proj_weight"], dtype=DType.float32
    )
    layer.a_log = Tensor(weights["a_log"], dtype=DType.float32)
    layer.dt_bias = Tensor(weights["dt_bias"], dtype=DType.float32)
    layer.o_norm.weight = Tensor(weights["o_norm_weight"], dtype=DType.float32)
    layer.o_proj.weight = Tensor(weights["o_proj_weight"], dtype=DType.float32)
    return layer


@pytest.mark.compile_on_gpu("kda_decode mutates its state in place")
# 5 is the length every other KDA test uses and is here only as a control that
# the harness agrees with them; the rest span the range where serving breaks.
@pytest.mark.parametrize("row_len", [5, 128, 291, 512])
def test_kda_row_output_is_independent_of_its_batch(row_len: int) -> None:
    weights = _weights()

    def run(rows: list[np.ndarray], slots: list[int]) -> np.ndarray:
        with default_device(Accelerator()):
            # Built inside the device scope: a tensor built outside it lands
            # on the host.
            layer = _layer(weights)
            # Pools start zeroed for both calls, so the only difference
            # between them is what else is packed into the batch.
            conv_pool = Tensor.zeros(
                [MAX_SLOTS, CONV_DIM, CONV_KERNEL - 1], dtype=DType.float32
            )
            state_pool = Tensor.zeros(
                [MAX_SLOTS, NUM_HEADS, HEAD_DIM, HEAD_DIM],
                dtype=DType.float32,
            )
            x = Tensor(
                np.ascontiguousarray(np.concatenate(rows, axis=0)),
                dtype=DType.float32,
            )
            offsets = np.cumsum(
                [0, *(len(r) for r in rows)], dtype=np.uint32
            ).astype(np.uint32)
            slot_idx = Tensor(
                np.asarray(slots, dtype=np.uint32), dtype=DType.uint32
            )
            out = layer(
                x,
                conv_pool,
                slot_idx,
                state_pool,
                slot_idx,
                Tensor(offsets, dtype=DType.uint32),
            )
            return out.cast(DType.float32).to_numpy().astype(np.float64)

    rng = np.random.default_rng(11)
    row_a = rng.standard_normal((row_len, HIDDEN)).astype(np.float32)
    row_b = rng.standard_normal((3, HIDDEN)).astype(np.float32)

    alone = run([row_a], [SLOT_A])
    batched = run([row_a, row_b], [SLOT_A, SLOT_B])[:row_len]

    assert not np.isnan(alone).any(), "row A alone produced NaN"
    assert not np.isnan(batched).any(), "row A in a batch produced NaN"

    scale = max(float(np.max(np.abs(alone))), 1e-12)
    per_token = np.max(np.abs(alone - batched), axis=1) / scale
    worst = float(np.max(per_token))
    first_bad = int(np.argmax(per_token > REL_TOL)) if worst > REL_TOL else -1
    assert worst < REL_TOL, (
        f"row of {row_len} tokens changed when a 3-token row joined the batch:"
        f" max relative difference {worst:.3e} at token {first_bad}"
    )
