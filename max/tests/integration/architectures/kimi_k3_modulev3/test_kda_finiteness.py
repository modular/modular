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
"""KDA must stay finite at the published checkpoint's width, gate and dtype.

Every other KDA test here runs the bring-up fixture's geometry -- 8 heads of
32 in fp32 -- and the published checkpoint shares none of those properties.
It is 96 heads of 128 in bf16 with fp32 recurrent state, and its
``linear_attn_config.gate_lower_bound: -5.0`` selects the bounded gate arm,
which no fixture test at this width exercises: the bounded arm folds
``exp(A_log)`` into the pre-activation and exponentiates
``bound * sigmoid(...)``, a different function of the same inputs.

Nothing in the recurrence normalizes the running state, so its only bound is
the delta rule's own non-expansiveness -- which holds only while the decay
stays at or below 1 and beta stays inside ``(0, 1)``.

The assertion is finiteness of BOTH halves of the kernel's output: the
per-token result and the recurrent state left in the pool. Checking every
token row is enough to catch a mid-sequence blow-up, because a non-finite
state at step ``t`` survives the decay into every later token too.

A PASS IS NOT AN EXONERATION. The weights here are RANDOM, so this is a stress
and guard test, not a reproduction. The serving failure it sits beside is
content-dependent on a TRAINED checkpoint, and random weights cannot reproduce
content dependence. Green says only that KDA does not blow up
unconditionally at this geometry.

Ported from the graph-API K3's ``test_kda_finiteness_gpu.py`` onto
:class:`KimiK3KDAAttention`. The recurrence beneath it is the same
``kda_decode`` op; what changed is the layer above, which now also routes a
prefill batch into the op's own chunk pipeline, so the published geometry is
worth re-asserting through it.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
)
from max.tensor import Tensor, default_device

# `moonshotai/Kimi-K3`'s own `linear_attn_config` and `hidden_size`. The hidden
# size is kept real so the projections produce gate, beta and value magnitudes
# of the right scale -- a narrowed one would understate every input to the
# recurrence.
HIDDEN = 7168
NUM_HEADS = 96
HEAD_DIM = 128
CONV_KERNEL = 4
INNER = NUM_HEADS * HEAD_DIM
CONV_DIM = 3 * INNER
MAX_SLOTS = 2
SLOT = 1

# Published K3 declares `gate_lower_bound: -5.0`, which is what selects the
# bounded arm. Any other value is rejected by the layer.
GATE_LOWER_BOUND = -5.0

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

# The checkpoint ships these fp32 and the layer declares them fp32 whatever
# the model dtype is; the rest follow the model dtype. Keeping the split
# faithful matters here because `b_proj` is what feeds beta, and beta is what
# bounds the state.
_FP32_WEIGHTS = frozenset(
    {"conv_weight", "b_proj_weight", "a_log", "dt_bias", "o_norm_weight"}
)


def _weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260913)
    weights = {
        name: (
            rng.standard_normal(shape, dtype=np.float32) / np.sqrt(shape[-1])
        ).astype(np.float32)
        for name, shape in _WEIGHT_SHAPES.items()
    }
    # Published K3's `A_log` is per-head, and the layer broadcasts it over
    # keys. The spread across heads covers both ends of the decay the bounded
    # arm can reach.
    weights["a_log"] = np.broadcast_to(
        np.linspace(-1.0, 1.5, NUM_HEADS, dtype=np.float32)[:, None],
        (NUM_HEADS, HEAD_DIM),
    ).copy()
    weights["o_norm_weight"] = (
        1.0 + 0.1 * weights["o_norm_weight"] * np.sqrt(HEAD_DIM)
    ).astype(np.float32)
    return weights


def _staged(values: np.ndarray, dtype: DType, device: Device) -> Tensor:
    """Stages ``values`` as ``dtype`` without a second full-precision copy.

    numpy has no bfloat16, so the values are truncated to their bf16 bit
    pattern on the host and the bytes are relabelled on the way to the device.
    ``>> 16`` rounds toward zero, not to nearest even as a cast does; that is
    immaterial for a finiteness guard but would matter in a parity test, and
    a bf16 checkpoint's loader rounds nothing at all -- its files already
    carry the bf16 values.
    """
    if dtype != DType.bfloat16:
        return Tensor.from_dlpack(
            Buffer.from_numpy(np.ascontiguousarray(values)).to(device)
        )
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    return Tensor.from_dlpack(
        Buffer.from_numpy(np.ascontiguousarray(bits))
        .view(DType.bfloat16)
        .to(device)
    )


def _layer(
    weights: dict[str, np.ndarray], device: Device
) -> KimiK3KDAAttention:
    """The ModuleV3 KDA layer at the published geometry.

    bf16 for every projection the layer fuses into its input GEMM, and fp32
    for the convs, the gate parameters and ``o_norm``.
    """
    dtype = DType.bfloat16
    layer = KimiK3KDAAttention(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
        gate_lower_bound=GATE_LOWER_BOUND,
    )
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "g_proj",
        "b_proj",
        "f_a_proj",
        "f_b_proj",
        "o_proj",
    ):
        getattr(layer, name).weight = _staged(
            weights[f"{name}_weight"], dtype, device
        )
    layer.q_conv1d, layer.k_conv1d, layer.v_conv1d = (
        _staged(conv.reshape(INNER, 1, CONV_KERNEL), DType.float32, device)
        for conv in np.split(weights["conv_weight"], 3)
    )
    layer.a_log = _staged(weights["a_log"][:, 0], DType.float32, device)
    layer.dt_bias = _staged(
        weights["dt_bias"].reshape(-1), DType.float32, device
    )
    layer.o_norm.weight = _staged(
        weights["o_norm_weight"], DType.float32, device
    )
    return layer


def _to_f64(tensor: Tensor) -> np.ndarray:
    """Widens a device tensor to host float64, bf16 included.

    numpy has no bfloat16, so ``to_numpy`` cannot decode one. Reinterpreting
    the buffer as ``uint16`` and shifting the pattern into float32's high half
    is exact and, unlike a cast, preserves the NaN and infinity encodings this
    test exists to detect.
    """
    host = Buffer.from_dlpack(tensor).to(CPU())
    if host.dtype != DType.bfloat16:
        return host.to_numpy().astype(np.float64)
    bits = host.view(DType.uint16, list(host.shape)).to_numpy()
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


# 543 is GPQA Diamond question 7, the prompt that degenerates 3/3 solo against
# the real checkpoint; 512 is the round length below it.
@pytest.mark.parametrize("row_len", [512, 543])
# The recurrence's only bound on the state is `sum(beta * |v|)`, so the state's
# reachable magnitude scales with the activations. 8x is well past anything a
# normalized residual stream produces and still must stay finite.
@pytest.mark.parametrize("x_scale", [1.0, 8.0])
@pytest.mark.compile_on_gpu("kda_decode mutates its state in place")
def test_kda_stays_finite_at_published_geometry(
    row_len: int, x_scale: float
) -> None:
    weights = _weights()
    rng = np.random.default_rng(11)
    row = (
        rng.standard_normal((row_len, HIDDEN), dtype=np.float32) * x_scale
    ).astype(np.float32)

    device = Accelerator()
    with default_device(device):
        layer = _layer(weights, device)
        conv_pool = Tensor.zeros(
            [MAX_SLOTS, CONV_DIM, CONV_KERNEL - 1], dtype=DType.bfloat16
        )
        state_pool = Tensor.zeros(
            [MAX_SLOTS, NUM_HEADS, HEAD_DIM, HEAD_DIM], dtype=DType.float32
        )
        x = _staged(row, DType.bfloat16, device)
        slot_idx = Tensor(
            np.asarray([SLOT], dtype=np.uint32), dtype=DType.uint32
        )
        offsets = Tensor(
            np.asarray([0, row_len], dtype=np.uint32), dtype=DType.uint32
        )
        out = layer(x, conv_pool, slot_idx, state_pool, slot_idx, offsets)
        out_np = _to_f64(out)
        state_np = _to_f64(state_pool)[SLOT]

    finite_rows = np.isfinite(out_np).all(axis=1)
    first_bad = int(np.argmin(finite_rows)) if not finite_rows.all() else -1
    assert first_bad == -1, (
        f"KDA output went non-finite at token {first_bad} of {row_len}"
        f" (x_scale={x_scale}); tokens before it peak at"
        f" {np.abs(out_np[:first_bad]).max() if first_bad else 0.0:.4g}"
    )
    assert np.isfinite(state_np).all(), (
        f"KDA recurrent state went non-finite after {row_len} steps"
        f" (x_scale={x_scale}) while every output token stayed finite"
    )
    # Not a tolerance -- a diagnostic ceiling. The delta rule bounds the state
    # by `sum(beta * |v|)`, so a run that lands orders of magnitude above the
    # inputs means the bound is not holding and a longer prompt would overflow.
    assert np.abs(state_np).max() < 1e30, (
        f"KDA state reached {np.abs(state_np).max():.4g} after {row_len} steps"
        f" (x_scale={x_scale}) -- finite, but on its way out of range"
    )
