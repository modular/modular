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
"""The ModuleV3 MiMo-V2 DFlash drafter against the checkpoint's own reference.

The drafter has no end-to-end gate today: the fused-graph test pins op counts,
the loading tests pin the weight names, and the graph-API drafter has an
attention test, so a divergence in the ModuleV3 port's *math* is invisible
until it shows up as lost acceptance. Acceptance hides it well -- a six-point
loss at one block position survives as a 0.2-token change in the aggregate.

This pins the arithmetic. One block is run over a context longer than the
per-query window, and every one of the block's rows is compared to a reference
transcribed from the checkpoint's ``dflash/dflash.py``: the context projection
and its per-layer K/V, the block's own K/V and queries, NeoX RoPE on the
leading ``rotary_dim``, the non-causal windowed attention with its learned
per-head sink, the ``attention_value_scale`` on both value paths, the MLP, and
the final norm.

``dflash.py`` itself omits the two knobs the checkpoint's ``config.json``
declares and both served engines apply (``attention_value_scale`` and
``attention_sink_bias``), so the reference here adds them the way vLLM's
``qwen3_dflash.py`` does.

Row 0 of the block is the anchor, a real token embedding; rows 1..7 are the
trained mask slot. A defect confined to the anchor row -- the row every step
writes twice, once from the target's taps and once from the anchor token's
embedding -- shows up here as a row-0-only disagreement, which the per-row
report names directly.

Two fixtures run, because one cannot pin everything:

``SERVED``
    The shipped geometry: the 1024-key window over a 1100-row context, which
    spans nine 128-token pages. This is the parity gate, and it is the only
    fixture that exercises a multiply-page context.

``NARROW``
    An 8-key window over a 40-row context, so a key near the window's edge
    takes real softmax mass instead of 1/1032 of it. Each convention the
    reference exposes is then separately falsifiable: :data:`CONVENTIONS`
    lists one wrong version of each, and
    ``test_every_convention_changes_the_block`` requires the reference to
    disagree with each wrong version by at least :data:`MUTATION`. Both sides
    of that comparison are the reference in numpy, so it is a
    hardware-independent statement about the fixture rather than a measurement
    of the port; the port is scored against the right reference by
    ``test_narrow_window_block_matches_the_reference``, and the two together
    put the port at least ``MUTATION - NARROW_PARITY`` away from every wrong
    version. At the served width no fixture can separate them -- one key of
    1024 moves a row by 0.1% -- so the narrow width is a unit scale chosen for
    separation, not a configuration anyone serves.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TypedDict

import ml_dtypes
import numpy as np
import numpy.typing as npt
import pytest
from max import functional as F
from max import tree
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    accelerator_count,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn import Module
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from max.nn.module import InputType
from max.pipelines.architectures.dflash_mimo_v2 import DFlashMiMoV2Config
from max.pipelines.architectures.dflash_mimo_v2.weight_adapters import (
    MASK_EMBEDDING,
    drafter_tensor_shapes,
)
from max.pipelines.architectures.dflash_mimo_v2_modulev3 import DFlashMiMoV2
from max.tensor import (
    Tensor,
    compilation,
    default_device,
    default_dtype,
)
from max.tensor.sharding import DeviceMesh, TensorLayout
from test_common.simple_kv_cache import paged_kv_cache_inputs

HIDDEN = 256
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 128
ROTARY = 64
INTERMEDIATE = 256
LAYERS = 2
BLOCK = 8
V_SCALE = 0.612
THETA = 1e4
EPS = 1e-6
TAPS = 2
VOCAB = 64

SERVED_WINDOW, SERVED_CONTEXT = 1024, 1100
"""The shipped drafter's window and a context past it. The block's queries
sit at 1100..1107, so their windows open at keys 77..84 and the first ~80
context rows are invisible to them -- a window that swallowed the whole
context would make the boundary unobservable. The context also spans nine
128-token pages, the only place a multi-page read is exercised."""
NARROW_WINDOW, NARROW_CONTEXT = 8, 40
"""The convention matrix's geometry: a window narrow enough that the key at its
edge takes a real share of a row's softmax mass instead of 1/1032. A row sees
8 to 15 keys here, so a uniform softmax would put 1/15 to 1/8 there; measured
on the fixture's own scores the edge key's share is 0.026 by median, from
0.000 to 0.57 across rows and heads -- heavy-tailed rather than uniform, and no
longer a rounding detail. The width is also what makes the window itself
separable: one key too many or too few moves a block row by 0.83."""

SERVED_GAIN = 1.1
"""Served fixture's norm gains drawn uniform in ``[1/1.1, 1.1]``, i.e. near
the 1.0 a trained norm sits at."""
NARROW_GAIN = 2.25
"""The matrix fixture's gains drawn uniform in ``[1/2.25, 2.25]``. Wide gains
are what make a dropped norm or RoPE visible: at gains of 1.0 dropping either
rescales a row without changing its direction, and the next projection absorbs
the scale."""
NARROW_HIDDEN_GAIN = 5.0
"""The matrix fixture draws the context projection's norm gain from a wider
``[1/5, 5]``. It is the one gain whose loss is otherwise nearly invisible: the
block's keys are re-normalized after projection, so a dropped context gain
survives mostly in the values, where the sink dilutes it. The real taps arrive
at scales spanning that range, so the width is a fixture choice about
separation, not a model assumption."""
NARROW_SINK_OFFSET = -1.0
"""Where the matrix fixture puts each layer's sink, in log space relative to
that layer's loudest key: at ``e**-1`` of it, so the sink takes a real share of
the softmax without drowning the keys. Sinks above the keys make a sink defect
obvious and every key-dependent convention invisible, which is the trade this
fixture cannot avoid. Both fixtures raise the sinks deliberately -- the
checkpoint's own are worth the note on :data:`SINK_MASS` -- so a sink placed
here is a fixture choice, not the checkpoint's value."""

SINK_MASS = 7.0
"""The served fixture's sinks, at the log of the window's key mass, so a sink
defect is visible at the served width too.

That is far above where the checkpoint's own sinks sit: the V2 gate measures
them taking ~3e-5 of a row's softmax mass, which no fixture of any width can
see a defect in. Both fixtures therefore raise the sinks on purpose, and the
convention they guard is the sink's presence and its effect on the
normalization, not its trained value."""

PARITY = 2.5e-2
"""Relative error allowed per block row on the served fixture.

Justified against measured noise rather than inherited: the reference runs in
float32/float64 while the port runs bf16, so the port's rounding floor is what
this must clear, and on B200 the served fixture's eight rows measure 0.0052 to
0.0065 -- the bar is 3.8x the worst of them.
"""
NARROW_PARITY = 4e-2
"""The matrix fixture's own bar, 1.6x :data:`PARITY`. It is not the arithmetic
gate -- the served fixture is -- but the mutations cannot be scored against a
port that is wrong at their own geometry. It is looser because the fixture's
wide gains sharpen the softmax, and rounding the scores to bfloat16 then costs
more: its rows measure 0.0089 to 0.0151.
"""
MUTATION = 10 * PARITY
"""How far a wrong version of the reference must move a block row before the
fixture counts as separating it: ten times the bar the gate scores the port
against.

Measured on this fixture at seed 0, ``half_sink`` is the weakest at 0.390 and
``swapped_mlp_gate`` the strongest at 1.03. Over seeds 0-5 the weakest runs
0.285 to 0.390, so the fixture's separation does not depend on the seed's
luck. It also does not move with the hardware: both sides of the comparison
are the reference in numpy on a seeded fixture, so what is asserted is a
property of the fixture, and the room above the bar is what the fixture has
left rather than what one port's rounding happened to leave it. The gate prints
the same eighteen numbers whether it runs on a GPU lane or a CPU-only rerun of
the reference, which is the check that this stayed true."""
LOGIT_PARITY = 5e-2
"""Relative error allowed per row through a 64-way head. The head sums 256
channels, so it adds its own rounding to the hidden states'."""


def _bf16(x: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Rounds float32 to the nearest bfloat16 and returns its bits."""
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)


def _f32(bits: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    return bits.view(ml_dtypes.bfloat16).astype(np.float32)


def _round(x: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return _f32(_bf16(x))


def _buffer(x: npt.ArrayLike) -> Buffer:
    return Buffer.from_numpy(_bf16(x)).view(DType.bfloat16)


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=LAYERS,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )


def _config(
    kv_params: MHAKVCacheParams, window: int, context: int
) -> DFlashMiMoV2Config:
    return DFlashMiMoV2Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        partial_rotary_factor=ROTARY / HEAD_DIM,
        sliding_window=window,
        block_size=BLOCK,
        target_layer_ids=list(range(TAPS)),
        mask_token_id=7,
        attention_value_scale=V_SCALE,
        max_seq_len=context + BLOCK + 8,
        devices=[DeviceRef.GPU()],
        kv_params=kv_params,
    )


def _weights(
    config: DFlashMiMoV2Config,
    rng: np.random.Generator,
    gain: float,
    hidden_gain: float,
) -> dict[str, npt.NDArray[np.float32]]:
    """BF16-exact random weights: projections that keep activations O(1)."""
    weights = {}
    for name, shape in drafter_tensor_shapes(config).items():
        if name.endswith("attention_sink_bias"):
            w = np.full(shape, SINK_MASS) + 0.5 * rng.standard_normal(shape)
        elif len(shape) == 2:
            w = rng.standard_normal(shape) / math.sqrt(shape[1])
        elif name == "hidden_norm.weight":
            w = rng.uniform(1.0 / hidden_gain, hidden_gain, shape)
        else:
            w = rng.uniform(1.0 / gain, gain, shape)
        weights[name] = _round(w.astype(np.float32))
    weights[MASK_EMBEDDING] = _round(
        rng.standard_normal(HIDDEN).astype(np.float32)
    )
    return weights


def _state(
    weights: dict[str, npt.NDArray[np.float32]],
) -> dict[str, WeightData]:
    return {
        name: WeightData(_buffer(w), name, DType.bfloat16, Shape(w.shape))
        for name, w in weights.items()
        if name != MASK_EMBEDDING
    }


def _rms(
    x: npt.NDArray[np.float32], gain: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * gain


def _rope(
    x: npt.NDArray[np.float32],
    positions: npt.NDArray[np.int64],
    *,
    interleaved: bool = False,
) -> npt.NDArray[np.float32]:
    """Rotation of the leading ``ROTARY`` dims of ``[rows, heads, dim]``.

    NeoX pairs dim ``i`` with ``i + ROTARY/2``; the other half of the space, the
    one a port could have transcribed instead, is the interleaved (GPT-J)
    pairing of ``2i`` with ``2i + 1``. The checkpoint and both served engines
    use NeoX.
    """
    inv_freq = 1.0 / THETA ** (np.arange(0, ROTARY, 2) / ROTARY)
    angles = positions[:, None, None] * inv_freq
    cos, sin = np.cos(angles), np.sin(angles)
    half = ROTARY // 2
    if interleaved:
        rotated = np.empty(x.shape[:-1] + (ROTARY,), np.float32)
        even, odd = x[..., :ROTARY:2], x[..., 1:ROTARY:2]
        rotated[..., ::2] = even * cos - odd * sin
        rotated[..., 1::2] = odd * cos + even * sin
        return np.concatenate([rotated, x[..., ROTARY:]], axis=-1)
    x1, x2 = x[..., :half], x[..., half:ROTARY]
    return np.concatenate(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., ROTARY:]], axis=-1
    )


def _silu(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def _tensor(buffer: Buffer) -> Tensor:
    return Tensor.from_dlpack(buffer)


def _project_kv(
    w: dict[str, npt.NDArray[np.float32]],
    x: npt.NDArray[np.float32],
    positions: npt.NDArray[np.int64],
    attn: str,
    *,
    normed: bool,
    roped: bool,
    v_scale: float,
    interleaved: bool = False,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """One layer's K and V for ``x`` at ``positions``."""
    k = (x @ w[attn + "k_proj.weight"].T).reshape(-1, KV_HEADS, HEAD_DIM)
    k = _rms(
        k,
        w[attn + "k_norm.weight"] if normed else np.ones(HEAD_DIM, np.float32),
    )
    v = (x @ w[attn + "v_proj.weight"].T).reshape(-1, KV_HEADS, HEAD_DIM)
    k = _rope(k, positions, interleaved=interleaved) if roped else k
    return k, v_scale * v


def _reference(
    config: DFlashMiMoV2Config,
    w: dict[str, npt.NDArray[np.float32]],
    taps: npt.NDArray[np.float32],
    anchor: npt.NDArray[np.float32],
    *,
    sinks: bool = True,
    sink_scale: float = 1.0,
    v_scale: float = V_SCALE,
    scale_block_v: bool = True,
    window: int | None = None,
    causal: bool = False,
    q_norm: bool = True,
    k_norm: bool = True,
    context_norm: bool = True,
    context_rope: bool = True,
    block_rope: bool = True,
    final_norm: bool = True,
    swap_gate: bool = False,
    rope_interleaved: bool = False,
    block_rope_from_zero: bool = False,
    gqa_modulo: bool = False,
    scaled_scores: bool = True,
    peaks: list[npt.NDArray[np.float32]] | None = None,
) -> npt.NDArray[np.float32]:
    """The block's final hidden states.

    Transcribed from ``dflash/dflash.py``'s ``DFlashDraftModel.forward`` and
    ``Qwen3DFlashAttention.forward``: the taps are projected once for every
    layer, the block's own rows run the layer stack pre-norm, and each layer
    attends ``[context, block]`` with the checkpoint's sink, value scale and
    per-query window.

    The keyword arguments default to the conventions both served engines
    implement, and each names the single alternative a port could have
    written instead; the matrix test requires the reference to reject every
    one of them. ``peaks``, when given, collects each layer's largest
    attention score per head so the caller can place a sink where it competes
    with the keys. Nothing here rounds: numpy promotes the RoPE table to
    float64 and the products stay in float32, both far below bfloat16's
    eight-bit significand, so what the comparison measures is the port's
    rounding.
    """
    context = len(taps)
    ctx_positions = np.arange(context)
    block_positions = np.arange(context, context + BLOCK)
    ctx = taps.reshape(context, TAPS * HIDDEN) @ w["fc.weight"].T
    ctx = (
        _rms(ctx, w["hidden_norm.weight"])
        if context_norm
        else _rms(ctx, np.ones(HIDDEN, np.float32))
    )

    h = np.concatenate(
        [anchor.reshape(1, HIDDEN), np.tile(w[MASK_EMBEDDING], (BLOCK - 1, 1))]
    )
    group = HEADS // KV_HEADS
    scores_scale = HEAD_DIM**-0.5 if scaled_scores else 1.0
    rope_block = np.arange(BLOCK) if block_rope_from_zero else block_positions
    keys = np.arange(context + BLOCK)
    # Non-causal: a key later than the query is always visible, so the window
    # only bounds it from below.
    visible = (
        keys[None, :] + (window or config.sliding_window)
        > (block_positions[:, None])
    )
    if causal:
        visible = visible & (keys[None, :] <= block_positions[:, None])

    for i in range(config.num_hidden_layers):
        attn = f"layers.{i}.self_attn."
        layer = f"layers.{i}."

        ctx_k, ctx_v = _project_kv(
            w,
            ctx,
            ctx_positions,
            attn,
            normed=k_norm,
            roped=context_rope,
            v_scale=v_scale,
            interleaved=rope_interleaved,
        )

        x = _rms(h, w[layer + "input_layernorm.weight"])
        q = (x @ w[attn + "q_proj.weight"].T).reshape(BLOCK, HEADS, HEAD_DIM)
        q_gain = (
            w[attn + "q_norm.weight"]
            if q_norm
            else np.ones(HEAD_DIM, np.float32)
        )
        q = _rope(_rms(q, q_gain), rope_block, interleaved=rope_interleaved)
        block_k, block_v = _project_kv(
            w,
            x,
            rope_block,
            attn,
            normed=k_norm,
            roped=block_rope,
            v_scale=v_scale if scale_block_v else 1.0,
            interleaved=rope_interleaved,
        )
        k, v = (
            np.concatenate([ctx_k, block_k]),
            np.concatenate([ctx_v, block_v]),
        )
        if gqa_modulo:
            # The other way to widen KV_HEADS to HEADS: head h reading kv head
            # h % KV_HEADS instead of h // group.
            pick = np.arange(HEADS) % KV_HEADS
            k, v = k[:, pick], v[:, pick]
        else:
            k, v = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)
        scores = np.einsum("qhd,khd->hqk", q, k) * scores_scale
        scores = np.where(visible[None], scores, -np.inf)
        if peaks is not None:
            peaks.append(scores.max(axis=(1, 2)))
        if sinks:
            sink = sink_scale * w[attn + "attention_sink_bias"][:, None, None]
            peak = np.maximum(scores.max(-1, keepdims=True), sink)
            mass = np.exp(scores - peak)
            total = mass.sum(-1, keepdims=True) + np.exp(sink - peak)
        else:
            peak = scores.max(-1, keepdims=True)
            mass = np.exp(scores - peak)
            total = mass.sum(-1, keepdims=True)
        out = np.einsum("hqk,khd->qhd", mass / total, v).reshape(BLOCK, -1)
        h = h + out @ w[attn + "o_proj.weight"].T

        y = _rms(h, w[layer + "post_attention_layernorm.weight"])
        gate_w, up_w = (
            ("mlp.up_proj.weight", "mlp.gate_proj.weight")
            if swap_gate
            else ("mlp.gate_proj.weight", "mlp.up_proj.weight")
        )
        gate = _silu(y @ w[layer + gate_w].T)
        up = y @ w[layer + up_w].T
        h = h + (gate * up) @ w[layer + "mlp.down_proj.weight"].T

    if not final_norm:
        return h
    return _rms(h, w["norm.weight"])


class _Conventions(TypedDict, total=False):
    """One wrong version of :func:`_reference`'s conventions, as keywords."""

    sinks: bool
    sink_scale: float
    v_scale: float
    scale_block_v: bool
    window: int
    causal: bool
    q_norm: bool
    k_norm: bool
    context_norm: bool
    context_rope: bool
    block_rope: bool
    final_norm: bool
    swap_gate: bool
    rope_interleaved: bool
    block_rope_from_zero: bool
    gqa_modulo: bool
    scaled_scores: bool


CONVENTIONS: list[tuple[str, _Conventions]] = [
    ("no_sink", dict(sinks=False)),
    ("half_sink", dict(sink_scale=0.5)),
    ("no_value_scale", dict(v_scale=1.0)),
    ("value_scale_on_context_only", dict(scale_block_v=False)),
    ("window_minus_one", dict(window=NARROW_WINDOW - 1)),
    ("window_plus_one", dict(window=NARROW_WINDOW + 1)),
    ("causal_block", dict(causal=True)),
    ("unscaled_scores", dict(scaled_scores=False)),
    ("no_q_norm", dict(q_norm=False)),
    ("no_k_norm", dict(k_norm=False)),
    ("gqa_head_modulo", dict(gqa_modulo=True)),
    ("no_context_norm", dict(context_norm=False)),
    ("no_context_rope", dict(context_rope=False)),
    ("no_block_rope", dict(block_rope=False)),
    ("interleaved_rope", dict(rope_interleaved=True)),
    ("block_rope_from_zero", dict(block_rope_from_zero=True)),
    ("no_final_norm", dict(final_norm=False)),
    ("swapped_mlp_gate", dict(swap_gate=True)),
]
"""One wrong version of each convention :func:`_reference` exposes.

Every one of them must move a block row by at least :data:`MUTATION` for the
fixture to count as pinning that convention;
``test_every_convention_changes_the_block`` asserts it, so a fixture edit that
weakens one fails here instead of quietly leaving the convention unguarded.

Three choices the reference also makes are deliberately not cases here: the
norm epsilon, the activation (``silu``), and the order the taps are
concatenated in. An epsilon two decades off moves a row by less than the
mutation bar, so a case for it would assert something the fixture cannot
resolve; the other two are one fixed choice in :func:`_reference` with no
keyword to flip, and a port that got either wrong fails served parity on its
own.
"""


def _caches(
    kv_params: KVCacheParams, kv: tuple[Tensor, ...]
) -> list[PagedCacheValues]:
    """The KV tree from the staged leaves.

    A ``BufferType`` leaf stages as a Tensor over a ``BufferValue``; the paged
    stores are in-place ops, so they need the buffer value itself rather than
    a loaded view of it.
    """
    return list(
        kv_params.unflatten_kv_inputs(
            iter([v for t in kv for v in t.graph_values])
        )
    )


class _WriteProbe(Module[..., Tensor]):
    """The drafter's context write as a graph body of its own."""

    def __init__(self, drafter: DFlashMiMoV2) -> None:
        self.drafter = drafter
        self.kv_params = drafter.config.kv_params

    def forward(
        self, tap0: Tensor, tap1: Tensor, offsets: Tensor, *kv: Tensor
    ) -> Tensor:
        """Writes every layer's context K/V and passes a tap through.

        The write returns nothing, and a staged body must return Tensors, so
        the second tap rides out as the graph's result.
        """
        self.drafter.write_context(
            [tap0, tap1], offsets, _caches(self.kv_params, kv)
        )
        return tap1


class _BlockProbe(Module[..., Tensor]):
    """The drafter's block forward as a graph body of its own."""

    def __init__(self, drafter: DFlashMiMoV2) -> None:
        self.drafter = drafter
        self.kv_params = drafter.config.kv_params

    def forward(self, anchor: Tensor, offsets: Tensor, *kv: Tensor) -> Tensor:
        """Runs the block and returns its final normed rows."""
        return self.drafter.forward_block(
            self.drafter.block_embeddings(anchor),
            _caches(self.kv_params, kv),
            offsets,
        )


def _run_drafter(
    config: DFlashMiMoV2Config,
    strict_state: dict[str, WeightData],
    mask: WeightData,
    taps: npt.NDArray[np.float32],
    anchor: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Runs the ModuleV3 drafter: context write, then the block.

    Both graphs are staged before either executes, so a ``precompile_mefs``
    record run reaches both compiles before its first device work. Returns the
    block's final normed rows, the surface that carries the context write with
    it.
    """
    context = len(taps)
    kv_params = config.kv_params
    mesh = DeviceMesh((Accelerator(0),), (1,), ("tp",))
    device = Accelerator(0)
    session = InferenceSession(devices=[device])
    kv_types = kv_params.flattened_kv_inputs()
    tap_layout = [
        TensorLayout(DType.bfloat16, ["rows", HIDDEN], mesh)
        for _ in range(TAPS)
    ]
    offsets_layout = TensorLayout(DType.uint32, ["offsets"], mesh)
    rows_layout = TensorLayout(DType.bfloat16, ["rows", HIDDEN], mesh)

    def staged(
        probe_type: Callable[[DFlashMiMoV2], Module[..., Tensor]],
        *layouts: tree.Tree[InputType],
    ) -> compilation.CompiledCallable[..., Tensor]:
        """Stages one probe with the drafter's weights bound by name.

        One drafter per graph: the RoPE table is a cached property, so a
        module reused across two graphs would carry the first graph's tensor
        into the second.
        """
        with F.lazy(), default_device(mesh), default_dtype(DType.bfloat16):
            probe = probe_type(DFlashMiMoV2(config, mask_embedding=mask))
        graph, registry = probe.stage(
            *layouts,
            # The drafter is a child module of the probe, so its weights
            # register under its attribute name.
            weights={f"drafter.{k}": v for k, v in strict_state.items()},
            name=f"dflash_v3_drafter_{probe_type.__name__}",
        )
        return compilation.CompiledCallable(graph, registry, session=session)

    write_call = staged(_WriteProbe, *tap_layout, offsets_layout, *kv_types)
    block_call = staged(_BlockProbe, rows_layout, offsets_layout, *kv_types)

    # The writer prefills the context and the block is one verify step over it,
    # which is what the two caches describe: `max_prompt_length` is a graph
    # input that picks the attention dispatch, so a block cache built as a
    # 1108-token prefill would run the block on the prefill kernel instead of
    # the served verify one.
    write_cache = paged_kv_cache_inputs(kv_params, [context], cache_lengths=[0])
    block_cache = replace(
        paged_kv_cache_inputs(kv_params, [BLOCK], cache_lengths=[context]),
        kv_blocks=write_cache.kv_blocks,
    )

    tap_buffers = [_buffer(taps[:, t, :]).to(device) for t in range(TAPS)]
    ctx_offsets = Buffer.from_numpy(np.array([0, context], np.uint32)).to(
        device
    )
    write_call(
        *[_tensor(b) for b in tap_buffers],
        _tensor(ctx_offsets),
        *[_tensor(b) for b in tree.leaves(write_cache)],
    )

    block_offsets = Buffer.from_numpy(np.array([0, BLOCK], np.uint32)).to(
        device
    )
    hidden = block_call(
        _tensor(_buffer(anchor).to(device)),
        _tensor(block_offsets),
        *[_tensor(b) for b in tree.leaves(block_cache)],
    )
    # numpy has no bfloat16: read the bits and widen through ml_dtypes.
    return _f32(
        np.asarray(
            np.from_dlpack(hidden.driver_tensor.to(CPU()).view(DType.uint16)),
            dtype=np.uint16,
        )
    )


@dataclass(frozen=True)
class _Drafter:
    """One seeded drafter: config, weights and inputs, and no hardware."""

    config: DFlashMiMoV2Config
    weights: dict[str, npt.NDArray[np.float32]]
    taps: npt.NDArray[np.float32]
    anchor: npt.NDArray[np.float32]

    def reference(
        self, conventions: _Conventions | None = None
    ) -> npt.NDArray[np.float32]:
        if conventions is None:
            conventions = {}
        return _reference(
            self.config, self.weights, self.taps, self.anchor, **conventions
        )


@dataclass(frozen=True)
class _Fixture(_Drafter):
    """One seeded drafter and the block's rows the port produced for it."""

    got: npt.NDArray[np.float32]


def _seeded(
    window: int,
    context: int,
    gain: float,
    *,
    hidden_gain: float | None = None,
    place_sinks: bool = False,
) -> _Drafter:
    """Seeds one drafter and, for the matrix fixture, places its sinks."""
    config = _config(_kv_params(), window, context)
    rng = np.random.default_rng(0)
    weights = _weights(config, rng, gain, hidden_gain or gain)
    taps = _round(rng.standard_normal((context, TAPS, HIDDEN)))
    anchor = _round(rng.standard_normal((1, HIDDEN)))
    if place_sinks:
        # Put each sink where it competes with its own head's loudest key. At
        # the checkpoint's value it would take 3e-5 of the softmax mass and no
        # fixture could see a defect in it. Two passes, because a layer's peak
        # moves once the layer below it has its sink.
        for _ in range(2):
            peaks: list[npt.NDArray[np.float32]] = []
            _reference(config, weights, taps, anchor, peaks=peaks)
            for i, peak in enumerate(peaks):
                weights[f"layers.{i}.self_attn.attention_sink_bias"] = _round(
                    peak + NARROW_SINK_OFFSET
                )
    return _Drafter(config, weights, taps, anchor)


def _fixture(
    window: int,
    context: int,
    gain: float,
    *,
    hidden_gain: float | None = None,
    place_sinks: bool = False,
) -> _Fixture:
    """Seeds one drafter and runs it on the GPU once."""
    seeded = _seeded(
        window,
        context,
        gain,
        hidden_gain=hidden_gain,
        place_sinks=place_sinks,
    )
    mask = WeightData(
        _buffer(seeded.weights[MASK_EMBEDDING]),
        MASK_EMBEDDING,
        DType.bfloat16,
        Shape([HIDDEN]),
    )
    got = _run_drafter(
        seeded.config,
        _state(seeded.weights),
        mask,
        seeded.taps,
        seeded.anchor,
    )
    return _Fixture(
        seeded.config,
        seeded.weights,
        seeded.taps,
        seeded.anchor,
        got=got,
    )


def _rel(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _rows(
    got: npt.NDArray[np.float32], want: npt.NDArray[np.float32]
) -> list[float]:
    """Each block row's relative error, row 0 first."""
    return [_rel(got[r], want[r]) for r in range(BLOCK)]


def _report(rows: Sequence[float]) -> str:
    return " ".join(f"row{r}={e:.4f}" for r, e in enumerate(rows))


@pytest.fixture(scope="module")
def served() -> _Fixture:
    """The shipped geometry: the real window over a nine-page context."""
    if accelerator_count() == 0:  # pragma: no cover - GPU lane only
        pytest.skip("needs one GPU")
    return _fixture(SERVED_WINDOW, SERVED_CONTEXT, SERVED_GAIN)


@pytest.fixture(scope="module")
def narrow() -> _Fixture:
    """The separating geometry: a window narrow enough to pin each convention."""
    if accelerator_count() == 0:  # pragma: no cover - GPU lane only
        pytest.skip("needs one GPU")
    return _fixture(
        NARROW_WINDOW,
        NARROW_CONTEXT,
        NARROW_GAIN,
        hidden_gain=NARROW_HIDDEN_GAIN,
        place_sinks=True,
    )


def test_drafter_rows_match_the_checkpoint_reference(served: _Fixture) -> None:
    """Every block row's hidden state, against ``dflash/dflash.py``.

    Row 0 is reported first and separately in the message: it is the anchor,
    the row every serving step writes twice. Its own bound is the maximum
    below, so a row-0-only defect fails with its own number in the report.
    """
    rows = _rows(served.got, served.reference())
    print(f"served rows: {_report(rows)}")
    assert max(rows) < PARITY, f"block rows: {_report(rows)}"


def test_drafter_top1_matches_the_checkpoint_reference(
    served: _Fixture,
) -> None:
    """The per-row argmax through a head, the surface serving compares."""
    head = np.random.default_rng(7).standard_normal((VOCAB, HIDDEN))
    got_logits, want_logits = served.got @ head.T, served.reference() @ head.T
    rows = _rows(got_logits, want_logits)
    print(f"served logit rows: {_report(rows)}")
    assert max(rows) < LOGIT_PARITY, f"per-row logits: {_report(rows)}"
    mismatch = [
        r
        for r in range(BLOCK)
        if got_logits[r].argmax() != want_logits[r].argmax()
    ]
    # One flip is a near-tie in a 64-way seeded head over bf16 activations;
    # more than one is the block's rows disagreeing, which the row check above
    # would already have failed.
    assert len(mismatch) <= 1, f"rows {mismatch} disagree: {_report(rows)}"


def test_narrow_window_block_matches_the_reference(narrow: _Fixture) -> None:
    """The matrix fixture's own parity, so its mutations are scored against a
    port that is already right at that geometry."""
    rows = _rows(narrow.got, narrow.reference())
    print(f"narrow rows: {_report(rows)}")
    assert max(rows) < NARROW_PARITY, f"block rows: {_report(rows)}"


@pytest.fixture(scope="module")
def narrow_reference() -> _Drafter:
    """The matrix fixture, seeded but not run.

    Separate from :func:`narrow` because the convention checks compare the
    reference against its own mutations, which needs no hardware and must not
    inherit the port's rounding.
    """
    return _seeded(
        NARROW_WINDOW,
        NARROW_CONTEXT,
        NARROW_GAIN,
        hidden_gain=NARROW_HIDDEN_GAIN,
        place_sinks=True,
    )


@pytest.mark.parametrize(
    ("convention", "wrong"),
    CONVENTIONS,
    ids=[name for name, _ in CONVENTIONS],
)
def test_every_convention_changes_the_block(
    narrow_reference: _Drafter, convention: str, wrong: _Conventions
) -> None:
    """The reference disagrees with each wrong version of itself.

    This is the fixture's own falsifiability check: if getting a convention
    wrong leaves the block where it is, the fixture cannot see that convention
    and cannot guard the port against it. Both sides are the reference, so the
    number is a numpy computation on a seeded fixture -- the same on every
    host, with no port rounding in it. What it bounds the *port* by is the
    pair of this and ``test_narrow_window_block_matches_the_reference``: the
    port is within ``NARROW_PARITY`` of the right reference on every row, so
    on the row a mutation moves most it is at least
    ``MUTATION - NARROW_PARITY`` away from the wrong version.
    """
    moved = max(
        _rows(narrow_reference.reference(), narrow_reference.reference(wrong))
    )
    print(f"{convention}: moves a block row by {moved:.3f}")
    assert moved > MUTATION, (
        f"{convention} moved a block row by only {moved:.3f}, under the"
        f" {MUTATION:.3f} the fixture must reach to pin it"
    )
