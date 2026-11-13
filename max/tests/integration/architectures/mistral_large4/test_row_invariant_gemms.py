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
"""A decode step's GEMM rows match the DSpark verify's rows bit for bit.

Greedy speculative decoding is lossless only if row ``r`` of the 4-row verify
equals what a 1-row decode step computes for the same token. The SM100 matmul
dispatch sends a one-row BF16 matmul to a CUDA-core GEMV and two or more rows
to the tensor-core GEMM, which reduce K in different orders, so Mistral Large 4
computes its BF16 decode projections with ``row_invariant_matmul``
(``mo.matmul.row_invariant``), which keeps a one-row launch on the GEMM. This
pins that every decode projection, computed as the model computes it, matches
across 1 to 4 rows.

Every dense projection of the decode step runs here at its per-rank TP8/EP8
shape, through the ops the model lowers to and with its weights as
device-resident graph constants, as the model's are: the FP8 block-scaled
chain (dynamic ``(1, 128)`` activation quantization, then the block-scaled
matmul) for ``wq``, ``wo``, the indexer's ``wq_b``, the shared expert and
layer 0's FFN; ``row_invariant_matmul`` for the RoPE and NoPE latent
projections, the indexer key and the LM head; the fused mixed-input gate GEMV
for the router, at its precise bound, and a float32 matmul for the indexer head
weights; and the gated FFNs end to end (gate_up, silu times up, down), as
``MistralLarge4MLP`` builds them. The graph runs at 1, 2, 3 and 4 rows over the
same leading rows, and each row must match the 4-row launch bit for bit.

The router's row-count invariance is not limited to a verify batch: up to its
precise bound it runs the same IEEE-fp32 GEMV at every row count, so a context
step's row and a verify step's row agree too. That wider sweep is
``test_fp32_cast_fold.py::test_a_rows_routing_is_the_same_at_every_row_count``,
which needs only the gate projection and so can afford the row counts at the
precise bound this graph cannot (its LM head is 25600 wide). Above that bound
the router returns to the TF32 tile GEMM, so prefill chunks keep today's
precision there, by decision.

Two controls keep the gate honest. The same BF16 projections as plain
``x @ weight.T`` take the GEMV at one row and must differ there, so the
comparison above can see the kernel change the opt-in avoids. And a 4-row
launch whose trailing rows are salted must leave row 0 unchanged: rows of one
launch are computed independently.

What the test cannot see: weights and activations are random, so it pins row
independence, not accuracy; graph-compiler fusions of the full model (residual
epilogues, fused norm and quantize) are not reproduced; the routed experts'
grouped NVFP4 matmul and batches beyond one request are out of scope. That the
model routes every BF16 decode projection through ``row_invariant_matmul`` is
checked by ``test_decode_capture_safe.py`` and end to end.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.nn._graph_ops import (
    MIXED_GEMV_EPILOGUE_MAX_M,
    _mixed_gemv_epilogue,
    _router_gate_mixed_gemv,
    dynamic_scaled_matmul,
    quantize_dynamic_scaled_float8,
    row_invariant_matmul,
)
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.architectures.mistral_large4_modulev3.layers.moe_gate import (
    _FUSED_GEMV_MAX_ROWS,
    _PRECISE_MAX_ROWS,
)
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    ML4_DSPARK_VERIFY_ROWS,
)

_BLOCK = 128
_INPUT_SPEC = InputScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    origin=ScaleOrigin.DYNAMIC,
    dtype=DType.float32,
    block_size=(1, _BLOCK),
)
_WEIGHT_SPEC = WeightScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    dtype=DType.float32,
    block_size=(_BLOCK, _BLOCK),
)


@dataclass(frozen=True)
class _Projection:
    name: str
    kind: str
    """``fp8`` (block-scaled), ``bf16``, ``fp32`` or ``fp32_fused``."""
    n: int
    k: int


# Per-rank shapes at TP8 attention / EP8 experts; the shared expert is listed
# both whole and TP8-split since either is a shape the dispatch may see.
_PROJECTIONS = (
    _Projection("wq", "fp8", 1024, 6144),
    _Projection("wo", "fp8", 6144, 1024),
    _Projection("indexer_wq_b", "fp8", 4096, 6144),
    _Projection("shared_gate_up", "fp8", 12288, 6144),
    _Projection("shared_down", "fp8", 6144, 6144),
    _Projection("shared_gate_up_tp8", "fp8", 1536, 6144),
    _Projection("shared_down_tp8", "fp8", 6144, 768),
    _Projection("dense_gate_up_tp8", "fp8", 6144, 6144),
    _Projection("dense_down_tp8", "fp8", 6144, 3072),
    _Projection("wkv_a", "bf16", 576, 6144),
    _Projection("wkv_a_nope", "bf16", 512, 6144),
    _Projection("indexer_wk", "bf16", 128, 6144),
    _Projection("lm_head_tp8", "bf16", 25600, 6144),
    _Projection("router", "fp32_fused", 128, 6144),
    _Projection("indexer_weights_proj", "fp32_head_weights", 32, 6144),
)

_BY_NAME = {p.name: p for p in _PROJECTIONS}
_FFNS = tuple(
    (_BY_NAME[f"{ffn}_gate_up{tp}"], _BY_NAME[f"{ffn}_down{tp}"])
    for ffn, tp in (("shared", ""), ("shared", "_tp8"), ("dense", "_tp8"))
)
_OUTPUTS = tuple(p.name for p in _PROJECTIONS) + tuple(
    f"{gate_up.name}+{down.name}" for gate_up, down in _FFNS
)
_BF16 = tuple(p for p in _PROJECTIONS if p.kind == "bf16")
_IN_DIMS = sorted({p.k for p in _PROJECTIONS})
_DEVICE = DeviceRef.GPU()


def _fp8_buffer(tensor: torch.Tensor) -> Buffer:
    return Buffer.from_dlpack(tensor.contiguous().view(torch.uint8)).view(
        DType.float8_e4m3fn
    )


def _weights(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Returns each projection's weight (and FP8 block scale) by graph name."""
    weights: dict[str, torch.Tensor] = {}
    for p in _PROJECTIONS:
        w = torch.randn(p.n, p.k, generator=generator) * 0.05
        if p.kind == "fp8":
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            tiled = w.reshape(p.n // _BLOCK, _BLOCK, p.k // _BLOCK, _BLOCK)
            scale = (tiled.abs().amax(dim=(1, 3)) / fp8_max).clamp(min=1e-12)
            q = (tiled / scale[:, None, :, None]).clamp(-fp8_max, fp8_max)
            weights[p.name] = q.reshape(p.n, p.k).to(torch.float8_e4m3fn)
            weights[f"{p.name}.scale"] = scale.float().contiguous()
        else:
            weights[p.name] = w.bfloat16()
    return weights


_GRAPH_WEIGHTS: dict[str, Weight] = {}
"""One `Weight` per name, so the FFN chains share the projections' weights."""


def _weight(g: Graph, name: str, dtype: DType, shape: list[int]) -> TensorValue:
    # Device-resident constants, as the model's weights are: a matmul's
    # dispatch depends on whether its weight is a constant.
    weight = _GRAPH_WEIGHTS.setdefault(
        name, Weight(name, dtype, shape, device=_DEVICE)
    )
    return g.add_weight(weight, force_initial_weight_on_host=False)


def _project(p: _Projection, x: TensorValue, w: TensorValue) -> TensorValue:
    if p.kind == "bf16":
        return row_invariant_matmul(x, w)
    if p.kind == "fp32_fused":
        # The router widens its activation inside the gate GEMV at every row
        # count this test runs, so that is the route to hold row-invariant; and
        # it asks the op for IEEE fp32 above the window too, so the rows of a
        # 4-row verify and a 1-row decode reduce the same way.
        return _router_gate_mixed_gemv(
            x,
            ops.cast(w, DType.float32),
            max_m=_FUSED_GEMV_MAX_ROWS,
            precise_max_m=_PRECISE_MAX_ROWS,
        )
    if p.kind == "fp32_head_weights":
        # The indexer's head-weight projection on SM100 is this op: one launch
        # that widens the activation in registers and applies the scale chain
        # in its epilogue, in place of the cast plus fp32 matmul. Its window is
        # the row count an fp32 matmul of this shape crosses to the TF32 tile
        # GEMM at, so it is bit-identical to that chain at every row count this
        # test runs and a 4-row verify still reduces like a 1-row decode.
        return _mixed_gemv_epilogue(
            x, ops.cast(w, DType.float32), max_m=MIXED_GEMV_EPILOGUE_MAX_M
        )
    # The model projects these in float32, as its reference does.
    return ops.cast(x, DType.float32) @ ops.cast(w, DType.float32).T


def _fp8_matmul(g: Graph, p: _Projection, x: TensorValue) -> TensorValue:
    w = _weight(g, p.name, DType.float8_e4m3fn, [p.n, p.k])
    w_scale = _weight(
        g, f"{p.name}.scale", DType.float32, [p.n // _BLOCK, p.k // _BLOCK]
    )
    x_fp8, x_scales = quantize_dynamic_scaled_float8(
        x,
        _INPUT_SPEC,
        _WEIGHT_SPEC,
        group_size_or_per_token=_BLOCK,
        scales_type=DType.float32,
        out_type=DType.float8_e4m3fn,
    )
    return dynamic_scaled_matmul(
        x_fp8,
        w,
        x_scales,
        w_scale,
        _INPUT_SPEC,
        _WEIGHT_SPEC,
        out_type=DType.bfloat16,
    )


def _build_graph() -> Graph:
    x_types = [
        TensorType(DType.bfloat16, ["rows", k], device=_DEVICE)
        for k in _IN_DIMS
    ]
    with Graph("ml4_row_invariant_gemms", input_types=x_types) as g:
        xs = {k: g.inputs[i].tensor for i, k in enumerate(_IN_DIMS)}
        outputs = []
        for p in _PROJECTIONS:
            x = xs[p.k]
            if p.kind == "fp8":
                outputs.append(_fp8_matmul(g, p, x))
            else:
                w = _weight(g, p.name, DType.bfloat16, [p.n, p.k])
                outputs.append(_project(p, x, w))
        # The gated FFNs as `MistralLarge4MLP` builds them, so the compiler
        # may fuse across the chain as it does in the model.
        for gate_up, down in _FFNS:
            h = _fp8_matmul(g, gate_up, xs[gate_up.k])
            gate, up = ops.split(h, [down.k, down.k], axis=-1)
            outputs.append(_fp8_matmul(g, down, ops.silu(gate) * up))
        # Controls: the BF16 projections without the opt-in.
        for p in _BF16:
            w = _weight(g, p.name, DType.bfloat16, [p.n, p.k])
            outputs.append(xs[p.k] @ w.T)
        g.output(*outputs)
    return g


@dataclass
class _Fixture:
    model: Model
    activations: dict[int, torch.Tensor]
    """``[4, k]`` canonical rows per input width."""
    salted: dict[int, torch.Tensor]
    """The same rows with rows 1-3 replaced."""


@pytest.fixture(scope="module")
def fixture() -> Iterator[_Fixture]:
    generator = torch.Generator().manual_seed(832)
    rows = ML4_DSPARK_VERIFY_ROWS
    activations = {
        k: torch.randn(rows, k, generator=generator).bfloat16()
        for k in _IN_DIMS
    }
    salted = {}
    for k, x in activations.items():
        s = x.clone()
        s[1:] = torch.randn(rows - 1, k, generator=generator).bfloat16() * 3
        salted[k] = s
    device = Accelerator()
    registry = {
        name: (
            _fp8_buffer(w)
            if w.dtype == torch.float8_e4m3fn
            else Buffer.from_dlpack(w.contiguous())
        ).to(device)
        for name, w in _weights(generator).items()
    }
    session = InferenceSession(devices=[device])
    model = session.load(_build_graph(), weights_registry=registry)
    yield _Fixture(model, activations, salted)


def _bits(out: object) -> torch.Tensor:
    assert isinstance(out, Buffer)
    t = torch.from_dlpack(out).cpu()
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _run_all(
    fx: _Fixture, rows: int, salted: bool = False
) -> list[torch.Tensor]:
    """Returns every graph output, as raw bits, for the first ``rows``."""
    device = fx.model.input_devices[0]
    source = fx.salted if salted else fx.activations
    xs = [
        Buffer.from_dlpack(source[k][:rows].contiguous()).to(device)
        for k in _IN_DIMS
    ]
    return [_bits(out) for out in fx.model(*xs)]


def _run(fx: _Fixture, rows: int, salted: bool = False) -> list[torch.Tensor]:
    """Returns each output as the model computes it, for the first ``rows``."""
    return _run_all(fx, rows, salted)[: len(_OUTPUTS)]


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a != b).sum())


def test_step_rows_match_verify_rows(fixture: _Fixture) -> None:
    verify = _run(fixture, ML4_DSPARK_VERIFY_ROWS)
    failures = []
    for rows in range(1, ML4_DSPARK_VERIFY_ROWS):
        step = _run(fixture, rows)
        for name, s, v in zip(_OUTPUTS, step, verify, strict=True):
            diffs = _differing(s, v[:rows])
            if diffs:
                failures.append(f"{name} rows={rows}: {diffs} elements")
    assert not failures, (
        "a launch below the verify height differs from the verify's rows: "
        + "; ".join(failures)
    )


def test_rows_are_independent_of_their_siblings(fixture: _Fixture) -> None:
    rows = ML4_DSPARK_VERIFY_ROWS
    plain = _run(fixture, rows)
    salted = _run(fixture, rows, salted=True)
    failures = [
        name
        for name, a, b in zip(_OUTPUTS, plain, salted, strict=True)
        if _differing(a[:1], b[:1])
    ]
    assert not failures, f"row 0 moved with its siblings in {failures}"


def test_control_plain_bf16_moves_with_the_row_count(
    fixture: _Fixture,
) -> None:
    """Without the opt-in every BF16 latent and head projection must move.

    Otherwise the step/verify comparison above could not fail. The indexer
    key projection is too narrow to move on every input, so it is reported
    but not required.
    """
    step = _run_all(fixture, 1)[len(_OUTPUTS) :]
    verify = _run_all(fixture, ML4_DSPARK_VERIFY_ROWS)[len(_OUTPUTS) :]
    moved = {
        p.name: _differing(s, v[:1])
        for p, s, v in zip(_BF16, step, verify, strict=True)
    }
    print(f"one-row plain BF16 elements that move: {moved}")
    missing = [
        name
        for name in ("wkv_a", "wkv_a_nope", "lm_head_tp8")
        if not moved[name]
    ]
    assert not missing, (
        f"plain {missing} no longer move at one row ({moved}); the opt-in may"
        " no longer be needed"
    )
