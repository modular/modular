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
"""GLM-5.3-Flash's KDA prefill routing, read off the graph.

The graphs are only built, never compiled, so every accelerator is checked
from any host with the driver queries patched. They are patched in the
op-wrapper module that owns the routing helpers, which the KDA core asks;
unpatched, it answers for the host GPU. The core takes traced tensors,
whose GPU inputs resolve to a virtual device of the patched accelerator, so
no GPU is needed either.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from max import functional as F
from max.driver import (
    CPU,
    Buffer,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn._graph_ops.state_space import kda_ring_record_elements, kimi_delta
from max.nn._graph_ops.state_space.kimi_delta import (
    KdaGateMode,
    amd_kda_chunk_workspace_bytes,
    kda_chunk_max_segments,
    kda_chunk_predicate,
    kda_chunk_scratch_rows,
    kda_chunk_workspace_bytes,
)
from max.pipelines.architectures.glm5_next.layers import kda
from max.pipelines.modeling.config_enums import PipelineRole
from max.tensor import Tensor
from max.tensor.realization_context import ensure_context

GPU = DeviceRef.GPU()
HEADS = 32
HEAD_DIM = 128
KERNEL = 4
RING_LEN = 4
QKV_DIM = HEADS * HEAD_DIM
SLOTS = 4


def _patch_accelerator(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    monkeypatch.setattr(kimi_delta, "accelerator_api", lambda: api)
    monkeypatch.setattr(
        kimi_delta, "accelerator_architecture_name", lambda: architecture
    )


@contextmanager
def _virtual_gpu() -> Iterator[None]:
    """One virtual GPU of the patched accelerator, for the traced inputs."""
    set_virtual_device_api(kimi_delta.accelerator_api())
    set_virtual_device_target_arch(kimi_delta.accelerator_architecture_name())
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def _core_mlir(
    gate_mode: KdaGateMode = "safe",
    pool_dtype: DType = DType.float32,
    verify: bool = False,
    qkv_dtype: DType = DType.float32,
) -> str:
    act = TensorType(DType.float32, ["total_tokens", QKV_DIM], device=GPU)
    qkv_part = TensorType(qkv_dtype, ["total_tokens", QKV_DIM], device=GPU)
    conv = TensorType(DType.float32, [QKV_DIM, KERNEL], device=GPU)
    rows = TensorType(DType.int32, [1, "batch"], device=GPU)
    ring = BufferType(
        DType.float32,
        [SLOTS, HEADS, RING_LEN, kda_ring_record_elements(HEAD_DIM, HEAD_DIM)],
        GPU,
    )
    with (
        Graph(
            "glm_kda_routing",
            input_types=[
                qkv_part,
                qkv_part,
                qkv_part,
                conv,
                conv,
                conv,
                BufferType(
                    DType.float32, [SLOTS, 3 * QKV_DIM, KERNEL - 1], device=GPU
                ),
                rows,
                act,
                TensorType(DType.float32, ["total_tokens", HEADS], device=GPU),
                act,
                TensorType(DType.float32, [HEADS], device=GPU),
                TensorType(DType.float32, [QKV_DIM], device=GPU),
                TensorType(DType.float32, [HEAD_DIM], device=GPU),
                BufferType(pool_dtype, [SLOTS, HEADS, HEAD_DIM, HEAD_DIM], GPU),
                rows,
                TensorType(DType.uint32, ["batch_plus_one"], device=GPU),
                ring,
            ],
        ) as graph,
        ensure_context(),
        _virtual_gpu(),
    ):
        i = [
            Tensor.from_graph_value(
                value.buffer
                if isinstance(value.type, BufferType)
                else value.tensor
            )
            for value in graph.inputs
        ]
        out = kda.kda_core(
            q=i[0],
            k=i[1],
            v=i[2],
            q_conv=i[3],
            k_conv=i[4],
            v_conv=i[5],
            conv_pool=i[6],
            conv_rows=i[7],
            raw_gate=i[8],
            beta_logits=i[9],
            output_gate=i[10],
            a_log=i[11],
            dt_bias=i[12],
            norm_weight=i[13],
            recurrent_pool=i[14],
            recurrent_rows=i[15],
            input_row_offsets=i[16],
            layer=F.constant(0, DType.int64, device=CPU()),
            head_dim=HEAD_DIM,
            norm_eps=1e-5,
            gate_mode=gate_mode,
            ring_pool=i[17] if verify else None,
            ring_rows=i[15] if verify else None,
            verify_width=(
                kda.verify_width_operand(RING_LEN - 1) if verify else None
            ),
            capture=kda.CaptureSink([]) if verify else None,
        )
        graph.output(TensorValue(out))
    return str(graph)


def _op_lines(mlir: str, symbol: str) -> list[str]:
    return [line for line in mlir.splitlines() if f'"{symbol}"' in line]


def _chunk_lines(mlir: str) -> list[str]:
    return _op_lines(mlir, "kda_chunk")


_VALUE_DTYPE = re.compile(r"!mo\.(?:tensor|buffer)<\[[^\]]*\], (\w+)")


def _signature(line: str) -> tuple[str, str]:
    """An op line's operand types and result types."""
    operands, results = line.split(") : (", 1)[1].split(" -> ", 1)
    return operands, results


def _operand_dtypes(line: str) -> list[str]:
    """Dtypes of an op's operands, in order."""
    return _VALUE_DTYPE.findall(_signature(line)[0])


def _chunk_io_dtypes(line: str) -> list[str]:
    """Dtypes of a ``kda_chunk`` op's q, k, v and output, in that order."""
    operands, results = _signature(line)
    return (
        _VALUE_DTYPE.findall(operands)[:3] + _VALUE_DTYPE.findall(results)[:1]
    )


def _assert_routed(
    mlir: str,
    io_dtype: str,
    output_dtype: str | None = None,
    fallback: str = "kda_decode",
) -> None:
    lines = _chunk_lines(mlir)
    assert lines, "prefill is not routed to kda_chunk"
    assert fallback in mlir, f"the other branch must stay on {fallback}"
    assert "mo.if" in mlir, "the two ops must sit behind a conditional"
    for line in lines:
        expected = [io_dtype] * 3 + [output_dtype or io_dtype]
        assert _chunk_io_dtypes(line) == expected, line


def _assert_sequential(mlir: str) -> None:
    assert not _chunk_lines(mlir)
    assert "mo.if" not in mlir
    assert "kda_decode" in mlir


def test_gfx950_prefill_takes_bf16_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    _assert_routed(_core_mlir(), "bf16")


@pytest.mark.parametrize("architecture", ["sm_100a", "sm_103"])
def test_sm100_prefill_takes_bf16_chunk(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """NVIDIA takes bf16 q/k/v and a float32 output: the fused kernel.

    It ignores the AMD-only gate and pool conditions.
    """
    _patch_accelerator(monkeypatch, "cuda", architecture)
    for mlir in (
        _core_mlir(),
        _core_mlir(gate_mode="original"),
        _core_mlir(pool_dtype=DType.bfloat16),
    ):
        _assert_routed(mlir, "bf16", "f32")
        assert "use_computebound = true" in mlir


def test_gfx950_chunk_is_not_computebound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD chunk op has no compute-bound variant."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    assert "use_computebound = true" not in _core_mlir()


@pytest.mark.parametrize(
    ("api", "architecture"),
    [("hip", "gfx942"), ("cuda", "sm_90a"), ("cuda", "sm_86")],
)
def test_unsupported_accelerator_stays_sequential(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    _patch_accelerator(monkeypatch, api, architecture)
    _assert_sequential(_core_mlir())


@pytest.mark.parametrize(
    ("api", "architecture"), [("hip", "gfx950"), ("cuda", "sm_100a")]
)
def test_unregistered_chunk_op_stays_sequential(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    """A build without the closed-source kernels lacks ``kda_chunk``."""
    _patch_accelerator(monkeypatch, api, architecture)
    monkeypatch.setattr(kimi_delta, "kda_chunk_is_registered", lambda: False)
    _assert_sequential(_core_mlir())


def test_gfx950_original_gate_stays_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD chunk kernels implement only the bounded ``"safe"`` gate."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    _assert_sequential(_core_mlir(gate_mode="original"))


def test_gfx950_bf16_pool_stays_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD chunk kernels take only a float32 state pool."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    _assert_sequential(_core_mlir(pool_dtype=DType.bfloat16))


def test_gfx950_decode_pins_its_fmas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain decode spells its multiply-adds as the verify ring does.

    The ring reproduces ``kda_decode`` bit for bit only under ``pin_fma``, so
    without it the model's MTP tokens drift from its own greedy tokens. NVIDIA
    pins inside the kernel; AMD takes the parameter.
    """
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    (line,) = _op_lines(_core_mlir(), "kda_decode")
    assert re.search(r"pin_fma\s*=\s*true", line), line


def _ops(mlir: str, symbol: str) -> int:
    return len(re.findall(rf'symbol\s*=\s*"{re.escape(symbol)}"', mlir))


@pytest.mark.parametrize(
    ("api", "architecture", "output_dtype", "computebound", "ring"),
    [
        ("hip", "gfx950", "bf16", False, "kda_verify_ring"),
        ("cuda", "sm_100a", "f32", True, "kda_verify_ring_kgroup_conv"),
    ],
)
def test_verify_keeps_drafts_on_the_ring(
    monkeypatch: pytest.MonkeyPatch,
    api: str,
    architecture: str,
    output_dtype: str,
    computebound: bool,
    ring: str,
) -> None:
    """A batch with drafts records into the ring the rollback folds.

    Only a draft-free prefill, which has nothing to roll back, takes
    ``kda_chunk``, behind the same conditional as the forward's, with the
    forward's chunk launch on either vendor. On NVIDIA the ring is the
    K-group kernel with the short conv in it.
    """
    _patch_accelerator(monkeypatch, api, architecture)
    mlir = _core_mlir(verify=True)
    _assert_routed(mlir, "bf16", output_dtype, fallback=f'"{ring}"')
    assert _ops(mlir, ring) == 1
    assert ("use_computebound = true" in mlir) == computebound
    assert "kda_decode" not in mlir


def test_gfx950_verify_asks_for_the_kgroup_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag the op ignores off NVIDIA, so both vendors trace one call."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    mlir = _core_mlir(verify=True)
    assert "kgroup = true" in mlir
    assert _ops(mlir, "kda_verify_ring_kgroup_conv") == 0


@pytest.mark.parametrize("architecture", ["sm_100a", "sm_90a"])
def test_nvidia_verify_runs_the_conv_in_the_ring(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """With drafts, the conv runs in the ring op, so the ring branch has
    neither a conv op nor the conv output's bf16 cast; the draft-free
    prefill's conv is ``kda_chunk_conv``, which writes the bf16 q, k and v
    itself, inside the chunk branch where ``kda_chunk`` is built."""
    _patch_accelerator(monkeypatch, "cuda", architecture)
    mlir = _core_mlir(verify=True)
    chunked = architecture == "sm_100a"
    assert _ops(mlir, "kda_verify_ring_kgroup_conv") == 1
    assert _ops(mlir, "kda_verify_ring") == 0
    assert _ops(mlir, "kda_chunk_conv") == int(chunked)
    assert _ops(mlir, "gated_delta_conv1d_verify_fwd") == 0
    assert ("mo.if" in mlir) == chunked


@pytest.mark.parametrize(
    ("qkv_dtype", "mlir_dtype"),
    [(DType.bfloat16, "bf16"), (DType.float32, "f32")],
)
def test_nvidia_verify_reads_qkv_in_the_model_dtype(
    monkeypatch: pytest.MonkeyPatch, qkv_dtype: DType, mlir_dtype: str
) -> None:
    """Both K-group conv ops widen ``qkv`` as they load it, so the layer
    hands them the projection's own dtype and makes no float32 copy."""
    _patch_accelerator(monkeypatch, "cuda", "sm_100a")
    mlir = _core_mlir(verify=True, qkv_dtype=qkv_dtype)
    for symbol in ("kda_verify_ring_kgroup_conv", "kda_chunk_conv"):
        (line,) = _op_lines(mlir, symbol)
        assert _operand_dtypes(line)[0] == mlir_dtype, line


def test_nvidia_verify_without_the_conv_op_keeps_the_conv_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build without the K-group conv op keeps the conv and the ring
    apart."""
    _patch_accelerator(monkeypatch, "cuda", "sm_100a")
    monkeypatch.setattr(
        kda,
        "kda_verify_ring_kgroup_conv_servable",
        lambda head_dim, gate_mode, pool_dtype: False,
    )
    mlir = _core_mlir(verify=True)
    assert _ops(mlir, "kda_verify_ring_kgroup_conv") == 0
    assert _ops(mlir, "kda_chunk_conv") == 0
    assert _ops(mlir, "kda_verify_ring") == 1
    assert _ops(mlir, "gated_delta_conv1d_verify_fwd") == 1
    assert "kgroup = true" in mlir


def _legacy_predicate(tokens: TensorValue, seqs: TensorValue) -> TensorValue:
    return kda_chunk_predicate(tokens, seqs)


def _layer_predicate(tokens: TensorValue, seqs: TensorValue) -> TensorValue:
    taken = kda._chunk_predicate(
        Tensor.from_graph_value(tokens), Tensor.from_graph_value(seqs)
    )
    return TensorValue(taken)


_Predicate = Callable[[TensorValue, TensorValue], TensorValue]

PREDICATES = pytest.mark.parametrize(
    "predicate", [_legacy_predicate, _layer_predicate], ids=["op", "layer"]
)
"""The op module's predicate, which the AMD launch mirrors, and the layer's
``F.functional`` wrapper of it, called on Tensors."""


def _takes_chunk(
    monkeypatch: pytest.MonkeyPatch,
    predicate: _Predicate,
    api: str,
    total_tokens: int,
    batch_size: int,
) -> bool:
    """Evaluates a chunk predicate on concrete counts."""
    monkeypatch.setattr(kimi_delta, "accelerator_api", lambda: api)
    scalar = TensorType(DType.int64, [], device=DeviceRef.CPU())
    with (
        Graph("kda_chunk_predicate", input_types=[scalar, scalar]) as graph,
        ensure_context(),
    ):
        tokens, seqs = graph.inputs
        graph.output(predicate(tokens.tensor, seqs.tensor))
    model = InferenceSession(devices=[CPU()]).load(graph)
    (taken,) = model.execute(
        Buffer.from_numpy(np.array(total_tokens, dtype=np.int64)),
        Buffer.from_numpy(np.array(batch_size, dtype=np.int64)),
    )
    assert isinstance(taken, Buffer)
    return bool(taken.to_numpy())


@PREDICATES
@pytest.mark.parametrize("api", ["hip", "cuda"])
def test_decode_batch_skips_chunk(
    monkeypatch: pytest.MonkeyPatch, predicate: _Predicate, api: str
) -> None:
    assert not _takes_chunk(monkeypatch, predicate, api, 8, 8)
    assert _takes_chunk(monkeypatch, predicate, api, 9, 8)


def _reserved(
    lower_bound: float | None = -5.0,
    state_dtype: DType = DType.float32,
    pipeline_role: PipelineRole = "prefill_and_decode",
    enable_chunked_prefill: bool = True,
) -> int:
    return amd_kda_chunk_workspace_bytes(
        num_heads=2 * HEADS,
        head_dim=HEAD_DIM,
        lower_bound=lower_bound,
        state_dtype=state_dtype,
        decode_only=pipeline_role == "decode_only",
        max_batch_input_tokens=8192,
        enable_chunked_prefill=enable_chunked_prefill,
        max_batch_size=8,
        max_length=16384,
    )


def test_gfx950_planner_reserves_the_chunk_workspaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launch allocates them per call, outside the graph memory planner."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    assert _reserved() == kda_chunk_workspace_bytes(
        2 * HEADS, HEAD_DIM, 8192, 8
    )
    assert _reserved(pipeline_role="prefill_only") == _reserved()


def test_planner_without_chunked_prefill_reserves_a_whole_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler then admits a full-length prompt on top of a full batch."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    assert _reserved(enable_chunked_prefill=False) == (
        kda_chunk_workspace_bytes(2 * HEADS, HEAD_DIM, 8192 + 16384, 8)
    )


def _launch_bytes(tokens: int, num_seqs: int, num_heads: int) -> int:
    """What one gfx950 ``kda_chunk`` launch on one device allocates."""
    rows = kda_chunk_scratch_rows(tokens, num_seqs, num_heads)
    return rows * num_heads * HEAD_DIM * 2


# (tokens, sequences, heads, segments, rows). `check_workspace_mirror` in
# `Kernels/test/kda/test_kda_amd_prefill.mojo` checks the launcher against the
# same table; change both together with the launcher's segment rule.
_LAUNCHER_PINNED_SHAPES = [
    (2048, 1, 8, 8, 15114),
    (2049, 1, 8, 6, 13716),
    (8192, 1, 8, 22, 52362),
    (8193, 1, 8, 9, 42516),
    (32768, 1, 12, 32, 166026),
    (81920, 1, 2, 80, 414858),
    (4992, 4, 8, 1, 22080),
    (5120, 4, 8, 8, 47208),
    (8192, 6, 8, 8, 73020),
    (8192, 8, 8, 1, 36432),
    (81920, 8, 12, 12, 428112),
    (16384, 64, 64, 1, 79488),
]


@pytest.mark.parametrize(
    ("tokens", "seqs", "heads", "segments", "rows"), _LAUNCHER_PINNED_SHAPES
)
def test_chunk_sizing_matches_the_launcher_table(
    tokens: int, seqs: int, heads: int, segments: int, rows: int
) -> None:
    """The Python sizing equals the table the launcher's test pins."""
    assert kda_chunk_max_segments(tokens, seqs, heads) == segments
    assert kda_chunk_scratch_rows(tokens, seqs, heads) == rows


@pytest.mark.parametrize("tensor_parallel", [1, 2, 4])
@pytest.mark.parametrize("num_seqs", [1, 8, 64])
@pytest.mark.parametrize("tokens", [64, 1000, 8192, 32768, 81920])
def test_chunk_workspace_bound_covers_every_batch(
    tensor_parallel: int, num_seqs: int, tokens: int
) -> None:
    """The bound covers every batch within the budget and stays near the largest.

    The full batch is not the largest: at 8192 tokens, six sequences split
    into eight segments each where eight sequences do not split at all.
    """
    if num_seqs > tokens:
        pytest.skip("a sequence holds at least one token")
    total_heads = 64
    bound = kda_chunk_workspace_bytes(total_heads, HEAD_DIM, tokens, num_seqs)
    # Both residues mod 32 so the sweep lands on every chunk-count edge and
    # on the single-sequence rule's steps at 2048 and 8192 tokens.
    token_counts = {*range(1, tokens + 1, 32), *range(32, tokens + 1, 32)}
    largest = 0
    for batch_tokens in sorted(token_counts | {tokens}):
        for seqs in range(1, min(num_seqs, batch_tokens) + 1):
            allocated = tensor_parallel * _launch_bytes(
                batch_tokens, seqs, total_heads // tensor_parallel
            )
            assert allocated <= bound, (batch_tokens, seqs)
            largest = max(largest, allocated)
    # The bound mirrors the launcher's mid-length segment fill without its
    # narrow-walk exception, so a few shapes carry up to ~26% slack.
    assert bound <= 1.3 * largest


@pytest.mark.parametrize(
    ("api", "architecture"),
    [("hip", "gfx942"), ("cuda", "sm_100a"), ("cuda", "sm_90a")],
)
def test_planner_reserves_nothing_off_gfx950(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    _patch_accelerator(monkeypatch, api, architecture)
    assert _reserved() == 0


def test_gfx950_planner_reserves_nothing_where_prefill_stays_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors the layer's fallbacks; a decode-only pipeline never prefills."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    assert _reserved(lower_bound=None) == 0
    assert _reserved(state_dtype=DType.bfloat16) == 0
    assert _reserved(pipeline_role="decode_only") == 0


@pytest.mark.parametrize("tensor_parallel", [1, 2, 4])
@pytest.mark.parametrize("tokens", [32768, 81920])
def test_chunk_workspace_bound_is_tight_for_one_long_sequence(
    tensor_parallel: int, tokens: int
) -> None:
    """One long sequence's bound stays near what its devices allocate.

    The bound takes the segment count of a one-head device, which a device
    holding more heads clamps lower; the per-chunk rows dominate, so the
    slack stays small.
    """
    total_heads = 64
    allocated = tensor_parallel * _launch_bytes(
        tokens, 1, total_heads // tensor_parallel
    )
    bound = kda_chunk_workspace_bytes(total_heads, HEAD_DIM, tokens, 1)
    assert allocated <= bound <= 1.2 * allocated
