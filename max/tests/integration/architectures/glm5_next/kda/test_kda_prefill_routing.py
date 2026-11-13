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
    kda_chunk_predicate,
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
) -> str:
    act = TensorType(DType.float32, ["total_tokens", QKV_DIM], device=GPU)
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
                act,
                act,
                act,
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
            qkv=F.concat([i[0], i[1], i[2]], axis=-1),
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


def _chunk_lines(mlir: str) -> list[str]:
    return [line for line in mlir.splitlines() if '"kda_chunk"' in line]


_VALUE_DTYPE = re.compile(r"!mo\.(?:tensor|buffer)<\[[^\]]*\], (\w+)")


def _chunk_io_dtypes(line: str) -> list[str]:
    """Dtypes of a ``kda_chunk`` op's q, k, v and output, in that order."""
    operands, results = line.split(") : (", 1)[1].split(" -> ", 1)
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


@pytest.mark.parametrize(
    ("api", "architecture", "output_dtype", "computebound"),
    [("hip", "gfx950", "bf16", False), ("cuda", "sm_100a", "f32", True)],
)
def test_verify_keeps_drafts_on_the_ring(
    monkeypatch: pytest.MonkeyPatch,
    api: str,
    architecture: str,
    output_dtype: str,
    computebound: bool,
) -> None:
    """A batch with drafts records into the ring the rollback folds.

    Only a draft-free prefill, which has nothing to roll back, takes
    ``kda_chunk``, behind the same conditional as the forward's, with the
    forward's chunk launch on either vendor.
    """
    _patch_accelerator(monkeypatch, api, architecture)
    mlir = _core_mlir(verify=True)
    _assert_routed(mlir, "bf16", output_dtype, fallback="kda_verify_ring")
    assert ("use_computebound = true" in mlir) == computebound
    assert "kda_decode" not in mlir


def _legacy_predicate(
    tokens: TensorValue, seqs: TensorValue, num_heads: int
) -> TensorValue:
    return kda_chunk_predicate(tokens, seqs, num_heads, HEAD_DIM)


def _layer_predicate(
    tokens: TensorValue, seqs: TensorValue, num_heads: int
) -> TensorValue:
    taken = kda._chunk_predicate(
        Tensor.from_graph_value(tokens),
        Tensor.from_graph_value(seqs),
        num_heads,
        HEAD_DIM,
    )
    return TensorValue(taken)


_Predicate = Callable[[TensorValue, TensorValue, int], TensorValue]

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
    num_heads: int = HEADS,
) -> bool:
    """Evaluates a chunk predicate on concrete counts."""
    monkeypatch.setattr(kimi_delta, "accelerator_api", lambda: api)
    scalar = TensorType(DType.int64, [], device=DeviceRef.CPU())
    with (
        Graph("kda_chunk_predicate", input_types=[scalar, scalar]) as graph,
        ensure_context(),
    ):
        tokens, seqs = graph.inputs
        graph.output(predicate(tokens.tensor, seqs.tensor, num_heads))
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


@PREDICATES
def test_amd_chunk_past_32_bit_indexing_falls_back(
    monkeypatch: pytest.MonkeyPatch, predicate: _Predicate
) -> None:
    """The AMD launch raises once ``kg_t`` passes 2^31 elements.

    It holds ``(T // 64 + N) * 64`` rows of ``H * 128``; at 32 heads one
    sequence fits up to ``T = 2^19 - 64`` under the predicate's bound.
    """
    limit = (1 << 31) // (HEADS * HEAD_DIM) - 64
    assert _takes_chunk(monkeypatch, predicate, "hip", limit, 1)
    assert not _takes_chunk(monkeypatch, predicate, "hip", limit + 1, 1)
    assert _takes_chunk(monkeypatch, predicate, "cuda", limit + 1, 1)


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


def _launch_groups(
    chunks_per_seq: int, num_seqs: int, num_heads: int, num_cus: int
) -> int:
    """``kda_amd_chunk_groups`` in ``Kernels/lib/kda/amd_chunk_launch.mojo``."""
    planes = max(num_seqs * num_heads, 1)
    fill1, fill2 = num_cus // (2 * planes), num_cus // planes
    groups = fill1
    if fill2 > 0 and chunks_per_seq // fill2 >= 12:
        groups = fill2
    groups = min(groups, chunks_per_seq // 4, 32)
    return groups if groups >= 4 else 1


def _launch_bytes(
    tokens: int, num_seqs: int, num_heads: int, num_cus: int = 256
) -> int:
    """What one ``kda_amd_chunk_launch`` on one device allocates."""
    chunk_len, plane_elems = 64, HEAD_DIM * HEAD_DIM
    max_chunks = tokens // chunk_len + num_seqs
    groups = _launch_groups(
        -(-tokens // chunk_len) // max(num_seqs, 1),
        num_seqs,
        num_heads,
        num_cus,
    )
    bf16 = (
        3 * tokens * num_heads * HEAD_DIM
        + tokens * num_heads * chunk_len
        + max_chunks * num_heads * HEAD_DIM * chunk_len
    )
    f32 = max_chunks * num_heads * HEAD_DIM
    if groups > 1:
        group_planes = groups * num_seqs * num_heads
        f32 += (
            group_planes * (2 * plane_elems + HEAD_DIM)
            + (group_planes * plane_elems + 1) // 2
        )
    return 2 * bf16 + 4 * f32 + 4 * num_seqs


@pytest.mark.parametrize("tensor_parallel", [1, 2, 4])
@pytest.mark.parametrize("num_seqs", [1, 8, 64])
@pytest.mark.parametrize("tokens", [64, 1000, 8192, 32768, 81920])
def test_chunk_workspace_bound_covers_the_launch(
    tensor_parallel: int, num_seqs: int, tokens: int
) -> None:
    """The planner's bound stays above, and near, every device's allocation."""
    if num_seqs > tokens:
        pytest.skip("a sequence holds at least one token")
    total_heads = 64
    allocated = tensor_parallel * _launch_bytes(
        tokens, num_seqs, total_heads // tensor_parallel
    )
    bound = kda_chunk_workspace_bytes(total_heads, HEAD_DIM, tokens, num_seqs)
    assert allocated <= bound <= 2 * allocated


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
def test_chunk_workspace_bound_caps_groups_per_sequence(
    tensor_parallel: int, tokens: int
) -> None:
    """One long sequence splits into at most 32 groups, not one per 4 chunks.

    Without that cap the split workspace alone puts the bound about 1.5x over
    the allocation here.
    """
    total_heads = 64
    allocated = tensor_parallel * _launch_bytes(
        tokens, 1, total_heads // tensor_parallel
    )
    bound = kda_chunk_workspace_bytes(total_heads, HEAD_DIM, tokens, 1)
    assert allocated <= bound <= 1.15 * allocated
