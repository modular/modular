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
"""KDA prefill routing in both GLM-5.3-Flash pipelines, read off the graph.

The graphs are only built, never compiled, so every accelerator is checked
from any host with the driver queries patched. They are patched in every
module that reads them: the layer, ModuleV3's core and the op wrapper each
ask on their own, and an unpatched one answers for the host GPU.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy.layer import Module
from max.nn.legacy.state_space import kimi_delta
from max.nn.legacy.state_space.kimi_delta import KdaGateMode
from max.pipelines.architectures.glm5_next.layers import (
    kimi_delta_attention,
)
from max.pipelines.architectures.glm5_next.layers.kimi_delta_attention import (
    KimiDeltaAttention,
    kda_chunk_predicate,
)
from max.pipelines.architectures.glm5_next.memory_planner import (
    amd_kda_chunk_workspace_bytes,
    kda_chunk_workspace_bytes,
)
from max.pipelines.architectures.glm5_next_modulev3.layers import (
    kda as modulev3_kda,
)
from max.pipelines.lib import PipelineRuntimeConfig
from max.pipelines.modeling.config_enums import PipelineRole

GPU = DeviceRef.GPU()
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
KERNEL = 4
QKV_DIM = HEADS * HEAD_DIM
SLOTS = 4


def _patch_accelerator(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    for module in (kimi_delta_attention, kimi_delta, modulev3_kda):
        monkeypatch.setattr(module, "accelerator_api", lambda: api)
    for module in (kimi_delta_attention, kimi_delta):
        monkeypatch.setattr(
            module, "accelerator_architecture_name", lambda: architecture
        )


def _qualify_weight_names(layer: Module, prefix: str = "kda") -> None:
    """Gives every weight its module path, as the enclosing model would.

    Standalone, the projections are all named "weight", and the graph refuses
    a duplicate.
    """
    for name, weight in layer.layer_weights.items():
        weight.name = f"{prefix}.{name}"
    for name, sub in layer.sublayers.items():
        _qualify_weight_names(sub, f"{prefix}.{name}")


def _layer_mlir(
    lower_bound: float | None = -5.0,
    pool_dtype: DType = DType.float32,
    replay: bool = False,
) -> str:
    layer = KimiDeltaAttention(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=KERNEL,
        dtype=DType.bfloat16,
        device=GPU,
        rms_norm_eps=1e-5,
        lower_bound=lower_bound,
    )
    _qualify_weight_names(layer)
    with Graph(
        "glm_kda_routing",
        input_types=[
            TensorType(DType.bfloat16, ["total_tokens", HIDDEN], device=GPU),
            BufferType(
                DType.float32, [SLOTS, 3 * QKV_DIM, KERNEL - 1], device=GPU
            ),
            TensorType(DType.int32, ["batch"], device=GPU),
            BufferType(pool_dtype, [SLOTS, HEADS, HEAD_DIM, HEAD_DIM], GPU),
            TensorType(DType.uint32, ["batch_plus_one"], device=GPU),
        ],
    ) as graph:
        x, conv_pool, row, rec_pool, offsets = graph.inputs
        graph.output(
            layer(
                x.tensor,
                conv_pool.buffer,
                row.tensor,
                rec_pool.buffer,
                row.tensor,
                offsets.tensor,
                replay_capture=[] if replay else None,
            )
        )
    return str(graph)


def _modulev3_mlir(
    gate_mode: KdaGateMode = "safe", pool_dtype: DType = DType.float32
) -> str:
    act = TensorType(DType.float32, ["total_tokens", QKV_DIM], device=GPU)
    conv = TensorType(DType.float32, [QKV_DIM, KERNEL], device=GPU)
    rows = TensorType(DType.int32, [1, "batch"], device=GPU)
    with Graph(
        "glm_kda_modulev3_routing",
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
        ],
    ) as graph:
        i = graph.inputs
        graph.output(
            modulev3_kda._kda_core_graph(
                q=i[0].tensor,
                k=i[1].tensor,
                v=i[2].tensor,
                q_conv=i[3].tensor,
                k_conv=i[4].tensor,
                v_conv=i[5].tensor,
                conv_pool=i[6].buffer,
                conv_rows=i[7].tensor,
                raw_gate=i[8].tensor,
                beta_logits=i[9].tensor,
                output_gate=i[10].tensor,
                a_log=i[11].tensor,
                dt_bias=i[12].tensor,
                norm_weight=i[13].tensor,
                recurrent_pool=i[14].buffer,
                recurrent_rows=i[15].tensor,
                input_row_offsets=i[16].tensor,
                layer=ops.constant(0, DType.int64, device=DeviceRef.CPU()),
                head_dim=HEAD_DIM,
                norm_eps=1e-5,
                gate_mode=gate_mode,
            )
        )
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


def _assert_routed(mlir: str, io_dtype: str) -> None:
    lines = _chunk_lines(mlir)
    assert lines, "prefill is not routed to kda_chunk"
    assert "kda_decode" in mlir, "decode must stay on kda_decode"
    assert "mo.if" in mlir, "the two ops must sit behind a conditional"
    for line in lines:
        assert _chunk_io_dtypes(line) == [io_dtype] * 4, line


def _assert_sequential(mlir: str) -> None:
    assert not _chunk_lines(mlir)
    assert "mo.if" not in mlir
    assert "kda_decode" in mlir


def test_gfx950_prefill_takes_bf16_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    _assert_routed(_layer_mlir(), "bf16")
    _assert_routed(_modulev3_mlir(), "bf16")


@pytest.mark.parametrize("architecture", ["sm_100a", "sm_103"])
def test_sm100_prefill_takes_float32_chunk(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """NVIDIA keeps float32 q/k/v and ignores the AMD-only conditions."""
    _patch_accelerator(monkeypatch, "cuda", architecture)
    _assert_routed(_layer_mlir(), "f32")
    _assert_routed(_layer_mlir(lower_bound=None), "f32")
    _assert_routed(_layer_mlir(pool_dtype=DType.bfloat16), "f32")


@pytest.mark.parametrize("architecture", ["sm_100a", "sm_103"])
def test_sm100_modulev3_prefill_stays_sequential(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """ModuleV3's NVIDIA chunk path has no numerics coverage yet."""
    _patch_accelerator(monkeypatch, "cuda", architecture)
    _assert_sequential(_modulev3_mlir())


@pytest.mark.parametrize(
    ("api", "architecture"),
    [("hip", "gfx942"), ("cuda", "sm_90a"), ("cuda", "sm_86")],
)
def test_unsupported_accelerator_stays_sequential(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    _patch_accelerator(monkeypatch, api, architecture)
    _assert_sequential(_layer_mlir())
    _assert_sequential(_modulev3_mlir())


@pytest.mark.parametrize(
    ("api", "architecture"), [("hip", "gfx950"), ("cuda", "sm_100a")]
)
def test_unregistered_chunk_op_stays_sequential(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    """A build without the closed-source kernels lacks ``kda_chunk``."""
    _patch_accelerator(monkeypatch, api, architecture)
    monkeypatch.setattr(
        kimi_delta_attention, "kda_chunk_is_registered", lambda: False
    )
    _assert_sequential(_layer_mlir())
    _assert_sequential(_modulev3_mlir())


def test_gfx950_original_gate_stays_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD chunk kernels implement only the bounded ``"safe"`` gate."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    _assert_sequential(_layer_mlir(lower_bound=None))
    _assert_sequential(_modulev3_mlir(gate_mode="original"))


def test_gfx950_bf16_pool_stays_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD chunk kernels take only a float32 state pool."""
    _patch_accelerator(monkeypatch, "hip", "gfx950")
    _assert_sequential(_layer_mlir(pool_dtype=DType.bfloat16))
    _assert_sequential(_modulev3_mlir(pool_dtype=DType.bfloat16))


@pytest.mark.parametrize(
    ("api", "architecture"), [("hip", "gfx950"), ("cuda", "sm_100a")]
)
def test_replay_capture_stays_sequential(
    monkeypatch: pytest.MonkeyPatch, api: str, architecture: str
) -> None:
    """A speculative replay must reproduce the forward's arithmetic."""
    _patch_accelerator(monkeypatch, api, architecture)
    _assert_sequential(_layer_mlir(replay=True))


def _takes_chunk(
    monkeypatch: pytest.MonkeyPatch,
    api: str,
    total_tokens: int,
    batch_size: int,
    num_heads: int = HEADS,
) -> bool:
    """Evaluates :func:`kda_chunk_predicate` on concrete counts."""
    monkeypatch.setattr(kimi_delta_attention, "accelerator_api", lambda: api)
    scalar = TensorType(DType.int64, [], device=DeviceRef.CPU())
    with Graph("kda_chunk_predicate", input_types=[scalar, scalar]) as graph:
        tokens, seqs = graph.inputs
        graph.output(
            kda_chunk_predicate(tokens.tensor, seqs.tensor, num_heads, HEAD_DIM)
        )
    model = InferenceSession(devices=[CPU()]).load(graph)
    (taken,) = model.execute(
        Buffer.from_numpy(np.array(total_tokens, dtype=np.int64)),
        Buffer.from_numpy(np.array(batch_size, dtype=np.int64)),
    )
    assert isinstance(taken, Buffer)
    return bool(taken.to_numpy())


@pytest.mark.parametrize("api", ["hip", "cuda"])
def test_decode_batch_skips_chunk(
    monkeypatch: pytest.MonkeyPatch, api: str
) -> None:
    assert not _takes_chunk(monkeypatch, api, 8, 8)
    assert _takes_chunk(monkeypatch, api, 9, 8)


def test_amd_chunk_past_32_bit_indexing_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD launch raises once ``kg_t`` passes 2^31 elements.

    It holds ``(T // 64 + N) * 64`` rows of ``H * 128``; at 32 heads one
    sequence fits up to ``T = 2^19 - 64`` under the predicate's bound.
    """
    limit = (1 << 31) // (HEADS * HEAD_DIM) - 64
    assert _takes_chunk(monkeypatch, "hip", limit, 1)
    assert not _takes_chunk(monkeypatch, "hip", limit + 1, 1)
    assert _takes_chunk(monkeypatch, "cuda", limit + 1, 1)


def _reserved(
    lower_bound: float | None = -5.0,
    state_dtype: DType = DType.float32,
    pipeline_role: PipelineRole = "prefill_and_decode",
    enable_chunked_prefill: bool = True,
) -> int:
    return amd_kda_chunk_workspace_bytes(
        PipelineRuntimeConfig(
            max_batch_input_tokens=8192,
            pipeline_role=pipeline_role,
            enable_chunked_prefill=enable_chunked_prefill,
        ),
        num_heads=2 * HEADS,
        head_dim=HEAD_DIM,
        lower_bound=lower_bound,
        state_dtype=state_dtype,
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
