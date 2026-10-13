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
"""The residual's token split must not read a device value back onto the host.

The split needs the packed row count as a bound
:func:`~max.graph.ops.slice_tensor` can take, and that bound has to be on CPU.
Taking it off the device row offsets would be a device-to-host copy, which
lowers to ``mgp.buffer.device_to_host`` followed by an ``mgp.sync``
(``GraphCompiler/lib/MGPDialect/Dialect/BufferBuilder.cpp``). A blocking sync
cannot be recorded into a capturing stream, so one such read anywhere in the
decode graph costs the whole model its device graph capture. The host copy of
the row offsets is already a graph input and carries the same values, so the
bound comes from there.

The split runs once at the stack entry rather than per MoE sublayer, so
this exercises :func:`~...layers.token_parallel.token_shard` directly.

Graph construction only: it builds IR for GPU device refs and never compiles
or executes, so it needs no GPU.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
)
from max.nn.kv_cache import MLAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.glm5_next.layers.token_parallel import (
    token_shard,
)
from max.pipelines.architectures.glm5_next.model_config import (
    Glm5NextConfig,
    token_shard_degree,
)

HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 512
NUM_HEADS = 8
PAGE_SIZE = 128
MAX_SEQ_LEN = 512


def _config(devices: list[DeviceRef]) -> Glm5NextConfig:
    """Enough config for the sublayer: it reads the data-parallel degree."""
    leaf = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=HIDDEN_SIZE,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=devices,
        num_q_heads=NUM_HEADS,
    )
    return Glm5NextConfig(
        dtype=DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"mla": leaf, "indexer": leaf}
        ),
        devices=devices,
        max_seq_len=MAX_SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_hidden_layers=1,
        qk_rope_head_dim=0,
        data_parallel_degree=1,
    )


def _transfers_to_host(graph: Graph) -> list[str]:
    """Every transfer in ``graph`` whose destination is the host.

    Read off the assembly because ``rmo.mo.transfer`` has no Python op class
    (it is excluded from ``_core/mcl/modules/dialects/allowlist.txt``,
    which is also why :func:`~max.graph.ops.transfer_to` stages it through
    ``Graph._add_op``). Each prints as ``... : <source type> to <"cpu", 0>``;
    the host-to-device direction weight placement stages is the same op with a
    ``gpu`` destination, and is not what this guards against.
    """
    return [
        line.strip()
        for line in str(graph._mlir_op).splitlines()
        if "rmo.mo.transfer" in line
        and line.rsplit(" to ", 1)[-1].strip().startswith('<"cpu"')
    ]


def test_token_split_reads_its_bound_from_the_host_offsets() -> None:
    """Staging the split leaves no device-to-host transfer behind."""
    devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
    degree = token_shard_degree(len(devices), data_parallel_degree=1)
    assert degree == len(devices), (
        "tensor-parallel attention must split the token axis, or this test"
        " never reaches the bound it is guarding"
    )

    with Graph(
        "glm5_next_token_split",
        input_types=[
            *(
                TensorType(
                    DType.float32,
                    shape=["total_seq_len", HIDDEN_SIZE],
                    device=device,
                )
                for device in devices
            ),
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.CPU(),
            ),
        ],
    ) as graph:
        *xs, host_row_offsets = graph.inputs
        # The last host offset is the packed row count, which is the bound.
        total = host_row_offsets.tensor[-1].cast(DType.int64)
        graph.output(
            *(
                token_shard(x.tensor, rank, degree, total)
                for rank, x in enumerate(xs)
            )
        )

    assert _transfers_to_host(graph) == []
