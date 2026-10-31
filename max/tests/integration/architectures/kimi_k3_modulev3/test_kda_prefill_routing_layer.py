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
"""KDA prefill routing in the tensor-parallel Kimi K3 layer, off its trace.

The layer is traced on virtual gfx950 devices, no GPU attached, at the
production head split: K3's 96 KDA heads over 8 tensor-parallel devices,
and the whole 96 again under the pure-EP mesh, where the KDA layers are
replicated. The layer builds one ``kda_decode`` call and carries no device
knowledge, so what these traces pin is that both meshes give every device
exactly one ``kda_decode``, with neither a ``kda_chunk`` nor a per-batch
conditional beside it; the op picks the recurrence at execute time.

Virtual devices must be configured before the process opens its first one,
so this is its own target: no test here may create a device outside the
context manager.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from max import functional as F
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KimiK3KDAAttention,
    tensor_parallel_kda_attention,
)
from max.tensor import default_device
from max.tensor.sharding import (
    BufferLayout,
    DeviceMesh,
    NamedMapping,
    TensorLayout,
)

HEADS = 96
"""K3's KDA head count across all devices."""

HEAD_DIM = 128
HIDDEN = 64
CONV_KERNEL = 4
BATCH = 2
SLOTS = 4


@contextmanager
def _virtual_gfx950(n_devices: int) -> Iterator[None]:
    """gfx950 virtual devices: graph staging with no GPU attached."""
    set_virtual_device_api("hip")
    set_virtual_device_target_arch("gfx950")
    set_virtual_device_count(n_devices)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def _trace_layer(n_devices: int, axis: str, shard_heads: bool) -> str:
    """Traces the KDA layer over a mesh of ``n_devices`` virtual devices.

    ``shard_heads`` splits the 96 heads across a ``tp`` axis, as the model
    does under tensor parallelism; otherwise the layers run replicated on an
    ``ep`` axis, as the pure-EP deployment builds them.
    """
    with _virtual_gfx950(n_devices):
        mesh = DeviceMesh(
            devices=tuple(Accelerator(i) for i in range(n_devices)),
            mesh_shape=(n_devices,),
            axis_names=(axis,),
        )
        with F.lazy(), default_device(mesh):
            layer = KimiK3KDAAttention(
                hidden_size=HIDDEN,
                num_heads=HEADS,
                head_dim=HEAD_DIM,
                conv_kernel_size=CONV_KERNEL,
                gate_lower_bound=-5.0,
            )
            if shard_heads:
                layer = tensor_parallel_kda_attention(layer, mesh)
            inner = HEADS * HEAD_DIM
            replicated = NamedMapping(mesh, (None,))
            head_axis = NamedMapping(mesh, (None, axis))
            staged = layer.trace(
                TensorLayout(
                    DType.bfloat16, ["total_seq_len", HIDDEN], replicated
                ),
                BufferLayout(
                    DType.float32,
                    [SLOTS, 3 * inner, CONV_KERNEL - 1],
                    head_axis,
                ),
                TensorLayout(DType.uint32, [BATCH], replicated),
                BufferLayout(
                    DType.float32,
                    [SLOTS, HEADS, HEAD_DIM, HEAD_DIM],
                    head_axis,
                ),
                TensorLayout(DType.uint32, [BATCH], replicated),
                TensorLayout(DType.uint32, [BATCH + 1], replicated),
            )
        return str(staged)


def _assert_one_decode_per_device(mlir: str, n_devices: int) -> None:
    n_decodes = mlir.count('"kda_decode"')
    assert n_decodes == n_devices, (
        f"each of {n_devices} devices must hold exactly one kda_decode,"
        f" got {n_decodes}"
    )
    assert '"kda_chunk"' not in mlir, (
        "the layer must not route to kda_chunk: the kda_decode op picks"
        " the recurrence itself"
    )
    assert "mo.if" not in mlir, "the layer needs no per-batch conditional"


def test_tp8_ranks_each_get_one_kda_decode() -> None:
    """Each of 8 devices runs 12 heads through its own ``kda_decode``.

    A layer that kept a per-batch conditional, or routed some batches to
    ``kda_chunk``, would leave a ``mo.if`` or a second op in the trace.
    """
    _assert_one_decode_per_device(_trace_layer(8, "tp", shard_heads=True), 8)


def test_ep_mesh_each_gets_one_kda_decode() -> None:
    """The pure-EP mesh replicates the KDA layers, 96 heads per device."""
    _assert_one_decode_per_device(_trace_layer(2, "ep", shard_heads=False), 2)
