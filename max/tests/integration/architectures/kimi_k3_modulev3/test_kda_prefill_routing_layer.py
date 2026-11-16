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
from max.tensor import Tensor, default_device
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


def _trace_layer(
    n_devices: int, axis: str, shard_heads: bool, narrow_weight: bool = False
) -> str:
    """Traces the KDA layer over a mesh of ``n_devices`` virtual devices.

    ``shard_heads`` splits the 96 heads across a ``tp`` axis, as the model
    does under tensor parallelism; otherwise the layers run replicated on an
    ``ep`` axis, as the pure-EP deployment builds them.

    A layer declares its fused q/k/v/g/beta weight wide enough for the gate's
    rows, the width the checkpoint adapter fills (``fuse_qkvgb_for_tp``).
    ``narrow_weight`` replaces it with the five-block weight a loader that
    does not fuse the gate leaves behind, which keeps the gate's own GEMV.
    """
    with _virtual_gfx950(n_devices):
        mesh = DeviceMesh(
            devices=tuple(Accelerator(i) for i in range(n_devices)),
            mesh_shape=(n_devices,),
            axis_names=(axis,),
        )
        with F.lazy(), default_device(mesh):
            tp_size = n_devices if shard_heads else 1
            layer = KimiK3KDAAttention(
                hidden_size=HIDDEN,
                num_heads=HEADS,
                head_dim=HEAD_DIM,
                conv_kernel_size=CONV_KERNEL,
                gate_lower_bound=-5.0,
                tp_size=tp_size,
            )
            if narrow_weight:
                layer.qkvg_proj.weight = Tensor.zeros(
                    [narrow_qkvg_rows(tp_size), HIDDEN], dtype=DType.bfloat16
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


def fused_qkvg_rows(tp_size: int) -> int:
    """The row count of the fused q/k/v/g/beta weight, gate rows included."""
    return 4 * HEADS * HEAD_DIM + HEADS + tp_size * HEAD_DIM


def narrow_qkvg_rows(tp_size: int) -> int:
    """The five checkpoint blocks only: q, k, v, g and beta."""
    del tp_size  # the five blocks are TP-independent in the global weight
    return 4 * HEADS * HEAD_DIM + HEADS


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


def test_staged_trace_folds_the_conv_operands() -> None:
    """The staged graph reaches ``kda_decode`` on the fused operands.

    The layer routes by shape at trace time only to decide operand
    conventions, and a served batch arrives with symbolic extents -- so a
    predicate that needs static shapes would leave every production batch
    on the activated path, where the graph pays the silu, cast and
    pre-activation elementwise passes this fusion removes. The staged trace
    here uses the symbolic input types the serving path builds, so it must
    carry ``input_mode = "conv_raw"`` and no silu elementwise op ahead of
    the decode.
    """
    mlir = _trace_layer(8, "tp", shard_heads=True)
    assert '"conv_raw"' in mlir, (
        "the staged kda_decode must take the fused conv_raw operands; a"
        " shape predicate left the served graph on the activated path"
    )
    assert "silu" not in mlir, (
        "the silu the fused prologue folds must not survive as its own"
        " elementwise op ahead of kda_decode"
    )


def test_tp8_ranks_each_get_one_kda_decode() -> None:
    """Each of 8 devices runs 12 heads through its own ``kda_decode``.

    A layer that kept a per-batch conditional, or routed some batches to
    ``kda_chunk``, would leave a ``mo.if`` or a second op in the trace.
    """
    _assert_one_decode_per_device(_trace_layer(8, "tp", shard_heads=True), 8)


def test_ep_mesh_each_gets_one_kda_decode() -> None:
    """The pure-EP mesh replicates the KDA layers, 96 heads per device."""
    _assert_one_decode_per_device(_trace_layer(2, "ep", shard_heads=False), 2)


def test_gate_down_row_rides_the_fused_input_gemm() -> None:
    """``f_a_proj`` is not a launch of its own.

    The gate's ``head_dim``-row down projection reads the same ``x`` as the
    fused q/k/v/g/beta projection, so the checkpoint adapter appends its rows
    to that weight once, at load time (``fuse_qkvgb_for_tp``), and one GEMM
    produces both -- a whole decode launch per layer per step less. Pinned at
    the production per-device shape: each rank's fused weight is its own
    ``q_r | k_r | v_r | g_r | b_r`` block plus the WHOLE replicated gate row
    (``f_a_proj`` is not head-sharded, because every rank's ``f_b_proj`` slice
    needs all ``head_dim`` of it), the GEMM's output is correspondingly wider,
    and the layer traces three matmuls per device -- this GEMM, ``f_b_proj``
    and ``o_proj`` -- instead of four.

    No weight concat appears in the graph: the stored weight already carries
    the gate's rows, so no layer holds a second resident copy of its fused
    weight.
    """
    n_devices = 8
    mlir = _trace_layer(n_devices, "tp", shard_heads=True)
    per_rank = fused_qkvg_rows(n_devices) // n_devices
    assert per_rank == (4 * HEADS * HEAD_DIM + HEADS) // n_devices + HEAD_DIM
    # The stored per-device weight, gate rows included: no concat of it.
    assert f"!mo.tensor<[{per_rank}, {HIDDEN}], bf16, gpu:0>" in mlir, (
        "the per-device fused weight does not carry the gate's rows"
    )
    assert (
        f"!mo.tensor<[{HEAD_DIM}, {HIDDEN}], bf16, gpu:0>"
        f" -> !mo.tensor<[{per_rank}, {HIDDEN}], bf16, gpu:0>"
    ) not in mlir, "the graph still concatenates the gate into the weight"
    # The one GEMM over the layer input, now `head_dim` columns wider.
    assert f"!mo.tensor<[total_seq_len, {per_rank}], bf16, gpu:0>" in mlir, (
        "the fused input GEMM does not carry the gate's down projection"
    )
    # `f_b_proj` still reads the WHOLE gate row on each device.
    assert (
        f"(!mo.tensor<[total_seq_len, {HEAD_DIM}], bf16, gpu:0>,"
        f" !mo.tensor<[{HEAD_DIM}, 1536], bf16, gpu:0>)"
    ) in mlir, "f_b_proj does not read the whole per-device gate row"
    n_matmul = mlir.count("rmo.matmul")
    assert n_matmul == 3 * n_devices, (
        f"each device must trace 3 matmuls (fused qkvg+gate, f_b, o_proj),"
        f" got {n_matmul} over {n_devices} devices"
    )


def test_unfused_layer_keeps_its_own_gate_gemm() -> None:
    """A layer loaded without the fused weight still projects the gate.

    The integration tests build a layer directly, and a checkpoint whose gate
    did not merge leaves ``f_a_proj`` on its own: that path must keep working,
    one launch heavier, rather than reading rows the weight does not have.
    """
    n_devices = 8
    mlir = _trace_layer(n_devices, "tp", shard_heads=True, narrow_weight=True)
    per_rank = (4 * HEADS * HEAD_DIM + HEADS) // n_devices
    assert f"!mo.tensor<[{per_rank}, {HIDDEN}], bf16, gpu:0>" in mlir, (
        "the narrow fused weight is not the five checkpoint blocks"
    )
    n_matmul = mlir.count("rmo.matmul")
    assert n_matmul == 4 * n_devices, (
        f"each device must trace 4 matmuls (qkvg, f_a, f_b, o_proj),"
        f" got {n_matmul} over {n_devices} devices"
    )


def test_declared_weight_width_matches_the_adapter() -> None:
    """The declared ``qkvg_proj`` width is the adapter's, or the load fails.

    ``check_fits`` rejects a checkpoint tensor whose shape differs from the
    parameter it fills, so the layer has to declare the fused width the
    adapter emits -- gate rows included, one replicated block per rank.
    """
    with _virtual_gfx950(1), F.lazy():
        for tp_size in (1, 2, 8):
            layer = KimiK3KDAAttention(
                hidden_size=HIDDEN,
                num_heads=HEADS,
                head_dim=HEAD_DIM,
                conv_kernel_size=CONV_KERNEL,
                gate_lower_bound=-5.0,
                tp_size=tp_size,
            )
            declared = int(layer.qkvg_proj.weight.shape[0])
            assert declared == fused_qkvg_rows(tp_size), (
                f"at tp={tp_size} the layer declares {declared} rows but the"
                f" adapter emits {fused_qkvg_rows(tp_size)}"
            )


def test_ep_mesh_fuses_the_gate_down_row_replicated() -> None:
    """On the pure-EP mesh the whole layer is replicated, gate row included."""
    n_devices = 2
    mlir = _trace_layer(n_devices, "ep", shard_heads=False)
    fused_rows = fused_qkvg_rows(1)
    assert f"!mo.tensor<[{fused_rows}, {HIDDEN}], bf16, gpu:0>" in mlir, (
        "the replicated fused weight does not carry the gate's down projection"
    )
    n_matmul = mlir.count("rmo.matmul")
    assert n_matmul == 3 * n_devices, (
        f"each device must trace 3 matmuls, got {n_matmul} over {n_devices}"
    )
