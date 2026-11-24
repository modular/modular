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

One more trace covers the other arm: a layer holding a speculative verify's
binding, whose graph runs the fused verify op. There the gate's up projection
is the op's own work, so the graph holds two matmuls per device instead of
three -- that trace is what pins the fold, and a control that turns it off
proves the assertion can fail.
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
from max.graph import ops
from max.nn._graph_ops.state_space import verify_width_operand
from max.pipelines.architectures.kimi_k3_modulev3.layers.kda_attention import (
    KdaVerifyBinding,
    KimiK3KDAAttention,
    tensor_parallel_kda_attention,
)
from max.tensor import Tensor, compilation, default_device
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
VERIFY_WINDOW = 8
"""The verify ring's length: one anchor row plus K3's seven drafts."""
RING_ROWS = 2 * BATCH + 3
"""Ring rows the trace binds: more than the batch, so no row is ambiguous."""
RING_STRIDE = 3 * HEAD_DIM
"""The ring record's `decay | key | update_factor` stride, unpadded."""


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


def fused_qkvg_rows(tp_size: int) -> int:
    """The global row count of the fused input GEMM's weight.

    Every rank's block holds its q/k/v/g/beta rows and the whole replicated
    gate down projection.
    """
    return 4 * HEADS * HEAD_DIM + HEADS + tp_size * HEAD_DIM


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
    q/k/v/g/beta projections, so the layer fuses its rows into their weight
    and one GEMM produces both -- a whole decode launch per layer per step
    less. Pinned at the production per-device shape: each rank's fused weight
    is its own ``q_r | k_r | v_r | g_r | b_r`` block plus the WHOLE
    replicated gate row (``f_a_proj`` is not head-sharded, because every
    rank's ``f_b_proj`` slice needs all ``head_dim`` of it), the GEMM's output
    is correspondingly wider, and the layer traces three matmuls per device
    -- this GEMM, ``f_b_proj`` and ``o_proj`` -- instead of four.
    """
    n_devices = 8
    mlir = _trace_layer(n_devices, "tp", shard_heads=True)
    per_rank = fused_qkvg_rows(n_devices) // n_devices
    assert per_rank == (4 * HEADS * HEAD_DIM + HEADS) // n_devices + HEAD_DIM
    assert f"!mo.tensor<[{per_rank}, {HIDDEN}], bf16, gpu:0>" in mlir, (
        "the per-device fused weight does not carry the gate's rows"
    )
    # The one GEMM over the layer input, `head_dim` columns wider.
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


# ===----------------------------------------------------------------------=== #
# The speculative-verify arm: the gate's up projection inside the fused op
# ===----------------------------------------------------------------------=== #


def _trace_verify_layer(dtype: DType, folds_gate_up: bool) -> str:
    """Traces the layer's verify forward on one virtual gfx950 device.

    The verify arm is the one a speculative decode step runs: the layer holds
    a :class:`KdaVerifyBinding` for its ring, so ``forward`` takes
    ``_fused_verify_forward`` rather than the plain decode chain. The op's own
    eligibility conditions are met -- a 128-wide head, a 4-tap conv and the
    ``-5.0`` safe gate on gfx950 -- which is what makes the fused arm
    reachable at all.

    The ring and the weights are declared *inputs* of the staged function
    rather than tensors built inside it: the op takes the ring as a buffer,
    which the tensor runtime realizes before the graph is finished, so a ring
    built here would have to execute a device op on hardware that is not
    attached. Weights ride the same way for the same reason, and neither
    changes what is being read off the graph -- which ops it holds.

    ``folds_gate_up`` false gives ``f_b_proj`` a dtype the fused input GEMM's
    weight cannot share, so ``_folds_gate_up`` is false: the reachable
    configuration in which the layer keeps the gate's own up-projection GEMM,
    and the arm the assertions below must be able to reject.
    """
    inner = HEADS * HEAD_DIM
    with _virtual_gfx950(1):
        mesh = DeviceMesh(
            devices=(Accelerator(0),), mesh_shape=(1,), axis_names=("ep",)
        )
        replicated = NamedMapping(mesh, (None,))
        up_dtype = dtype
        if not folds_gate_up:
            up_dtype = (
                DType.bfloat16 if dtype != DType.bfloat16 else DType.float32
            )
        # The layer's weights, by attribute, as the checkpoint stores them.
        module_weights = {
            "q_proj": TensorLayout(dtype, [inner, HIDDEN], replicated),
            "k_proj": TensorLayout(dtype, [inner, HIDDEN], replicated),
            "v_proj": TensorLayout(dtype, [inner, HIDDEN], replicated),
            "g_proj": TensorLayout(dtype, [inner, HIDDEN], replicated),
            "b_proj": TensorLayout(dtype, [HEADS, HIDDEN], replicated),
            "f_a_proj": TensorLayout(dtype, [HEAD_DIM, HIDDEN], replicated),
            "f_b_proj": TensorLayout(up_dtype, [inner, HEAD_DIM], replicated),
            "o_proj": TensorLayout(dtype, [HIDDEN, inner], replicated),
            "o_norm": TensorLayout(DType.float32, [HEAD_DIM], replicated),
        }
        conv = TensorLayout(DType.float32, [inner, 1, CONV_KERNEL], replicated)
        tensors = {
            "q_conv1d": conv,
            "k_conv1d": conv,
            "v_conv1d": conv,
            "a_log": TensorLayout(DType.float32, [HEADS], replicated),
            "dt_bias": TensorLayout(DType.float32, [inner], replicated),
        }

        def build_graph(
            x: Tensor,
            conv_pool: Tensor,
            conv_row_id: Tensor,
            state_pool: Tensor,
            recurrent_row_id: Tensor,
            input_row_offsets: Tensor,
            ring_pool: Tensor,
            ring_row_id: Tensor,
            weights: dict[str, Tensor],
            layer_tensors: dict[str, Tensor],
        ) -> Tensor:
            # The layer's own initializers draw from the graph's seed.
            ops.random.set_seed(0)
            layer = KimiK3KDAAttention(
                hidden_size=HIDDEN,
                num_heads=HEADS,
                head_dim=HEAD_DIM,
                conv_kernel_size=CONV_KERNEL,
                gate_lower_bound=-5.0,
            )
            for name, weight in weights.items():
                getattr(layer, name).weight = weight
            for name, tensor in layer_tensors.items():
                setattr(layer, name, tensor)
            layer.verify_binding = KdaVerifyBinding(
                verify_width=verify_width_operand(VERIFY_WINDOW - 1),
                ring_pool=ring_pool,
                ring_row_id=ring_row_id,
                captures=[[]],
            )
            return layer(
                x,
                conv_pool,
                conv_row_id,
                state_pool,
                recurrent_row_id,
                input_row_offsets,
            )

        with default_device(mesh):
            staged = compilation.stage(build_graph)(
                TensorLayout(dtype, ["total_seq_len", HIDDEN], replicated),
                BufferLayout(
                    DType.float32,
                    [SLOTS, 3 * inner, CONV_KERNEL - 1],
                    replicated,
                ),
                TensorLayout(DType.uint32, [BATCH], replicated),
                BufferLayout(
                    DType.float32,
                    [SLOTS, HEADS, HEAD_DIM, HEAD_DIM],
                    replicated,
                ),
                TensorLayout(DType.uint32, [BATCH], replicated),
                TensorLayout(DType.uint32, [BATCH + 1], replicated),
                BufferLayout(
                    DType.float32,
                    [RING_ROWS, HEADS, VERIFY_WINDOW, RING_STRIDE],
                    replicated,
                ),
                TensorLayout(DType.uint32, [BATCH], replicated),
                module_weights,
                tensors,
            )
        return str(staged)


GATE_UP_OP = '"kda_verify_ring_conv_packed_gate_up"'
"""The fused op that runs the gate's up projection itself."""

PACKED_OP = '"kda_verify_ring_conv_packed"'
"""The same fused chain without it; quotes keep it out of ``GATE_UP_OP``."""


def test_verify_folds_the_gate_up_projection() -> None:
    """A verify's graph runs the gate's up projection inside the fused op.

    The raw gate is two projections: the fused input GEMM emits the down one
    at the tail of the packed row, and the op reduces it against
    ``f_b_proj``'s weight in the recurrence launch. So the graph holds that
    one op and two matmuls per device -- the fused qkvg GEMM and ``o_proj`` --
    where the unfolded chain holds the plain verify op and three: the up
    projection's own GEMM comes back, and so does the copy that made its
    non-contiguous input contiguous (a ``mogg_foreach`` pass at run time).
    """
    for dtype in (DType.bfloat16, DType.float32):
        mlir = _trace_verify_layer(dtype, folds_gate_up=True)
        n_gate_up = mlir.count(GATE_UP_OP)
        assert n_gate_up == 1, (
            f"the {dtype} verify graph must run"
            " kda_verify_ring_conv_packed_gate_up once, got"
            f" {n_gate_up}"
        )
        assert PACKED_OP not in mlir, (
            "the folded verify must not trace the up-projection-free op too"
        )
        n_matmul = mlir.count("rmo.matmul")
        assert n_matmul == 2, (
            f"the {dtype} verify graph must trace 2 matmuls (the fused qkvg"
            f" GEMM and o_proj), got {n_matmul}"
        )


def test_verify_keeps_the_up_projection_gemm_without_the_fold() -> None:
    """RED control: with ``_folds_gate_up`` false the GEMM and the plain op
    come back, so the assertions above can fail."""
    mlir = _trace_verify_layer(DType.bfloat16, folds_gate_up=False)
    assert GATE_UP_OP not in mlir
    assert mlir.count(PACKED_OP) == 1, (
        "the unfolded verify must trace kda_verify_ring_conv_packed"
    )
    n_matmul = mlir.count("rmo.matmul")
    assert n_matmul == 3, (
        "the unfolded verify must trace 3 matmuls (fused qkvg, f_b, o_proj),"
        f" got {n_matmul}"
    )
