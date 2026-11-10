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
"""Wiring tests for the fused MoE routing-glue ops.

The kernels' byte-identity against the gather+quantize chain is covered by
``max/kernels/test/gpu/nn/test_moe_gather_quantize.mojo`` on the GPU. Per
repo policy this target builds graphs without compiling them, so it checks
what the Python side controls: the wrappers' output types and fail-fast
checks, and which ops ``KimiK3LatentMoE`` stages on its AMD W4A8 path
versus every other path -- the sort+gather+quantize op replaces
``moe_create_indices`` plus ``F.gather`` plus the MXFP8 quantize there,
and only there. Which of the op's arms runs is decided inside the op at
execution (the serving graph's token count is dynamic), so the graph
stages the one node at every batch size.

The layer is traced on virtual gfx950 devices, no GPU attached, so the
AMD path resolves without one; virtual devices must be configured before
the process opens its first real device, which is why this is its own
target.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from max import functional as F
from max.driver import (
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import (
    moe_gather_quantize_dynamic_block_scaled,
    moe_sort_gather_quantize_dynamic_block_scaled,
)
from max.nn.legacy.quant_config import QuantConfig
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
    tensor_parallel_latent_moe,
)
from max.tensor import default_device
from max.tensor.sharding import DeviceMesh, NamedMapping, TensorLayout

GPU = DeviceRef.GPU()
FUSED_OP = "mo.moe.gather.quantize.dynamic.block.scaled"
SORT_FUSED_OP = "mo.moe.sort.gather.quantize.dynamic.block.scaled"

# Kernel-valid layer sizes; the latent and FFN widths are multiples of
# the CDNA4 GEMM's 128-element K granule.
HIDDEN = 512
LATENT = 128
MOE_INTER = 128
NUM_EXPERTS = 64
TOPK = 4
K = 3584


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


def test_wrapper_output_types() -> None:
    """The wrapper emits `[P, K]` fp8 plus `[P, K // 32]` E8M0 scales."""
    with Graph(
        "fused_moe_gather_quantize",
        input_types=[
            TensorType(DType.bfloat16, ["m", K], device=GPU),
            TensorType(DType.uint32, ["p"], device=GPU),
        ],
    ) as graph:
        latent, order = (i.tensor for i in graph.inputs)
        quantized, scales = moe_gather_quantize_dynamic_block_scaled(
            latent, order, topk=16
        )
        assert quantized.dtype == DType.float8_e4m3fn
        assert scales.dtype == DType.float8_e8m0fnu
        assert quantized.shape[0] == order.shape[0]
        assert int(quantized.shape[1]) == K
        assert scales.shape[0] == order.shape[0]
        assert int(scales.shape[1]) == K // 32


def test_sort_wrapper_output_types() -> None:
    """The sort-fused wrapper emits the create-indices contract plus fp8."""
    num_experts = 896
    with Graph(
        "fused_moe_sort_gather_quantize",
        input_types=[
            TensorType(DType.bfloat16, ["m", K], device=GPU),
            TensorType(DType.int32, ["p"], device=GPU),
        ],
    ) as graph:
        latent, topk_ids = (i.tensor for i in graph.inputs)
        outputs = moe_sort_gather_quantize_dynamic_block_scaled(
            latent, topk_ids, topk=16, num_experts=num_experts
        )
        assert len(outputs) == 7
        (
            order,
            start,
            restore,
            ids,
            stats,
            quantized,
            scales,
        ) = outputs
        assert order.dtype == DType.uint32
        assert order.shape[0] == topk_ids.shape[0]
        assert start.dtype == DType.uint32
        assert int(start.shape[0]) == num_experts + 1
        assert restore.dtype == DType.uint32
        assert restore.shape[0] == topk_ids.shape[0]
        assert ids.dtype == DType.int32
        assert int(ids.shape[0]) == num_experts
        assert stats.dtype == DType.uint32
        assert int(stats.shape[0]) == 2
        assert quantized.dtype == DType.float8_e4m3fn
        assert quantized.shape[0] == topk_ids.shape[0]
        assert int(quantized.shape[1]) == K
        assert scales.dtype == DType.float8_e8m0fnu
        assert scales.shape[0] == topk_ids.shape[0]
        assert int(scales.shape[1]) == K // 32


def test_sort_wrapper_rejects_bad_inputs() -> None:
    """Every unsupported combination fails fast rather than mis-launching."""

    good_input = TensorType(DType.bfloat16, [8, K], device=GPU)
    good_ids = TensorType(DType.int32, [8 * 16], device=GPU)

    def build(
        input_type: TensorType = good_input,
        ids_type: TensorType = good_ids,
        *,
        topk: int = 16,
        num_experts: int = 896,
    ) -> None:
        with Graph(
            "fused_moe_sort_gather_quantize_bad",
            input_types=[input_type, ids_type],
        ) as graph:
            latent, topk_ids = (i.tensor for i in graph.inputs)
            moe_sort_gather_quantize_dynamic_block_scaled(
                latent, topk_ids, topk=topk, num_experts=num_experts
            )

    rank3 = TensorType(DType.bfloat16, [2, 4, K], device=GPU)
    fp32 = TensorType(DType.float32, [8, K], device=GPU)
    uint32_ids = TensorType(DType.uint32, [128], device=GPU)
    rank2_ids = TensorType(DType.int32, [128, 1], device=GPU)
    dynamic_cols = TensorType(DType.bfloat16, [8, "k"], device=GPU)
    misaligned = TensorType(DType.bfloat16, [8, K - 4], device=GPU)
    wrong_slots = TensorType(DType.int32, [100], device=GPU)

    cases: list[tuple[Callable[[], None], str]] = [
        (lambda: build(input_type=rank3), "rank 3 input"),
        (lambda: build(input_type=fp32), "non-bfloat16 input"),
        (lambda: build(ids_type=uint32_ids), "uint32 topk_ids"),
        (lambda: build(ids_type=rank2_ids), "rank 2 topk_ids"),
        (
            lambda: build(input_type=dynamic_cols),
            "dynamic column count",
        ),
        (lambda: build(input_type=misaligned), "K not a multiple of 32"),
        (lambda: build(ids_type=wrong_slots), "slot count != tokens * topk"),
        (
            lambda: build(input_type=good_input, num_experts=0),
            "zero experts",
        ),
    ]
    for attempt, bad in cases:
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError(f"sort wrapper accepted {bad}")


def test_wrapper_rejects_bad_inputs() -> None:
    """Every unsupported combination fails fast rather than mis-launching."""

    good_input = TensorType(DType.bfloat16, [8, K], device=GPU)
    good_order = TensorType(DType.uint32, [8 * 16], device=GPU)

    def build(
        input_type: TensorType,
        order_type: TensorType = good_order,
        *,
        topk: int = 16,
    ) -> None:
        with Graph(
            "fused_moe_gather_quantize_bad",
            input_types=[input_type, order_type],
        ) as graph:
            latent, order = (i.tensor for i in graph.inputs)
            moe_gather_quantize_dynamic_block_scaled(latent, order, topk=topk)

    rank3 = TensorType(DType.bfloat16, [2, 4, K], device=GPU)
    fp32 = TensorType(DType.float32, [8, K], device=GPU)

    cases: list[tuple[Callable[[], None], str]] = [
        (lambda: build(input_type=rank3), "rank 3 input"),
        (lambda: build(input_type=fp32), "non-bfloat16 input"),
        (
            lambda: build(
                input_type=good_input,
                order_type=TensorType(DType.int32, [128], device=GPU),
            ),
            "int32 order",
        ),
        (
            lambda: build(
                input_type=good_input,
                order_type=TensorType(DType.uint32, [128, 1], device=GPU),
            ),
            "rank 2 order",
        ),
        (
            lambda: build(
                input_type=TensorType(DType.bfloat16, [8, K - 4], device=GPU),
                order_type=good_order,
            ),
            "K not a multiple of 32",
        ),
        (
            lambda: build(
                input_type=good_input,
                order_type=TensorType(DType.uint32, [100], device=GPU),
            ),
            "slot count != tokens * topk",
        ),
    ]
    for attempt, bad in cases:
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError(f"wrapper accepted {bad}")


def _trace_moe(amd_w4a8: bool, tp_size: int = 1) -> str:
    """Traces the latent MoE layer over ``tp_size`` virtual gfx950 devices."""
    # TP splits the FFN width, so each rank's share must still be a
    # multiple of the CDNA4 GEMM's 128-element K granule.
    moe_inter = MOE_INTER * tp_size
    with _virtual_gfx950(tp_size):
        mesh = DeviceMesh(
            devices=tuple(Accelerator(i) for i in range(tp_size)),
            mesh_shape=(tp_size,),
            axis_names=("tp",),
        )
        with F.lazy(), default_device(mesh):
            layer = KimiK3LatentMoE(
                hidden_size=HIDDEN,
                latent_size=LATENT,
                moe_intermediate_size=moe_inter,
                shared_intermediate_size=moe_inter,
                num_experts=NUM_EXPERTS,
                num_experts_per_token=TOPK,
                beta=1.5,
                linear_beta=1.5,
                rms_norm_eps=1e-5,
                routed_scaling_factor=1.0,
                num_expert_group=1,
                topk_group=1,
                moe_renormalize=True,
                quant_config=(
                    QuantConfig.mxfp4(mlp_quantized_layers=[0])
                    if amd_w4a8
                    else None
                ),
                amd_w4a8=amd_w4a8,
            )
            layer = tensor_parallel_latent_moe(layer, mesh)
            staged = layer.trace(
                TensorLayout(
                    DType.bfloat16,
                    ["seq", HIDDEN],
                    NamedMapping(mesh, (None, None)),
                )
            )
        return str(staged)


def test_amd_w4a8_path_stages_the_fused_op() -> None:
    """The AMD path stages the sort-fused op and drops what it fused in."""
    ir = _trace_moe(amd_w4a8=True)
    assert SORT_FUSED_OP in ir
    # The op registers as "moe.create.indices" (no "mo." prefix), so the
    # literal must match the registration or the assert never fires.
    assert "moe.create.indices" not in ir, (
        "the AMD path must not also run the index build it fused in"
    )
    assert "mo.gather" not in ir, (
        "the AMD path must not also run the fan-out gather it fused in"
    )
    assert FUSED_OP not in ir, (
        "the AMD path must not also run the gather+quantize op; the"
        " sort-fused op covers that work in both its arms"
    )


def test_amd_w4a8_tp2_trace_builds() -> None:
    """The AMD path traces at TP=2; the index outputs need placements.

    The custom op has no sharding rule, so the layer restates every
    output's placement itself. ``moe_finalize``'s sharding rule needs
    ``restore_token_order`` replicated; at TP=1 the single-device mesh
    never runs the rule, which is why this traces over two devices.
    """
    assert SORT_FUSED_OP in _trace_moe(amd_w4a8=True, tp_size=2)


def test_dense_path_keeps_the_unfused_chain() -> None:
    """Every other path keeps the unfused chain and never stages a fused op.

    Main folds the fan-out gather into the ragged GEMM's A load, so the
    dense path stages the index build and the ragged GEMM, not a gather op.
    """
    ir = _trace_moe(amd_w4a8=False)
    assert "moe.create.indices" in ir
    assert "matmul.grouped.ragged" in ir
    assert FUSED_OP not in ir
    assert SORT_FUSED_OP not in ir
