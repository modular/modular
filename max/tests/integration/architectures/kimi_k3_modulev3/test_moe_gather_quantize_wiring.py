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

The kernels' numerics are covered on the GPU by
``max/kernels/test/gpu/nn/test_moe_gather_quantize.mojo`` (byte identity
against the gather+quantize chain) and ``test_moe_exit_act_concat.mojo``
(the exit activation). Per repo policy this target builds graphs without
compiling them, so it checks what the Python side controls: the wrappers'
output types and fail-fast checks, and which ops ``KimiK3LatentMoE``
stages on its AMD W4A8 path versus every other path -- the
sort+gather+quantize op replaces ``moe_create_indices`` plus ``F.gather``
plus the MXFP8 quantize there, and the exit writes the shared activation
into the concatenation's own destination, and only there. Which of the
quantize op's arms runs is decided inside the op at execution (the
serving graph's token count is dynamic), so the graph stages the one node
at every batch size.

The layer is traced on virtual gfx950 devices, no GPU attached, so the
AMD path resolves without one; virtual devices must be configured before
the process opens its first real device, which is why this is its own
target.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

import pytest
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
from max.pipelines.architectures.kimi_k3_modulev3.layers.mlp import (
    KimiK3DenseMLP,
)
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
    _moe_exit_act_concat,
    tensor_parallel_latent_moe,
)
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh, NamedMapping, TensorLayout

GPU = DeviceRef.GPU()
FUSED_OP = "mo.moe.gather.quantize.dynamic.block.scaled"
SORT_FUSED_OP = "mo.moe.sort.gather.quantize.dynamic.block.scaled"
EXIT_ACT_OP = "moe.exit.act.concat"

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
            latent,
            topk_ids,
            topk=16,
            num_experts=num_experts,
            cols=K - 32,
            cast_cols=32,
        )
        assert len(outputs) == 8
        (
            order,
            start,
            restore,
            ids,
            stats,
            quantized,
            scales,
            window,
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
        assert int(quantized.shape[1]) == K - 32
        assert scales.dtype == DType.float8_e8m0fnu
        assert scales.shape[0] == topk_ids.shape[0]
        assert int(scales.shape[1]) == (K - 32) // 32
        assert window.dtype == DType.bfloat16
        assert window.shape[0] == latent.shape[0]
        assert int(window.shape[1]) == 32


def test_exit_act_concat_output_type() -> None:
    """The exit activation is `[T, latent + half]`, in its inputs' dtype."""
    latent_cols = 448
    half_cols = 768
    tokens = 32
    for linear_beta in (1.5, None):
        with Graph(
            "moe_exit_act_concat",
            input_types=[
                TensorType(DType.bfloat16, [tokens, latent_cols], device=GPU),
                TensorType(DType.bfloat16, [tokens, 2 * half_cols], device=GPU),
            ],
        ) as graph:
            latent, gate_up = (i.tensor for i in graph.inputs)
            act = _moe_exit_act_concat(
                latent, gate_up, beta=1.5, linear_beta=linear_beta
            )
            assert act.dtype == DType.bfloat16
            assert tuple(int(d) for d in act.shape) == (
                tokens,
                latent_cols + half_cols,
            )
            graph.output(act)


def test_exit_act_concat_rejects_bad_inputs() -> None:
    """Mismatched dtypes and widths fail fast rather than mis-launching."""
    latent_cols = 448
    half_cols = 768
    tokens = 32
    bf16_latent = TensorType(DType.bfloat16, [tokens, latent_cols], device=GPU)
    bf16_gate_up = TensorType(
        DType.bfloat16, [tokens, 2 * half_cols], device=GPU
    )

    def build(latent_type: TensorType, gate_up_type: TensorType) -> None:
        with Graph(
            "moe_exit_act_concat_bad",
            input_types=[latent_type, gate_up_type],
        ) as graph:
            latent, gate_up = (i.tensor for i in graph.inputs)
            _moe_exit_act_concat(latent, gate_up, beta=1.5, linear_beta=None)

    cases: list[tuple[TensorType, TensorType, str]] = [
        (
            bf16_latent,
            TensorType(DType.float32, [tokens, 2 * half_cols], device=GPU),
            "mixed dtypes",
        ),
        (
            TensorType(DType.bfloat16, [tokens, "latent"], device=GPU),
            bf16_gate_up,
            "dynamic latent width",
        ),
        (
            bf16_latent,
            TensorType(DType.bfloat16, [tokens, 2 * half_cols + 1], device=GPU),
            "odd gate/up width",
        ),
    ]
    for latent_type, gate_up_type, bad in cases:
        try:
            build(latent_type, gate_up_type)
        except ValueError:
            continue
        raise AssertionError(f"exit activation accepted {bad}")


def test_sort_wrapper_rejects_bad_inputs() -> None:
    """Every unsupported combination fails fast rather than mis-launching."""

    good_input = TensorType(DType.bfloat16, [8, K + 32], device=GPU)
    good_ids = TensorType(DType.int32, [8 * 16], device=GPU)

    def build(
        input_type: TensorType = good_input,
        ids_type: TensorType = good_ids,
        *,
        topk: int = 16,
        num_experts: int = 896,
        cols: int | None = None,
        cast_cols: int = 32,
    ) -> None:
        with Graph(
            "fused_moe_sort_gather_quantize_bad",
            input_types=[input_type, ids_type],
        ) as graph:
            latent, topk_ids = (i.tensor for i in graph.inputs)
            moe_sort_gather_quantize_dynamic_block_scaled(
                latent,
                topk_ids,
                topk=topk,
                num_experts=num_experts,
                cols=cols,
                cast_cols=cast_cols,
            )

    rank3 = TensorType(DType.bfloat16, [2, 4, K + 32], device=GPU)
    fp16 = TensorType(DType.float16, [8, K + 32], device=GPU)
    uint32_ids = TensorType(DType.uint32, [128], device=GPU)
    rank2_ids = TensorType(DType.int32, [128, 1], device=GPU)
    dynamic_cols = TensorType(DType.bfloat16, [8, "k"], device=GPU)
    misaligned = TensorType(DType.bfloat16, [8, K - 4 + 32], device=GPU)
    wrong_slots = TensorType(DType.int32, [100], device=GPU)

    cases: list[tuple[Callable[[], None], str]] = [
        (lambda: build(input_type=rank3), "rank 3 input"),
        (lambda: build(input_type=fp16), "non-bfloat16/float32 input"),
        (lambda: build(ids_type=uint32_ids), "uint32 topk_ids"),
        (lambda: build(ids_type=rank2_ids), "rank 2 topk_ids"),
        (
            lambda: build(input_type=dynamic_cols),
            "dynamic column count",
        ),
        (lambda: build(input_type=misaligned), "K not a multiple of 32"),
        (lambda: build(cols=K + 32), "cols past the input's width"),
        (lambda: build(cast_cols=12), "cast_cols not a multiple of 8"),
        (lambda: build(cast_cols=64), "cast window past the input's width"),
        (lambda: build(cols=K - 4), "cols not a multiple of 32"),
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


def test_amd_w4a8_reads_the_merged_fp32_latent() -> None:
    """The AMD path feeds the sort op the merged entry GEMM's fp32 result.

    The op rounds each value through bfloat16 itself, so the separate
    ``.cast(DType.bfloat16)`` launch the latent slice needed is gone: its
    operand is the GEMM's own fp32 output, wide enough to hold the shared
    expert and router columns behind the latent.
    """
    tp_size = 2
    ir = _trace_moe(amd_w4a8=True, tp_size=tp_size)
    op_lines = [line for line in ir.splitlines() if SORT_FUSED_OP in line]
    assert op_lines, f"{SORT_FUSED_OP} not staged"
    # The operand's declared type, not just "some f32": the merged entry at
    # TP2 is the latent's leading columns followed by this rank's shared
    # gate/up and the whole router. Passing the rounded latent slice, or the
    # wrong width, would still leave an f32 tensor in the line.
    # Per device: the latent, this rank's shared gate/up, the router.
    merged_cols = LATENT + 2 * MOE_INTER + NUM_EXPERTS
    assert f"!mo.tensor<[seq, {merged_cols}], f32, gpu:0>" in op_lines[0], (
        "the sort op does not read the whole merged entry row: " + op_lines[0]
    )
    # And the width it quantizes is the latent's, not the merged row's.
    assert f"NUM_COLS = {LATENT} " in op_lines[0], (
        "the sort op quantizes the wrong column count: " + op_lines[0]
    )
    # No cast or slice may feed it: the merged tensor is the operand.
    assert ".cast" not in op_lines[0] and "slice" not in op_lines[0], (
        "a cast or slice still feeds the sort op: " + op_lines[0]
    )


def test_amd_w4a8_tp_exit_calls_the_fused_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The AMD TP exit goes through the fused exit collective's functional.

    ``allreduce_rms_norm_exit_concat`` reduces, norms, slices and activates
    in one launch, so the exit stages the fused op and neither the
    standalone activation concat nor a separate norm. The spy checks the
    layer called it once, on the bfloat16 latent that moe_finalize writes;
    the kernel reads the norm's bfloat16 weight as is.
    """
    partial_dtypes: list[DType] = []
    fused = F.allreduce_rms_norm_exit_concat

    def spy(
        t: Tensor,
        gate_up: Tensor,
        gamma: Tensor,
        *,
        epsilon: float,
        beta: float,
        linear_beta: float,
        mesh_axis: int | str | Sequence[int | str] = 0,
    ) -> Tensor:
        partial_dtypes.append(t.dtype)
        return fused(
            t,
            gate_up,
            gamma,
            epsilon=epsilon,
            beta=beta,
            linear_beta=linear_beta,
            mesh_axis=mesh_axis,
        )

    monkeypatch.setattr(F, "allreduce_rms_norm_exit_concat", spy)
    ir = _trace_moe(amd_w4a8=True, tp_size=2)
    assert partial_dtypes == [DType.bfloat16]
    assert "mo.composite.distributed.allreduce_rms_norm_exit_concat" in ir
    assert EXIT_ACT_OP not in ir
    assert "mo.composite.reduce.rms_norm" not in ir


def test_amd_w4a8_tp2_trace_builds() -> None:
    """The AMD path traces at TP=2; the index outputs need placements.

    The custom op has no sharding rule, so the layer restates every
    output's placement itself. ``moe_finalize``'s sharding rule needs
    ``restore_token_order`` replicated; at TP=1 the single-device mesh
    never runs the rule, which is why this traces over two devices.
    """
    assert SORT_FUSED_OP in _trace_moe(amd_w4a8=True, tp_size=2)


def test_sharded_non_amd_exit_activates_the_narrowed_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every sharded exit but the AMD one narrows the window itself.

    The merged entry GEMM writes float32 whenever the router rides it, so
    the shared expert's gate/up comes back float32. Only the AMD W4A8
    routing-glue op rounds that window while it walks the same rows; on
    every other sharded exit -- another vendor at TP>1, or AMD without the
    MXFP4 quant config -- ``_merged_entry`` has to narrow it, as the base
    did, so the shared activation runs on the model dtype.
    """
    seen: list[DType] = []
    narrow = KimiK3DenseMLP.situ_and_mul

    def spy(self: KimiK3DenseMLP, gate_up: Tensor) -> Tensor:
        seen.append(gate_up.dtype)
        return narrow(self, gate_up)

    monkeypatch.setattr(KimiK3DenseMLP, "situ_and_mul", spy)
    ir = _trace_moe(amd_w4a8=False, tp_size=2)
    # This path activates the shared half in its own concat, not inside the
    # AMD exit collective, so the one call is the window's only reader.
    assert "allreduce_rms_norm_exit_concat" not in ir
    assert seen == [DType.bfloat16], (
        "the shared expert's activation read the merged entry GEMM's own"
        f" dtype instead of the model's: {seen}"
    )


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
    assert EXIT_ACT_OP not in ir


def test_amd_w4a8_keeps_the_combine_in_the_model_dtype() -> None:
    """The combine stays bfloat16 even though the entry GEMM writes float32.

    The merged entry GEMM writes fp32 whenever the router rides it, so the
    latent the sort op reads is fp32. The MoE's *output* must not follow it:
    ``moe_finalize`` combines in the model dtype, which keeps the exit
    identical to every other path and keeps the exit's
    ``allreduce_rms_norm`` on its fused bfloat16 kernel -- that op is
    bfloat16-only and silently falls back to two launches otherwise.
    """
    ir = _trace_moe(amd_w4a8=True, tp_size=2)
    finalize = [line for line in ir.splitlines() if "moe.finalize" in line]
    assert finalize, "the AMD path does not stage moe.finalize"
    # The result type is what follows `->`; the operands before it include
    # the router weight, which stays float32.
    result = finalize[0].split("->")[-1]
    assert "bf16" in result, (
        "the MoE combine left the model dtype: " + finalize[0]
    )
    assert "f32" not in result, (
        "the MoE combine runs in float32, which un-fuses the exit's"
        " allreduce+norm: " + finalize[0]
    )
    assert "mo.composite.distributed.allreduce_rms_norm" in ir, (
        "the exit lost its fused allreduce+norm launch"
    )


def test_amd_w4a8_narrows_the_gate_up_inside_the_routing_glue() -> None:
    """The exit collective reads the routing glue's own bf16 window.

    The merged entry GEMM writes the shared expert's gate/up as float32
    columns behind the latent; the routing glue narrows them while it walks
    the same rows. So the graph must both carry the bf16 window out of the
    glue into the exit collective and contain no separate cast of it.
    """
    tp_size = 2
    ir = _trace_moe(amd_w4a8=True, tp_size=tp_size)
    # Per rank: the shared expert's whole gate/up, at this rank's share of the
    # FFN width.
    window_cols = 2 * MOE_INTER
    bf16_window = f"!mo.tensor<[seq, {window_cols}], bf16"
    f32_window = f"!mo.tensor<[seq, {window_cols}], f32"

    op_lines = [line for line in ir.splitlines() if SORT_FUSED_OP in line]
    assert op_lines, f"{SORT_FUSED_OP} not staged"
    assert any(bf16_window in line.split("->")[-1] for line in op_lines), (
        "the routing glue does not return the gate/up window: " + op_lines[0]
    )
    # At TP2 the rank's window is wider than the latent, which is the shape a
    # per-column write bound would have truncated; this trace is the graph-build
    # check for it.
    params = {
        name: int(value)
        for name, value in re.findall(r"(\w+) = (\d+) : index", op_lines[0])
    }
    assert params["CAST_COLS"] > params["NUM_COLS"], (
        "this fixture must exercise a window wider than the latent, at TP2: "
        + op_lines[0]
    )
    exit_lines = [
        line
        for line in ir.splitlines()
        if "allreduce_rms_norm_exit_concat" in line
    ]
    assert exit_lines, "the exit collective is not staged"
    assert any(bf16_window in line for line in exit_lines), (
        "the exit collective does not read the bf16 window: " + exit_lines[0]
    )

    # A cast of the window in a launch of its own reads the f32 window and
    # writes the bf16 one. The glue reads the merged row instead, so it
    # matches neither half of this.
    def narrows_in_its_own_launch(line: str) -> bool:
        sides = line.split("->")
        if len(sides) != 2:
            return False
        operands, results = sides
        return f32_window in operands and bf16_window in results

    casts = [
        line for line in ir.splitlines() if narrows_in_its_own_launch(line)
    ]
    assert not casts, (
        "a separate cast of the gate/up window is back; the routing glue must"
        " narrow it in place: " + casts[0]
    )
