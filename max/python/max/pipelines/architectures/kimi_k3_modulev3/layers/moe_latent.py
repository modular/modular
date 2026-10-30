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
"""Kimi K3's latent sparse MoE block.

It runs BF16, or AMD CDNA4 MXFP4 W4A8 under tensor parallelism; expert
parallelism runs BF16 only.
Mirrors ``kimi_k3/layers/moe_latent.py``'s ``KimiK3LatentMoE``.
:func:`expert_parallel_latent_moe` rejects an MXFP4 layer: expert
parallelism's quantized dispatch needs its own transport kernels, a
separate piece of work.

The real production checkpoint ships its routed experts MXFP4-only, so
the W4A8 arm below is what actually runs the real model on this hardware
(AMD CDNA4 / MI355X). See ``lossy_mxfp4_activations_available`` for why
the alternatives (BF16 dequant, W4A4) aren't built here instead.

Its AMD W4A8 validation (``validate_amd_tp_w4a8``, ``validate_amd_packed_k``)
is copied from ``kimi_k3.layers.moe_latent`` rather than imported, so this
port has no build dependency on V2.

``tensor_parallel_latent_moe`` shards the per-expert FFN width across the
``tp`` mesh axis. Unlike the other sharded layers, this block returns a
complete value: ``routed_expert_norm`` cannot read a partial sum, so
``forward()`` reduces the routed branch before the norm and the shared
branch before adding it.

``expert_parallel_latent_moe`` gives each device whole experts instead,
exchanging tokens through :mod:`ep` (mirroring ``minimax_m3_modulev3/ep.py``).
It dispatches the LATENT activations, not the hidden state, since that is
what the experts actually consume. The block needs no allreduce under
expert parallelism: the combine already returns final, correctly-ordered,
router-weighted output.

``max.nn.common_layers.moe.QuantizedMoE`` cannot be wrapped
whole for this: its
``forward(x)`` calls ``self.gate(x)`` on the SAME ``x`` it dispatches to
experts, so there is no seam to insert K3's entry projection (which must run
BEFORE the gate sees anything, at the full ``hidden_size``, while experts run
at the narrower ``routed_expert_hidden_size`` latent) without also starving
the gate of the width it needs. This block instead calls the same shared
primitives ``QuantizedMoE`` itself calls (``moe_create_indices``,
``grouped_matmul_ragged``, ``moe_finalize``) directly, the same way K3's V2
``KimiK3LatentMoE`` does against ``max.graph.ops`` -- reusing the kernel-level
infrastructure without reusing the monolithic orchestration class.

Three non-obvious things this mirrors exactly from the V2 implementation and
the official reference (``modeling_kimi_linear.py:776-832``):

* ``routed_expert_down_proj`` [latent, hidden] is the ENTRY (hidden -> latent)
  and ``routed_expert_up_proj`` [hidden, latent] is the EXIT (latent ->
  hidden) -- inverted from what the names suggest.
* ``routed_expert_norm`` sits BETWEEN the experts and the exit, normalizing
  the weighted-combined LATENT, not the expert input.
* The per-token routing-weight multiply happens IN THE LATENT, via
  :func:`moe_finalize` on the latent-width expert outputs, before the norm
  and before the exit projection -- not after.

``shared_experts`` runs on the original identity at the full hidden width,
never in the latent, and is added last.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import Linear, Module
from max.nn.common_layers.functional_kernels import (
    grouped_matmul_ragged,
    moe_create_indices,
    moe_finalize,
)
from max.nn.common_layers.mesh_axis import EP, TP
from max.nn.legacy.comm.ep import EPConfig
from max.nn.legacy.kernels import (
    grouped_dynamic_block_scaled_matmul_amd,
    quantize_dynamic_block_scaled,
)
from max.nn.legacy.quant_config import QuantConfig
from max.nn.norm import RMSNorm
from max.support.math import ceildiv
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Partial,
)

from . import ep
from .amd_w4a8_validation import validate_amd_packed_k, validate_amd_tp_w4a8
from .mlp import KimiK3DenseMLP, kimi_situ_and_mul, tensor_parallel_dense_mlp
from .moe_router import make_kimi_k3_moe_router
from .tensor_parallel import check_divisible, shard_interleaved_blocks

# Scale-factor block width for MXFP4/MXFP8: one E8M0 scale per this many
# elements along the contraction axis. Matches V2's ``_MXFP8_GROUP``.
_SF_VECTOR_SIZE = 32

# Per-device K-granule the AMD block-scaled GEMM tiles at; matches V2's
# ``_AMD_K_GRANULE``. ``validate_amd_packed_k`` checks against this.
_AMD_K_GRANULE = 128

# Fallback per-expert token cap; serving passes
# ``KimiK3Config.routed_expert_row_bound`` instead (see ``model_config.py``).
_DEFAULT_MAX_TOKENS_PER_EXPERT = 8192


def _amd_moe_usage_stats(
    expert_ids: Tensor, latent: Tensor, max_tokens_per_expert: int
) -> TensorValue:
    """The host ``[max_tokens_per_expert, num_active_experts]`` pair.

    The AMD grouped GEMM reads it.
    Direct port of V2's ``_host_usage_stats``: ``moe_create_indices`` leaves
    its own usage stats on device, while the AMD kernel reads element 0 and 1
    of a HOST ``uint32`` pair, so the bound is computed host-side here. A
    bound too small for this step's real token count silently drops an
    expert's tail rows rather than erroring -- see that V2 docstring for the
    real incident (98% of one kernel's launched workgroups were dead grid
    rows from an unbounded per-expert cap, 47% of decode kernel time).

    ``rows`` is the minimum of the static per-expert budget and this step's
    real token count, since no expert can hold more rows than the batch has
    tokens (a token's ``top_k`` picks are distinct).
    """
    rows = ops.min(
        ops.shape_to_tensor(TensorValue(latent).shape)[0],
        ops.constant(
            max_tokens_per_expert, dtype=DType.int64, device=DeviceRef.CPU()
        ),
    )
    return ops.stack(
        [
            ops.cast(rows, DType.uint32),
            ops.constant(
                int(TensorValue(expert_ids).shape[0]),
                dtype=DType.uint32,
                device=DeviceRef.CPU(),
            ),
        ]
    )


amd_moe_usage_stats = F.functional(_amd_moe_usage_stats)


def _quantize_mxfp8(activations: Tensor) -> tuple[TensorValue, TensorValue]:
    """BF16 activations -> (E4M3 values, E8M0 scales), row-major.

    MXFP8 is purely block scaled (no per-tensor factor), so ``tensor_sf``
    stays the identity; matches V2's ``_amd_w4a8_matmul``'s quantize call.
    """
    return quantize_dynamic_block_scaled(
        TensorValue(activations),
        tensor_sf=1.0,
        sf_vector_size=_SF_VECTOR_SIZE,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )


quantize_mxfp8 = F.functional(_quantize_mxfp8)


def _amd_w4a8_grouped_matmul(
    activations: Tensor,
    weight: Tensor,
    a_scales: Tensor,
    b_scales: Tensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    usage_stats: Tensor,
    *,
    out_dtype: DType,
) -> TensorValue:
    """E4M3 activations against packed-MXFP4 weights, row-major (no preshuffle).

    Direct port of V2's ``_amd_w4a8_matmul`` call, without its
    ``preshuffled_b`` option -- an unmeasured optimization this first
    MXFP4 commit doesn't add yet.
    """
    return grouped_dynamic_block_scaled_matmul_amd(
        TensorValue(activations),
        TensorValue(weight),
        TensorValue(a_scales),
        TensorValue(b_scales),
        TensorValue(expert_start_indices),
        TensorValue(expert_ids),
        TensorValue(usage_stats),
        out_type=out_dtype,
        preshuffled_b=False,
    )


amd_w4a8_grouped_matmul = F.functional(_amd_w4a8_grouped_matmul)


class KimiK3LatentMoE(Module[..., Tensor]):
    """One ``KimiSparseMoeBlock`` whose routed experts live in a latent.

    Single-device, tensor-parallel after :func:`tensor_parallel_latent_moe`,
    or expert-parallel after :func:`expert_parallel_latent_moe` shards it.
    Expert-parallel runs through :meth:`expert_parallel_forward` instead of
    :meth:`forward`, since its exchange needs an ``ep_config``/``ep_buffers``
    pair ``forward``'s plain-Tensor signature has no room for.
    """

    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int,
        beta: float,
        linear_beta: float | None,
        rms_norm_eps: float,
        routed_scaling_factor: float,
        num_expert_group: int,
        topk_group: int,
        moe_renormalize: bool,
        dtype: DType = DType.bfloat16,
        quant_config: QuantConfig | None = None,
        amd_w4a8: bool | None = None,
        max_tokens_per_expert: int = _DEFAULT_MAX_TOKENS_PER_EXPERT,
    ) -> None:
        """Builds the latent MoE block.

        Args:
            hidden_size: ``config.hidden_size`` -- the block's in/out width and
            the width the shared experts and the gate run at.
            latent_size: ``config.routed_expert_hidden_size``; the routed
            experts' width.
            moe_intermediate_size: ``config.moe_intermediate_size``; a routed
            expert's FFN width.
            shared_intermediate_size: ``config.moe_intermediate_size *
            config.num_shared_experts``.
            num_experts: ``config.num_experts``.
            num_experts_per_token: ``config.num_experts_per_token``.
            beta: ``config.activation_situ_beta``.
            linear_beta: ``config.activation_situ_linear_beta``.
            rms_norm_eps: ``config.rms_norm_eps`` for ``routed_expert_norm``.
            routed_scaling_factor: ``config.routed_scaling_factor``.
            num_expert_group: ``config.num_expert_group``.
            topk_group: ``config.topk_group``.
            moe_renormalize: ``config.moe_renormalize``.
            dtype: Dense (non-expert) weight and activation dtype; the packed
            routed experts only reach this width once dequantized.
            quant_config: When MXFP4, the routed experts are packed uint8 with
            E8M0 block scales; see the module docstring for the compute
            arm this selects.
            amd_w4a8: Whether to run the AMD CDNA4 W4A8 path. ``None`` derives
            it from the host accelerator via :func:`validate_amd_tp_w4a8`;
            pass a value to build the graph on a host without one (e.g. a
            shape-only CPU test).
            max_tokens_per_expert: Rows the W4A8 grouped GEMM sizes its M grid
            for, per expert -- pass ``KimiK3Config.routed_expert_row_bound``.
            Too small silently drops an expert's tail.
        """
        super().__init__()
        self.latent_size = latent_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.beta = beta
        self.linear_beta = linear_beta
        self.moe_intermediate_size = moe_intermediate_size
        self.dtype = dtype
        self.max_tokens_per_expert = max_tokens_per_expert
        self.mxfp4 = quant_config is not None and quant_config.is_mxfp4
        self.amd_tp_w4a8 = validate_amd_tp_w4a8(
            mxfp4=self.mxfp4,
            expert_parallel=False,
            dtype=dtype,
            requested=amd_w4a8,
        )

        self.gate = make_kimi_k3_moe_router(
            hidden_dim=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            routed_scaling_factor=routed_scaling_factor,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            moe_renormalize=moe_renormalize,
        )

        # ENTRY: hidden -> latent, despite the name (`Linear` stores
        # [out, in], so loading the exit weight here raises rather than
        # silently working).
        self.routed_expert_down_proj = Linear(
            in_dim=hidden_size, out_dim=latent_size, bias=False
        )
        # EXIT: latent -> hidden.
        self.routed_expert_up_proj = Linear(
            in_dim=latent_size, out_dim=hidden_size, bias=False
        )
        self.routed_expert_norm = RMSNorm(latent_size, eps=rms_norm_eps)

        if self.mxfp4:
            # Packed along the INPUT (K) dim, two 4-bit values per byte, one
            # E8M0 scale per 32 input elements. The output (N) dim is
            # untouched, so the gate/up fusion is still a plain row
            # concatenation -- matches V2's `experts_gate_up_proj` layout.
            self.experts_gate_up_proj = Tensor.zeros(
                [num_experts, 2 * moe_intermediate_size, latent_size // 2],
                dtype=DType.uint8,
            )
            self.experts_gate_up_proj_scale = Tensor.zeros(
                [
                    num_experts,
                    2 * moe_intermediate_size,
                    ceildiv(latent_size, _SF_VECTOR_SIZE),
                ],
                dtype=DType.float8_e8m0fnu,
            )
            self.experts_down_proj = Tensor.zeros(
                [num_experts, latent_size, moe_intermediate_size // 2],
                dtype=DType.uint8,
            )
            self.experts_down_proj_scale = Tensor.zeros(
                [
                    num_experts,
                    latent_size,
                    ceildiv(moe_intermediate_size, _SF_VECTOR_SIZE),
                ],
                dtype=DType.float8_e8m0fnu,
            )
        else:
            # Stacked on a leading expert axis for `grouped_matmul_ragged`.
            self.experts_gate_up_proj = Tensor.zeros(
                [num_experts, 2 * moe_intermediate_size, latent_size]
            )
            self.experts_down_proj = Tensor.zeros(
                [num_experts, latent_size, moe_intermediate_size]
            )

        self.shared_experts = KimiK3DenseMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_intermediate_size,
            beta=beta,
            linear_beta=linear_beta,
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        ep_config: EPConfig | None = None,
        ep_buffers: ep.EPBuffers | None = None,
    ) -> Tensor:
        """Runs the block over ``hidden_states``.

        Args:
            hidden_states: ``[T, hidden_size]``.
            ep_config: This block's expert-parallel exchange setup. Given
                with ``ep_buffers``, it routes to
                :meth:`expert_parallel_forward` instead of running this
                method's own body.
            ep_buffers: The exchange's graph buffers.

        Returns:
            ``[T, hidden_size]``.
        """
        if ep_config is not None:
            assert ep_buffers is not None
            return self.expert_parallel_forward(
                hidden_states, ep_config, ep_buffers
            )

        topk_idx, topk_weight = self.gate(hidden_states)

        latent = self.routed_expert_down_proj(hidden_states)

        flat_idx = F.cast(F.reshape(topk_idx, [-1]), DType.int32)
        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(flat_idx, self.num_experts)

        permuted = F.gather(
            latent,
            F.cast(
                token_expert_order // self.num_experts_per_token,
                DType.int32,
            ),
            axis=0,
        )
        if self.amd_tp_w4a8:
            mesh = permuted.mesh

            # Custom ops have no sharding rule, so every output's placement
            # is restated immediately after the call -- same convention as
            # `kda_attention.py`'s `kda_decode`/`kimi_short_conv_qkv` calls.
            def replicated(t: Tensor) -> Tensor:
                return t.rebind_mapping(NamedMapping(mesh, (None,) * t.rank))

            def ffn_sharded(t: Tensor) -> Tensor:
                return t.rebind_mapping(NamedMapping(mesh, (None, TP)))

            # The GEMM reads its usage bound on the HOST; `moe_create_indices`
            # leaves its own on device. See `_amd_moe_usage_stats`. Lives on
            # the CPU, off the GPU mesh entirely -- not rebound, like the
            # dense path's own `expert_start_indices`/`expert_ids`.
            usage_stats = amd_moe_usage_stats(
                expert_ids, latent, self.max_tokens_per_expert
            )
            quantized, act_scales = quantize_mxfp8(permuted)
            quantized, act_scales = (
                replicated(quantized),
                replicated(act_scales),
            )
            gate_up = ffn_sharded(
                amd_w4a8_grouped_matmul(
                    quantized,
                    self.experts_gate_up_proj,
                    act_scales,
                    self.experts_gate_up_proj_scale,
                    expert_start_indices,
                    expert_ids,
                    usage_stats,
                    out_dtype=self.dtype,
                )
            )
            activated = kimi_situ_and_mul(
                gate_up,
                self.beta,
                self.linear_beta,
                self.moe_intermediate_size,
            )
            # SiTU hands back BF16, so the down projection re-quantizes
            # rather than chaining a quantized intermediate. The FFN-width
            # axis is still TP-sharded here (SiTU is elementwise), so the
            # quantized activations and their scales carry the same split.
            quantized, act_scales = quantize_mxfp8(activated)
            quantized, act_scales = (
                ffn_sharded(quantized),
                ffn_sharded(act_scales),
            )
            expert_out = amd_w4a8_grouped_matmul(
                quantized,
                self.experts_down_proj,
                act_scales,
                self.experts_down_proj_scale,
                expert_start_indices,
                expert_ids,
                usage_stats,
                out_dtype=self.dtype,
            )
            # `experts_down_proj` is split on its K (FFN) axis, so this is a
            # PARTIAL sum over the latent axis, same as the dense row-parallel
            # case -- `moe_finalize` below needs that stated explicitly.
            expert_out = expert_out.rebind_mapping(
                DeviceMapping(mesh, (Partial(),))
            )
        else:
            gate_up = grouped_matmul_ragged(
                permuted,
                self.experts_gate_up_proj,
                expert_start_indices,
                expert_ids,
                expert_usage_stats,
            )
            activated = kimi_situ_and_mul(
                gate_up,
                self.beta,
                self.linear_beta,
                self.moe_intermediate_size,
            )
            expert_out = grouped_matmul_ragged(
                activated,
                self.experts_down_proj,
                expert_start_indices,
                expert_ids,
                expert_usage_stats,
            )

        # The weight multiply and the unpermute both happen here, IN THE
        # LATENT, before the norm and before the exit projection.
        combined = moe_finalize(
            expert_out, restore_token_order, topk_weight, latent.dtype
        )
        mesh = self.experts_gate_up_proj.mesh
        sharded = TP in mesh.axis_names and mesh.axis_size(TP) > 1
        if sharded:
            # The norm rides the allreduce in one collective launch. The
            # attributes mirror `routed_expert_norm` exactly:
            # `RMSNorm.forward` is `ops.rms_norm` with its defaults
            # (weight_offset 0.0, multiply_before_cast False).
            normed = F.allreduce_rms_norm(
                combined,
                self.routed_expert_norm.weight,
                self.routed_expert_norm.eps,
                weight_offset=0.0,
                multiply_before_cast=False,
                mesh_axis=TP,
            )
        else:
            normed = self.routed_expert_norm(combined)
        routed = self.routed_expert_up_proj(normed)

        # The shared branch reads the ORIGINAL full-width input, not the
        # latent, and is added last.
        shared = self.shared_experts(hidden_states)
        if sharded:
            shared = F.allreduce_sum(shared, mesh_axis=TP)
        return routed + shared

    def expert_parallel_forward(
        self,
        hidden_states: Tensor,
        ep_config: EPConfig,
        ep_buffers: ep.EPBuffers,
    ) -> Tensor:
        """The expert-parallel forward; see :func:`expert_parallel_latent_moe`.

        Args:
            hidden_states: ``[T, hidden_size]``, replicated across the mesh.
            ep_config: Sized for this block's latent width -- see
                :func:`expert_parallel_latent_moe`.
            ep_buffers: This block's own dispatch/combine exchange buffers.

        Returns:
            ``[T, hidden_size]``, replicated, placed as ``hidden_states`` was.
        """
        topk_idx, topk_weight = self.gate(hidden_states)
        latent = self.routed_expert_down_proj(hidden_states)

        received, src_info = ep.dispatch(
            ep_config, ep_buffers, latent, F.cast(topk_idx, DType.int32)
        )
        usage = ep.grouped_matmul_bounds(ep_config)
        gate_up = grouped_matmul_ragged(
            received.tokens,
            self.experts_gate_up_proj,
            received.start,
            received.ids,
            usage,
        )
        activated = kimi_situ_and_mul(
            gate_up, self.beta, self.linear_beta, self.moe_intermediate_size
        )
        expert_out = grouped_matmul_ragged(
            activated,
            self.experts_down_proj,
            received.start,
            received.ids,
            usage,
        )

        # The combine already restores token order and applies
        # `topk_weight`, so unlike `forward`, there is no `moe_finalize`
        # here -- the result is the complete, weighted-combined latent.
        combined = ep.combine(
            ep_config, ep_buffers, expert_out, src_info, latent, topk_weight
        )
        normed = self.routed_expert_norm(combined)
        routed = self.routed_expert_up_proj(normed)
        return routed + self.shared_experts(hidden_states)


def tensor_parallel_latent_moe(
    layer: KimiK3LatentMoE, mesh: DeviceMesh
) -> KimiK3LatentMoE:
    """Shards a latent MoE block's FFN widths along the ``tp`` mesh axis.

    The router and the latent entry, exit and norm stay replicated: they
    work at the hidden or latent width, not the FFN width.
    """
    tp_size = mesh.axis_size(TP)
    if tp_size == 1:
        return layer
    check_divisible("MoE expert width", layer.moe_intermediate_size, tp_size)
    check_divisible(
        "Shared expert width",
        layer.shared_experts.intermediate_size,
        tp_size,
    )
    if layer.mxfp4:
        # The down weight is packed two values per byte along the expert FFN
        # axis and carries one E8M0 scale per 32 of those values, so a shard
        # boundary that is not a multiple of 32 would split a scale block and
        # silently rescale the wrong elements -- same check V2 runs before
        # sharding, via the same shared helper.
        validate_amd_packed_k(
            latent_size=layer.latent_size,
            per_device_ffn=layer.moe_intermediate_size // tp_size,
            opt_in_env="MAX_KIMI_K3_AMD_W4A8",
        )
    layer.experts_gate_up_proj = shard_interleaved_blocks(
        layer.experts_gate_up_proj, mesh, num_blocks=2, axis=1
    )
    layer.experts_down_proj = layer.experts_down_proj.to(
        NamedMapping(mesh, (None, None, TP))
    )
    if layer.mxfp4:
        # Scales carry one entry per 32 INPUT elements, so they split on the
        # same axis as the weight they scale.
        layer.experts_gate_up_proj_scale = shard_interleaved_blocks(
            layer.experts_gate_up_proj_scale, mesh, num_blocks=2, axis=1
        )
        layer.experts_down_proj_scale = layer.experts_down_proj_scale.to(
            NamedMapping(mesh, (None, None, TP))
        )
    tensor_parallel_dense_mlp(layer.shared_experts, mesh)
    return layer


def expert_parallel_latent_moe(
    layer: KimiK3LatentMoE, mesh: DeviceMesh
) -> KimiK3LatentMoE:
    """Gives each device whole experts, across ``mesh``'s ``ep`` axis.

    Only the experts shard, on the expert axis. Everything else -- entry,
    exit, norm, shared experts -- stays replicated, since the exchange is
    this block's only communication under expert parallelism. Named by
    axis, not position, since ``mesh`` may also carry a ``tp`` axis of
    size 1.
    """
    ep_size = mesh.axis_size(EP)
    if ep_size == 1:
        return layer
    if layer.mxfp4:
        raise NotImplementedError(
            "Kimi K3 MoE: expert parallelism does not support MXFP4 routed"
            " experts yet"
        )
    if layer.num_experts % ep_size != 0:
        raise ValueError(
            f"Kimi K3 MoE: {layer.num_experts} experts must divide by the"
            f" expert-parallel degree {ep_size}"
        )
    row_mapping = NamedMapping(mesh, (EP, None, None))
    layer.experts_gate_up_proj = layer.experts_gate_up_proj.to(row_mapping)
    layer.experts_down_proj = layer.experts_down_proj.to(row_mapping)
    return layer


def expert_parallel_config(
    layer: KimiK3LatentMoE,
    mesh: DeviceMesh,
    *,
    max_tokens_per_rank: int,
) -> EPConfig:
    """The ``EPConfig`` this block's exchange needs.

    ``hidden_size`` is the LATENT width, not the model's -- see the module
    docstring for why.
    """
    return EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=layer.latent_size,
        top_k=layer.num_experts_per_token,
        n_experts=layer.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=mesh.axis_size(EP),
        n_nodes=1,
    )
