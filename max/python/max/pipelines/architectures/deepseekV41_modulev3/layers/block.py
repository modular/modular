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
"""Composes attention and MoE with V4.1's shifted mHC boundaries.

Each layer receives the previous FFN output before writeback and returns
its own output in the same deferred form. The trunk materializes the
pending writeback before Engram injection and after the final layer."""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMapping, DeviceMesh
from max.experimental.tensor import Tensor
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues

from ..layer_resolver import LayerInfo, block_group
from .attention import Attention, RaggedRows
from .compressor import Compressor
from .indexer import Indexer
from .mhc import claim_replicated, mhc_shifted_post_pre
from .moe import RoutedExperts, StackedSharedExpert
from .norm import RMSNorm
from .rope import RopeParams
from .router import Router

if TYPE_CHECKING:
    from ..model_config import DeepseekV41Config

__all__ = [
    "DecoderLayer",
    "FeedForward",
    "LayerPlan",
    "SiteInputs",
    "layer_plan",
    "leaf_values",
    "slot_constants",
]

# The routed experts and the shared expert clamp SwiGLU at the training value
# (config ``swiglu_limit``).
_SWIGLU_LIMIT = 10.0


@dataclass(frozen=True)
class LayerPlan:
    """The KV leaves one layer's body touches and its slot in each.

    ``leaves`` is in operand order. ``slots`` maps a leaf to the layer's
    leaf-local layer index: the window leaf is indexed by the trunk layer, a
    compressed leaf by the position of the nearest KV source at or before the
    layer among the sources of its ratio (the layers that own it).
    """

    leaves: tuple[str, ...]
    slots: dict[str, int]
    comp: str
    """The leaf the attention reads compressed entries from."""


def layer_plan(config: DeepseekV41Config, layer: int) -> LayerPlan:
    """The leaves and leaf-local slots of ``layer``."""
    group = block_group(config, layer)
    owners = {
        2: [
            l
            for l in config.kv_source_layer_ids
            if config.compress_ratios[l] == 2
        ],
        1: [
            l
            for l in config.kv_source_layer_ids
            if config.compress_ratios[l] == 1
        ],
    }

    def slot(ratio: int) -> int:
        return max(bisect.bisect_right(owners[ratio], layer) - 1, 0)

    if group == "swa_only":
        # A window-only layer selects nothing, but the attention call still
        # takes a compressed leaf; any leaf of the model will do.
        comp = "c2a" if owners[2] else "c1a"
        return LayerPlan(("swa", comp), {"swa": layer, comp: 0}, comp)
    if group == "full_r2":
        names: tuple[str, ...] = (
            "swa",
            "c2a",
            "c2a_state",
            "idx_c2a",
        )
        return LayerPlan(
            names, {n: layer if n == "swa" else slot(2) for n in names}, "c2a"
        )
    if group == "reuse_r2":
        return LayerPlan(("swa", "c2a"), {"swa": layer, "c2a": slot(2)}, "c2a")
    if group in ("full_r1_cand", "reindex_r1"):
        names = ("swa", "c1a", "idx_c1a")
        return LayerPlan(
            names, {n: layer if n == "swa" else slot(1) for n in names}, "c1a"
        )
    assert group == "reuse_r1", group
    return LayerPlan(("swa", "c1a"), {"swa": layer, "c1a": slot(1)}, "c1a")


def _on_each_device(
    mesh: DeviceMesh,
    fn: Callable[..., Any],
    *args: Any,
    mapping: DeviceMapping | None = None,
) -> Any:
    """Runs single-device module code once per rank, on that rank's shards.

    The router, compressor and indexer build their graph with single-device
    tensors (``x.device``, per-device constants). A ``Module`` is a pytree, so
    passing it through ``call_on_mesh`` hands ``fn`` the module with each
    parameter replaced by the rank's own shard; every rank holds the whole
    replicated operand, computes the same result, and the result is claimed
    ``Replicated``. A cache write through a shard is carried back to the
    distributed leaf by ``call_on_mesh``.

    Under data parallelism the rows differ per rank, so ``mapping`` claims
    the result on the rows' own placement instead.
    """
    if mesh.num_devices == 1:
        return fn(*args)
    out = mapping if mapping is not None else DeviceMapping.replicated(mesh)
    return F.call_on_mesh(fn, mesh, out_specs=out)(*args)


@dataclass
class SiteInputs:
    """Everything a layer body reads besides the stream; never crosses a call.

    The subgraph body rebuilds it from its operands. ``kv`` and ``slots`` are
    keyed by leaf name, and ``routed`` is the selection a reuse or reindex
    layer reads: the nearest source's top-k, or the candidate pool.
    """

    rows: RaggedRows
    swa_begin: Tensor
    global_write_floor: Tensor
    image_token_mask: Tensor
    kv: dict[str, PagedCacheValues]
    slots: dict[str, Tensor]
    routed: Tensor | None = None


class FeedForward(Module[..., Tensor]):
    """The mixture of experts: the router, the routed and the shared expert.

    The routed and shared outputs are added unreduced on each rank and the
    sum is all-reduced once in float32 (the shared expert is unscaled,
    ``y += shared_experts(x)``, as in the HF reference's MoE block), so a
    layer pays one cross-GPU barrier instead of two, and the result is
    rounded to bf16 once.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        mesh: DeviceMesh,
        *,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.mesh = mesh
        self.data_parallel = data_parallel
        h, ffn = config.hidden_size, config.moe_intermediate_size
        self.gate = Router(
            h,
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.routed_scaling_factor,
        )
        # The router builds its leaves on the module's default device (CPU);
        # the gate's matmul runs on the mesh, so replicate them there.
        replicated = DeviceMapping.replicated(mesh)
        self.gate.weight = self.gate.weight.to(replicated)
        self.gate.bias = self.gate.bias.to(replicated)
        # The trunk's shared expert holds the adapter's stacked ``w13``
        # leaf. Under data parallelism it runs whole per rank (TP=1), so
        # its weights stay unsharded and nothing reduces.
        self.shared_experts = StackedSharedExpert(
            h,
            ffn,
            mesh=mesh,
            swiglu_limit=_SWIGLU_LIMIT,
            data_parallel=data_parallel,
        )
        # Name-transparent: the experts' leaves stay ``ffn.experts.E.wN``.
        self.routed = RoutedExperts(
            h,
            ffn,
            config.n_routed_experts,
            mesh=mesh,
            swiglu_limit=_SWIGLU_LIMIT,
        )

    def forward(self, x: Tensor, image_token_mask: Tensor) -> Tensor:
        """The mixture of experts, reduced per the parallelism layout."""
        ids, weights = _on_each_device(
            self.mesh,
            lambda gate, x, mask: gate(x, mask),
            self.gate,
            x,
            image_token_mask,
            mapping=x.mapping,
        )
        routed = self.routed.partial(x, ids, weights)
        shared = self.shared_experts.partial(x)
        # Both are Partial; adding them locally keeps the generic add from
        # reducing each one before the sum.
        total = F.call_on_mesh(F.add, self.mesh, out_specs=routed.mapping)(
            routed, shared
        )
        return F.allreduce_sum(total).cast(DType.bfloat16)


def _select(
    group: str,
    compressor: Compressor | None,
    indexer: Indexer,
    x: Tensor,
    qr: Tensor,
    rows: RaggedRows,
    kv: dict[str, PagedCacheValues],
    slots: dict[str, Tensor],
    pool: Tensor | None,
    global_write_floor: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """One layer's compressor and indexer on one device (see ``_index``)."""
    if group == "reindex_r1":
        # No compressor here: the layer rescoring the candidate source's
        # blocks reads its index leaf and its compressed leaf.
        assert pool is not None
        return indexer(
            x,
            qr,
            None,
            None,
            rows,
            {"idx_source": kv["idx_c1a"], "comp": kv["c1a"]},
            {"idx_source": slots["idx_c1a"], "comp": slots["c1a"]},
            pool=pool,
        )
    assert compressor is not None
    raw = compressor.project(x)
    if group == "full_r2":
        # The ratio-2 ring save runs inside the pooling op, against this
        # layer's own ``c2a_state`` scratch leaf.
        latent, valid = compressor(
            raw, rows, kv["c2a_state"], slots["c2a_state"]
        )
        return indexer(
            x,
            qr,
            latent,
            valid,
            rows,
            {"idx": kv["idx_c2a"], "comp": kv["c2a"]},
            {"idx": slots["idx_c2a"], "comp": slots["c2a"]},
            global_write_floor=global_write_floor,
        )
    assert group == "full_r1_cand", group
    latent, valid = compressor(raw, rows)
    return indexer(
        x,
        qr,
        latent,
        valid,
        rows,
        {"idx": kv["idx_c1a"], "comp": kv["c1a"]},
        {"idx": slots["idx_c1a"], "comp": slots["c1a"]},
        global_write_floor=global_write_floor,
    )


class DecoderLayer(Module[..., Any]):
    """One trunk layer; layers of one block group share one subgraph body.

    Args:
        config: The resolved model config.
        info: This layer's resolved facts.
        mesh: The tensor-parallel mesh.
        rope: The checkpoint-wide RoPE settings.
        max_position: The context bound; sizes the indexer's RoPE table
            and entry cap.
        candidate_block_size: The candidate pool's block size in entries.
        engram: The layer's Engram module on ``engram_layer_ids``.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        info: LayerInfo,
        mesh: DeviceMesh,
        *,
        rope: RopeParams,
        max_position: int,
        candidate_block_size: int,
        engram: Module[..., Any] | None = None,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        layer = info.layer_id
        self.layer_id = layer
        self.group = block_group(config, layer)
        self.plan = layer_plan(config, layer)
        self.hc_mult = config.hc_mult
        # The mHC gate RMS shares the checkpoint's ``rms_norm_eps`` with
        # the norms; ``mhc_shifted_post_pre`` would otherwise default it.
        self.norm_eps = config.rms_norm_eps
        self.mesh = mesh
        self.data_parallel = data_parallel
        h = config.hidden_size
        mix = (2 + config.hc_mult) * config.hc_mult

        self.attn_norm = RMSNorm(h, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(h, eps=config.rms_norm_eps)
        # The dynamically-assigned mHC coefficient leaves (setattr below
        # follows the checkpoint's attribute names).
        self.hc_attn_fn: Tensor
        self.hc_attn_base: Tensor
        self.hc_attn_scale: Tensor
        self.hc_ffn_fn: Tensor
        self.hc_ffn_base: Tensor
        self.hc_ffn_scale: Tensor
        for role in ("attn", "ffn"):
            setattr(
                self,
                f"hc_{role}_fn",
                Tensor.zeros(
                    [mix, config.hc_mult * h], dtype=DType.float32, device=mesh
                ),
            )
            setattr(
                self,
                f"hc_{role}_base",
                Tensor.zeros([mix], dtype=DType.float32, device=mesh),
            )
            setattr(
                self,
                f"hc_{role}_scale",
                Tensor.zeros([3], dtype=DType.float32, device=mesh),
            )

        ratio = config.compress_ratios[layer]
        compressor: Compressor | None = None
        indexer: Indexer | None = None
        if ratio and layer in config.kv_source_layer_ids:
            compressor = Compressor(config, ratio, mesh)
        if ratio and layer in config.index_source_layer_ids:
            indexer = Indexer(
                config,
                mesh,
                self.group,
                layer_info=info,
                rope=rope,
                max_position=max_position,
                candidate_block_size=candidate_block_size,
            )
        self.attn = Attention(
            config,
            info,
            mesh,
            compressor=compressor,
            indexer=indexer,
            rope=rope,
            data_parallel=data_parallel,
        )
        self.ffn = FeedForward(
            config,
            mesh,
            data_parallel=data_parallel,
        )
        if engram is not None:
            self.engram = engram

    def _attention(
        self, x: Tensor, s: SiteInputs
    ) -> tuple[Tensor, list[Tensor]]:
        """Selects compressed entries where this layer does, then attends."""
        attn = self.attn
        group = self.group
        comp_indices: Tensor | None = None
        produced: list[Tensor] = []
        plan = self.plan
        projected: tuple[Tensor, Tensor] | None = None
        if group in ("full_r2", "full_r1_cand", "reindex_r1"):
            # One stacked projection for the whole site: the q-lora slice
            # feeds the indexer's qr here, and the exact pair passes to
            # ``attn`` below, so the layer traces exactly one project
            # call -- the attention's own q path never re-projects.
            projected = attn.project(x)
            qr = attn.q_norm(projected[0])
            comp_indices, pool = self._index(x, qr, s)
            produced = [comp_indices]
            if pool is not None:
                produced.append(pool)
        elif group in ("reuse_r2", "reuse_r1"):
            comp_indices = s.routed
        y = attn(
            x,
            (s.kv["swa"], s.kv[plan.comp]),
            s.rows,
            (s.slots["swa"], s.slots[plan.comp]),
            s.swa_begin,
            comp_indices,
            projected=projected,
        )
        return y, produced

    def _index(
        self, x: Tensor, qr: Tensor, s: SiteInputs
    ) -> tuple[Tensor, Tensor | None]:
        """Compressor and indexer: store the new entries, pick the top-k."""
        attn = self.attn
        kv, slots = s.kv, s.slots
        # The modules a group does not own are ``None`` (not tensors, so
        # they pass through ``call_on_mesh`` unchanged).
        compressor = getattr(attn, "compressor", None)
        return _on_each_device(
            self.mesh,
            _select,
            self.group,
            compressor,
            attn.indexer,
            x,
            qr,
            s.rows,
            kv,
            slots,
            s.routed,
            s.global_write_floor,
            mapping=x.mapping,
        )

    def forward(
        self,
        stream: Tensor,
        pre_mix: Tensor,
        y: Tensor,
        post: Tensor,
        comb: Tensor,
        rows: RaggedRows,
        swa_begin: Tensor,
        global_write_floor: Tensor,
        image_token_mask: Tensor,
        kv: list[PagedCacheValues],
        slots: list[Tensor],
        routed: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[Tensor]]:
        """One layer: the subgraph body its block group shares.

        ``stream`` is the stream *before* the previous layer's feed-forward
        write-back, which ``y`` (that output), ``post`` and ``comb`` (its
        row-major ``[rows, hc_mult**2]`` gates) describe;
        :func:`~.mhc.mhc_identity_writeback` stands in where the stream is
        already written back. ``kv`` and ``slots`` follow
        ``layer_plan(...).leaves``; every index is a graph operand, never a
        constant baked into the shared body. ``routed`` holds the selection
        a reuse or reindex layer reads and is empty elsewhere.

        Returns:
            ``(stream, pre, y, post, comb, produced)``: the stream before
            this layer's feed-forward write-back, the feed-forward's
            ``pre`` (the next layer's carried ``pre_mix``), the
            feed-forward output and its ``post`` and ``comb``, and what a
            source layer hands down -- its selection, plus the candidate
            pool at the candidate source.
        """
        names = self.plan.leaves
        s = SiteInputs(
            rows,
            swa_begin,
            global_write_floor,
            image_token_mask,
            dict(zip(names, kv, strict=True)),
            dict(zip(names, slots, strict=True)),
            routed[0] if routed else None,
        )
        # Write the previous layer's feed-forward output back onto the
        # stream and start the attention site, collapsing with the
        # carried ``pre_mix``.
        stream, post, comb, pre, x = mhc_shifted_post_pre(
            stream,
            y,
            post,
            comb.reshape([stream.shape[0], self.hc_mult, self.hc_mult]),
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            pre_mix=pre_mix,
            norm_eps=self.norm_eps,
        )
        post = claim_replicated(post, stream)
        comb = claim_replicated(comb, stream)
        pre = claim_replicated(pre, stream)
        y, produced = self._attention(self.attn_norm(x), s)
        # Write the attention output back and start the feed-forward
        # site, collapsing with the attention site's ``pre``.
        stream, post, comb, pre, x = mhc_shifted_post_pre(
            stream,
            y,
            post,
            comb.reshape([stream.shape[0], self.hc_mult, self.hc_mult]),
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            pre_mix=pre,
            norm_eps=self.norm_eps,
        )
        post = claim_replicated(post, stream)
        comb = claim_replicated(comb, stream)
        pre = claim_replicated(pre, stream)
        y = self.ffn(self.ffn_norm(x), s.image_token_mask)
        return stream, pre, y, post, comb, produced


def slot_constants(plan: LayerPlan) -> list[Tensor]:
    """The layer's leaf slots as host scalars, one graph value per leaf."""
    return [
        F.constant(plan.slots[name], DType.uint32, device=CPU())
        for name in plan.leaves
    ]


def leaf_values(
    plan: LayerPlan, leaves: dict[str, PagedCacheValues]
) -> list[PagedCacheValues]:
    """The layer's leaves in operand order."""
    return [leaves[name] for name in plan.leaves]
