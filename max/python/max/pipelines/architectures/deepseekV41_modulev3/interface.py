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
"""The frozen DeepSeek-V4.1-Flash ModuleV3 interface: tree, inputs and KV.

Everything the sliced test fixtures and Mach build against is derived
here from the model config, with no graph and no weights: the parameter
tree (names, dtypes, global shapes, placements), the KV leaf geometry,
the forward inputs after the KV leaves, and the shared-subgraph group of
each layer. The stub graph in ``stub.py`` is built from these functions,
and the recorded ABI contract is checked against them.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from max import tree
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.legacy.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.lib import KVCacheConfig

ExpertForm = Literal["per_expert", "stacked"]

PAGE_SIZE = 128
"""Tokens per KV cache page. This interface fixes the page size at 128; any
other value is rejected in :func:`build_kv_params`."""

# The packed-cache record formats ``max.nn.kernels`` names for
# ``store_packed_cache_ragged`` (MXFP8 window records, NVFP4 compressed
# records, MXFP4 index keys). The kernel-side constants arrive with the
# packed-store kernels; the ids and row pitches are frozen here with the
# rest of the ABI.
PACKED_MXFP8 = 0
PACKED_NVFP4 = 1
PACKED_MXFP4 = 2

# DeepSeek-V4.1 stores its dense weights in 32x32 E8M0-scaled FP8 blocks and
# its experts in 16-wide NVFP4 blocks.
_FP8_BLOCK = 32
_NVFP4_BLOCK = 16

# Engram's key/value projection rows, read off the checkpoint census
# (testdata/census_patterns.tsv); the config states neither.
_ENGRAM_WKV_SHAPE = (25600, 6144)
_ENGRAM_Q_ROWS = 4
_BATCH_SYMBOL = "replica_0_batch_size"
_NUM_LAST_TOKEN_LOOKBACK = 3

# The raw-byte dtypes the checkpoint stores float8 and E8M0 tensors in.
_E4M3 = DType.float8_e4m3fn
_E8M0 = DType.float8_e8m0fnu

# DSpark draft stages when the checkpoint config names none, mirroring V2's
# ``deepseekV41.model_config._DEFAULT_DSPARK_STAGES`` so the two pipelines
# resolve the same window-leaf geometry.
_DEFAULT_DSPARK_STAGES = 3


@dataclass(frozen=True)
class V41Config:
    """The slice of the V4.1 text config the interface depends on."""

    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_head_dim: int
    engram_num_embeddings: tuple[int, ...]
    engram_head_dim: int
    num_dspark_stages: int
    hidden_size: int
    moe_intermediate_size: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    n_routed_experts: int
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int
    index_n_heads: int
    index_topk: int
    candidate_topk_blocks: int
    hc_mult: int
    engram_layer_ids: tuple[int, ...]
    has_vision: bool

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> V41Config:
        """Reads the interface fields off an HF ``config.json`` mapping."""
        text = config["text_config"]
        return cls(
            vocab_size=text["vocab_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            head_dim=text["head_dim"],
            sliding_window=text["sliding_window"],
            compress_ratios=tuple(text["compress_ratios"]),
            kv_source_layer_ids=tuple(text["kv_source_layer_ids"]),
            index_head_dim=text["index_head_dim"],
            engram_num_embeddings=tuple(text["engram_num_embeddings"]),
            engram_head_dim=text["engram_head_dim"],
            # The draft-stage count that sizes the window leaf's extra
            # rows, resolved the way V2's
            # ``deepseekV41.model_config.num_dspark_stages`` resolves it.
            # The checkpoint's tap list (``dspark_target_layer_ids``) is a
            # different quantity and only coincides with it.
            num_dspark_stages=(
                text.get("n_mtp_layers")
                or text.get("num_nextn_predict_layers")
                or _DEFAULT_DSPARK_STAGES
            ),
            hidden_size=text["hidden_size"],
            moe_intermediate_size=text["moe_intermediate_size"],
            q_lora_rank=text["q_lora_rank"],
            o_lora_rank=text["o_lora_rank"],
            o_groups=text["o_groups"],
            n_routed_experts=text["n_routed_experts"],
            index_source_layer_ids=tuple(text["index_source_layer_ids"]),
            candidate_source_layer_id=text["candidate_source_layer_id"],
            index_n_heads=text["index_n_heads"],
            index_topk=text["index_topk"],
            candidate_topk_blocks=text["candidate_topk_blocks"],
            hc_mult=text["hc_mult"],
            engram_layer_ids=tuple(text["engram_layer_ids"]),
            has_vision="vision_config" in config,
        )

    def ratio(self, layer: int) -> int:
        """Returns the compression ratio of one trunk layer.

        Args:
            layer: The trunk layer id.

        Returns:
            The layer's compression ratio; 0 means sliding window only.
        """
        return self.compress_ratios[layer]

    @property
    def kv_layers_r2(self) -> tuple[int, ...]:
        """The KV source layers that run a ratio-2 compressor."""
        return tuple(l for l in self.kv_source_layer_ids if self.ratio(l) == 2)

    @property
    def kv_layers_r1(self) -> tuple[int, ...]:
        """The KV source layers that run a ratio-1 compressor."""
        return tuple(l for l in self.kv_source_layer_ids if self.ratio(l) == 1)


@dataclass(frozen=True)
class ParamSpec:
    """One parameter of the stub tree.

    ``placement`` is ``"rep"`` (replicated over the mesh), ``"s0"`` / ``"s1"``
    (sharded along that axis over the mesh), ``"gpu:r"`` (whole on rank ``r``)
    or ``"cpu"``.
    """

    name: str
    dtype: DType
    shape: tuple[int, ...]
    placement: str


def _ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _fp8_linear(
    prefix: str, out_dim: int, in_dim: int, placement: str
) -> Iterator[ParamSpec]:
    """A dense FP8 linear: compact E8M0 32x32 scales, sharded with the weight."""
    yield ParamSpec(f"{prefix}.weight", _E4M3, (out_dim, in_dim), placement)
    yield ParamSpec(
        f"{prefix}.scale",
        _E8M0,
        (_ceildiv(out_dim, _FP8_BLOCK), _ceildiv(in_dim, _FP8_BLOCK)),
        placement,
    )


def _bf16(
    name: str, shape: tuple[int, ...], placement: str = "rep"
) -> ParamSpec:
    return ParamSpec(name, DType.bfloat16, shape, placement)


def _f32(
    name: str, shape: tuple[int, ...], placement: str = "rep"
) -> ParamSpec:
    return ParamSpec(name, DType.float32, shape, placement)


def _layer_specs(config: V41Config, layer: int) -> Iterator[ParamSpec]:
    """The trunk parameters of ``layer`` outside the routed experts."""
    h = config.hidden_size
    p = f"layers.{layer}"
    ratio = config.ratio(layer)
    is_kv = layer in config.kv_source_layer_ids
    is_ix = layer in config.index_source_layer_ids
    hc_mix = (2 + config.hc_mult) * config.hc_mult

    yield _bf16(f"{p}.attn_norm.weight", (h,))
    yield _bf16(f"{p}.ffn_norm.weight", (h,))
    for role in ("attn", "ffn"):
        yield _f32(f"{p}.hc_{role}_fn", (hc_mix, config.hc_mult * h))
        yield _f32(f"{p}.hc_{role}_base", (hc_mix,))
        yield _f32(f"{p}.hc_{role}_scale", (3,))

    a = f"{p}.attn"
    n_heads = config.num_attention_heads
    qk_dim = n_heads * config.head_dim
    yield _f32(f"{a}.attn_sink", (n_heads,), "s0")
    yield from _fp8_linear(f"{a}.wq_a", config.q_lora_rank, h, "rep")
    yield _bf16(f"{a}.q_norm.weight", (config.q_lora_rank,))
    yield from _fp8_linear(f"{a}.wq_b", qk_dim, config.q_lora_rank, "s0")
    yield from _fp8_linear(f"{a}.wkv", config.head_dim, h, "rep")
    yield _bf16(f"{a}.kv_norm.weight", (config.head_dim,))
    o_rows = config.o_groups * config.o_lora_rank
    yield from _fp8_linear(f"{a}.wo_a", o_rows, qk_dim // config.o_groups, "s0")
    yield from _fp8_linear(f"{a}.wo_b", h, o_rows, "s1")

    if ratio and is_kv:
        c = f"{a}.compressor"
        yield _bf16(f"{c}.wkv.weight", (config.head_dim, h))
        yield _bf16(f"{c}.norm.weight", (config.head_dim,))
        if ratio == 2:
            yield _bf16(f"{c}.wgate.weight", (config.head_dim, h))
    if ratio and is_ix:
        ix = f"{a}.indexer"
        yield from _fp8_linear(
            f"{ix}.wq_b",
            config.index_n_heads * config.index_head_dim,
            config.q_lora_rank,
            "s0",
        )
        yield _bf16(
            f"{ix}.weights_proj.weight", (config.index_n_heads, h), "s0"
        )
    if ratio and is_kv:
        ix = f"{a}.indexer"
        yield _bf16(f"{ix}.wk.weight", (config.index_head_dim, config.head_dim))
        yield _bf16(f"{ix}.k_norm.weight", (config.index_head_dim,))

    f = f"{p}.ffn"
    yield _bf16(f"{f}.gate.weight", (config.n_routed_experts, h))
    yield _f32(f"{f}.gate.bias", (config.n_routed_experts,))
    if config.has_vision:
        yield _f32(f"{f}.gate.bias_vl", (config.n_routed_experts,))
    ffn = config.moe_intermediate_size
    s = f"{f}.shared_experts"
    yield from _fp8_linear(f"{s}.w1", ffn, h, "s0")
    yield from _fp8_linear(f"{s}.w3", ffn, h, "s0")
    yield from _fp8_linear(f"{s}.w2", h, ffn, "s1")

    if layer in config.engram_layer_ids:
        e = f"{p}.engram"
        yield _bf16(f"{e}.q_weight", (_ENGRAM_Q_ROWS, h))
        yield _bf16(f"{e}.k_weight", (_ENGRAM_Q_ROWS, h))
        yield from _fp8_linear(f"{e}.wkv", *_ENGRAM_WKV_SHAPE, "rep")


def _expert_projections(config: V41Config) -> Iterator[tuple[str, int, int]]:
    """``(name, out_dim, in_dim)`` of the three expert projections."""
    ffn, h = config.moe_intermediate_size, config.hidden_size
    yield "w1", ffn, h
    yield "w3", ffn, h
    yield "w2", h, ffn


def _nvfp4_specs(
    prefix: str,
    out_dim: int,
    in_dim: int,
    placement: str,
    lead: tuple[int, ...] = (),
) -> Iterator[ParamSpec]:
    """The four leaves of one ``NVFP4Tensor``, optionally behind a leading axis."""
    yield ParamSpec(
        f"{prefix}.weight",
        DType.uint8,
        (*lead, out_dim, in_dim // 2),
        placement,
    )
    yield ParamSpec(
        f"{prefix}.weight_scale",
        _E4M3,
        (*lead, out_dim, in_dim // _NVFP4_BLOCK),
        placement,
    )
    for scalar in ("weight_scale_2", "input_scale"):
        yield ParamSpec(f"{prefix}.{scalar}", DType.float32, lead, placement)


def _expert_specs(
    config: V41Config, layer: int, n_devices: int, experts: ExpertForm
) -> Iterator[ParamSpec]:
    p = f"layers.{layer}.ffn.experts"
    n = config.n_routed_experts
    per = n // n_devices
    if experts == "stacked":
        for name, out_dim, in_dim in _expert_projections(config):
            yield from _nvfp4_specs(
                f"{p}.{name}.weight", out_dim, in_dim, "s0", lead=(n,)
            )
        return
    for e in range(n):
        for name, out_dim, in_dim in _expert_projections(config):
            yield from _nvfp4_specs(
                f"{p}.{e}.{name}.weight",
                out_dim,
                in_dim,
                f"gpu:{e // per}",
            )


def _checkpoint_specs(
    config: V41Config, n_devices: int, experts: ExpertForm = "per_expert"
) -> list[ParamSpec]:
    """The parameter tree at checkpoint granularity, in module-attribute order.

    Names are the checkpoint's, except the NVFP4 expert leaves (see
    :func:`to_checkpoint_name`). ``experts="stacked"`` is the fallback
    form with one stacked leaf per expert projection and layer.
    """
    if config.n_routed_experts % n_devices:
        raise ValueError(
            f"{config.n_routed_experts} experts do not split over "
            f"{n_devices} devices"
        )
    h, v = config.hidden_size, config.vocab_size
    specs = [
        _bf16("embed.weight", (v, h), "s0"),
        *(
            spec
            for layer in range(config.num_hidden_layers)
            for spec in (
                *_layer_specs(config, layer),
                *_expert_specs(config, layer, n_devices, experts),
            )
        ),
        _bf16("norm.weight", (h,)),
        _bf16("head.weight", (v, h), "s0"),
    ]
    return specs


# (first, second, stacked, sharded) module-path suffixes: two projections
# that read the same activation, so the weight adapter stacks them into one
# leaf that runs as one GEMM. ``stacked`` holds ``[first; second]`` along N
# over one device; ``sharded`` marks a pair whose leaf shards along N over
# the mesh, where the adapter interleaves the parts at the per-rank row
# block so each rank's slice is ``[first_r; second_r]``.
STACKED_PROJECTIONS = (
    (".attn.wq_a", ".attn.wkv", ".attn.wqa_wkv", False),
    (
        ".ffn.shared_experts.w1",
        ".ffn.shared_experts.w3",
        ".ffn.shared_experts.w13",
        True,
    ),
)


def _derived_global_scale_spec(layer: int) -> ParamSpec:
    """The uniform NVFP4 dispatch scale of one MoE layer's routed experts.

    A derived leaf, never a checkpoint name: the weight adapter reduces
    it at load from the layer's experts' static ``w1.input_scale`` values
    (the max), which is what the EP dispatch quantizes with. Replicated,
    so every rank's graph region reads its own copy. It follows the
    module's expert block because :class:`RoutedExperts` declares it
    right after its expert list.
    """
    return ParamSpec(
        f"layers.{layer}.ffn.global_input_scale",
        DType.float32,
        (1,),
        "rep",
    )


def _with_derived_leaves(specs: list[ParamSpec]) -> list[ParamSpec]:
    """Inserts each layer's derived dispatch scale after its expert block.

    Every trunk layer is a MoE layer, so the walk is one pass over the
    folded list: a layer's experts end where the next non-expert leaf
    begins (or the list does), which is where :class:`RoutedExperts`
    declares its ``global_input_scale`` in module-attribute order.
    """
    out: list[ParamSpec] = []
    pending: int | None = None
    for spec in specs:
        match = _EXPERT_BLOCK.match(spec.name)
        if match is not None:
            pending = int(match["layer"])
        elif pending is not None:
            out.append(_derived_global_scale_spec(pending))
            pending = None
        out.append(spec)
    if pending is not None:
        out.append(_derived_global_scale_spec(pending))
    return out


_EXPERT_BLOCK = re.compile(r"^layers\.(?P<layer>\d+)\.ffn\.experts\.")


def _fold_stacked_projections(specs: list[ParamSpec]) -> list[ParamSpec]:
    """Folds each :data:`STACKED_PROJECTIONS` pair into its stacked leaf.

    The stacked leaf takes the first part's position (module-attribute
    order), the summed leading dim, and the parts' placement, which must
    agree: a stacked pair shards as one block or not at all.
    """
    by_name = {spec.name: spec for spec in specs}
    drop: set[str] = set()
    stacked_by_first: dict[str, ParamSpec] = {}
    for first, second, stacked, _ in STACKED_PROJECTIONS:
        for leaf in ("weight", "scale"):
            for part in (
                spec for spec in specs if spec.name.endswith(f"{first}.{leaf}")
            ):
                prefix = part.name[: -len(f"{first}.{leaf}")]
                other_name = f"{prefix}{second}.{leaf}"
                other = by_name.get(other_name)
                if other is None:
                    raise ValueError(f"{part.name} has no {other_name}")
                if (
                    part.dtype != other.dtype
                    or part.placement != other.placement
                    or part.shape[1:] != other.shape[1:]
                ):
                    raise ValueError(
                        f"{part.name} and {other_name} do not stack: "
                        f"{part} vs {other}"
                    )
                stacked_by_first[part.name] = ParamSpec(
                    f"{prefix}{stacked}.{leaf}",
                    part.dtype,
                    (part.shape[0] + other.shape[0], *part.shape[1:]),
                    part.placement,
                )
                drop.add(other_name)
    return [
        stacked_by_first.get(spec.name, spec)
        for spec in specs
        if spec.name not in drop
    ]


def parameter_specs(
    config: V41Config, n_devices: int, experts: ExpertForm = "per_expert"
) -> list[ParamSpec]:
    """The parameter tree of the stub, in module-attribute order.

    Names are the checkpoint's, except the NVFP4 expert leaves (see
    :func:`to_checkpoint_name`), the stacked projections
    (:data:`STACKED_PROJECTIONS`, whose pairs share one leaf) and each MoE
    layer's derived uniform dispatch scale (see
    :func:`_derived_global_scale_spec`).
    ``experts="stacked"`` is the fallback form with one stacked leaf per
    expert projection and layer.
    """
    return _with_derived_leaves(
        _fold_stacked_projections(_checkpoint_specs(config, n_devices, experts))
    )


def unmerged_parameter_specs(
    config: V41Config, n_devices: int, experts: ExpertForm = "per_expert"
) -> list[ParamSpec]:
    """The tree at checkpoint granularity: the stacked pairs unfolded.

    The one-to-one partner of the checkpoint census, which audits every
    checkpoint tensor against a leaf it feeds; a stacked leaf is fed by a
    pair, so the audit runs before the fold.
    """
    return _checkpoint_specs(config, n_devices, experts)


_EXPERT_LEAF = re.compile(
    r"^(?P<head>.*\.experts(?:\.\d+)?\.w\d)\.weight\."
    r"(?P<leaf>weight|weight_scale|weight_scale_2|input_scale)$"
)


def to_checkpoint_name(name: str) -> str:
    """Maps a tree path back to its checkpoint tensor name.

    Only the NVFP4 expert leaves differ: they nest under the projection's
    ``weight``, so ``w1.weight.weight`` is the checkpoint's ``w1.weight``
    and ``w1.weight.weight_scale`` its ``w1.weight_scale``.
    """
    match = _EXPERT_LEAF.match(name)
    if match is None:
        return name
    return f"{match['head']}.{match['leaf']}"


class _BlockGroupConfig(Protocol):
    """The config fields :func:`block_group` reads.

    Structural and read-only, so the interface's frozen :class:`V41Config`
    and the trunk's ``DeepseekV41Config`` both feed the grouping without
    either importing the other.
    """

    @property
    def compress_ratios(self) -> Sequence[int]: ...

    @property
    def kv_source_layer_ids(self) -> Sequence[int]: ...

    @property
    def index_source_layer_ids(self) -> Sequence[int]: ...

    @property
    def candidate_source_layer_id(self) -> int: ...


def block_group(config: _BlockGroupConfig, layer: int) -> str:
    """The shared-subgraph body a layer runs: a window-only layer, a
    KV-source owner (the candidate source among them), a reindex layer
    that rescores the candidate source's keys, or a reuse layer that
    inherits a source's selection.
    """
    ratio = config.compress_ratios[layer]
    if ratio == 0:
        return "swa_only"
    if layer in config.kv_source_layer_ids:
        if layer == config.candidate_source_layer_id:
            return f"full_r{ratio}_cand"
        return f"full_r{ratio}"
    if layer in config.index_source_layer_ids:
        return f"reindex_r{ratio}"
    return f"reuse_r{ratio}"


BLOCK_GROUPS = (
    "swa_only",
    "full_r2",
    "reuse_r2",
    "full_r1_cand",
    "reindex_r1",
    "reuse_r1",
)


# Bytes of one packed row per format (``max.nn.kernels.store_packed_cache_ragged``):
# the record's raw payload of codes then scales, with no alignment slack. The
# tiling question the old padded pitches (576/288/72, whose exact LCM was
# 580.5 MiB) answered by baking slack into every row is answered by the pool
# instead: a leaf whose kernels address pages through their stride declares a
# ``stride_alignment`` and the pool pads its page strides, so the huge block
# shrinks without widening every row. The index-key and state leaves are the
# padded ones; the window and compressed leaves stay packed (their readers
# fold the flat row view, which only a packed page stride expresses).
PACKED_ROW_BYTES = {PACKED_MXFP8: 528, PACKED_NVFP4: 288, PACKED_MXFP4: 68}

# Rows of the compressor state ring. A verify step of ``k`` drafts writes
# ``k + 1`` rows and a rejection resumes on one of them, so the ring takes
# the power of two at or above ``k + 2``; the interface's recorded
# speculative depth is DSpark ``k = 5``, so eight (#105708's
# ``state_ring_rows``).
STATE_RING_ROWS = 8

# The index-key and state page strides their kernels can address: any
# multiple of 16 bytes, so the pool may pad their pages. The packed
# window and compressed leaves declare none -- their readers fold the
# flat row view, which only a packed page stride expresses.
INDEX_KEY_STRIDE_ALIGNMENT = 16
STATE_STRIDE_ALIGNMENT = 16


@dataclass(frozen=True)
class KVLeafSpec:
    """One paged-cache leaf of the V4.1 KV state.

    ``ratio`` is the compression of a zone leaf (``c2a``, ``idx_c2a``, ...),
    whose page holds ``PAGE_SIZE // ratio`` slots; window and state leaves
    keep one slot per token. A packed leaf (``packed_format`` set) stores
    one opaque ``uint8`` record per slot -- the exact bytes the CSA2
    store, attention and indexer kernels write and read -- as a
    single-head leaf of ``row_bytes`` channels.

    A ring leaf (``ring_rows`` set) is per-request scratch instead: one
    block a request, ``ring_rows`` slots of ``head_dim`` values, never
    published -- invisible to prefix hits, commits and eviction, the way
    #105708's ``state_ring_rows`` declares the compressor state.
    """

    key: str
    ratio: int
    window_size: int | None
    head_dim: int
    kv_dim: int
    """1 (K-only / MLA-shaped) or 2 (K and V planes); ``build_kv_params``
    passes ``is_mla=kv_dim == 1``, which selects the params class."""
    num_layers: int
    float32: bool = False
    packed_format: int | None = None
    stride_alignment: int | None = None
    """See :attr:`~max.nn.legacy.kv_cache.KVLeafRegion.stride_alignment`."""
    ring_rows: int | None = None
    """Rows of a ``state`` leaf's per-request ring, or ``None`` for a leaf
    paged by token."""

    @property
    def slots_per_page(self) -> int:
        if self.ring_rows is not None:
            return self.ring_rows
        return PAGE_SIZE // self.ratio if self.ratio else PAGE_SIZE

    @property
    def row_bytes(self) -> int:
        """The leaf's channels: packed bytes, or ``head_dim`` values."""
        if self.packed_format is None:
            return self.head_dim
        return PACKED_ROW_BYTES[self.packed_format]

    @property
    def dtype(self) -> DType:
        if self.packed_format is not None:
            return DType.uint8
        return DType.float32 if self.float32 else DType.bfloat16

    @property
    def scratch(self) -> bool:
        """Whether the leaf is per-request scratch, never published."""
        return self.ring_rows is not None


def kv_leaf_specs(config: V41Config) -> list[KVLeafSpec]:
    """Returns the V4.1 KV leaves in params-tree order, dropping unused
    ratios.

    Every layer owns the sliding-window leaf. Each compression ratio a KV
    source layer runs adds a compressed-record leaf, an index-key leaf
    and, at ratio 2, the compressor-state ring. The packed leaves hold
    one opaque record per slot in their frozen format -- an MXFP8 record
    per window token, an NVFP4 record per closed compression group and
    an MXFP4 index key per group (:data:`PACKED_ROW_BYTES`) -- and the
    ring keeps :data:`STATE_RING_ROWS` slots, the depth the recorded
    speculative decoding needs, so the ABI never moves when the draft
    count lands.

    Args:
        config: The interface config the leaf geometry derives from.

    Returns:
        The leaf specs, in the order :func:`build_kv_params` keys them
        into the params tree.
    """
    specs = [
        KVLeafSpec(
            "swa",
            0,
            config.sliding_window,
            config.head_dim,
            1,
            # DSpark stages share the window leaf, so adding them later does
            # not change the shape of ``kv_blocks``. The stage count sizes
            # the extra rows, the way V2's ``model_config.num_dspark_stages``
            # does -- not the tap list, which only coincides with it on the
            # published checkpoint.
            config.num_hidden_layers + config.num_dspark_stages,
            packed_format=PACKED_MXFP8,
        )
    ]
    for ratio, layers in ((2, config.kv_layers_r2), (1, config.kv_layers_r1)):
        n = len(layers)
        if not n:
            continue
        specs.append(
            KVLeafSpec(
                f"c{ratio}a",
                ratio,
                None,
                config.head_dim,
                1,
                n,
                packed_format=PACKED_NVFP4,
            )
        )
        if ratio == 2:
            # The ratio-2 compressor's open pair state is a per-request
            # scratch ring (#105708): one block a request, never published,
            # one row per ring slot holding the pair's float32 KV row then
            # its score row. The registered ``mo.csa.compress.paged`` op
            # writes and carries it, and the ring keeps ``STATE_RING_ROWS``
            # slots so the ABI never moves when the draft count lands.
            specs.append(
                KVLeafSpec(
                    "c2a_state",
                    ratio,
                    None,
                    2 * config.head_dim,
                    1,
                    n,
                    True,
                    stride_alignment=STATE_STRIDE_ALIGNMENT,
                    ring_rows=STATE_RING_ROWS,
                )
            )
        specs.append(
            KVLeafSpec(
                f"idx_c{ratio}a",
                ratio,
                None,
                config.index_head_dim,
                1,
                n,
                packed_format=PACKED_MXFP4,
                stride_alignment=INDEX_KEY_STRIDE_ALIGNMENT,
            )
        )
    return specs


def build_kv_params(
    config: V41Config,
    kv_cache_config: KVCacheConfig,
    devices: Sequence[DeviceRef],
    data_parallel_degree: int = 1,
) -> MultiKVCacheParams:
    """Builds one ``MultiKVCacheParams`` with a leaf per :func:`kv_leaf_specs`
    entry.

    Every leaf shares the token page size, so one block table serves all
    of them; per-page slots come from each spec: one per token on the
    window leaf, one per closed compression group on the compressed and
    index-key leaves, and the ring's rows on the scratch state leaf.
    Record leaves are opaque ``uint8`` rows; the state ring is ``float32``
    and scratch -- drawn once per request, never published. The packed
    rows carry their own scales, so no scale inputs exist. The pool's own
    settings -- prefix caching, quantization, connectors -- flow through
    ``kv_cache_config.to_params`` untouched.

    ``data_parallel_degree`` names each replica's row dim after its replica
    (``replica_r_batch_size``, the deepseekV3 ``DP_EP`` pattern): replicas
    carry different request counts, so their KV metadata rows must be
    distinct symbolic dims. Degree 1 keeps every device on
    ``replica_0_batch_size`` — the tensor-parallel shape, where all devices
    do hold the same rows.

    Args:
        config: The interface config the leaf geometry derives from.
        kv_cache_config: The pool settings that flow through
            ``kv_cache_config.to_params`` untouched.
        devices: The devices the cache leaves live on.
        data_parallel_degree: The number of data-parallel replicas; 1
            keeps the tensor-parallel row dim naming.

    Returns:
        The params tree holding one child per leaf spec.

    Raises:
        ValueError: If ``kv_cache_config``'s page size is not
            :data:`PAGE_SIZE`. The leaves' slot counts are fixed to
            128-token pages and the same page size also drives allocation
            outside these params, so a mismatched config is rejected rather
            than silently overridden.
    """
    if kv_cache_config.kv_cache_page_size != PAGE_SIZE:
        raise ValueError(
            "DeepSeek-V4.1 requires a KV cache page size of "
            f"{PAGE_SIZE}, got {kv_cache_config.kv_cache_page_size}. Set "
            f"--kv-cache-page-size {PAGE_SIZE}."
        )
    children: dict[str, KVCacheParams] = {}
    for spec in kv_leaf_specs(config):
        children[spec.key] = kv_cache_config.to_params(
            dtype=spec.dtype,
            n_kv_heads=1,
            head_dim=spec.row_bytes,
            num_layers=spec.num_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=spec.kv_dim == 1,
            num_q_heads=config.num_attention_heads,
            # Reserved for a kv_dim 2 (K-and-V) leaf; every V4.1 leaf is
            # K-only, so no leaf replicates heads today.
            allow_kv_head_replication=spec.kv_dim == 2,
            slots_per_page=spec.slots_per_page,
            window_size=spec.window_size,
            stride_alignment=spec.stride_alignment,
            scratch=spec.scratch,
        )
    return MultiKVCacheParams.from_params(children)


def kv_cache_config() -> KVCacheConfig:
    """The KV cache config the interface is recorded under."""
    return KVCacheConfig(kv_cache_page_size=PAGE_SIZE)


def extra_inputs(
    config: V41Config,
    device_refs: Sequence[DeviceRef],
    data_parallel_degree: int = 1,
) -> list[tuple[str, TensorType]]:
    """Returns the forward inputs after the KV leaves, in graph order.

    The four per-request fields name their rows _BATCH_SYMBOL under
    tensor parallelism -- every device holds the same rows, so the name
    unifies them with each replica's KV metadata. Under data parallelism
    the engine feeds them the GLOBAL row count (the graph broadcasts and
    slices them per rank from ``data_parallel_splits``), so they need a
    dim of their own: ``replica_0_batch_size`` here is the per-replica
    request count, and sharing it would force the global rows onto
    replica 0.

    Args:
        config: The interface config the input extents derive from.
        device_refs: The devices the graph runs on; the per-row fields
            name their rows after the first.
        data_parallel_degree: The number of data-parallel replicas; above
            1 the per-request fields take a global-row dim of their own.

    Returns:
        The ``(name, tensor type)`` pairs, in the order the forward reads
        them.
    """
    gpu0 = device_refs[0]
    cpu = DeviceRef.CPU()
    i32, u8 = DType.int32, DType.uint8
    batch_rows = (
        _BATCH_SYMBOL if data_parallel_degree == 1 else "global_batch_rows"
    )
    extras = [
        ("swa_begin", TensorType(i32, [batch_rows], gpu0)),
        ("global_write_floor", TensorType(i32, [batch_rows], gpu0)),
        ("image_token_mask", TensorType(u8, ["total_seq_len"], gpu0)),
        (
            "engram_lookback",
            TensorType(i32, [batch_rows, _NUM_LAST_TOKEN_LOOKBACK], gpu0),
        ),
        (
            "engram_lookback_dead",
            TensorType(u8, [batch_rows, _NUM_LAST_TOKEN_LOOKBACK], gpu0),
        ),
    ]
    for ordinal, rows in enumerate(config.engram_num_embeddings):
        extras.append(
            (
                f"engram_table_{ordinal}_weight",
                TensorType(
                    DType.float8_e4m3fn,
                    [rows, config.engram_head_dim],
                    cpu,
                ),
            )
        )
        extras.append(
            (
                f"engram_table_{ordinal}_scale",
                TensorType(
                    DType.float8_e8m0fnu,
                    [rows, config.engram_head_dim // 32],
                    cpu,
                ),
            )
        )
    extras.append(
        ("engram_token_map", TensorType(i32, [config.vocab_size], gpu0))
    )
    # Only the extent matters: the graph reads the decoder's row count off it.
    extras.append(("decoder_rows", TensorType(u8, ["decoder_rows"], gpu0)))
    return extras


def extra_input_names(config: V41Config) -> list[str]:
    """Returns the names of :func:`extra_inputs`, in order.

    Args:
        config: The interface config the input extents derive from.

    Returns:
        The extra input names, in graph order.
    """
    return [name for name, _ in extra_inputs(config, [DeviceRef.GPU(0)])]


def kv_input_names(kv_params: MultiKVCacheParams) -> list[str]:
    """``kv.<leaf>.<device>.<field>`` for each KV input, in graph order.

    A per-device record flattens to its non-``None`` fields in declaration
    order, which is how the names are read off it.
    """
    names = [
        f"kv.{path}.{field.name}"
        for path, record in tree.nodes(
            kv_params.get_symbolic_inputs(), KVCacheInputsPerDevice
        ).items()
        for field in dataclasses.fields(record)
        if getattr(record, field.name) is not None
    ]
    assert len(names) == len(kv_params.flattened_kv_inputs())
    return names


def input_names(
    config: V41Config, kv_params: MultiKVCacheParams, n_devices: int
) -> list[str]:
    """Names every forward input in graph order, signal buffers last."""
    names = ["tokens", "return_n_logits", "input_row_offsets"]
    names += kv_input_names(kv_params)
    names += extra_input_names(config)
    if n_devices > 1:
        names += [f"signal_buffer.{i}" for i in range(n_devices)]
    return names
