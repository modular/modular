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
"""Mistral Large 4 config (ModuleV3), forking the DeepSeek-V3.2 DSA spine.

Every per-layer decision arrives here as an explicit list read from the
generated ``config.json`` (see :mod:`.model_config`), so nothing downstream
re-derives a rope/NoPE or indexer-owner rule with a modulo. On top of the
DeepSeek-V3.2 fields this adds:

* ``rope_layers`` -- 54 rope layers, 18 NoPE.
* ``kv_source_layers`` -- 9 layers (39..71) read layer 35's latent instead of
  writing a cache row of their own; 63 layers own theirs.
* A KV tree with one leaf per storage group rather than DeepSeek's single MLA
  leaf: a 576-wide rope latent, a 512-wide NoPE latent, and the indexer's FP8
  keys with their FP32 per-token scales.
* The quantization mix this checkpoint needs at once: FP8 block for ``wq``,
  ``wo``, ``indexer.wq_b``, the shared experts and layer 0; NVFP4 for the
  routed experts; BF16 for the MLA projections, the router and the indexer's
  ``wk`` and ``weights_proj``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.quant_config import (
    DISABLE_FUSED_SWIGLU_NVFP4_ENV,
    QuantConfig,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.speculative.config import SpeculativeMethod
from transformers import AutoConfig
from typing_extensions import Self, override

from ..deepseekV3_2.layers.sparse_mla import sparse_selection_width
from ..deepseekV3_2_modulev3.model_config import (
    DeepseekV3_2Config,
    resolve_indexer_types,
)
from .checkpoint import LocalCheckpoint, params_json_candidates
from .model_config import (
    MistralLarge4NormalizedConfig,
    MistralLarge4Yarn,
    read_params_json,
)

ROPE_MLA_GROUP = "mla_rope"
"""Cache leaf holding the 576-wide FP8 latent of rope-layer owners."""
NOPE_MLA_GROUP = "mla_nope"
"""Cache leaf holding the 512-wide FP8 latent of NoPE-layer owners."""
INDEXER_GROUP = "indexer"
"""Cache leaf holding the indexer's FP8 keys and their FP32 scales."""

LATENT_HEAD_DIM = 576
"""A rope layer's latent row width: 512 latent dims plus a 64-wide rope key."""
NOPE_LATENT_HEAD_DIM = 512
"""A NoPE layer's latent row width: the latent alone, no rope key."""

logger = logging.getLogger("max.pipelines")


def _dspark_target_layers(text_config: Any) -> list[int]:
    """Returns the target layers an in-checkpoint drafter reads, if any.

    The generated ``config.json`` carries the list as
    ``dspark_target_activation_layers`` (a list, or the checkpoint's
    comma-separated ``params.json`` spelling); a config that has not been
    through the generator may instead carry the checkpoint's ``mtp`` block.
    An empty list means nothing captures taps.

    Args:
        text_config: The generated text config.

    Returns:
        The tap layers, ascending.

    Raises:
        ValueError: If a declared list is malformed.
    """
    declared = getattr(text_config, "dspark_target_activation_layers", None)
    if declared is None:
        mtp = getattr(text_config, "mtp", None)
        declared = (
            getattr(mtp, "target_activation_layers", None) if mtp else None
        )
    if declared is None:
        return []
    if isinstance(declared, str):
        declared = [int(part) for part in declared.split(",") if part]
    return [int(layer) for layer in declared]


def _params_target_layers(model_config: LocalCheckpoint) -> list[int]:
    """Reads ``mtp.target_activation_layers`` from the checkpoint's ``params.json``.

    The generated ``config.json`` this graph is normally built from does not
    carry the drafter's tap list -- it reproduces the checkpoint family's
    published config, which has no such key -- so the checkpoint's
    ``params.json`` is read here instead, fetched from its repo when a Hub
    serve has not downloaded it. A checkpoint with no local files, or one with
    no ``params.json``, yields no taps: a drafter-less model must still serve.

    A ``params.json`` that is there and malformed is not the same thing as a
    missing one, and is reported rather than served past.

    Args:
        model_config: The pipeline's model config, which names the checkpoint.

    Returns:
        The tap layers, or an empty list when no candidate holds a
        ``params.json`` with an ``mtp`` block.

    Raises:
        ValueError: If a ``params.json`` it finds is not JSON, or declares
            ``mtp.target_activation_layers`` that are not integers. The message
            names the file.
    """
    for path in params_json_candidates(model_config):
        if not path.is_file():
            continue
        try:
            mtp = read_params_json(path).get("mtp")
        except ValueError as error:
            raise ValueError(f"{path} is not a readable params.json") from error
        if not isinstance(mtp, dict):
            continue
        declared = mtp.get("target_activation_layers")
        if declared is None:
            continue
        if isinstance(declared, str):
            declared = [part for part in declared.split(",") if part]
        try:
            return [int(layer) for layer in declared]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path} declares mtp.target_activation_layers={declared!r},"
                " which are not layer indices"
            ) from error
    logger.info(
        "no params.json with an mtp block beside %r or its weights; this graph"
        " captures no drafter taps",
        model_config.model_path,
    )
    return []


def _served_taps(text_config: Any, model_config: LocalCheckpoint) -> list[int]:
    """The tap layers a served graph captures, or none.

    The generated ``config.json`` declares the list when it came from a
    checkpoint generator that writes it; otherwise the checkpoint's own
    ``params.json`` is read. Either way the list is all-or-nothing: a schedule
    cut shorter than the last tap (a staged or minimized graph) captures
    nothing rather than a mis-ordered subset of the drafter's columns.

    Args:
        text_config: The resolved text config.
        model_config: The pipeline's model config, for the local checkpoint
            path.

    Returns:
        The taps, ascending, or an empty list when this graph carries no
        drafter.
    """
    declared = _dspark_target_layers(text_config)
    if not declared:
        declared = _params_target_layers(model_config)
    if any(layer >= int(text_config.num_hidden_layers) for layer in declared):
        return []
    return declared


def _fitted_taps(normalized: MistralLarge4NormalizedConfig) -> list[int]:
    """The checkpoint's tap layers that exist in the schedule being built.

    Args:
        normalized: The checkpoint's normalized config.

    Returns:
        The taps, or an empty list for a schedule cut shorter than one of
        them -- such a graph carries no drafter.
    """
    if normalized.mtp is None:
        return []
    taps = list(normalized.mtp.target_layers)
    if any(layer >= normalized.n_layers for layer in taps):
        return []
    return taps


def mla_group(rope: bool) -> str:
    """Returns the MLA cache leaf a layer with this rope flag stores in.

    Args:
        rope: Whether the layer rotates its key.
    """
    return ROPE_MLA_GROUP if rope else NOPE_MLA_GROUP


def mla_head_dim(rope: bool) -> int:
    """Returns a layer's MLA latent row width.

    Args:
        rope: Whether the layer rotates its key, and so stores one.
    """
    return LATENT_HEAD_DIM if rope else NOPE_LATENT_HEAD_DIM


def kv_slot_map(
    rope_layers: list[bool], kv_source_layers: list[int]
) -> dict[str, dict[int, int]]:
    """Maps each layer to its slot in the MLA leaf its source layer owns.

    Slots are numbered in layer order within each leaf, so a reader takes its
    source's slot: the rows it attends are the rows the source wrote.

    Args:
        rope_layers: One flag per layer, true where the layer rotates its key.
        kv_source_layers: One entry per layer, the layer whose latent it reads.

    Returns:
        ``{"mla_rope": {layer: slot}, "mla_nope": {layer: slot}}``, with an
        entry for every layer that maps into that leaf.

    Raises:
        ValueError: If a layer's rope flag disagrees with its source's.
    """
    slots: dict[str, dict[int, int]] = {
        ROPE_MLA_GROUP: {},
        NOPE_MLA_GROUP: {},
    }
    next_slot = {ROPE_MLA_GROUP: 0, NOPE_MLA_GROUP: 0}
    for layer, source in enumerate(kv_source_layers):
        if rope_layers[layer] != rope_layers[source]:
            raise ValueError(
                f"layer {layer} rotates its key but reads layer {source},"
                " whose rows do not"
            )
        group = mla_group(rope_layers[source])
        if source == layer:
            slots[group][layer] = next_slot[group]
            next_slot[group] += 1
        else:
            slots[group][layer] = slots[group][source]
    return slots


def indexer_slot_map(indexer_types: list[str]) -> dict[int, int]:
    """Maps each full-indexer layer to its slot in the indexer key cache.

    Args:
        indexer_types: One ``"full"`` or ``"shared"`` per layer.

    Returns:
        The slot of every layer that owns an indexer.
    """
    slots: dict[int, int] = {}
    for layer, indexer_type in enumerate(indexer_types):
        if indexer_type == "full":
            slots[layer] = len(slots)
    return slots


def latent_cache_dtype(kv_cache_format: str | None) -> DType:
    """Returns the latent cache's dtype, the one answer both trees read.

    The framework builds the cache tree it flattens the graph's inputs from,
    and the decoder's own ops are told the config tree's leaves; the two must
    be one tree, so both resolve the dtype here rather than one deriving it
    from the quantization encoding and the other naming its own. This
    checkpoint's latent rows are FP8 in both of its releases' serving intent,
    and the row-invariant decode pins are compile-time checks that hold only
    over an FP8 latent cache, so an unset ``kv_cache_format`` means FP8. An
    operator who names another format gets that format, and the decode pins
    then switch off, which
    :func:`~.layers.transformer_block.row_invariant_decode_blocker` reports
    rather than hides.

    Args:
        kv_cache_format: The operator's ``--kv-cache-format``, or ``None``.
    """
    if kv_cache_format is None:
        return DType.float8_e4m3fn
    return cache_dtype_for_encoding(None, kv_cache_format)


def build_kv_params(
    *,
    rope_layers: list[bool],
    kv_source_layers: list[int],
    indexer_types: list[str],
    num_attention_heads: int,
    index_head_dim: int,
    index_topk: int,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
    cache_dtype: DType,
    data_parallel_degree: int = 1,
    speculative_method: SpeculativeMethod | None = None,
    num_draft_tokens: int = 0,
) -> MultiKVCacheParams:
    """Builds the cache tree: one leaf per latent group plus the indexer.

    Each leaf keys the way the model launches it: the latents carry the
    selection width and the physical-row form the sparse op passes, and the
    indexer keeps the scaled FP8 form its own cache implies. The capture key
    the runner resolves from this tree is then a property of the model and not
    of the cache the step happens to hold.

    Args:
        rope_layers: One flag per layer.
        kv_source_layers: One entry per layer, the layer whose latent it reads.
        indexer_types: One ``"full"`` or ``"shared"`` per layer.
        num_attention_heads: Query heads, for the MLA leaves' dispatch metadata.
        index_head_dim: Width of one indexer key row (128).
        index_topk: Keys an indexer keeps (2048).
        devices: Devices the cache spans.
        kv_cache_config: Page-size and budget configuration.
        cache_dtype: Dtype of the latent rows.
        data_parallel_degree: Number of attention replicas.
        speculative_method: The pipeline's speculative method, if any.
        num_draft_tokens: Draft tokens a verify step carries.

    Raises:
        ValueError: If a layer's rope flag disagrees with its KV source's.
    """
    leaves: dict[str, MLAKVCacheParams] = {}
    for group, rope in ((ROPE_MLA_GROUP, True), (NOPE_MLA_GROUP, False)):
        owners = [
            layer
            for layer, source in enumerate(kv_source_layers)
            if source == layer and mla_group(rope_layers[layer]) == group
        ]
        if not owners:
            continue
        leaf = kv_cache_config.to_params(
            dtype=cache_dtype,
            # One KV head: an MLA cache holds a single latent vector.
            n_kv_heads=1,
            head_dim=mla_head_dim(rope),
            num_layers=len(owners),
            devices=devices,
            is_mla=True,
            num_q_heads=num_attention_heads,
            data_parallel_degree=data_parallel_degree,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )
        assert isinstance(leaf, MLAKVCacheParams)
        # The sparse op passes this leaf's selection width as its index stride
        # and declares its rows physical, so the leaf keys the same way: left
        # at the dense defaults the capture key would size itself from the
        # live cache on every step while the launch took one pinned split.
        leaves[group] = replace(
            leaf,
            sparse_indices_stride=sparse_selection_width(leaf, index_topk),
            sparse_physical_indices=True,
        )

    # The indexer's keys are FP8 with one FP32 scale per token.
    indexer_leaf = kv_cache_config.to_params(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=1,
        head_dim=index_head_dim,
        # Only ``full`` layers run the indexer, so only they hold a slot.
        num_layers=sum(1 for t in indexer_types if t == "full"),
        devices=devices,
        data_parallel_degree=data_parallel_degree,
        is_mla=True,
        num_q_heads=num_attention_heads,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=index_head_dim,
        ),
        speculative_method=speculative_method,
        num_draft_tokens=num_draft_tokens,
    )
    # No kernel reads the indexer leaf's dispatch metadata, but it is part of
    # the composite capture key, so it has to key off the selection its layer
    # gathers rather than off the live cache. Only the stride is set: the leaf
    # keeps its own dispatch form, and a scaled FP8 cache keys through the
    # quantized form, which sizes the split from that stride alone. Joining it
    # onto a latent leaf instead borrows the latent's plain FP8 form, whose
    # verify record reads the cache length -- the one input the key must not
    # depend on -- and the latent's row width, which is 576 for the rope group
    # and 512 for the nope group, so which groups the checkpoint has would
    # decide the width the indexer's key assumes.
    assert isinstance(indexer_leaf, MLAKVCacheParams)
    leaves[INDEXER_GROUP] = replace(
        indexer_leaf,
        sparse_indices_stride=sparse_selection_width(indexer_leaf, index_topk),
    )
    return MultiKVCacheParams.from_params(leaves)


@dataclass(kw_only=True)
class MistralLarge4Config(DeepseekV3_2Config):
    """Configuration for Mistral Large 4 models (ModuleV3)."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float4_e2m1fnx2"
    # The spine's quant configs are built from the architecture, never from this
    # set: ``from_normalized`` hard-codes NVFP4 routed experts and FP8 block
    # attention, so a bf16 request would be accepted here and silently ignored.
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float4_e2m1fnx2"}

    kv_params: KVCacheParamInterface
    """A ``MultiKVCacheParams`` tree: one leaf per latent group and one for
    the indexer's keys."""

    rope_layers: list[bool] = field(default_factory=list)
    """One flag per layer: true where the layer rotates its query and key."""
    kv_source_layers: list[int] = field(default_factory=list)
    """One entry per layer: the layer whose latent rows it attends."""
    moe_layers: list[bool] = field(default_factory=list)
    """One flag per layer: true for a sparse (MoE) FFN."""
    lerope_alpha_dim: int = 32
    """Width of the ``lerope.alpha`` multiplier, ``qk_rope_head_dim // 2``."""
    index_norm_eps: float = 1e-6
    """Epsilon of the indexer's two LayerNorms: the checkpoint's
    ``dsa.norm_eps``, which differs from the block-wide ``rms_norm_eps``."""
    yarn: MistralLarge4Yarn | None = None
    """The YaRN geometry the rope table is built from."""
    q_lora_rank: int | None = None
    """Null: Mistral Large 4 has no query LoRA."""
    final_logit_softcapping: float = 30.0
    """``softmax_tanh_gating``: ``cap * tanh(logits / cap)``, on the head only."""
    target_layer_ids: list[int] = field(default_factory=list)
    """The target layers whose post-layer residual streams an in-checkpoint
    drafter reads, in the checkpoint's declared order. Layer ``idx`` means the
    residual the graph holds *after* layer ``idx`` -- ``hidden_states[idx +
    1]`` in the HF convention, the un-normalized tensor the next layer's
    attention reads -- which is the tensor the drafter was trained on.

    The list is all-or-nothing: a schedule cut shorter than the last tap
    captures nothing, because a partial concat would misalign the drafter's
    projection columns. An empty list therefore means nothing captures taps,
    which is the non-speculative path."""

    encodes_images: bool = False
    """Whether this graph's ABI carries the vision-merge inputs.

    The pipeline's decoder always declares them -- its batches carry image
    embeddings through the vision-encoder cache, empties on a text-only or
    decode step. A caller that builds the module outside the pipeline
    leaves this off: the fused DSpark graph traces ``verify`` and its own
    ABI has no vision slots, and a direct module compile keeps the
    text-only input list.
    """

    def __post_init__(self) -> None:
        """Validates the per-layer schedule against the rest of the config.

        Raises:
            ValueError: If a per-layer list does not cover every layer, the
                YaRN geometry is missing, or the shared-expert count is not the
                single expert the block builds.
        """
        super().__post_init__()
        for name, values in (
            ("rope_layers", self.rope_layers),
            ("kv_source_layers", self.kv_source_layers),
            ("moe_layers", self.moe_layers),
        ):
            if len(values) != self.num_hidden_layers:
                raise ValueError(
                    f"{name} has {len(values)} entries, expected"
                    f" num_hidden_layers={self.num_hidden_layers}"
                )
        # A reader attends rows the source layer wrote in this same step, so the
        # source has to own a latent: a chained reader emits none of its own.
        for layer, source in enumerate(self.kv_source_layers):
            if not 0 <= source < self.num_hidden_layers:
                raise ValueError(
                    f"kv_source_layers[{layer}]={source} is not a layer index"
                )
            if self.kv_source_layers[source] != source:
                raise ValueError(
                    f"layer {layer} reads layer {source}, which reads layer"
                    f" {self.kv_source_layers[source]} in turn; a reader"
                    " cannot also be a source"
                )
            if source > layer:
                raise ValueError(
                    f"layer {layer} shares KV with a later layer {source}"
                )
        if self.yarn is None:
            raise ValueError("Mistral Large 4 requires its YaRN parameters")
        if self.target_layer_ids:
            if list(self.target_layer_ids) != sorted(
                set(self.target_layer_ids)
            ):
                raise ValueError(
                    "target_layer_ids must be ascending and unique, got"
                    f" {self.target_layer_ids}"
                )
            for target in self.target_layer_ids:
                if not 0 <= target < self.num_hidden_layers:
                    raise ValueError(
                        f"target_layer_ids names layer {target}, which is not"
                        f" one of the {self.num_hidden_layers} layers"
                    )
        # Every MoE block builds exactly one shared expert beside the routed
        # ones; the checkpoint's mixture has no other count.
        if self.n_shared_experts != 1:
            raise ValueError(
                f"n_shared_experts={self.n_shared_experts} is unsupported;"
                " Mistral Large 4 has exactly one shared expert"
            )

    @classmethod
    def estimate_vision_cache_entry_bytes(cls, huggingface_config: Any) -> int:
        """Worst-case bytes for one vision encoder cache entry.

        The largest image the processor admits is the tower's
        ``image_size`` on a side: 1540 pixels is 110 patches of 14, which
        the 2x2 merge folds into 55x55 merged rows, and the projector
        emits one BF16 row per merged square at the text hidden width --
        55 * 55 * 6144 * 2 bytes. A checkpoint without a vision tower has
        no entry to size.

        Args:
            huggingface_config: The pipeline's HuggingFace config.

        Returns:
            Estimated bytes per vision cache entry, or 0 without a tower.
        """
        spec = cls.get_vision_cache_row_spec(huggingface_config)
        vision = getattr(huggingface_config, "vision_config", None)
        if spec is None or vision is None:
            return 0
        hidden, dtype = spec
        merge = int(getattr(huggingface_config, "spatial_merge_size", 2))
        tokens_per_side = int(vision.image_size) // (
            int(vision.patch_size) * merge
        )
        return tokens_per_side**2 * hidden * dtype.size_in_bytes

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: Any
    ) -> tuple[int, DType] | None:
        """One embedding row per merged vision token: text width, bfloat16.

        The vision tower's projector emits one BF16 row per merged patch
        square at the text model's hidden width, which is what the
        pipeline-owned vision encoder cache stores and what
        ``empty_vision_embeddings`` allocates.
        """
        text_config = cls.get_text_config(huggingface_config)
        return int(text_config.hidden_size), DType.bfloat16

    @property
    def qk_head_dim(self) -> int:
        """The full query/key head width: the nope and rope slices summed."""
        return int(self.qk_nope_head_dim) + int(self.qk_rope_head_dim)

    @property
    def attention_scale(self) -> float:
        """The explicit softmax scale, ``qk_head_dim ** -0.5`` on every layer.

        Taken from the config's widths, so a NoPE layer -- whose heads carry
        128 content dims and no rope dims -- still scales by ``1/sqrt(128)``
        rather than by the 576-wide latent geometry's ``192 ** -0.5``.
        """
        return float(self.qk_head_dim**-0.5)

    @property
    def fp8_config(self) -> QuantConfig:
        """The FP8 block config every FP8 site in this checkpoint loads with."""
        n = self.num_hidden_layers
        return QuantConfig.blockscaled_fp8(
            mlp_quantized_layers=range(n), attn_quantized_layers=range(n)
        )

    @property
    def nvfp4_config(self) -> QuantConfig:
        """The NVFP4 config the routed experts load with.

        ``shared_experts_weight_dtype`` names the shared expert's real dtype,
        FP8 block, rather than leaving it to default to the routed experts':
        that default would tell the expert-parallel path to fold the shared
        expert into the NVFP4 expert bundle, which its bytes cannot fill.

        The routed experts take the fused SwiGLU gate/up kernel, as the
        checkpoint-parsed NVFP4 configs do. That hands the graph compiler the
        gate/up and down pair it fuses into one MegaFFN launch, in place of a
        gate/up matmul, a SiLU requantize and a down matmul.
        """
        config = QuantConfig.nvfp4(
            mlp_quantized_layers=range(self.num_hidden_layers),
            attn_quantized_layers=(),
            shared_experts_weight_dtype=DType.float8_e4m3fn,
        )
        config.can_use_fused_swiglu = (
            os.environ.get(DISABLE_FUSED_SWIGLU_NVFP4_ENV) != "1"
        )
        return config

    @staticmethod
    def get_text_config(huggingface_config: Any) -> Any:
        """Normalizes any supported config form to the text config.

        The checkpoint's ``config.json`` is the multimodal
        ``MistralLarge4Config``, whose every text field sits under
        ``text_config``; a bare text config (or the mapping a caller built from
        ``params.json``) has no such attribute and is returned unchanged.
        """
        return getattr(huggingface_config, "text_config", huggingface_config)

    @override
    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """The longest sequence that fits the KV cache.

        The bound is the text config's ``max_position_embeddings``; the
        DeepSeek spine this forked reads the top-level object, which on this
        multimodal config carries no such field.
        """
        return super().calculate_max_seq_len(
            cls.get_text_config(huggingface_config), model_config
        )

    @override
    def get_kv_params(self) -> KVCacheParamInterface:  # type: ignore[override]
        return self.kv_params

    def mla_leaf(self, rope: bool) -> KVCacheParams:
        """Returns the MLA cache leaf a layer with this rope flag uses.

        Args:
            rope: Whether the layer rotates its key.
        """
        assert isinstance(self.kv_params, MultiKVCacheParams)
        leaf = self.kv_params.children[mla_group(rope)]
        assert isinstance(leaf, KVCacheParams)
        return leaf

    @property
    def indexer_kv_params(self) -> KVCacheParams:
        """The indexer key-cache leaf."""
        assert isinstance(self.kv_params, MultiKVCacheParams)
        leaf = self.kv_params.children[INDEXER_GROUP]
        assert isinstance(leaf, KVCacheParams)
        return leaf

    @staticmethod
    def construct_kv_params(  # type: ignore[override]
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Builds the cache tree from the generated ``config.json``.

        Args:
            huggingface_config: The generated ``config.json``, read for the
                explicit per-layer schedule.
            pipeline_config: The pipeline configuration.
            devices: Devices the cache spans.
            kv_cache_config: Page-size and budget configuration.
            cache_dtype: Dtype of the latent rows.
        """
        speculative_method = None
        num_draft_tokens = 0
        if pipeline_config.speculative:
            speculative_method = pipeline_config.speculative.speculative_method
            num_draft_tokens = (
                pipeline_config.speculative.num_speculative_tokens or 0
            )
        text_config = MistralLarge4Config.get_text_config(huggingface_config)
        return build_kv_params(
            rope_layers=[bool(flag) for flag in text_config.no_rope_layers],
            kv_source_layers=list(text_config.kv_source_layers),
            indexer_types=resolve_indexer_types(
                text_config, int(text_config.num_hidden_layers)
            ),
            num_attention_heads=int(text_config.num_attention_heads),
            index_head_dim=int(text_config.index_head_dim),
            index_topk=int(text_config.index_topk),
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            data_parallel_degree=(pipeline_config.model.data_parallel_degree),
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )

    @classmethod
    def from_normalized(
        cls,
        normalized: MistralLarge4NormalizedConfig,
        *,
        kv_params: KVCacheParamInterface,
        devices: list[DeviceRef],
        max_seq_len: int,
        dtype: DType = DType.float8_e4m3fn,
        graph_mode: str = "auto",
        max_batch_context_length: int = 131072,
    ) -> Self:
        """Builds the config straight from the checkpoint's ``params.json``.

        Args:
            normalized: The resolved per-layer schedule.
            kv_params: The cache tree to trace against.
            devices: Devices the model runs on.
            max_seq_len: Longest sequence the pipeline serves.
            dtype: Quantization encoding dtype.
            graph_mode: ``"prefill"``, ``"decode"`` or ``"auto"``.
            max_batch_context_length: Longest batch context the graph sizes for.
        """
        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=devices,
            max_seq_len=max_seq_len,
            vocab_size=normalized.vocab_size,
            hidden_size=normalized.hidden_size,
            intermediate_size=normalized.intermediate_size,
            moe_intermediate_size=normalized.moe_intermediate_size,
            moe_layer_freq=1,
            num_hidden_layers=normalized.n_layers,
            num_attention_heads=normalized.n_heads,
            num_key_value_heads=normalized.n_kv_heads,
            n_shared_experts=normalized.n_shared_experts,
            n_routed_experts=normalized.n_routed_experts,
            routed_scaling_factor=normalized.routed_scaling_factor,
            kv_lora_rank=normalized.kv_lora_rank,
            q_lora_rank=None,
            qk_rope_head_dim=normalized.qk_rope_head_dim,
            v_head_dim=normalized.v_head_dim,
            qk_nope_head_dim=normalized.qk_nope_head_dim,
            topk_method="",
            n_group=1,
            topk_group=1,
            num_experts_per_tok=normalized.num_experts_per_tok,
            first_k_dense_replace=normalized.first_k_dense_replace,
            norm_topk_prob=True,
            hidden_act="silu",
            max_position_embeddings=normalized.max_position_embeddings,
            rms_norm_eps=normalized.rms_norm_eps,
            tie_word_embeddings=normalized.tie_word_embeddings,
            rope_interleave=True,
            scoring_func="softmax",
            attention_bias=False,
            attention_dropout=0.0,
            graph_mode=graph_mode,
            max_batch_context_length=max_batch_context_length,
            data_parallel_degree=1,
            index_head_dim=normalized.index_head_dim,
            index_n_heads=normalized.index_n_heads,
            index_topk=normalized.index_topk,
            indexer_types=[
                "full" if owns else "shared"
                for owns in normalized.layers.owns_indexer
            ],
            index_norm_eps=normalized.index_norm_eps,
            quantization_encoding="float4_e2m1fnx2",
            final_logit_softcapping=normalized.final_logit_softcapping,
            rope_layers=list(normalized.layers.rope),
            kv_source_layers=list(normalized.layers.kv_source),
            moe_layers=list(normalized.layers.moe),
            lerope_alpha_dim=normalized.yarn.lerope_alpha_dim,
            yarn=normalized.yarn,
            # A checkpoint whose ``mtp`` block declares taps carries the
            # drafter that reads them; a minimized fixture may declare none,
            # and a schedule cut shorter than a tap (a staged or fixture
            # graph) cannot host one, so it captures nothing.
            target_layer_ids=_fitted_taps(normalized),
        )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes the config from the pipeline's ``config.json``.

        Args:
            pipeline_config: The pipeline configuration.
            model_config: The model config to read; defaults to the pipeline's.
            max_seq_len: Longest sequence the pipeline was sized for.

        Raises:
            ValueError: If the model config carries no Hugging Face config.
        """
        model_config = model_config or pipeline_config.model
        config = model_config.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required for"
                f" '{model_config.model_path}', but config could not be"
                " loaded. Please ensure the model repository contains a valid"
                " config.json file."
            )
        config = cls.get_text_config(config)
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=model_config.kv_cache,
            cache_dtype=latent_cache_dtype(
                model_config.kv_cache.kv_cache_format
            ),
        )
        if pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        lerope_alpha_dim = int(config.qk_rope_head_dim) // 2
        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            moe_layer_freq=1,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=None,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            # Mistral's router is a softmax top-k with no groups, no bias
            # correction and an unconditional renormalization.
            topk_method="",
            n_group=1,
            topk_group=1,
            num_experts_per_tok=config.num_experts_per_tok,
            first_k_dense_replace=config.first_k_dense_replace,
            norm_topk_prob=True,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings
            + spec_decode_cache_slack(kv_params),
            max_seq_len=max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_interleave=True,
            scoring_func="softmax",
            attention_bias=False,
            attention_dropout=0.0,
            graph_mode=graph_mode,
            max_batch_context_length=131072,
            data_parallel_degree=model_config.data_parallel_degree,
            final_logit_softcapping=float(
                getattr(config, "final_logit_softcapping", 30.0)
            ),
            index_head_dim=int(config.index_head_dim),
            index_n_heads=int(config.index_n_heads),
            index_topk=int(config.index_topk),
            indexer_types=resolve_indexer_types(
                config, int(config.num_hidden_layers)
            ),
            index_norm_eps=float(getattr(config, "indexer_norm_eps", 1e-6)),
            quantization_encoding=quantization_encoding,
            rope_layers=[bool(flag) for flag in config.no_rope_layers],
            kv_source_layers=list(config.kv_source_layers),
            moe_layers=[t == "sparse" for t in config.mlp_layer_types],
            lerope_alpha_dim=lerope_alpha_dim,
            yarn=MistralLarge4Yarn(
                rope_theta=float(config.rope_parameters["rope_theta"]),
                factor=float(config.rope_parameters["factor"]),
                original_max_position_embeddings=int(
                    config.rope_parameters["original_max_position_embeddings"]
                ),
                beta_fast=float(config.rope_parameters["beta_fast"]),
                beta_slow=float(config.rope_parameters["beta_slow"]),
                attention_factor=float(
                    config.rope_parameters.get("attention_factor", 1.0)
                ),
                lerope=True,
                lerope_alpha_dim=lerope_alpha_dim,
            ),
            target_layer_ids=_served_taps(config, model_config),
        )
