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
"""Mixed-precision map for GLM-5.3-Flash checkpoints.

GLM-5.3-Flash is DeepSeek-style blockwise FP8 -- ``quant_method: "fp8"``,
``fmt: "e4m3"``, ``weight_block_size: [128, 128]``, ``activation_scheme:
"dynamic"`` -- but unlike GLM-5.2-FP8 or DeepSeek-V3 it leaves whole *linear*
layers unquantized: every KDA projection, the whole DSA indexer, ``kv_b_proj``
and the whole vision tower are BF16. MAX's shared blockscaled-FP8 parser
assumes the opposite and describes every layer as quantized::

    # All layers use FP8 in this format (e.g. Qwen3-30B-A3B FP8, DeepSeekV3).
    # modules_to_not_convert only lists layernorms and router gate, ...
    all_layers = set(range(num_hidden_layers))

max/python/max/pipelines/weights/quant.py:631

The map here is derived from the *presence of a weight scale* in the
checkpoint rather than from ``modules_to_not_convert``. The checkpoint's
exclusion list has 1,509 entries with per-layer duplicates and several
plausible-but-wrong readings; the scale tensors are unambiguous and
self-consistent. The diagnostic that settles it: ``o_proj`` appears 46 times
in the weight index but ``o_proj.weight_scale_inv`` only 12 -- exactly the
sparse-MLA layers.

Anything this module cannot account for raises. A checkpoint that does not
match the shape described above is one MAX has not been taught to read, and
that has to fail at load time rather than in the logits.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

__all__ = [
    "UNQUANTIZED_ATTN_PROJECTIONS",
    "Glm5NextQuantScheme",
    "parse_quant_scheme",
]

# Block shape MAX's blockscaled-FP8 path implements. Any other shape misreads
# every scale, so it is checked rather than assumed.
_WEIGHT_BLOCK_SIZE = [128, 128]

#: Projections inside a *quantized* sparse-MLA layer that are nonetheless
#: BF16 in the checkpoint. ``kv_b_proj`` is the one that matters: the absorb
#: folds it into the query and output paths at graph-build time, so a BF16
#: source needs no dequant -- but MAX's FP8 MLA class asserts that weight and
#: input scale block sizes exist
#: (max/python/max/nn/attention/multi_latent_attention_fp8.py:146), so the
#: sparse-MLA constructor has to take a per-projection dtype rather than one
#: ``dtype`` for the block.
UNQUANTIZED_ATTN_PROJECTIONS = frozenset(
    {
        "kv_b_proj",
        # The whole indexer is BF16: wq_b, wk, k_norm, weights_proj,
        # index_kpool_compress_gate, index_kpool_compress_ape.
        "indexer",
    }
)

# Post-adapter MAX weight names: `layers.<idx>.<block>....weight_scale`.
# `mlp.experts.<n>.` and `mlp.shared_experts.` both fall under `mlp`.
_SCALED_WEIGHT = re.compile(
    r"^layers\.(\d+)\.(mlp|self_attn)\.(.+)\.weight_scale$"
)


@dataclass(frozen=True)
class Glm5NextQuantScheme:
    """Which GLM-5.3-Flash modules are FP8, and how."""

    config: QuantConfig
    """Blockwise-FP8 config for the modules the layer sets below name."""

    mlp_layers: frozenset[int]
    """Layers whose MLP -- dense, shared expert and routed experts -- is FP8."""

    attn_layers: frozenset[int]
    """Layers whose ``q_a_proj`` / ``q_b_proj`` / ``kv_a_proj_with_mqa`` /
    ``o_proj`` are FP8. On GLM-5.3-Flash this is exactly the sparse-MLA layer
    subset plus the MTP layer; the 34 KDA layers hold no FP8 weight at all."""

    compute_dtype: DType
    """Dtype of everything unquantized: activations, norms, embeddings, every
    KDA projection and conv, ``kv_b_proj``, the indexer, the vision tower and
    the mHC mapping. Distinct from :attr:`Glm5NextConfig.dtype`, which is the
    *storage* dtype the resolved encoding implies."""

    def mlp_config(self, layer_idx: int) -> QuantConfig | None:
        """Returns the config for layer ``layer_idx``'s MLP, or None if BF16."""
        return self.config if layer_idx in self.mlp_layers else None

    def attn_config(self, layer_idx: int) -> QuantConfig | None:
        """Returns the config for layer ``layer_idx``'s attention projections.

        Applies to the four FP8 projections only; callers must keep
        :obj:`UNQUANTIZED_ATTN_PROJECTIONS` at :attr:`compute_dtype` even in a
        layer this returns a config for.
        """
        return self.config if layer_idx in self.attn_layers else None


def _scaled_layers(
    state_dict: Mapping[str, WeightData],
) -> tuple[frozenset[int], frozenset[int]]:
    """Returns the (mlp, attn) layer indices that carry FP8 weight scales."""
    mlp: set[int] = set()
    attn: set[int] = set()
    for name in state_dict:
        match = _SCALED_WEIGHT.match(name)
        if match is None:
            continue
        layer_idx, block, projection = match.groups()
        if block == "mlp":
            mlp.add(int(layer_idx))
        elif any(p in projection for p in UNQUANTIZED_ATTN_PROJECTIONS):
            # A scale on a projection the checkpoint is documented to leave
            # BF16 means the map is being derived from the wrong checkpoint.
            raise ValueError(
                f"'{name}' carries an FP8 weight scale, but GLM-5.3-Flash "
                f"stores {sorted(UNQUANTIZED_ATTN_PROJECTIONS)} in bfloat16. "
                "Re-derive the precision map from this checkpoint's weight "
                "index before loading it."
            )
        else:
            attn.add(int(layer_idx))
    return frozenset(mlp), frozenset(attn)


def parse_quant_scheme(
    hf_quantization_config: Mapping[str, object] | None,
    state_dict: Mapping[str, WeightData],
    compute_dtype: DType,
) -> Glm5NextQuantScheme | None:
    """Builds the mixed-precision map for a GLM-5.3-Flash checkpoint.

    Args:
        hf_quantization_config: The checkpoint's resolved ``quantization_config``.
            ``None`` for the BF16 variant, which has no quantized modules.
        state_dict: Post-adapter MAX weights, read for ``weight_scale`` presence.
        compute_dtype: Dtype of the unquantized modules.

    Returns:
        The scheme, or ``None`` when the checkpoint declares no quantization.

    Raises:
        ValueError: If the checkpoint declares a quantization shape this
            module does not implement, or if no FP8 scales are found in a
            checkpoint that declares FP8.
    """
    if not hf_quantization_config:
        return None

    method = hf_quantization_config.get("quant_method")
    if method != "fp8":
        raise ValueError(
            f"GLM-5.3-Flash expects quant_method 'fp8', got {method!r}"
        )
    fmt = hf_quantization_config.get("fmt")
    if fmt != "e4m3":
        raise ValueError(f"GLM-5.3-Flash expects fmt 'e4m3', got {fmt!r}")
    scheme = hf_quantization_config.get("activation_scheme")
    if scheme != "dynamic":
        raise ValueError(
            f"GLM-5.3-Flash expects a dynamic activation scheme, got {scheme!r}"
        )
    block_size = hf_quantization_config.get("weight_block_size")
    if (
        not isinstance(block_size, (list, tuple))
        or list(block_size) != _WEIGHT_BLOCK_SIZE
    ):
        raise ValueError(
            f"GLM-5.3-Flash expects weight_block_size {_WEIGHT_BLOCK_SIZE}, "
            f"got {block_size!r}"
        )

    mlp_layers, attn_layers = _scaled_layers(state_dict)
    if not mlp_layers and not attn_layers:
        raise ValueError(
            "The checkpoint declares blockwise FP8 but no weight scales were "
            "found in the loaded state dict. The weight adapter renames "
            "'weight_scale_inv' to 'weight_scale'; check that it ran."
        )

    scale_dtype = DType.float32
    for name, weight in state_dict.items():
        if name.endswith(".weight_scale"):
            scale_dtype = weight.dtype
            break

    config = QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=scale_dtype,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=scale_dtype,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(mlp_layers),
        attn_quantized_layers=set(attn_layers),
        embedding_output_dtype=compute_dtype,
        format=QuantFormat.BLOCKSCALED_FP8,
    )
    return Glm5NextQuantScheme(
        config=config,
        mlp_layers=mlp_layers,
        attn_layers=attn_layers,
        compute_dtype=compute_dtype,
    )
