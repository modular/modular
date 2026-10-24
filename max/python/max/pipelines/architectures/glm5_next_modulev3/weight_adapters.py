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
"""Maps a GLM-5.3-Flash checkpoint onto the ModuleV3 module tree.

The DeepSeek ModuleV3 adapter cannot be reused as-is: it decides how to name a
projection from the projection's *name* alone, and GLM-5.3-Flash has a ``q_proj``
in every KDA layer that is a plain BF16 linear, not the raw MLA projection that
name means to DeepSeek. Here an FP8 weight is recognized by its scale instead,
since the checkpoint is mixed precision layer by layer.
"""

from __future__ import annotations

from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from transformers import AutoConfig

__all__ = ["convert_glm5_next_state_dict"]

_LM_PREFIXES = (
    ("model.language_model.", "language_model."),
    ("model.", "language_model."),
    ("lm_head.", "language_model.lm_head."),
)

# MLA projections the module holds as raw (unwrapped) weights.
_MLA_RAW = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj")

# Parameters the model reads in float32. `hc_*_fn` is BF16 in the checkpoint;
# widening it is exact and matches what the mapping computes in.
_FLOAT32_SUFFIXES = (
    ".A_log",
    ".dt_bias",
    ".o_norm.weight",
    ".hc_fn",
    ".hc_base",
    ".hc_scale",
)


def _lm_name(name: str) -> str | None:
    for before, after in _LM_PREFIXES:
        if name.startswith(before):
            return after + name[len(before) :]
    return None


def _rename(name: str) -> str:
    # The mHC sites are flat tensors in the checkpoint and modules here.
    for site in ("hc_attn", "hc_ffn"):
        for part in ("fn", "base", "scale"):
            name = name.replace(f".{site}_{part}", f".{site}.hc_{part}")
    name = name.replace("mlp.gate.weight", "mlp.gate.gate_score.weight")
    name = name.replace(
        "self_attn.kv_a_layernorm.weight", "self_attn.kv_a_proj_layernorm"
    )
    # The k-pool gate is a bare `[128, 4096]` tensor applied as a linear.
    name = name.replace(
        "indexer.index_kpool_compress_gate",
        "indexer.index_kpool_compress_gate.weight",
    )
    # KDA's depthwise convs are held as raw `[channels, kernel]` tensors.
    for which in "qkv":
        name = name.replace(
            f"self_attn.{which}_conv1d.weight", f"self_attn.{which}_conv1d"
        )
    return name


def _squeeze_conv(weight: WeightData) -> WeightData:
    """Drops the singleton group axis of a ``[C, 1, K]`` depthwise conv."""
    if len(weight.shape) != 3:
        return weight
    channels, _, kernel = (int(d) for d in weight.shape)
    return WeightData(
        data=Buffer.from_dlpack(weight.data).view(
            dtype=weight.dtype, shape=(channels, kernel)
        ),
        name=weight.name,
        dtype=weight.dtype,
        shape=Shape([channels, kernel]),
        quantization_encoding=weight.quantization_encoding,
    )


def convert_glm5_next_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Converts a GLM-5.3-Flash checkpoint to the ModuleV3 weight names.

    Args:
        state_dict: The raw checkpoint weights.
        huggingface_config: The checkpoint's config, read for the layer count
            that identifies the MTP draft layer.

    Returns:
        The renamed weights. The vision tower and the MTP draft layer are
        dropped: the graph is text-only and has no draft head.
    """
    text_config = getattr(huggingface_config, "text_config", huggingface_config)
    mtp_prefix = f"language_model.layers.{text_config.num_hidden_layers}."
    scaled = {
        name[: -len(".weight_scale_inv")]
        for name in state_dict
        if name.endswith(".weight_scale_inv")
    }

    out: dict[str, WeightData] = {}
    for checkpoint_name, value in state_dict.items():
        if checkpoint_name.startswith("model.visual."):
            continue
        name = _lm_name(checkpoint_name)
        if name is None or name.startswith(mtp_prefix):
            continue
        weight = value.data()

        is_scale = checkpoint_name.endswith(".weight_scale_inv")
        base = None
        if is_scale:
            base = checkpoint_name[: -len(".weight_scale_inv")]
        elif checkpoint_name.endswith(".weight"):
            base = checkpoint_name[: -len(".weight")]

        if base is not None and base in scaled:
            # An FP8 weight keeps its checkpoint name: the FP8BlockTensor keys
            # its packed data `weight` beside `weight_scale_inv`, for a raw
            # MLA projection and a linear alike.
            module = _rename(_lm_name(base) or base)
            name = f"{module}.{'weight_scale_inv' if is_scale else 'weight'}"
        else:
            name = _rename(name)
            stem = name.removesuffix(".weight")
            if stem != name and stem.rsplit(".", 1)[-1] in _MLA_RAW:
                # A BF16 raw projection, `kv_b_proj` on this checkpoint.
                name = stem
            if name.endswith("_conv1d"):
                weight = _squeeze_conv(weight)
        if name.endswith(_FLOAT32_SUFFIXES) and weight.dtype != DType.float32:
            weight = weight.astype(DType.float32)
        out[name] = weight
    return out
