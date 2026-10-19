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
"""Weight loading for the unified DSpark Nemotron-H pipeline.

The draft module tree mirrors its checkpoint's names, so tensors pass through
as stored, quantized ones included. What differs is:

- ``embed_tokens`` is a frozen copy of the target's embedding, with one row
  trained: the mask token's. The draft embeds its anchor with the target's
  table and its mask slots with that row, so the table is dropped after
  checking it matches the target's everywhere else, and the row is kept as
  ``mask_embedding``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData

from .dspark_qwen3 import DSparkQwen3DraftConfig, draft_nvfp4_modules

__all__ = [
    "DRAFT_PREFIX",
    "MASK_EMBEDDING",
    "SHARED_DRAFT_PREFIXES",
    "SKIPPED_DRAFT_PREFIXES",
    "TARGET_PREFIX",
    "classify_draft_weight",
    "convert_draft_state_dict",
    "convert_unified_state_dict",
    "expected_draft_weight_names",
]

TARGET_PREFIX = "target."
DRAFT_PREFIX = "draft."
MASK_EMBEDDING = "mask_embedding"
"""The draft module's weight holding the mask token's trained embedding."""

SHARED_DRAFT_PREFIXES = ("embed_tokens.", "lm_head.")
"""Draft checkpoint weights the draft takes from the target instead."""

SKIPPED_DRAFT_PREFIXES = ("confidence_head.", "mask_embedding")
"""Draft checkpoint weights nothing in this pipeline reads: the confidence
head gates dynamic proposal lengths, which MAX does not use."""

_QUANT_SUFFIXES = ("weight", "weight_scale", "weight_scale_2")


def classify_draft_weight(name: str) -> str:
    """Says what the pipeline does with one draft checkpoint tensor.

    Returns:
        ``"shared"`` if the draft reuses the target's copy, ``"skipped"`` if
        nothing reads it, and ``"loaded"`` otherwise.
    """
    if name.startswith(SHARED_DRAFT_PREFIXES):
        return "shared"
    if name.startswith(SKIPPED_DRAFT_PREFIXES):
        return "skipped"
    return "loaded"


def expected_draft_weight_names(
    config: DSparkQwen3DraftConfig,
) -> set[str]:
    """Returns the checkpoint tensor names the draft module reads.

    The names are those the module's attributes spell, so this is the
    contract between the checkpoint and :class:`~.dspark_qwen3.DSparkQwen3`.
    """
    quantized = draft_nvfp4_modules(config)
    names: set[str] = set()

    def linear(module: str) -> None:
        if module in quantized:
            names.update(f"{module}.{s}" for s in _QUANT_SUFFIXES)
        else:
            names.add(f"{module}.weight")

    linear("fc")
    names.add(MASK_EMBEDDING)
    names.add("hidden_norm.weight")
    names.add("norm.weight")
    names.add("markov_head.markov_w1.weight")
    linear("markov_head.markov_w2")
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        names.add(f"{prefix}.input_layernorm.weight")
        names.add(f"{prefix}.post_attention_layernorm.weight")
        attn = f"{prefix}.self_attn"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"{attn}.{proj}.weight")
        names.add(f"{attn}.q_norm.weight")
        names.add(f"{attn}.k_norm.weight")
        names.add(f"{attn}.attention_sink_bias")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            linear(f"{prefix}.mlp.{proj}")
    return names


def _rows(weight: WeightData) -> np.ndarray:
    """A BF16 table's bits as ``[rows, width]`` uint16."""
    rows, width = (int(d) for d in weight.shape)
    buffer = weight.to_buffer().view(DType.uint16, (rows, width))
    return np.from_dlpack(buffer)


def _mask_embedding(
    embedding: WeightData,
    target_embedding: WeightData | None,
    mask_token_id: int,
) -> WeightData:
    """Returns the mask token's row of the draft's embedding table.

    The draft's table is the target's with the mask row trained. Every other
    row must equal the target's, or the draft would embed its anchor
    differently from the target it reads.

    Raises:
        ValueError: If the shapes or any other row differ.
    """
    shape = tuple(int(d) for d in embedding.shape)
    if embedding.dtype != DType.bfloat16:
        raise ValueError(
            f"The DSpark draft's embedding is {embedding.dtype}, not BF16."
        )
    table = _rows(embedding)
    if target_embedding is not None:
        target_shape = tuple(int(d) for d in target_embedding.shape)
        if shape != target_shape:
            raise ValueError(
                f"The DSpark draft's embedding has shape {shape}, which does"
                " not match the target's embedding"
                f" {target_shape}; the draft cannot share it."
            )
        differing = np.flatnonzero(
            (table != _rows(target_embedding)).any(axis=1)
        )
        others = [int(r) for r in differing if int(r) != mask_token_id]
        if others:
            raise ValueError(
                "The DSpark draft's embedding differs from the target's at"
                f" rows {others[:5]} beyond the mask token {mask_token_id},"
                " so the draft cannot share the target's table."
            )
    row = np.ascontiguousarray(table[mask_token_id])
    return WeightData(
        data=Buffer.from_numpy(row.view(np.uint8)).view(
            DType.bfloat16, (shape[1],)
        ),
        name=MASK_EMBEDDING,
        dtype=DType.bfloat16,
        shape=Shape((shape[1],)),
    )


def convert_draft_state_dict(
    state_dict: Mapping[str, WeightData],
    config: DSparkQwen3DraftConfig,
    *,
    target_embedding: WeightData | None = None,
) -> dict[str, WeightData]:
    """Returns the draft tensors its module reads.

    Args:
        state_dict: The draft checkpoint's tensors.
        config: The parsed draft config.
        target_embedding: The target's embedding table, which the draft's
            must equal bar the mask token's row.

    Raises:
        ValueError: If the draft's embedding does not match the target's, or
            the checkpoint lacks or adds tensors.
    """
    weights: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        match classify_draft_weight(name):
            case "shared":
                if name == "embed_tokens.weight":
                    weights[MASK_EMBEDDING] = _mask_embedding(
                        value, target_embedding, config.mask_token_id
                    )
            case "skipped":
                continue
            case _:
                weights[name] = value
    expected = expected_draft_weight_names(config)
    missing, extra = expected - weights.keys(), weights.keys() - expected
    if missing or extra:
        raise ValueError(
            "The DSpark draft checkpoint does not match its module. Missing:"
            f" {sorted(missing)[:5]}; unexpected: {sorted(extra)[:5]}."
        )
    return weights


def convert_unified_state_dict(
    target_state_dict: Mapping[str, WeightData],
    draft_state_dict: Mapping[str, WeightData],
) -> dict[str, WeightData]:
    """Names the target's and the draft's tensors for the unified module."""
    unified = {f"{TARGET_PREFIX}{k}": v for k, v in target_state_dict.items()}
    for name, value in draft_state_dict.items():
        unified[f"{DRAFT_PREFIX}{name}"] = value
    return unified
