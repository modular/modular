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
"""Weight adapters for Step-3.5-Flash models.

Checkpoint layout (HuggingFace) -> MAX layout:
  - Strip "model." prefix
  - Stacked MoE expert weights split into individual experts
  - MoE gate: "moe.gate.weight" -> "mlp.moe.gate.gate_score.weight"
  - MoE router bias: "moe.router_bias" -> "mlp.moe.gate.router_bias"
  - Shared expert: "share_expert." -> "mlp.share_expert."
  - Norm weights passed as-is (RMSNorm with weight_offset=1.0 adds 1 at runtime)
"""

from __future__ import annotations

import re

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig


def _build_partial_rope_perm(head_dim: int, rotary_dim: int) -> list[int]:
    """Build per-head dim permutation for partial RoPE with interleaved=False.

    When partial_rotary_factor < 1.0, the HuggingFace reference applies
    ``rotate_half`` within the first ``rotary_dim`` dimensions, pairing
    ``(q[i], q[i + rotary_dim//2])`` for ``i = 0 .. rotary_dim//2 - 1``.

    With ``interleaved=False``, the MAX kernel pairs across the full
    ``head_dim``:  ``(q[i], q[i + head_dim//2])``.  These are incompatible
    when ``rotary_dim < head_dim``.

    This function returns a permutation that swaps
    ``[rotary_dim//2, rotary_dim)`` with ``[rotary_dim, rotary_dim + rotary_dim//2)``
    so the kernel's full-head pairing produces the same rotation as the
    reference's subspace pairing.  The permutation is its own inverse.
    """
    half_rot = rotary_dim // 2
    perm = list(range(head_dim))
    for i in range(half_rot):
        perm[half_rot + i], perm[rotary_dim + i] = (
            perm[rotary_dim + i],
            perm[half_rot + i],
        )
    return perm


def _buf_to_numpy(
    buf: Buffer, dtype: DType, shape: Shape | tuple[int, ...] | list[int]
) -> np.ndarray:
    """Convert a Buffer to a writable numpy array.

    bfloat16 is reinterpreted as uint16 (same byte width) since numpy
    has no native bfloat16 support.
    """
    if dtype == DType.bfloat16:
        buf = buf.view(DType.uint16)
    arr = buf.to_numpy()
    # Shape may contain Dim objects; convert to plain ints for numpy.
    int_shape = tuple(int(d) for d in shape)
    return arr.reshape(int_shape)


def _numpy_to_weight(
    arr: np.ndarray,
    name: str,
    dtype: DType,
    shape: Shape,
    orig_buf_shape: tuple[int, ...] | list[int],
) -> WeightData:
    """Convert a numpy array back to WeightData, restoring original dtype."""
    arr = np.ascontiguousarray(arr)
    buf = Buffer.from_numpy(arr)
    if dtype == DType.bfloat16:
        buf = buf.view(dtype, [int(d) for d in orig_buf_shape])
    return WeightData(buf, name, dtype, shape)


def _apply_head_perm(
    weight_data: WeightData,
    perm: list[int],
    num_heads: int,
    head_dim: int,
) -> WeightData:
    """Permute the output (row) dimension of a ``[num_heads*head_dim, in_dim]`` weight."""
    buf = Buffer.from_dlpack(weight_data.data)
    orig_buf_shape = buf.shape
    arr = _buf_to_numpy(buf, weight_data.dtype, weight_data.shape)
    orig_shape = arr.shape
    arr = arr.reshape(num_heads, head_dim, *orig_shape[1:])
    arr = arr[:, perm, ...]
    arr = arr.reshape(orig_shape)
    return _numpy_to_weight(
        arr,
        weight_data.name,
        weight_data.dtype,
        weight_data.shape,
        orig_buf_shape,
    )


def _apply_norm_perm(
    weight_data: WeightData,
    perm: list[int],
) -> WeightData:
    """Permute a ``[head_dim]`` norm weight."""
    buf = Buffer.from_dlpack(weight_data.data)
    orig_buf_shape = buf.shape
    arr = _buf_to_numpy(buf, weight_data.dtype, weight_data.shape)
    arr = arr[perm]
    return _numpy_to_weight(
        arr,
        weight_data.name,
        weight_data.dtype,
        weight_data.shape,
        orig_buf_shape,
    )


def convert_step3p5_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Convert Step-3.5-Flash checkpoint weights to MAX format.

    Args:
        state_dict: Raw checkpoint weights.
        huggingface_config: HuggingFace model configuration.
        pipeline_config: Pipeline configuration.

    Returns:
        Transformed weights for the MAX Step-3.5 model.
    """
    num_hidden_layers = huggingface_config.num_hidden_layers
    new_state_dict: dict[str, WeightData] = {}

    # Pre-compute partial RoPE weight permutation.
    # Full-attention layers have partial_rotary_factor < 1.0.  Their Q/K
    # weights must be permuted so the halved-pairing RoPE kernel rotates
    # the correct dimension pairs.  See _build_partial_rope_perm docstring.
    head_dim = getattr(huggingface_config, "head_dim", 128)
    partial_rotary_factors = getattr(
        huggingface_config, "partial_rotary_factors", []
    )
    partial_rope_perm: list[int] | None = None
    partial_rope_layers: set[int] = set()
    for i in range(min(len(partial_rotary_factors), num_hidden_layers)):
        prf = float(partial_rotary_factors[i])
        if prf < 1.0:
            if partial_rope_perm is None:
                rotary_dim = int(head_dim * prf)
                partial_rope_perm = _build_partial_rope_perm(
                    head_dim, rotary_dim
                )
            partial_rope_layers.add(i)

    num_q_heads_full = huggingface_config.num_attention_heads
    num_kv_heads = getattr(huggingface_config, "num_attention_groups", 8)

    for safetensor_name, value in state_dict.items():
        # Skip MTP layers (layer indices >= num_hidden_layers)
        if "model.layers." in safetensor_name:
            parts = safetensor_name.split(".")
            layer_idx_str = parts[2] if len(parts) > 2 else ""
            if (
                layer_idx_str.isdigit()
                and int(layer_idx_str) >= num_hidden_layers
            ):
                continue

        # Skip non-model weights
        if safetensor_name.startswith("mtp."):
            continue

        weight_data = value.data()
        max_name = safetensor_name

        # Strip "model." prefix
        max_name = max_name.removeprefix("model.")

        # Stacked MoE expert weights: layers.{i}.moe.{proj}.weight [num_experts, ...]
        # -> individual: layers.{i}.mlp.moe.experts.{j}.{proj}.weight
        if (
            ".moe.gate_proj.weight" in max_name
            or ".moe.up_proj.weight" in max_name
            or ".moe.down_proj.weight" in max_name
        ):
            prefix, rest = max_name.split(".moe.", 1)
            proj_name = rest  # e.g. "gate_proj.weight"
            buf = Buffer.from_dlpack(weight_data.data)
            num_experts = buf.shape[0]
            expert_shape = list(buf.shape[1:])
            # Slice one expert at a time: [j:j+1, :, :] -> reshape to [dims...]
            remaining = tuple(slice(None) for _ in range(len(buf.shape) - 1))
            for j in range(num_experts):
                expert_name = f"{prefix}.mlp.moe.experts.{j}.{proj_name}"
                sliced = buf[(slice(j, j + 1), *remaining)]
                expert_buf = sliced.view(weight_data.dtype, expert_shape)
                new_state_dict[expert_name] = WeightData(
                    expert_buf,
                    expert_name,
                    weight_data.dtype,
                    Shape(expert_shape),
                )
            continue

        # MoE gate: moe.gate.weight -> mlp.moe.gate.gate_score.weight
        if ".moe.gate.weight" in max_name:
            max_name = max_name.replace(
                ".moe.gate.weight", ".mlp.moe.gate.gate_score.weight"
            )

        # MoE router bias: moe.router_bias -> mlp.moe.gate.router_bias
        if ".moe.router_bias" in max_name:
            max_name = max_name.replace(
                ".moe.router_bias", ".mlp.moe.gate.router_bias"
            )
            # Router bias must be float32
            if weight_data.dtype != DType.float32:
                weight_data = weight_data.astype(DType.float32)

        # Shared expert: share_expert. -> mlp.share_expert.
        if ".share_expert." in max_name:
            max_name = max_name.replace(".share_expert.", ".mlp.share_expert.")

        # Partial RoPE weight permutation for full-attention layers.
        # Q/K projection and QK-norm weights must be permuted so the
        # halved-pairing kernel rotates the correct dimension pairs
        # when partial_rotary_factor < 1.0.
        if partial_rope_perm is not None:
            qk_match = re.match(
                r"layers\.(\d+)\.self_attn\.(q_proj|k_proj)\.weight$",
                max_name,
            )
            if qk_match and int(qk_match.group(1)) in partial_rope_layers:
                n_heads = (
                    num_q_heads_full
                    if qk_match.group(2) == "q_proj"
                    else num_kv_heads
                )
                weight_data = _apply_head_perm(
                    weight_data, partial_rope_perm, n_heads, head_dim
                )

            norm_match = re.match(
                r"layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight$",
                max_name,
            )
            if norm_match and int(norm_match.group(1)) in partial_rope_layers:
                weight_data = _apply_norm_perm(weight_data, partial_rope_perm)

        new_state_dict[max_name] = weight_data

    return new_state_dict
