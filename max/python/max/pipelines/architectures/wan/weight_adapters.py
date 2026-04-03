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

"""Weight adapters for Wan transformer models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import WeightData, Weights

from .wan_transformer import WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

# Weight key remapping from diffusers -> MAX module naming.
WAN_SAFETENSOR_MAP = [
    (".attn1.to_out.0.", ".attn1.to_out."),
    (".attn2.to_out.0.", ".attn2.to_out."),
    (".ffn.net.0.proj.", ".ffn.proj."),
    (".ffn.net.2.", ".ffn.linear_out."),
    # Image embedder GELU FFN: diffusers nested structure -> flat Linear layers.
    ("image_embedder.ff.net.0.proj.", "image_embedder.ff_proj."),
    ("image_embedder.ff.net.2.", "image_embedder.ff_out."),
]

# Keys to skip (non-persistent buffers computed at runtime).
WAN_SKIP_PREFIXES = ("rope.freqs_cos", "rope.freqs_sin")
WAN_FLOAT32_STATE_DICT_KEYS = frozenset({"motion_encoder.q_matrix"})


def _to_float32_numpy(tensor: Any) -> np.ndarray:
    """Materialize a host tensor-like value as contiguous float32 NumPy."""
    if isinstance(tensor, np.ndarray):
        return np.ascontiguousarray(tensor, dtype=np.float32)

    src_dtype = tensor.dtype
    dlpack_obj = tensor.to_buffer() if hasattr(tensor, "to_buffer") else tensor
    return np.ascontiguousarray(
        np.from_dlpack(
            cast_dlpack_to(dlpack_obj, src_dtype, DType.float32, CPU())
        )
    )


def _to_host_buffer(key: str, tensor: Any, target_dtype: DType) -> Any:
    """Convert a raw host tensor-like value to a CPU Buffer."""
    if isinstance(tensor, np.ndarray):
        src_dtype = DType.float32
        dlpack_obj = np.ascontiguousarray(tensor, dtype=np.float32)
    else:
        src_dtype = tensor.dtype
        dlpack_obj = (
            tensor.to_buffer() if hasattr(tensor, "to_buffer") else tensor
        )

    dst_dtype = (
        DType.float32 if key in WAN_FLOAT32_STATE_DICT_KEYS else target_dtype
    )
    return cast_dlpack_to(dlpack_obj, src_dtype, dst_dtype, CPU())


def convert_safetensor_state_dict(
    weights: Weights,
    *,
    target_dtype: DType = DType.bfloat16,
    is_animate: bool = False,
) -> dict[str, Any]:
    """Convert Wan diffusers safetensors to MAX-native transformer weights."""
    state_dict: dict[str, Any] = {}

    # First pass: collect all weights with key remapping.
    raw_dict: dict[str, Any] = {}
    for key, value in weights.items():
        if any(key.startswith(prefix) for prefix in WAN_SKIP_PREFIXES):
            continue

        new_key = key
        for old, new in WAN_SAFETENSOR_MAP:
            new_key = new_key.replace(old, new)

        tensor = value.data()

        # Conv3d weight permutation for patch embedding:
        # Diffusers [F, C, D, H, W] -> MAX [D, H, W, C, F].
        if len(tensor.shape) == 5 and new_key == "patch_embedding.weight":
            permuted: WeightData | np.ndarray = np.ascontiguousarray(
                _to_float32_numpy(tensor).transpose(2, 3, 4, 1, 0)
            )
            raw_dict[new_key] = permuted
        else:
            raw_dict[new_key] = tensor

    # Second pass: fuse attn2.to_k + attn2.to_v into attn2.to_kv.
    fused_keys: set[str] = set()
    for key in list(raw_dict.keys()):
        if ".attn2.to_k." not in key:
            continue

        k_key = key
        v_key = key.replace(".attn2.to_k.", ".attn2.to_v.")
        kv_key = key.replace(".attn2.to_k.", ".attn2.to_kv.")
        if v_key not in raw_dict:
            continue

        k_np = _to_float32_numpy(raw_dict[k_key])
        v_np = _to_float32_numpy(raw_dict[v_key])
        kv_np = np.ascontiguousarray(np.concatenate([k_np, v_np], axis=0))
        state_dict[kv_key] = kv_np
        fused_keys.add(k_key)
        fused_keys.add(v_key)

    for key, tensor in raw_dict.items():
        if key not in fused_keys:
            state_dict[key] = tensor

    if is_animate:
        preprocess_face_encoder_weights(state_dict)
        preprocess_motion_encoder_weights(state_dict)

    for key in state_dict:
        state_dict[key] = _to_host_buffer(key, state_dict[key], target_dtype)

    return state_dict


def preprocess_motion_encoder_weights(
    state_dict: dict[str, Any],
) -> None:
    """Pre-process Wan-Animate motion encoder weights in-place."""
    me_keys = [key for key in state_dict if key.startswith("motion_encoder.")]
    if not me_keys:
        return

    for key in me_keys:
        sub = key[len("motion_encoder.") :]
        arr = _to_float32_numpy(state_dict[key])
        shape = arr.shape

        if sub == "motion_synthesis_weight":
            q, _ = np.linalg.qr((arr + 1e-8).astype(np.float64))
            state_dict["motion_encoder.q_matrix"] = np.ascontiguousarray(
                q.astype(np.float32)
            )
            del state_dict[key]
            continue

        if sub.endswith(".weight") and len(shape) == 4:
            _, in_channels, kernel_h, kernel_w = shape
            scale = 1.0 / math.sqrt(in_channels * kernel_h * kernel_w)
            state_dict[key] = np.ascontiguousarray(
                (arr * scale).transpose(2, 3, 1, 0)
            )
        elif sub.endswith(".weight") and len(shape) == 2:
            _, in_dim = shape
            scale = 1.0 / math.sqrt(in_dim)
            state_dict[key] = np.ascontiguousarray(arr * scale)

    k1d = np.array([1, 3, 3, 1], dtype=np.float32)
    k2d = np.outer(k1d, k1d)
    k2d = k2d / k2d.sum()

    channels = WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES
    size = 512
    log_size = int(math.log(size, 2))
    in_channels = channels[size]
    for block_idx in range(log_size - 2):
        out_channels = channels[2 ** (log_size - 1 - block_idx)]
        blur_arr = np.tile(
            k2d[:, :, np.newaxis, np.newaxis], [1, 1, 1, in_channels]
        )
        prefix = f"motion_encoder.res_blocks.{block_idx}"
        state_dict[f"{prefix}.conv2.blur_filter"] = np.ascontiguousarray(
            blur_arr
        )
        state_dict[f"{prefix}.conv_skip.blur_filter"] = np.ascontiguousarray(
            blur_arr
        )
        in_channels = out_channels


def preprocess_face_encoder_weights(
    state_dict: dict[str, Any],
) -> None:
    """Pre-process Wan-Animate face encoder weights in-place."""
    for key in list(state_dict.keys()):
        if not (key.startswith("face_encoder.") and key.endswith(".weight")):
            continue
        arr = _to_float32_numpy(state_dict[key])
        if arr.ndim != 3:
            continue
        arr = arr.transpose(2, 1, 0)
        kernel_size, in_channels, out_channels = arr.shape
        state_dict[key] = np.ascontiguousarray(
            arr.reshape(kernel_size * in_channels, out_channels)
        )
