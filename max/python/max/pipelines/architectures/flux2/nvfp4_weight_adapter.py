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
"""NVFP4 weight adapter for FLUX2 transformer models.

Converts BFL (Black Forest Labs) single-file checkpoint naming to the
MAX/diffusers parameter naming convention used by the FLUX2 model.
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData

BFL_TO_MAX_MAPPING = {
    "img_in.": "x_embedder.",
    "txt_in.": "context_embedder.",
    "time_in.in_layer.": "time_guidance_embed.timestep_embedder.linear_1.",
    "time_in.out_layer.": "time_guidance_embed.timestep_embedder.linear_2.",
    "guidance_in.in_layer.": "time_guidance_embed.guidance_embedder.linear_1.",
    "guidance_in.out_layer.": "time_guidance_embed.guidance_embedder.linear_2.",
    "final_layer.adaLN_modulation.1.": "norm_out.linear.",
    "final_layer.linear.": "proj_out.",
    ".lin.": ".linear.",
    "double_blocks.": "transformer_blocks.",
    "single_blocks.": "single_transformer_blocks.",
    ".img_attn.qkv.": ".attn.qkv_proj.",
    ".txt_attn.qkv.": ".attn.add_qkv_proj.",
    ".img_attn.proj.": ".attn.to_out.0.",
    ".txt_attn.proj.": ".attn.to_add_out.",
    ".img_attn.norm.query_norm.scale": ".attn.norm_q.weight",
    ".img_attn.norm.key_norm.scale": ".attn.norm_k.weight",
    ".txt_attn.norm.query_norm.scale": ".attn.norm_added_q.weight",
    ".txt_attn.norm.key_norm.scale": ".attn.norm_added_k.weight",
    ".norm.query_norm.scale": ".attn.norm_q.weight",
    ".norm.key_norm.scale": ".attn.norm_k.weight",
    ".img_mlp.0.": ".ff.linear_in.",
    ".img_mlp.2.": ".ff.linear_out.",
    ".txt_mlp.0.": ".ff_context.linear_in.",
    ".txt_mlp.2.": ".ff_context.linear_out.",
    ".linear1.": ".attn.to_qkv_mlp_proj.",
    ".linear2.": ".attn.to_out.",
}


def _swap_fp4_nibbles(value: WeightData) -> WeightData:
    """Swap the nibble order of packed FP4 weight bytes.

    The FLUX.2-NVFP4 checkpoint packs two FP4 values per byte with the
    first value in the high nibble (bits 7:4) and the second in the low
    nibble (bits 3:0).  The block-scaled matmul kernel and our FP4
    unpacking code both expect lo-first ordering: low nibble first,
    high nibble second.  This function swaps each byte so the packed
    weights match the expected convention.
    """
    raw = np.from_dlpack(value.data).view(np.uint8)
    swapped = (
        ((raw & np.uint8(0x0F)) << np.uint8(4))
        | ((raw >> np.uint8(4)) & np.uint8(0x0F))
    ).astype(np.uint8)
    return WeightData(swapped, value.name, value.dtype, value.shape)


def _swap_adaln_weight_halves(value: WeightData) -> WeightData:
    """Swap the first and second halves of an AdaLN modulation weight.

    BFL's ``LastLayer`` unpacks as ``shift, scale = chunk(2)`` (shift
    first), but diffusers' ``AdaLayerNormContinuous`` — which our model
    uses — unpacks as ``scale, shift = chunk(2)`` (scale first).  When
    loading a BFL-format checkpoint we must swap the two halves of the
    linear weight (and bias) so the output order matches our code.
    """
    buf = value.to_buffer()
    # View bfloat16 as uint16 so numpy can handle it, as numpy doesn't
    # support bfloat16.
    buf_u16 = buf.view(DType.uint16)
    arr = buf_u16.to_numpy()
    half = arr.shape[0] // 2
    swapped = np.ascontiguousarray(
        np.concatenate([arr[half:], arr[:half]], axis=0)
    )
    swapped_buf = Buffer.from_numpy(swapped).view(value.dtype, buf.shape)
    return WeightData(swapped_buf, value.name, value.dtype, value.shape)


def _deinterleave_scales(value: WeightData) -> WeightData:
    """Convert BFL TCGEN5-pre-interleaved scales back to true ``[M, K//16]``.

    BFL ships ``weight_scale`` with shape ``(M, K//16)`` but the underlying
    memory is the TCGEN5 5D layout ``(M//128, K//64, 32, 4, 4)`` flattened
    in row-major. Per ``fp4_utils.mojo`` ``set_scale_factor``, the mapping
    is::

        row  -> (i0, i2, i3) = (row // 128, row % 32, (row % 128) // 32)
        nblk -> (i1, i4)     = (nblk // 4, nblk % 4)

    So storage ``[i_M, i_N]`` does **not** index ``[row, nblk]`` directly —
    slicing axis 1 of the storage to shard K produces a jagged, semantically
    incorrect K-slice that mixes rows across the M-tile.

    Reinterpret the storage as 5D, transpose to ``(i0, i3, i2, i1, i4)``,
    and reshape back to ``(M, K//16)`` to yield a true row-major scale
    tensor that shards cleanly along axis 1. The runtime
    ``block_scales_interleave`` op then re-interleaves per-shard for the
    matmul.
    """
    SF_MN_GROUP_SIZE = 128
    SF_ATOM_M0 = 32
    SF_ATOM_K = 4

    M, K_div16 = int(value.shape[0]), int(value.shape[1])
    if M % SF_MN_GROUP_SIZE != 0 or K_div16 % SF_ATOM_K != 0:
        return value

    # View FP8 bytes as uint8 so numpy can permute them.
    buf = value.to_buffer()
    buf_u8 = buf.view(DType.uint8)
    arr = buf_u8.to_numpy()
    scales_5d = arr.reshape(
        M // SF_MN_GROUP_SIZE,
        K_div16 // SF_ATOM_K,
        SF_ATOM_M0,
        SF_MN_GROUP_SIZE // SF_ATOM_M0,
        SF_ATOM_K,
    )
    deinterleaved = np.ascontiguousarray(
        scales_5d.transpose(0, 3, 2, 1, 4).reshape(M, K_div16)
    )
    deinterleaved_buf = Buffer.from_numpy(deinterleaved).view(
        value.dtype, buf.shape
    )
    return WeightData(deinterleaved_buf, value.name, value.dtype, value.shape)


def convert_nvfp4_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Convert a BFL NVFP4 single-file checkpoint to MAX parameter naming."""
    if not any(k.startswith("double_blocks.") for k in state_dict):
        return state_dict

    new_state_dict: dict[str, WeightData] = {}

    for key, value in state_dict.items():
        max_name = key
        for before, after in BFL_TO_MAX_MAPPING.items():
            max_name = max_name.replace(before, after)

        # The checkpoint packs FP4 nibbles in the opposite order from
        # what cuBLAS / PTX expect.  Swap them at load time.
        if value.dtype == DType.uint8:
            value = _swap_fp4_nibbles(value)

        # BFL ships block-scaled weight scales in 5D TCGEN5 interleaved
        # layout flattened to 2D. That storage cannot be K-sharded by
        # slicing axis 1, so deinterleave to true row-major here and let
        # the runtime re-interleave per shard.
        if (
            max_name.endswith(".weight_scale")
            and value.dtype == DType.float8_e4m3fn
            and len(value.shape) == 2
        ):
            value = _deinterleave_scales(value)

        # BFL's LastLayer uses ``shift, scale = chunk(2)`` but our
        # AdaLayerNormContinuous uses ``scale, shift = chunk(2)``.
        # Swap the weight halves so the linear output matches our code.
        if max_name in ("norm_out.linear.weight", "norm_out.linear.bias"):
            value = _swap_adaln_weight_halves(value)

        new_state_dict[max_name] = value

    return new_state_dict
