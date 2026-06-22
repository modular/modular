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

from __future__ import annotations

import dataclasses

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from max.nn._nvfp4_repack_host import nvfp4_repack_host
from max.nn.quant_ops import _NVFP4_USE_SKELETON_GEMM

GEMMA4_LANGUAGE_SAFETENSOR_MAP: dict[str, str] = {
    "model.language_model.": "",
    "language_model.model.": "",
    "router.proj.weight": "moe_block.gate.gate_score.weight",
    "router.scale": "moe_block.gate.scale",
    "router.per_expert_scale": "moe_block.gate.per_expert_scale",
    "pre_feedforward_layernorm_2.weight": "moe_block.pre_expert_norm.weight",
    "experts.": "moe_block.experts.",
}

GEMMA4_VISION_SAFETENSOR_MAP: dict[str, str] = {
    "model.vision_tower.": "",
    "model.embed_vision": "embed_vision",
    ".linear.": ".",
}


def convert_safetensor_language_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the language model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        # modelopt checkpoints may carry FP8 KV-cache scales (k_scale /
        # v_scale) that MAX's BF16 KV cache does not consume; drop them.
        if weight_name.endswith((".k_scale", ".v_scale")):
            continue
        if not (
            weight_name.startswith("language_model.")
            or weight_name.startswith("model.language_model.")
        ):
            continue

        max_name = weight_name
        for before, after in GEMMA4_LANGUAGE_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        # compressed-tensors NVFP4 checkpoints (e.g. RedHatAI/gemma-4-*-NVFP4)
        # store the same block-scaled E2M1 weights as the modelopt export but
        # under different tensor names. Reconcile them to the modelopt names
        # the quantized Linear expects so both NVFP4 formats load through the
        # same graph weights. The shared block-scale tensor (``weight_scale``)
        # already matches and is left untouched.
        ct_global_scale = False
        if max_name.endswith(".weight_packed"):
            max_name = max_name.removesuffix(".weight_packed") + ".weight"
        elif max_name.endswith(".weight_global_scale"):
            max_name = (
                max_name.removesuffix(".weight_global_scale")
                + ".weight_scale_2"
            )
            ct_global_scale = True
        elif max_name.endswith(".input_global_scale"):
            max_name = (
                max_name.removesuffix(".input_global_scale") + ".input_scale"
            )
            ct_global_scale = True

        data = value.data()

        if max_name.endswith(".weight_scale") and data.dtype == DType.uint8:
            data = dataclasses.replace(data, dtype=DType.float8_e8m0fnu)

        # compressed-tensors stores the per-tensor global scales as the
        # reciprocal of the modelopt convention (llm-compressor stores
        # ``(FP4_MAX * FP8_MAX) / amax`` while modelopt stores
        # ``amax / (FP4_MAX * FP8_MAX)``), shaped ``[1]`` rather than scalar.
        # The NVFP4 kernels multiply activations/weights by ``input_scale`` /
        # ``weight_scale_2``, so invert to the modelopt convention and squeeze
        # ``[1]`` -> scalar to match the shape the quantized Linear registers.
        # modelopt checkpoints arrive scalar under these names and skip this.
        if ct_global_scale:
            inv = (1.0 / np.from_dlpack(data.data).astype(np.float32)).reshape(
                ()
            )
            data = WeightData(inv, max_name, data.dtype, Shape([]))

        # Stacked MoE expert weights: split into individual per-expert weights.
        # HF stores gate_up_proj [num_experts, 2*moe_dim, hidden_dim]
        # and down_proj [num_experts, hidden_dim, moe_dim] as single tensors.
        if "moe_block.experts.gate_up_proj" in max_name:
            prefix = max_name.split("moe_block.experts.")[0]
            buf = Buffer.from_dlpack(data.data)
            num_experts = buf.shape[0]
            half = buf.shape[1] // 2
            expert_shape = [half, buf.shape[2]]
            for j in range(num_experts):
                for proj, s in [
                    ("gate_proj", slice(None, half)),
                    ("up_proj", slice(half, None)),
                ]:
                    name = f"{prefix}moe_block.experts.{j}.{proj}.weight"
                    proj_buf = buf[j : j + 1, s, :].view(
                        data.dtype, expert_shape
                    )
                    new_state_dict[name] = WeightData(
                        proj_buf, name, data.dtype, Shape(expert_shape)
                    )
            continue

        if "moe_block.experts.down_proj" in max_name:
            prefix = max_name.split("moe_block.experts.")[0]
            buf = Buffer.from_dlpack(data.data)
            num_experts = buf.shape[0]
            expert_shape = list(buf.shape[1:])
            for j in range(num_experts):
                name = f"{prefix}moe_block.experts.{j}.down_proj.weight"
                expert_buf = buf[j : j + 1, :, :].view(data.dtype, expert_shape)
                new_state_dict[name] = WeightData(
                    expert_buf, name, data.dtype, Shape(expert_shape)
                )
            continue

        new_state_dict[max_name] = data

    if _NVFP4_USE_SKELETON_GEMM:
        new_state_dict = _fuse_nvfp4_skeleton_weights(new_state_dict)

    return new_state_dict


def _as_uint8_numpy(data: WeightData) -> np.ndarray:
    """Return the raw bytes of ``data`` as a 2D uint8 numpy array.

    Works for both the canonical packed weight (already uint8) and the FP8
    block scales (1 byte/element) -- ``view``-as-uint8 keeps the exact bit
    pattern the host repack consumes.
    """
    buf = Buffer.from_dlpack(data.data)
    n = int(data.shape[0])
    row_bytes = buf.num_elements // n * buf.element_size
    return buf.view(DType.uint8, [n, row_bytes]).to_numpy()


def _fuse_nvfp4_skeleton_weights(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Repack NVFP4 ``(.weight, .weight_scale, .weight_scale_2)`` triples into a
    single combined skeleton buffer at load time.

    For every NVFP4 linear (a uint8 ``.weight`` with sibling ``.weight_scale``
    [float8_e4m3fn block scales] and ``.weight_scale_2`` [scalar global]), emit
    the byte-exact combined buffer under ``.weight`` and DROP both scale tensors
    so they are never registered/uploaded. The global scale uses the multiply
    convention (the already-inverted modelopt value), matching the kernel which
    multiplies weights by ``weight_scale_2``.
    """
    out: dict[str, WeightData] = {}
    # Identify NVFP4 layers by the presence of the modelopt-named scale tensors.
    nvfp4_prefixes: set[str] = set()
    for name, data in state_dict.items():
        if name.endswith(".weight_scale") and data.dtype == DType.float8_e4m3fn:
            prefix = name.removesuffix(".weight_scale")
            if (
                f"{prefix}.weight" in state_dict
                and f"{prefix}.weight_scale_2" in state_dict
            ):
                nvfp4_prefixes.add(prefix)

    for name, data in state_dict.items():
        # Drop the scale tensors of fused layers -- never registered/uploaded.
        if name.endswith(".weight_scale"):
            if name.removesuffix(".weight_scale") in nvfp4_prefixes:
                continue
        if name.endswith(".weight_scale_2"):
            if name.removesuffix(".weight_scale_2") in nvfp4_prefixes:
                continue

        if name.endswith(".weight"):
            prefix = name.removesuffix(".weight")
            if prefix in nvfp4_prefixes:
                weight_u8 = _as_uint8_numpy(data)  # [N, K/2]
                scale_data = state_dict[f"{prefix}.weight_scale"]
                scale_fp8 = _as_uint8_numpy(scale_data)  # [N, K/16]
                gs_data = state_dict[f"{prefix}.weight_scale_2"]
                global_scale = float(
                    np.from_dlpack(gs_data.data).astype(np.float32).reshape(())
                )
                combined = nvfp4_repack_host(
                    weight_u8, scale_fp8, global_scale, group_size=16
                )
                out[name] = WeightData(
                    combined,
                    name,
                    DType.uint8,
                    Shape(combined.shape),
                )
                continue

        out[name] = data

    return out


def convert_safetensor_vision_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format for the vision model."""
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        if not (
            weight_name.startswith("model.vision_tower.")
            or weight_name.startswith("model.embed_vision.")
        ):
            continue

        max_name = weight_name
        for before, after in GEMMA4_VISION_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        new_state_dict[max_name] = value.data()

    return new_state_dict
