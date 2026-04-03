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

"""LoRA utilities for WAN diffusion transformers.

Provides download and state-dict merge for LoRA safetensors (e.g.
Wan2.2-Lightning).  Merged at load time into the base state dict so
the compiled graphs see pre-merged weights — no runtime adapter slots.

The LoRA safetensors from lightx2v/Wan2.2-Lightning use ComfyUI-style
key naming::

    diffusion_model.blocks.N.self_attn.{q,k,v,o}
    diffusion_model.blocks.N.cross_attn.{q,k,v,o}
    diffusion_model.blocks.N.ffn.{0,2}

The MAX state dict (after convert_safetensor_state_dict in
weight_adapters.py) uses::

    blocks.N.attn1.to_{q,k,v}.weight       (self-attention)
    blocks.N.attn1.to_out.weight            (self-attention output)
    blocks.N.attn2.to_q.weight              (cross-attention query)
    blocks.N.attn2.to_kv.weight             (cross-attention K+V fused)
    blocks.N.attn2.to_out.weight            (cross-attention output)
    blocks.N.ffn.proj.weight                (ffn gate/up projection)
    blocks.N.ffn.linear_out.weight          (ffn down projection)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Remap ComfyUI-style LoRA base keys to MAX state_dict base keys.
# Applied after stripping `diffusion_model.` prefix and before
# appending `.weight`.
_LORA_KEY_REMAP = [
    # Norm keys (must come before generic attn remaps to avoid partial match)
    (".self_attn.norm_q", ".attn1.norm_q"),
    (".self_attn.norm_k", ".attn1.norm_k"),
    (".cross_attn.norm_q", ".attn2.norm_q"),
    (".cross_attn.norm_k", ".attn2.norm_k"),
    # Self-attention
    (".self_attn.q", ".attn1.to_q"),
    (".self_attn.k", ".attn1.to_k"),
    (".self_attn.v", ".attn1.to_v"),
    (".self_attn.o", ".attn1.to_out"),
    # Cross-attention
    (".cross_attn.q", ".attn2.to_q"),
    (".cross_attn.k", ".attn2.to_k"),
    (".cross_attn.v", ".attn2.to_v"),
    (".cross_attn.o", ".attn2.to_out"),
    # FFN
    (".ffn.0", ".ffn.proj"),
    (".ffn.2", ".ffn.linear_out"),
]


def download_wan_lora(
    repo_id: str,
    subfolder: str,
    filenames: list[str] | None = None,
) -> dict[str, Path]:
    """Download LoRA safetensors from a HuggingFace repo.

    Args:
        repo_id: HF repo id, e.g. ``"lightx2v/Wan2.2-Lightning"``.
        subfolder: Subfolder inside the repo, e.g.
            ``"Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1"``.
        filenames: Specific safetensors files to download.  Defaults to
            ``["high_noise_model.safetensors",
              "low_noise_model.safetensors"]``.

    Returns:
        Mapping from stem name (e.g. ``"high_noise_model"``) to local
        file path.
    """
    from huggingface_hub import hf_hub_download

    if filenames is None:
        filenames = [
            "high_noise_model.safetensors",
            "low_noise_model.safetensors",
        ]

    result: dict[str, Path] = {}
    for fname in filenames:
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=fname,
            )
            stem = Path(fname).stem
            result[stem] = Path(local)
            logger.info(
                "Downloaded LoRA: %s/%s/%s -> %s",
                repo_id,
                subfolder,
                fname,
                local,
            )
        except Exception:
            logger.warning(
                "LoRA file not found, skipping: %s/%s/%s",
                repo_id,
                subfolder,
                fname,
            )
    return result


def _remap_lora_key(key: str) -> str:
    """Remap a ComfyUI-style LoRA base key to MAX state_dict naming.

    Strips ``diffusion_model.`` prefix, then applies _LORA_KEY_REMAP.
    """
    # Strip the diffusion_model. prefix
    key = key.removeprefix("diffusion_model.")

    for old, new in _LORA_KEY_REMAP:
        key = key.replace(old, new)
    return key


def load_and_merge_lora(
    state_dict: dict[str, Any],
    lora_path: str | Path,
    lora_scale: float = 1.0,
) -> dict[str, Any]:
    """Merge a LoRA safetensors file into an existing state dict in-place.

    The LoRA file contains keys of the form::

        diffusion_model.blocks.N.{self_attn,cross_attn}.{q,k,v,o}.lora_{down,up}.weight
        diffusion_model.blocks.N.ffn.{0,2}.lora_{down,up}.weight
        ...and corresponding .alpha scalars

    For cross-attention keys ``attn2.to_k`` and ``attn2.to_v``, deltas
    are computed separately then concatenated to match the fused
    ``attn2.to_kv`` weight in the MAX state dict.

    All arithmetic is done in float32 for numerical stability; the
    result is cast back to the base parameter's dtype (typically
    bfloat16).

    Args:
        state_dict: The base state dict (modified in-place and returned).
        lora_path: Path to the ``.safetensors`` LoRA file.
        lora_scale: Global multiplier for LoRA strength.

    Returns:
        The modified state_dict.
    """
    # safetensors.numpy can't read bfloat16; fall back to torch if needed.
    try:
        from safetensors.numpy import load_file

        lora_weights = load_file(str(lora_path))
    except TypeError:
        from safetensors.torch import load_file as load_file_torch

        torch_weights = load_file_torch(str(lora_path), device="cpu")
        lora_weights = {k: v.float().numpy() for k, v in torch_weights.items()}
    logger.info(
        "Merging LoRA from %s (%d keys, scale=%.2f)",
        lora_path,
        len(lora_weights),
        lora_scale,
    )

    # Group LoRA weights by base key
    # {base_key: {"down": ndarray, "up": ndarray, "alpha": float}}
    groups: dict[str, dict[str, Any]] = {}
    # Collect bias deltas (.diff_b) and norm deltas (.diff) separately
    bias_deltas: dict[str, np.ndarray] = {}
    norm_deltas: dict[str, np.ndarray] = {}
    for key, tensor in lora_weights.items():
        if key.endswith(".lora_down.weight"):
            base = key[: -len(".lora_down.weight")]
            groups.setdefault(base, {})["down"] = tensor
        elif key.endswith(".lora_up.weight"):
            base = key[: -len(".lora_up.weight")]
            groups.setdefault(base, {})["up"] = tensor
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            groups.setdefault(base, {})["alpha"] = float(tensor.flat[0])
        elif key.endswith(".diff_b"):
            # Bias delta (Wan 2.1 distill LoRA)
            base = key[: -len(".diff_b")]
            bias_deltas[base] = np.asarray(tensor, dtype=np.float32)
        elif key.endswith(".diff"):
            # Weight delta (e.g. norm_q.diff, norm_k.diff).
            # Only merge 1D deltas (norms); skip multi-dim deltas
            # (patch_embedding, head) which need permutation handling.
            base = key[: -len(".diff")]
            arr = np.asarray(tensor, dtype=np.float32)
            if arr.ndim == 1:
                norm_deltas[base] = arr

    # Collect cross-attn K/V deltas per block for fusion
    # key: "blocks.{i}.attn2" -> {"k_delta": ndarray, "v_delta": ndarray}
    kv_pending: dict[str, dict[str, np.ndarray]] = {}

    merged_count = 0
    skipped_keys: list[str] = []

    for base_key, parts in groups.items():
        down = parts.get("down")
        up = parts.get("up")
        if down is None or up is None:
            continue

        down_f32 = np.asarray(down, dtype=np.float32)
        up_f32 = np.asarray(up, dtype=np.float32)
        rank = down_f32.shape[0]
        alpha = float(parts.get("alpha") or rank)
        scale = lora_scale * (alpha / rank)

        # delta = scale * (up @ down)  — shape [out_dim, in_dim]
        delta = scale * (up_f32 @ down_f32)

        # Remap the LoRA key to MAX naming
        remapped = _remap_lora_key(base_key)

        # Handle attn2.to_k / attn2.to_v → fused attn2.to_kv
        if ".attn2.to_k" in remapped:
            kv_base = remapped.replace(".attn2.to_k", ".attn2")
            kv_pending.setdefault(kv_base, {})["k_delta"] = delta
            continue
        elif ".attn2.to_v" in remapped:
            kv_base = remapped.replace(".attn2.to_v", ".attn2")
            kv_pending.setdefault(kv_base, {})["v_delta"] = delta
            continue

        # Direct merge
        sd_key = remapped + ".weight"
        if sd_key not in state_dict:
            skipped_keys.append(sd_key)
            continue

        _merge_delta_into(state_dict, sd_key, delta)
        merged_count += 1

    # Fuse K/V deltas and merge into attn2.to_kv
    for kv_base, deltas in kv_pending.items():
        k_delta = deltas.get("k_delta")
        v_delta = deltas.get("v_delta")
        if k_delta is None or v_delta is None:
            logger.warning("Incomplete K/V LoRA pair for %s, skipping", kv_base)
            continue

        kv_delta = np.concatenate([k_delta, v_delta], axis=0)
        sd_key = kv_base + ".to_kv.weight"
        if sd_key not in state_dict:
            skipped_keys.append(sd_key)
            continue

        _merge_delta_into(state_dict, sd_key, kv_delta)
        merged_count += 1

    # Merge bias deltas (.diff_b) — Wan 2.1 distill LoRA
    for base_key, delta in bias_deltas.items():
        remapped = _remap_lora_key(base_key)
        sd_key = remapped + ".bias"
        if sd_key not in state_dict:
            skipped_keys.append(sd_key)
            continue
        _merge_delta_into(state_dict, sd_key, lora_scale * delta)
        merged_count += 1

    # Merge norm deltas (.diff) — e.g. norm_q.diff, norm_k.diff
    for base_key, delta in norm_deltas.items():
        remapped = _remap_lora_key(base_key)
        sd_key = remapped + ".weight"
        if sd_key not in state_dict:
            skipped_keys.append(sd_key)
            continue
        _merge_delta_into(state_dict, sd_key, lora_scale * delta)
        merged_count += 1

    if skipped_keys:
        logger.warning(
            "LoRA: %d keys not found in state_dict: %s",
            len(skipped_keys),
            skipped_keys[:5],
        )
    logger.info("LoRA merge complete: %d parameters updated", merged_count)
    return state_dict


def _merge_delta_into(
    state_dict: dict[str, Any],
    key: str,
    delta: np.ndarray,
) -> None:
    """Add float32 delta to an existing state_dict entry, respecting dtype.

    The base weights are typically bfloat16 (which numpy can't handle
    natively), so we cast to float32 via cast_dlpack_to, add the delta,
    then cast back.
    """
    from max.driver import CPU
    from max.dtype import DType
    from max.graph.buffer_utils import cast_dlpack_to

    cpu = CPU()

    # Cast bfloat16 → float32 so numpy can operate on it
    base_f32_tensor = cast_dlpack_to(
        state_dict[key], DType.bfloat16, DType.float32, cpu
    )
    base_f32 = np.from_dlpack(base_f32_tensor)
    result_f32 = np.ascontiguousarray(base_f32 + delta)

    # Cast back to bfloat16
    state_dict[key] = cast_dlpack_to(
        result_f32, DType.float32, DType.bfloat16, cpu
    )
