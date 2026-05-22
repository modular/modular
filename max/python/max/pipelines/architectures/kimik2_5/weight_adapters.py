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

"""Weight adapters for Kimi K2.5 components.

Maps HuggingFace checkpoint keys to the MAX module state_dict keys for:
- Language model (``language_model.*`` → DeepseekV3)
- Vision encoder (``vision_tower.*`` + projector → Transformer)

Two top-level adapters correspond to the two architectures in ``arch.py``:

- :func:`convert_kimik2_5_safetensor_state_dict` —
  ``KimiK25ForConditionalGeneration`` (e.g. nvidia/Kimi-K2.5-NVFP4).
  Uses ``mm_projector.*`` projector naming; drops ``.k_scale``/``.v_scale``.
- :func:`convert_kimivl_safetensor_state_dict` —
  ``KimiVLForConditionalGeneration`` (e.g. moonshotai/Kimi-VL-A3B).
  Uses ``multi_modal_projector.*`` projector naming.

Both adapters share :func:`_convert_merged_state_dict`, which processes
vision and language keys in a single loop over the raw checkpoint.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import WeightData, Weights
from max.nn.kernels import mxfp4_preshuffle_b_5d
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language model rename map
# ---------------------------------------------------------------------------

KIMI_SAFETENSOR_MAP: dict[str, str] = {
    "language_model.model.": "language_model.",
    "gate.weight": "gate.gate_score.weight",
    "weight_scale_inv": "weight_scale",
}

# ---------------------------------------------------------------------------
# Vision tower + projector rename map
# ---------------------------------------------------------------------------

# String-replacement mapping for vision-related checkpoint keys.
# Replacements are applied in insertion order.  The first three entries
# swap HuggingFace prefixes for MAX module prefixes; the remaining
# entries rename sub-keys within those modules.
KIMIK2_5_VISION_MAPPING: dict[str, str] = {
    # Prefix swaps (must come first)
    "vision_tower.": "vision_encoder.",
    "mm_projector.": "vision_encoder.patch_merger.",
    "multi_modal_projector.": "vision_encoder.patch_merger.",
    # Vision tower sub-key renames
    "encoder.final_layernorm.": "encoder.norm.",
    ".mlp.fc0.": ".mlp.up_proj.",
    ".mlp.fc1.": ".mlp.down_proj.",
    # Projector sub-key renames — two conventions:
    #   mm_projector: Sequential indices (proj.0 / proj.2)
    #   multi_modal_projector: underscore names (linear_1 / linear_2)
    "proj.0.": "linear1.",
    "proj.2.": "linear2.",
    "linear_1.": "linear1.",
    "linear_2.": "linear2.",
}

# Regex patterns for attention renames that need the block index captured.
# HF: ``blocks.N.wqkv.*`` → MAX: ``blocks.N.attn.wqkv.*``
# HF: ``blocks.N.wo.*``   → MAX: ``blocks.N.attn.wo.*``
_ATTN_RENAME_PATTERNS = [
    (re.compile(r"blocks\.(\d+)\.wqkv\."), r"blocks.\1.attn.wqkv."),
    (re.compile(r"blocks\.(\d+)\.wo\."), r"blocks.\1.attn.wo."),
]

_VISION_PREFIXES = ("vision_tower.", "mm_projector.", "multi_modal_projector.")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rename_vision_key(checkpoint_name: str) -> str:
    """Apply KIMIK2_5_VISION_MAPPING and attention regex renames."""
    name = checkpoint_name
    for before, after in KIMIK2_5_VISION_MAPPING.items():
        name = name.replace(before, after)
    for pattern, replacement in _ATTN_RENAME_PATTERNS:
        name = pattern.sub(replacement, name)
    return name


def _cast_vision_weight(checkpoint_name: str, weight: Weights) -> WeightData:
    """Return WeightData, casting floats (non-FP8, non-scale) to bfloat16."""
    weight_data = weight.data()
    is_scale = checkpoint_name.endswith(
        ".weight_scale"
    ) or checkpoint_name.endswith(".input_scale")
    if (
        weight_data.dtype.is_float()
        and not weight_data.dtype.is_float8()
        and not is_scale
    ):
        weight_data = weight_data.astype(DType.bfloat16)
    return weight_data


_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.layers\.\d+\.mlp\.experts)"
    r"\.(?P<idx>\d+)"
    r"\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def _stack_experts(arrays: list[np.ndarray]) -> np.ndarray:
    """Build [E, N, K_BYTES] from per-expert uint8 views via a
    multi-threaded memcpy stack
    """
    E = len(arrays)
    N, K_BYTES = arrays[0].shape
    dest = np.empty((E, N, K_BYTES), dtype=np.uint8)

    def _copy(j: int) -> None:
        dest[j] = arrays[j]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(_copy, range(E)):
            pass
    return dest


def _warm_expert_safetensor_caches(expert_weights: list[Weights]) -> None:
    """Pre-fault safetensor file pages into the OS page cache.

    The default mmap demand-fault-in pattern touches ~4 KB pages on first
    access with poor kernel readahead, which on cold-cache loads can cost
    far more than the actual shuffle work. Hint sequential access and
    issue a chunked sequential read up front so all subsequent reads hit
    RAM. Linux-only hints; silently no-op on other platforms.
    """
    paths: set[str] = set()
    for w in expert_weights:
        filepaths = getattr(w, "_filepaths", None)
        idx_map = getattr(w, "_tensors_to_file_idx", None)
        if filepaths is None or idx_map is None:
            continue
        idx = idx_map.get(w.name)
        if idx is None:
            continue
        paths.add(os.fspath(filepaths[idx]))

    if not paths:
        return

    def _warm(path: str) -> None:
        with open(path, "rb") as f:
            fd = f.fileno()
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            except (AttributeError, OSError):
                pass
            while f.read(64 * 1024 * 1024):
                pass

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        for _ in pool.map(_warm, paths):
            pass
    print(
        f"[MXFP4 preshuffle] warmed page cache for {len(paths)} "
        f"safetensor file(s) in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


def _batch_preshuffle_experts(
    state_dict: dict[str, Weights],
) -> dict[str, np.ndarray]:
    """Group expert weights by ``(layer prefix, proj)`` and shuffle each
    group on-GPU via ``mxfp4_preshuffle_b_5d`` (one ``[E, N, K_BYTES]``
    op per group).

    Returns a dict mapping original checkpoint name to that expert's
    slice of the shuffled stack.
    """
    # Group: (prefix, proj) -> {expert_idx: (checkpoint_name, weight)}
    groups: dict[tuple[str, str], dict[int, tuple[str, Weights]]] = {}
    for name, weight in state_dict.items():
        m = _EXPERT_WEIGHT_RE.match(name)
        if m:
            key = (m.group("prefix"), m.group("proj"))
            groups.setdefault(key, {})[int(m.group("idx"))] = (name, weight)

    if not groups:
        return {}

    _warm_expert_safetensor_caches(
        [w for group in groups.values() for _, w in group.values()]
    )

    if accelerator_count() == 0:
        raise RuntimeError(
            "MXFP4 expert preshuffle requires a GPU device; none found."
        )
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    # Cache the compiled one-op graph per `(E, N, K_BYTES)` shape — Kimi
    # has only a handful of unique shapes (gate/up share one, down has
    # another) so total compile cost stays negligible.
    model_cache: dict[tuple[int, int, int], Model] = {}

    def _get_model(E: int, N: int, K_BYTES: int) -> Model:
        key = (E, N, K_BYTES)
        if key in model_cache:
            return model_cache[key]
        in_type = TensorType(
            dtype=DType.uint8,
            shape=(E, N, K_BYTES),
            device=device_ref,
        )
        with Graph("mxfp4_preshuffle_b_5d_eager", input_types=(in_type,)) as g:
            raw = g.inputs[0].tensor
            g.output(mxfp4_preshuffle_b_5d(raw))
        model = session.load(g)
        model_cache[key] = model
        return model

    result: dict[str, np.ndarray] = {}
    totals = {
        "materialize": 0.0,
        "stack": 0.0,
        "upload": 0.0,
        "gpu": 0.0,
        "download": 0.0,
    }
    for (prefix, proj), experts in groups.items():
        idxs = sorted(experts.keys())

        t0 = time.perf_counter()
        arrays: list[np.ndarray] = []
        skip = False
        for i in idxs:
            data = experts[i][1].data()
            if data.dtype != DType.uint8:
                skip = True
                break
            arrays.append(np.from_dlpack(data.data))
        if skip or not arrays or arrays[0].ndim != 2:
            continue
        N, K_BYTES = arrays[0].shape
        if N % 16 != 0 or K_BYTES % 64 != 0:
            continue
        t1 = time.perf_counter()

        # Multi-threaded stack: each expert is in its own safetensor,
        # so they aren't contiguous in any single mmap region and we
        # have to copy. Threads parallelize the per-expert memcpy.
        stacked = _stack_experts(arrays)
        E = stacked.shape[0]
        t2 = time.perf_counter()

        model = _get_model(E, N, K_BYTES)
        raw_buf = Buffer.from_dlpack(stacked).to(device)
        t3 = time.perf_counter()

        outputs = model.execute(raw_buf)
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        assert isinstance(out, Buffer)
        t4 = time.perf_counter()

        # to_numpy() blocks on any pending GPU work, so it captures both
        # the GPU→host transfer and any GPU work the execute() above
        # may have only enqueued.
        shuffled = out.to_numpy()
        t5 = time.perf_counter()

        totals["materialize"] += t1 - t0
        totals["stack"] += t2 - t1
        totals["upload"] += t3 - t2
        totals["gpu"] += t4 - t3
        totals["download"] += t5 - t4

        print(
            f"[MXFP4 preshuffle] {prefix}.*.{proj}.weight "
            f"shape=({E},{N},{K_BYTES}) "
            f"materialize={t1 - t0:.2f}s stack={t2 - t1:.2f}s "
            f"upload={t3 - t2:.2f}s gpu={t4 - t3:.2f}s "
            f"download={t5 - t4:.2f}s",
            flush=True,
        )
        for j, i in enumerate(idxs):
            result[experts[i][0]] = np.ascontiguousarray(shuffled[j])

    print(
        f"[MXFP4 preshuffle] totals: "
        f"materialize={totals['materialize']:.1f}s "
        f"stack={totals['stack']:.1f}s "
        f"upload={totals['upload']:.1f}s "
        f"gpu={totals['gpu']:.1f}s "
        f"download={totals['download']:.1f}s",
        flush=True,
    )
    return result


def _convert_merged_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    drop_kv_scales: bool = False,
) -> dict[str, WeightData]:
    """Convert a full Kimi checkpoint to MAX module keys in a single pass.

    Vision keys (``vision_tower.*``, ``mm_projector.*``,
    ``multi_modal_projector.*``) are renamed and cast to bfloat16.
    Language-model keys (``language_model.*``) are renamed via
    :data:`KIMI_SAFETENSOR_MAP` with MTP-layer pruning.
    All other checkpoint keys are silently ignored.

    Args:
        state_dict: Raw HuggingFace checkpoint.
        huggingface_config: Model config.  If the config wraps a language-model
            sub-config under ``.text_config`` (as in KimiK25Config), that
            sub-config is used for MTP layer-index lookup; otherwise the
            top-level config is used directly.
        drop_kv_scales: When ``True``, language-model keys ending in
            ``.k_scale`` or ``.v_scale`` are dropped (required for
            FP8 nvidia/Kimi-K2.5-NVFP4 checkpoints).

    Returns:
        Single merged state dict whose keys match the ``KimiK2_5`` module:
        ``vision_encoder.*`` and ``language_model.*``.
    """
    print(
        f"[MXFP4 preshuffle] _convert_merged_state_dict ENTRY "
        f"n_keys={len(state_dict)}",
        flush=True,
    )
    # Batched preshuffle: group all per-expert MXFP4 weights by
    # (layer, proj) and shuffle each group in a single numpy call. ~385x
    # less Python overhead per shuffle and lets numpy SIMD-vectorize over
    # the new E dim.
    shuffled_experts = _batch_preshuffle_experts(state_dict)
    print(
        f"[MXFP4 preshuffle] batched shuffle: {len(shuffled_experts)} "
        f"expert weights ready",
        flush=True,
    )

    llm_config = getattr(huggingface_config, "text_config", huggingface_config)
    mtp_layer_idx = llm_config.num_hidden_layers

    result: dict[str, WeightData] = {}

    for checkpoint_name, weight in state_dict.items():
        # --- Vision tower + projector keys ---
        if any(checkpoint_name.startswith(p) for p in _VISION_PREFIXES):
            name = _rename_vision_key(checkpoint_name)
            result[name] = _cast_vision_weight(checkpoint_name, weight)
            continue

        # --- Language model keys ---
        if not checkpoint_name.startswith("language_model."):
            continue

        name = checkpoint_name
        for before, after in KIMI_SAFETENSOR_MAP.items():
            name = name.replace(before, after)

        # TODO(E2EOPT-673): Support MTP.  Prune the speculative MTP layer.
        # See https://github.com/deepseek-ai/DeepSeek-V3/blob/4592be48c07f036b32ef971474068aebc489e3e7/inference/convert.py#L53-L54
        if name.startswith(f"language_model.layers.{mtp_layer_idx}."):
            continue
        if name.endswith(".self_attn.rotary_emb.inv_freq"):
            continue
        if drop_kv_scales and (
            name.endswith(".k_scale") or name.endswith(".v_scale")
        ):
            continue

        data = weight.data()
        # Safetensors stores E8M0 scales as uint8; reinterpret.
        if name.endswith(".weight_scale") and data.dtype == DType.uint8:
            data = dataclasses.replace(data, dtype=DType.float8_e8m0fnu)
        # AMD MXFP4 expert weight preshuffle: if this key was batch-shuffled
        # in the pre-pass, swap the raw bytes for the shuffled bytes. The
        # batched math is byte-equivalent to per-expert shuffle because the
        # n0 stride depends only on K_BYTES; stacking pre-shuffled per-expert
        # bytes is the same as shuffling the stacked tensor.
        if checkpoint_name in shuffled_experts:
            data = WeightData.from_numpy(
                shuffled_experts[checkpoint_name], name=data.name
            )
        result[name] = data

    return result


# ---------------------------------------------------------------------------
# Public adapters — one per architecture in arch.py
# ---------------------------------------------------------------------------


def convert_kimik2_5_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig | None = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert a ``KimiK25ForConditionalGeneration`` safetensor checkpoint.

    Handles the ``mm_projector.*`` projector naming convention used by
    nvidia/Kimi-K2.5-NVFP4 checkpoints, and drops FP8 kv-scale weights.

    Args:
        state_dict: Raw checkpoint as dict from weight name to :class:`Weights`.
        huggingface_config: HuggingFace config (required for MTP pruning).

    Returns:
        State dict with ``vision_encoder.*`` and ``language_model.*`` keys
        matching the ``KimiK2_5`` MAX module.
    """
    if huggingface_config is None:
        raise ValueError(
            "convert_kimik2_5_safetensor_state_dict requires huggingface_config"
            " for language model weight conversion (MTP layer pruning)."
        )
    return _convert_merged_state_dict(
        state_dict,
        huggingface_config=huggingface_config,
        drop_kv_scales=True,
    )


def convert_kimivl_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig | None = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert a ``KimiVLForConditionalGeneration`` safetensor checkpoint.

    Handles the ``multi_modal_projector.*`` projector naming convention used
    by moonshotai/Kimi-VL-A3B checkpoints.

    Args:
        state_dict: Raw checkpoint as dict from weight name to :class:`Weights`.
        huggingface_config: HuggingFace config (required for MTP pruning).

    Returns:
        State dict with ``vision_encoder.*`` and ``language_model.*`` keys
        matching the ``KimiK2_5`` MAX module.
    """
    if huggingface_config is None:
        raise ValueError(
            "convert_kimivl_safetensor_state_dict requires huggingface_config"
            " for language model weight conversion (MTP layer pruning)."
        )
    return _convert_merged_state_dict(
        state_dict,
        huggingface_config=huggingface_config,
        drop_kv_scales=False,
    )


# ---------------------------------------------------------------------------
# Eagle3 draft checkpoint adapter
# ---------------------------------------------------------------------------

# Eagle3 checkpoint key prefix -> Eagle3KimiK25 module path.
_EAGLE3_KEY_MAP: dict[str, str] = {
    "layers.0.hidden_norm.": "hidden_norm.",
    "layers.0.input_layernorm.": "decoder_layer.input_layernorm.",
    "layers.0.self_attn.": "decoder_layer.self_attn.",
    "layers.0.post_attention_layernorm.": "decoder_layer.post_attention_layernorm.",
    "layers.0.mlp.": "decoder_layer.mlp.",
}


def convert_eagle3_draft_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert an Eagle3 draft checkpoint to Eagle3KimiK25 module keys.

    The Eagle3 checkpoint (``nvidia/Kimi-K2.5-Thinking-Eagle3``) has keys:
    ``fc.*``, ``layers.0.*``, ``norm.*``, ``lm_head.*``.

    All weights are loaded; ``norm`` and ``lm_head`` are kept independent
    from the target model.  Only ``embed_tokens``
    is shared from the target.

    Args:
        state_dict: Raw Eagle3 checkpoint.

    Returns:
        State dict with keys matching ``Eagle3KimiK25`` module hierarchy.
    """
    result: dict[str, WeightData] = {}

    for name, weight in state_dict.items():
        # Apply key mapping
        mapped = False
        for before, after in _EAGLE3_KEY_MAP.items():
            if name.startswith(before):
                new_name = after + name[len(before) :]
                result[new_name] = weight.data()
                mapped = True
                break

        if not mapped:
            # fc.*, norm.*, lm_head.* pass through directly
            result[name] = weight.data()

    return result


# ---------------------------------------------------------------------------
# Llama-style Eagle3 draft checkpoint adapter
# ---------------------------------------------------------------------------

# Llama Eagle3 checkpoint key prefix -> Eagle3MHAKimiK25 module path.
# The MHA draft module's layer is flat (single block, no ``decoder_layer``
# namespace), so the mapping strips the single-layer prefix and inlines
# norms.
#
# Checkpoints in the wild use different single-layer prefix conventions
# (``midlayer.``, ``model.layers.0.``, ``layers.0.``). Handle all three.
_LLAMA_EAGLE3_KEY_MAP: dict[str, str] = {
    "midlayer.hidden_norm.": "hidden_norm.",
    "midlayer.input_layernorm.": "input_layernorm.",
    "midlayer.self_attn.": "self_attn.",
    "midlayer.post_attention_layernorm.": "post_attention_layernorm.",
    "midlayer.mlp.": "mlp.",
    "model.layers.0.hidden_norm.": "hidden_norm.",
    "model.layers.0.input_layernorm.": "input_layernorm.",
    "model.layers.0.self_attn.": "self_attn.",
    "model.layers.0.post_attention_layernorm.": "post_attention_layernorm.",
    "model.layers.0.mlp.": "mlp.",
    "model.norm.": "norm.",
    "model.embed_tokens.": "embed_tokens.",
    "layers.0.hidden_norm.": "hidden_norm.",
    "layers.0.input_layernorm.": "input_layernorm.",
    "layers.0.self_attn.": "self_attn.",
    "layers.0.post_attention_layernorm.": "post_attention_layernorm.",
    "layers.0.mlp.": "mlp.",
}


def convert_llama_eagle3_draft_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert a ``LlamaForCausalLMEagle3`` checkpoint to ``Eagle3MHAKimiK25``.

    Handles both ``model.*``-prefixed (standard HF Llama) and
    ``layers.0.*``-prefixed (EAGLE-export) checkpoints. ``fc.*``,
    ``norm.*``, and ``lm_head.*`` pass through.
    """
    result: dict[str, WeightData] = {}
    for name, weight in state_dict.items():
        mapped = False
        for before, after in _LLAMA_EAGLE3_KEY_MAP.items():
            if name.startswith(before):
                new_name = after + name[len(before) :]
                result[new_name] = weight.data()
                mapped = True
                break
        if not mapped:
            # fc.*, norm.*, lm_head.* and any other top-level keys.
            result[name] = weight.data()
    return result
