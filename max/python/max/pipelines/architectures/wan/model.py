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

import logging
import threading
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel
from max.profiler import Tracer

from .model_config import WanConfig
from .wan_transformer import (
    WanTransformerBlock,
    WanTransformerBlockSequence,
    WanTransformerPostProcess,
    WanTransformerPreProcess,
)

logger = logging.getLogger(__name__)

# Weight key remapping from diffusers -> MAX module naming
_KEY_REMAP = [
    (".attn1.to_out.0.", ".attn1.to_out."),
    (".attn2.to_out.0.", ".attn2.to_out."),
    (".ffn.net.0.proj.", ".ffn.proj."),
    (".ffn.net.2.", ".ffn.linear_out."),
    # Image embedder GELU FFN: diffusers nested structure → flat Linear layers
    ("image_embedder.ff.net.0.proj.", "image_embedder.ff_proj."),
    ("image_embedder.ff.net.2.", "image_embedder.ff_out."),
]

# Keys to skip (non-persistent buffers computed at runtime)
_SKIP_PREFIXES = ("rope.freqs_cos", "rope.freqs_sin")


def _remap_state_dict(
    weights: Weights,
    target_dtype: DType = DType.bfloat16,
) -> dict[str, Any]:
    """Remap diffusers weight keys to MAX module naming, permute Conv3d,
    and cast weights to target dtype.

    Some WAN checkpoints store weights as float32 (A14B), others as
    bfloat16 (5B). We cast all to target_dtype to match the module
    parameter declarations.
    """
    state_dict: dict[str, Any] = {}

    # First pass: collect all weights with key remapping.
    raw_dict: dict[str, Any] = {}
    for key, value in weights.items():
        if any(key.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        new_key = key
        for old, new in _KEY_REMAP:
            new_key = new_key.replace(old, new)

        tensor = value.data()

        # Conv3d weight permutation for patch_embedding
        # Diffusers: [F, C, D, H, W] (PyTorch FCDHW)
        # MAX Conv3d(permute=False): [D, H, W, C, F] (QRSCF)
        if new_key == "patch_embedding.weight" and len(tensor.shape) == 5:
            buf = tensor.to_buffer() if hasattr(tensor, "to_buffer") else tensor
            t_f32 = cast_dlpack_to(buf, tensor.dtype, DType.float32, CPU())
            permuted: WeightData | np.ndarray = np.ascontiguousarray(
                np.from_dlpack(t_f32).transpose(2, 3, 4, 1, 0)
            )
            raw_dict[new_key] = permuted
        else:
            raw_dict[new_key] = tensor

    # Second pass: fuse attn2.to_k + attn2.to_v into attn2.to_kv
    fused_keys: set[str] = set()
    for key in list(raw_dict.keys()):
        if ".attn2.to_k." in key:
            k_key = key
            v_key = key.replace(".attn2.to_k.", ".attn2.to_v.")
            kv_key = key.replace(".attn2.to_k.", ".attn2.to_kv.")
            if v_key in raw_dict:
                k_data = raw_dict[k_key]
                v_data = raw_dict[v_key]
                k_buf = (
                    k_data.to_buffer()
                    if hasattr(k_data, "to_buffer")
                    else k_data
                )
                v_buf = (
                    v_data.to_buffer()
                    if hasattr(v_data, "to_buffer")
                    else v_data
                )
                k_f32 = cast_dlpack_to(
                    k_buf, k_data.dtype, DType.float32, CPU()
                )
                v_f32 = cast_dlpack_to(
                    v_buf, v_data.dtype, DType.float32, CPU()
                )
                k_np = np.from_dlpack(k_f32)
                v_np = np.from_dlpack(v_f32)
                kv_np = np.ascontiguousarray(
                    np.concatenate([k_np, v_np], axis=0)
                )
                state_dict[kv_key] = kv_np
                fused_keys.add(k_key)
                fused_keys.add(v_key)

    for key, tensor in raw_dict.items():
        if key not in fused_keys:
            state_dict[key] = tensor

    cpu_device = CPU()
    for key in state_dict:
        tensor = state_dict[key]
        if isinstance(tensor, WeightData):
            src_dtype = tensor.dtype
            dlpack_obj = tensor.to_buffer()
        else:
            src_dtype = DType.float32
            dlpack_obj = tensor
        state_dict[key] = cast_dlpack_to(
            dlpack_obj, src_dtype, target_dtype, cpu_device
        )

    return state_dict


def _get_1d_rotary_pos_embed_np(
    dim: int,
    pos: np.ndarray,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D rotary position embeddings (numpy, for eager RoPE)."""
    freq_exponent = np.arange(0, dim, 2, dtype=np.float64) / dim
    freqs = 1.0 / (theta**freq_exponent)
    angles = np.outer(pos.astype(np.float64), freqs)
    cos_emb = np.cos(angles).astype(np.float32)
    sin_emb = np.sin(angles).astype(np.float32)
    cos_emb = np.repeat(cos_emb, 2, axis=1)
    sin_emb = np.repeat(sin_emb, 2, axis=1)
    return cos_emb, sin_emb


@lru_cache(maxsize=8)
def _compute_wan_rope_cached(
    num_frames: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D RoPE cos/sin arrays for Wan transformer (cached by resolution)."""
    p_t, p_h, p_w = patch_size
    ppf = num_frames // p_t
    pph = height // p_h
    ppw = width // p_w

    d_h = (head_dim // 3 // 2) * 2
    d_w = d_h
    d_t = head_dim - d_h - d_w

    cos_t, sin_t = _get_1d_rotary_pos_embed_np(d_t, np.arange(ppf), theta)
    cos_h, sin_h = _get_1d_rotary_pos_embed_np(d_h, np.arange(pph), theta)
    cos_w, sin_w = _get_1d_rotary_pos_embed_np(d_w, np.arange(ppw), theta)

    cos_t = np.broadcast_to(cos_t[:, None, None, :], (ppf, pph, ppw, d_t))
    sin_t = np.broadcast_to(sin_t[:, None, None, :], (ppf, pph, ppw, d_t))
    cos_h = np.broadcast_to(cos_h[None, :, None, :], (ppf, pph, ppw, d_h))
    sin_h = np.broadcast_to(sin_h[None, :, None, :], (ppf, pph, ppw, d_h))
    cos_w = np.broadcast_to(cos_w[None, None, :, :], (ppf, pph, ppw, d_w))
    sin_w = np.broadcast_to(sin_w[None, None, :, :], (ppf, pph, ppw, d_w))

    rope_cos = np.concatenate([cos_t, cos_h, cos_w], axis=-1)
    rope_sin = np.concatenate([sin_t, sin_h, sin_w], axis=-1)

    seq_len = ppf * pph * ppw
    rope_cos = np.ascontiguousarray(rope_cos.reshape(seq_len, head_dim))
    rope_sin = np.ascontiguousarray(rope_sin.reshape(seq_len, head_dim))
    return rope_cos, rope_sin


class BlockLevelModel:
    """Executes transformer forward pass as pre -> N blocks -> post.

    Supports two modes:

    * **Combined** (``combined_blocks`` is set): All transformer blocks
      are compiled into a single ``Model`` graph, so the runtime
      allocates one shared workspace.
    * **Per-block** (``blocks`` list): Each block is a separate
      ``Model``.  Kept for backwards compatibility and MoE weight-swap.
    """

    def __init__(
        self,
        pre: Model,
        blocks: list[Model],
        post: Model,
        *,
        combined_blocks: Model | None = None,
    ) -> None:
        self.pre = pre
        self.blocks = blocks
        self.post = post
        self.combined_blocks = combined_blocks

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        i2v_condition: Buffer | None = None,
    ) -> Buffer:
        pre_args = [hidden_states, timestep, encoder_hidden_states]
        if i2v_condition is not None:
            pre_args.append(i2v_condition)
        with Tracer("dit_pre"):
            pre_out = self.pre.execute(*pre_args)
            hs, temb, timestep_proj, text_emb = (
                pre_out[0],
                pre_out[1],
                pre_out[2],
                pre_out[3],
            )
        with Tracer("dit_blocks"):
            if self.combined_blocks is not None:
                block_out = self.combined_blocks.execute(
                    hs, text_emb, timestep_proj, rope_cos, rope_sin
                )
                hs = block_out[0]
            else:
                for block in self.blocks:
                    block_out = block.execute(
                        hs, text_emb, timestep_proj, rope_cos, rope_sin
                    )
                    hs = block_out[0]
        with Tracer("dit_post"):
            post_out = self.post.execute(hs, temb, spatial_shape)
        return post_out[0]


class WanTransformerModel(ComponentModel):
    """MAX-native Wan DiT interface with block-level compilation.

    Each block is compiled independently so only one block's workspace
    is live at any time, keeping peak VRAM low.
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession | None = None,
        eager_load: bool = True,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = WanConfig.generate(config, encoding, devices)
        self.config.dtype = DType.bfloat16
        self._state_dict: dict[str, Any] | None = None
        self.model: BlockLevelModel | None = None
        self._weight_registry_cache: dict[
            int,
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
        ] = {}
        self.session = session or InferenceSession(devices=devices)
        self._load_lock = threading.Lock()
        if eager_load:
            self.prepare_state_dict()

    def _ensure_state_dict(self) -> dict[str, Any]:
        if self._state_dict is None:
            if self.weights is None:
                raise RuntimeError(
                    "WanTransformerModel weights are unavailable "
                    "while state_dict is not initialized."
                )
            self._state_dict = _remap_state_dict(
                self.weights, target_dtype=DType.bfloat16
            )
            self.weights = None  # type: ignore[assignment]

        return self._state_dict

    def prepare_state_dict(self) -> dict[str, Any]:
        """Materialize the remapped state dict without compiling graphs."""
        with self._load_lock:
            return self._ensure_state_dict()

    def _split_state_dict(
        self, state_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Split flat state dict into pre/block/post weight groups."""
        pre_weights: dict[str, Any] = {}
        post_weights: dict[str, Any] = {}
        block_weights_list: list[dict[str, Any]] = [
            {} for _ in range(self.config.num_layers)
        ]

        for key, value in state_dict.items():
            if key.startswith("patch_embedding.") or key.startswith(
                "condition_embedder."
            ):
                pre_weights[key] = value
            elif key.startswith("blocks."):
                rest = key[len("blocks.") :]
                dot = rest.index(".")
                block_idx = int(rest[:dot])
                sub_key = rest[dot + 1 :]
                block_weights_list[block_idx][sub_key] = value
            else:
                post_weights[key] = value

        return pre_weights, block_weights_list, post_weights

    def _build_weight_registries(
        self, state_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Build module-level weight registries for pre/block/post."""
        dim = self.config.num_attention_heads * self.config.attention_head_dim
        dtype = self.config.dtype
        dev_ref = DeviceRef.from_device(self.config.device)
        pre_weights, block_weights_list, post_weights = self._split_state_dict(
            state_dict
        )

        pre_module = WanTransformerPreProcess(
            self.config, dtype=dtype, device=dev_ref
        )
        pre_module.load_state_dict(pre_weights, weight_alignment=1, strict=True)

        block_registries: list[dict[str, Any]] = []
        block_module = WanTransformerBlock(
            dim=dim,
            ffn_dim=self.config.ffn_dim,
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.attention_head_dim,
            text_dim=dim,
            cross_attn_norm=self.config.cross_attn_norm,
            eps=self.config.eps,
            added_kv_proj_dim=self.config.added_kv_proj_dim,
            dtype=dtype,
            device=dev_ref,
        )
        for block_weights in block_weights_list:
            block_module.load_state_dict(
                block_weights, weight_alignment=1, strict=True
            )
            block_registries.append(block_module.state_dict())

        post_module = WanTransformerPostProcess(
            self.config, dtype=dtype, device=dev_ref
        )
        post_module.load_state_dict(
            post_weights, weight_alignment=1, strict=True
        )

        return (
            pre_module.state_dict(),
            block_registries,
            post_module.state_dict(),
        )

    def _get_cached_weight_registries(
        self, state_dict: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Return weight registries, caching by state_dict identity."""
        target_state_dict = state_dict or self._ensure_state_dict()
        cache_key = id(target_state_dict)
        cached = self._weight_registry_cache.get(cache_key)
        if cached is not None:
            return cached

        registries = self._build_weight_registries(target_state_dict)
        self._weight_registry_cache[cache_key] = registries
        return registries

    def reload_model_weights(
        self, state_dict: dict[str, Any] | None = None
    ) -> None:
        """Reload weights into already-compiled models for MoE weight switching."""
        with self._load_lock:
            if self.model is None:
                raise RuntimeError("Wan transformer model not compiled.")

            pre_registry, block_registries, post_registry = (
                self._get_cached_weight_registries(state_dict)
            )

            self.model.pre.reload(pre_registry)
            if self.model.combined_blocks is not None:
                # Build combined registry with LayerList prefixes.
                combined_registry: dict[str, Any] = {}
                for i, block_reg in enumerate(block_registries):
                    for k, v in block_reg.items():
                        combined_registry[f"blocks.{i}.{k}"] = v
                self.model.combined_blocks.reload(combined_registry)
            else:
                for compiled_block, block_registry in zip(
                    self.model.blocks, block_registries, strict=True
                ):
                    compiled_block.reload(block_registry)
            self.model.post.reload(post_registry)

    def load_model(  # type: ignore[override]
        self,
        *,
        seq_text_len: int,
        seq_len: int,
        batch_size: int = 1,
    ) -> Callable[..., Any]:
        """Compile the transformer as separate pre/block/post graphs.

        Block graphs are compiled with symbolic ``seq_len`` and concrete
        ``batch_size`` / ``seq_text_len``. Pre/post graphs use symbolic
        spatial dims.
        """
        with self._load_lock:
            if self.model is not None:
                return self.__call__

            state_dict = self._ensure_state_dict()

            dim = (
                self.config.num_attention_heads * self.config.attention_head_dim
            )
            dtype = self.config.dtype
            dev = self.config.device
            dev_ref = DeviceRef.from_device(dev)

            pre_weights, block_weights_list, post_weights = (
                self._split_state_dict(state_dict)
            )
            pre_input_types = [
                TensorType(
                    dtype,
                    [
                        "batch",
                        self.config.in_channels,
                        "frames",
                        "height",
                        "width",
                    ],
                    device=dev,
                ),
                TensorType(DType.float32, ["batch"], device=dev),
                TensorType(
                    dtype,
                    ["batch", seq_text_len, self.config.text_dim],
                    device=dev,
                ),
            ]
            pre_module = WanTransformerPreProcess(
                self.config, dtype=dtype, device=dev_ref
            )
            pre_module.load_state_dict(
                pre_weights, weight_alignment=1, strict=True
            )
            with Graph("wan_pre", input_types=pre_input_types) as pre_graph:
                outs = pre_module(*(v.tensor for v in pre_graph.inputs))
                pre_graph.output(*outs)
            pre_model = self.session.load(
                pre_graph, weights_registry=pre_module.state_dict()
            )
            # Combined blocks graph: all blocks in a single Model
            # so the runtime allocates one shared workspace.
            block_seq_len_dim: str = "seq_len"
            block_input_types = [
                TensorType(
                    dtype, [batch_size, block_seq_len_dim, dim], device=dev
                ),
                TensorType(dtype, [batch_size, seq_text_len, dim], device=dev),
                TensorType(dtype, [batch_size, 6, dim], device=dev),
                TensorType(
                    DType.float32,
                    [block_seq_len_dim, self.config.attention_head_dim],
                    device=dev,
                ),
                TensorType(
                    DType.float32,
                    [block_seq_len_dim, self.config.attention_head_dim],
                    device=dev,
                ),
            ]
            all_blocks = [
                WanTransformerBlock(
                    dim=dim,
                    ffn_dim=self.config.ffn_dim,
                    num_heads=self.config.num_attention_heads,
                    head_dim=self.config.attention_head_dim,
                    text_dim=dim,
                    cross_attn_norm=self.config.cross_attn_norm,
                    eps=self.config.eps,
                    added_kv_proj_dim=self.config.added_kv_proj_dim,
                    dtype=dtype,
                    device=dev_ref,
                )
                for _ in range(self.config.num_layers)
            ]
            block_sequence = WanTransformerBlockSequence(all_blocks)
            # Load per-block weights with LayerList prefix.
            combined_block_weights: dict[str, Any] = {}
            for i, block_weights in enumerate(block_weights_list):
                for k, v in block_weights.items():
                    combined_block_weights[f"blocks.{i}.{k}"] = v
            block_sequence.load_state_dict(
                combined_block_weights, weight_alignment=1, strict=True
            )
            with Graph(
                "wan_blocks_combined", input_types=block_input_types
            ) as blocks_graph:
                block_out = block_sequence(
                    *(v.tensor for v in blocks_graph.inputs)
                )
                blocks_graph.output(block_out)
            combined_blocks_model = self.session.load(
                blocks_graph,
                weights_registry=block_sequence.state_dict(),
            )
            logger.info(
                "Compiled combined block graph (batch=%d, seq_len=symbolic "
                "default=%d, seq_text=%d, %d layers)",
                batch_size,
                seq_len,
                seq_text_len,
                self.config.num_layers,
            )
            post_input_types = [
                TensorType(dtype, ["batch", "seq_len", dim], device=dev),
                TensorType(dtype, ["batch", dim], device=dev),
                TensorType(DType.int8, ["ppf", "pph", "ppw"], device=dev),
            ]
            post_module = WanTransformerPostProcess(
                self.config, dtype=dtype, device=dev_ref
            )
            post_module.load_state_dict(
                post_weights, weight_alignment=1, strict=True
            )
            with Graph("wan_post", input_types=post_input_types) as post_graph:
                post_out = post_module(*(v.tensor for v in post_graph.inputs))
                post_graph.output(post_out)
            post_model = self.session.load(
                post_graph, weights_registry=post_module.state_dict()
            )
            self.model = BlockLevelModel(
                pre_model,
                [],
                post_model,
                combined_blocks=combined_blocks_model,
            )
            return self.__call__

    def compute_rope(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[Buffer, Buffer]:
        """Compute 3D RoPE cos/sin tensors and transfer to device."""
        rope_cos_np, rope_sin_np = _compute_wan_rope_cached(
            num_frames,
            height,
            width,
            self.config.patch_size,
            self.config.attention_head_dim,
        )
        device = self.devices[0]
        return (
            Buffer.from_numpy(rope_cos_np).to(device),
            Buffer.from_numpy(rope_sin_np).to(device),
        )

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
    ) -> Buffer:
        if self.model is None:
            raise RuntimeError(
                "Wan transformer model not compiled. Call load_model() first."
            )
        return self.model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            spatial_shape,
        )
