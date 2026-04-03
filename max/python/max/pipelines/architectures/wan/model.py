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
from pathlib import Path
from typing import Any

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .layers.embeddings import compute_wan_rope_cached
from .layers.transformer import WanTransformerBlock
from .model_config import WanConfig
from .wan_transformer import (
    WanTransformerPostProcess,
    WanTransformerPreProcess,
)
from .weight_adapters import convert_safetensor_state_dict

logger = logging.getLogger(__name__)


class BlockLevelModel:
    """Executes transformer forward pass as pre -> N blocks -> post.

    Each component is a separately compiled graph, so only one block's
    workspace is live at any time. This keeps peak VRAM low.
    """

    def __init__(
        self,
        pre: Model,
        blocks: list[Model],
        post: Model,
    ) -> None:
        self.pre = pre
        self.blocks = blocks
        self.post = post

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
    ) -> Buffer:
        pre_out = self.pre.execute(
            hidden_states, timestep, encoder_hidden_states
        )
        hs, temb, timestep_proj, text_emb = (
            pre_out[0],
            pre_out[1],
            pre_out[2],
            pre_out[3],
        )
        for block in self.blocks:
            block_out = block.execute(
                hs, text_emb, timestep_proj, rope_cos, rope_sin
            )
            hs = block_out[0]
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
        lora_path: str | Path | None = None,
        lora_scale: float = 1.0,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = WanConfig.generate(config, encoding, devices)
        self.config.dtype = DType.bfloat16
        self._state_dict: dict[str, Any] | None = None
        self._lora_path = lora_path
        self._lora_scale = lora_scale
        self._lora_merged = False
        self.model: BlockLevelModel | None = None
        self._weight_registry_cache: dict[
            int,
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
        ] = {}
        self.session = session or InferenceSession(devices=devices)
        self._load_lock = threading.Lock()

    def _ensure_state_dict(self) -> dict[str, Any]:
        if self._state_dict is None:
            if self.weights is None:
                raise RuntimeError(
                    "WanTransformerModel weights are unavailable "
                    "while state_dict is not initialized."
                )
            self._state_dict = convert_safetensor_state_dict(
                self.weights, target_dtype=DType.bfloat16
            )
            self.weights = None  # type: ignore[assignment]

        if self._lora_path and not self._lora_merged:
            from .lora_utils import load_and_merge_lora

            self._state_dict = load_and_merge_lora(
                self._state_dict, self._lora_path, self._lora_scale
            )
            self._lora_merged = True

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

            self.model.pre._load(pre_registry)
            for compiled_block, block_registry in zip(
                self.model.blocks, block_registries, strict=True
            ):
                compiled_block._load(block_registry)
            self.model.post._load(post_registry)

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
            block_template = WanTransformerBlock(
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
            block_template.load_state_dict(
                block_weights_list[0], weight_alignment=1, strict=True
            )
            with Graph(
                "wan_block", input_types=block_input_types
            ) as block_graph:
                block_out = block_template(
                    *(v.tensor for v in block_graph.inputs)
                )
                block_graph.output(block_out)

            block_models: list[Model] = [
                self.session.load(
                    block_graph,
                    weights_registry=block_template.state_dict(),
                )
            ]
            for i in range(1, self.config.num_layers):
                block_template.load_state_dict(
                    block_weights_list[i],
                    weight_alignment=1,
                    strict=True,
                )
                block_models.append(
                    self.session.load(
                        block_graph,
                        weights_registry=block_template.state_dict(),
                    )
                )
            logger.info(
                "Compiled block graph (batch=%d, seq_len=symbolic "
                "default=%d, seq_text=%d, %d layers)",
                batch_size,
                seq_len,
                seq_text_len,
                len(block_models),
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
            self.model = BlockLevelModel(pre_model, block_models, post_model)
            return self.__call__

    def compute_rope(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[Buffer, Buffer]:
        """Compute 3D RoPE cos/sin tensors and transfer to device."""
        rope_cos_np, rope_sin_np = compute_wan_rope_cached(
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
