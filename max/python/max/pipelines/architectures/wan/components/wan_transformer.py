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

"""Wan transformer component with block-level compilation for WanExecutor."""

from __future__ import annotations

import logging
import threading
from typing import Any

from max.driver import Buffer, Device, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import load_weights
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced

from ..model import (
    BlockLevelModel,
    _compute_wan_rope_cached,
    _remap_state_dict,
)
from ..model_config import WanConfig
from ..wan_transformer import (
    WanTransformerBlock,
    WanTransformerPostProcess,
    WanTransformerPreProcess,
)

logger = logging.getLogger("max.pipelines")


class WanTransformer(CompiledComponent):
    """Wan DiT transformer with block-level compilation.

    Each block is compiled independently so only one block's workspace
    is live at any time, keeping peak VRAM low for 14B-parameter models.

    Supports MoE (dual expert) via weight swapping or dual-load.
    """

    # Diffusers pads to 512 but trims embeddings to 226 for cross-attn.
    embed_seq_len: int = 226

    # Default resolution for block graph compilation (height, width, frames).
    default_resolution: tuple[int, int, int] = (720, 1280, 81)

    @traced(message="WanTransformer.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
    ) -> None:
        super().__init__(manifest, session)

        config_entry = manifest["transformer"]
        config_dict = config_entry.huggingface_config.to_dict()
        encoding = config_entry.quantization_encoding or "bfloat16"
        devices = load_devices(config_entry.device_specs)

        self.config = WanConfig.generate(config_dict, encoding, devices)
        self.config.dtype = DType.bfloat16
        self._device = devices[0]

        self._load_lock = threading.Lock()
        self._model: BlockLevelModel | None = None
        self._state_dict: dict[str, Any] | None = None
        self._weight_registry_cache: dict[
            int,
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]],
        ] = {}

        # Load and remap weights.
        paths = config_entry.resolved_weight_paths()
        weights = load_weights(paths)
        self._state_dict = _remap_state_dict(
            weights, target_dtype=DType.bfloat16
        )

        # MoE: secondary transformer state dict (lazy).
        self._transformer_2_state_dict: dict[str, Any] | None = None
        self._moe_dual_loaded = False
        self._active_weights = "primary"
        self._model_2: BlockLevelModel | None = None
        self._setup_moe(manifest)

        # Compile block-level graphs.
        h, w, nf = self.default_resolution
        seq_len = self._compute_seq_len(h, w, nf)
        self._compile_model(
            seq_text_len=self.embed_seq_len,
            seq_len=seq_len,
        )

        # Attempt MoE dual-load after primary compilation.
        if self._transformer_2_state_dict is not None:
            if self._try_dual_load():
                self._moe_dual_loaded = True
                logger.info(
                    "MoE dual-load enabled: transformer_2 stays resident."
                )
            else:
                logger.info(
                    "MoE swap mode: transformer_2 will use weight swap."
                )

    def _setup_moe(self, manifest: ModelManifest) -> None:
        """Load optional transformer_2 weights for MoE."""
        if "transformer_2" not in manifest:
            return

        config_entry_2 = manifest["transformer_2"]
        paths_2 = config_entry_2.resolved_weight_paths()
        weights_2 = load_weights(paths_2)
        self._transformer_2_state_dict = _remap_state_dict(
            weights_2, target_dtype=DType.bfloat16
        )

    def _compile_model(
        self,
        *,
        seq_text_len: int,
        seq_len: int,
        batch_size: int = 1,
    ) -> None:
        """Compile the transformer as separate pre/block/post graphs."""
        with self._load_lock:
            if self._model is not None:
                return

            assert self._state_dict is not None
            state_dict = self._state_dict

            dim = (
                self.config.num_attention_heads * self.config.attention_head_dim
            )
            dtype = self.config.dtype
            dev = self._device
            dev_ref = DeviceRef.from_device(dev)

            pre_weights, block_weights_list, post_weights = (
                self._split_state_dict(state_dict)
            )

            # Pre-processor graph.
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
            pre_model = self._session.load(
                pre_graph, weights_registry=pre_module.state_dict()
            )

            # Block graph (shared across all layers).
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
                self._session.load(
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
                    self._session.load(
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

            # Post-processor graph.
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
            post_model = self._session.load(
                post_graph, weights_registry=post_module.state_dict()
            )
            self._model = BlockLevelModel(pre_model, block_models, post_model)

    @traced(message="WanTransformer.__call__")
    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
    ) -> Buffer:
        """Execute the block-level transformer forward pass.

        Args:
            hidden_states: Latents in model dtype,
                shape ``(B, C, T, H, W)``.
            timestep: Timestep, shape ``(B,)`` float32.
            encoder_hidden_states: Text embeddings,
                shape ``(B, seq_text, dim)``.
            rope_cos: RoPE cosine, shape ``(seq_len, head_dim)`` float32.
            rope_sin: RoPE sine, shape ``(seq_len, head_dim)`` float32.
            spatial_shape: Shape carrier ``(ppf, pph, ppw)`` int8.

        Returns:
            Noise prediction, shape ``(B, C, T, H, W)`` in model dtype.
        """
        if self._model is None:
            raise RuntimeError(
                "WanTransformer model not compiled. "
                "This should not happen — __init__ compiles it."
            )
        return self._model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            spatial_shape,
        )

    def call_secondary(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
    ) -> Buffer:
        """Execute the secondary (low-noise) transformer for dual-load MoE."""
        if self._model_2 is None:
            raise RuntimeError(
                "Secondary transformer not compiled for dual-load MoE."
            )
        return self._model_2(
            hidden_states,
            timestep,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            spatial_shape,
        )

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
        return (
            Buffer.from_numpy(rope_cos_np).to(self._device),
            Buffer.from_numpy(rope_sin_np).to(self._device),
        )

    def activate_weights(self, *, use_secondary: bool) -> None:
        """Switch active weights for MoE weight-swap mode.

        No-op if dual-load is active (both models resident).
        """
        if self._moe_dual_loaded:
            return  # Both models are resident; no swap needed.

        if not use_secondary:
            if self._active_weights != "primary":
                assert self._state_dict is not None
                self._reload_weights(self._state_dict)
                self._active_weights = "primary"
        else:
            if self._active_weights != "secondary":
                assert self._transformer_2_state_dict is not None
                self._reload_weights(self._transformer_2_state_dict)
                self._active_weights = "secondary"

    @property
    def has_moe(self) -> bool:
        """Whether this transformer has MoE (dual expert) support."""
        return self._transformer_2_state_dict is not None

    @property
    def moe_dual_loaded(self) -> bool:
        """Whether both MoE models are compiled and resident on device."""
        return self._moe_dual_loaded

    @property
    def device(self) -> Device:
        return self._device

    # -- Internal helpers ---------------------------------------------------

    def _compute_seq_len(self, height: int, width: int, num_frames: int) -> int:
        """Compute the latent sequence length for a given resolution."""
        p_t, p_h, p_w = self.config.patch_size
        # Compute video latent shape.
        vae_scale_t = 4  # Default temporal scale factor
        vae_scale_s = 8  # Default spatial scale factor
        adjusted = max(1, num_frames)
        if adjusted > 1:
            remainder = (adjusted - 1) % vae_scale_t
            if remainder != 0:
                adjusted += vae_scale_t - remainder
        latent_frames = (adjusted - 1) // vae_scale_t + 1
        latent_h = 2 * (height // (vae_scale_s * 2))
        latent_w = 2 * (width // (vae_scale_s * 2))
        return (latent_frames // p_t) * (latent_h // p_h) * (latent_w // p_w)

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
        dev_ref = DeviceRef.from_device(self._device)
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
        self, state_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Return weight registries, caching by state_dict identity."""
        cache_key = id(state_dict)
        cached = self._weight_registry_cache.get(cache_key)
        if cached is not None:
            return cached
        registries = self._build_weight_registries(state_dict)
        self._weight_registry_cache[cache_key] = registries
        return registries

    def _reload_weights(self, state_dict: dict[str, Any]) -> None:
        """Reload weights into already-compiled primary model."""
        with self._load_lock:
            if self._model is None:
                raise RuntimeError("Transformer model not compiled.")

            pre_registry, block_registries, post_registry = (
                self._get_cached_weight_registries(state_dict)
            )

            self._model.pre.reload(pre_registry)
            for compiled_block, compiled_reg in zip(
                self._model.blocks, block_registries, strict=True
            ):
                compiled_block.reload(compiled_reg)
            self._model.post.reload(post_registry)

    def _try_dual_load(self) -> bool:
        """Try to compile secondary transformer on GPU if VRAM sufficient."""
        if self._transformer_2_state_dict is None or self._model is None:
            return False

        free_vram = self._get_free_vram_bytes()
        if free_vram is None:
            return False

        # Estimate required VRAM as the primary transformer's weight size.
        assert self._state_dict is not None
        estimated_bytes = 0
        for v in self._state_dict.values():
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                num_elements = 1
                for d in v.shape:
                    num_elements *= d
                estimated_bytes += num_elements * 2  # bfloat16

        margin = 1.2  # 20% headroom
        if free_vram < estimated_bytes * margin:
            logger.info(
                "Insufficient VRAM for dual load: need %.1f GB, free %.1f GB",
                estimated_bytes * margin / 1e9,
                free_vram / 1e9,
            )
            return False

        assert self._model is not None
        # Reload secondary weights into a clone of the block models.
        # For dual-load we re-use the compiled graphs from the primary
        # but load with secondary weight registries.
        h, w, nf = self.default_resolution
        seq_len = self._compute_seq_len(h, w, nf)

        # The simplest approach: compile the same graph structure
        # with secondary weights.
        saved_state_dict = self._state_dict
        self._state_dict = self._transformer_2_state_dict
        saved_model = self._model
        self._model = None
        try:
            self._compile_model(
                seq_text_len=self.embed_seq_len,
                seq_len=seq_len,
            )
            self._model_2 = self._model
        finally:
            self._model = saved_model
            self._state_dict = saved_state_dict
        return self._model_2 is not None

    @staticmethod
    def _get_free_vram_bytes() -> int | None:
        """Query free GPU VRAM in bytes via nvidia-smi."""
        import subprocess

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            return int(out.strip().split("\n")[0]) * 1024 * 1024
        except Exception:
            return None
