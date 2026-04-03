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

"""Wan-Animate transformer model with single-graph compilation.

Extends the base WanTransformerModel with:
- Pose patch embedding injection in pre-processing
- CLIP image embedding in pre-processing (passed to block cross-attention)
- Face adapter injection every `inject_face_latents_blocks` blocks
- Face encoder (run once per segment, before denoising loop)
- Motion encoder (MAX graph, run once per segment)
"""

from __future__ import annotations

import logging
from typing import Any

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .layers.embeddings import compute_wan_rope_cached
from .model_config import WanConfig
from .wan_transformer import (
    WanAnimateFaceEncoder,
    WanAnimateMotionEncoder,
    WanAnimateTransformer3DModel,
)
from .weight_adapters import convert_safetensor_state_dict

logger = logging.getLogger(__name__)


class WanAnimateTransformerModel(ComponentModel):
    """MAX-native Wan-Animate transformer with single-graph compilation.

    Extends WanTransformerModel with pose injection, CLIP conditioning,
    face encoder, and face adapter blocks.
    """

    def __init__(
        self,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession | None = None,
    ) -> None:
        super().__init__(config, encoding, devices, weights)
        self.config = WanConfig.generate(config, encoding, devices)
        self.model: Model | None = None
        self.session = session or InferenceSession(devices=devices)
        # Face encoder model (compiled MAX graph, run once per segment)
        self.face_encoder_model: Model | None = None
        # Motion encoder model (compiled MAX graph, run once per segment)
        self.motion_encoder_model: Model | None = None
        self.load_model()

    def load_model(self) -> None:
        """Compile the animate transformer as a single graph."""
        if self.weights is None:
            raise RuntimeError(
                "WanAnimateTransformerModel weights unavailable."
            )

        state_dict = convert_safetensor_state_dict(
            self.weights,
            target_dtype=DType.bfloat16,
            is_animate=True,
        )
        dim = self.config.num_attention_heads * self.config.attention_head_dim
        dtype = self.config.dtype
        dev = self.config.device
        dev_ref = DeviceRef.from_device(dev)

        (
            transformer_weights,
            face_encoder_weights,
            motion_encoder_weights,
        ) = self._split_animate_state_dict(state_dict)
        self.model = self._load_transformer_model(
            transformer_weights,
            dtype=dtype,
            dev=dev,
            dev_ref=dev_ref,
        )
        self.face_encoder_model = self._load_face_encoder_model(
            face_encoder_weights,
            dim=dim,
            dtype=dtype,
            dev=dev,
            dev_ref=dev_ref,
        )
        self.motion_encoder_model = self._load_motion_encoder_model(
            motion_encoder_weights,
            dim=dim,
            dtype=dtype,
            dev=dev,
            dev_ref=dev_ref,
        )

    def _split_animate_state_dict(
        self, state_dict: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Split state dict into transformer, face_encoder, motion_encoder weights."""
        # NOTE: Split these weights to compile each component as a separate graph.
        # face_endoer and motion_encoder can be executed just once per segment,
        # rather than once per denoising step within the transformer.
        transformer_weights: dict[str, Any] = {}
        face_encoder_weights: dict[str, Any] = {}
        motion_encoder_weights: dict[str, Any] = {}

        for key, value in state_dict.items():
            if (
                key.startswith("patch_embedding.")
                or key.startswith("condition_embedder.")
                or key.startswith("pose_patch_embedding.")
            ):
                transformer_weights[f"pre.{key}"] = value
            elif key.startswith("blocks."):
                transformer_weights[key] = value
            elif key.startswith("face_adapter."):
                transformer_weights[key] = value
            elif key.startswith("face_encoder."):
                sub_key = key[len("face_encoder.") :]
                face_encoder_weights[sub_key] = value
            elif key.startswith("motion_encoder."):
                sub_key = key[len("motion_encoder.") :]
                motion_encoder_weights[sub_key] = value
            else:
                # proj_out, scale_shift_table -> post weights
                transformer_weights[f"post.{key}"] = value

        return (
            transformer_weights,
            face_encoder_weights,
            motion_encoder_weights,
        )

    def _load_transformer_model(
        self,
        transformer_weights: dict[str, Any],
        *,
        dtype: DType,
        dev: Device,
        dev_ref: DeviceRef,
    ) -> Model:
        transformer_module = WanAnimateTransformer3DModel(
            self.config, dtype=dtype, device=dev_ref
        )
        transformer_module.load_state_dict(
            transformer_weights, weight_alignment=1, strict=True
        )
        # Finalize fully qualified weight names before graph construction.
        weights_registry = transformer_module.state_dict()

        dim = self.config.num_attention_heads * self.config.attention_head_dim
        model_input_types = [
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
                ["batch", "seq_text", self.config.text_dim],
                device=dev,
            ),
            TensorType(
                dtype,
                ["batch", "clip_seq", self.config.image_dim or 1280],
                device=dev,
            ),
            TensorType(
                dtype,
                [
                    "batch",
                    self.config.latent_channels,
                    "pose_frames",
                    "height",
                    "width",
                ],
                device=dev,
            ),
            TensorType(
                DType.float32,
                ["seq_len", self.config.attention_head_dim],
                device=dev,
            ),
            TensorType(
                DType.float32,
                ["seq_len", self.config.attention_head_dim],
                device=dev,
            ),
            TensorType(DType.int8, ["ppf", "pph", "ppw"], device=dev),
            TensorType(dtype, ["batch", "t_face", 5, dim], device=dev),
            TensorType(DType.int32, [1], device=dev),
        ]

        with Graph(
            "wan_animate_transformer", input_types=model_input_types
        ) as graph:
            out = transformer_module(*(v.tensor for v in graph.inputs))
            graph.output(out)
        return self.session.load(graph, weights_registry=weights_registry)

    def _load_face_encoder_model(
        self,
        face_encoder_weights: dict[str, Any],
        *,
        dim: int,
        dtype: DType,
        dev: Device,
        dev_ref: DeviceRef,
    ) -> Model:
        face_enc_dim = self.config.motion_encoder_dim  # 512
        face_enc_out_dim = dim  # 5120
        face_enc_hidden = self.config.face_encoder_hidden_dim  # 1024
        face_enc_heads = self.config.face_encoder_num_heads  # 4

        face_enc_input_types = [
            # motion_vectors [B, T, 512]
            TensorType(dtype, ["batch", "t_motion", face_enc_dim], device=dev),
        ]
        face_enc_module = WanAnimateFaceEncoder(
            in_dim=face_enc_dim,
            hidden_dim=face_enc_hidden,
            out_dim=face_enc_out_dim,
            num_heads=face_enc_heads,
            dtype=dtype,
            device=dev_ref,
        )
        face_enc_module.load_state_dict(
            face_encoder_weights, weight_alignment=1, strict=True
        )
        with Graph(
            "wan_face_encoder", input_types=face_enc_input_types
        ) as face_enc_graph:
            face_out = face_enc_module(face_enc_graph.inputs[0].tensor)
            face_enc_graph.output(face_out)
        return self.session.load(
            face_enc_graph, weights_registry=face_enc_module.state_dict()
        )

    def _load_motion_encoder_model(
        self,
        motion_encoder_weights: dict[str, Any],
        *,
        dim: int,
        dtype: DType,
        dev: Device,
        dev_ref: DeviceRef,
    ) -> Model:
        me_size = self.config.motion_encoder_size  # 512
        me_style_dim = self.config.motion_style_dim  # 512
        me_motion_dim = self.config.motion_dim  # 20
        me_blocks = 5  # fixed in diffusers

        me_input_types = [
            # face_image [B, 3, size, size] in [-1, 1]
            TensorType(dtype, ["batch", 3, me_size, me_size], device=dev),
        ]
        me_module = WanAnimateMotionEncoder(
            size=me_size,
            style_dim=me_style_dim,
            motion_dim=me_motion_dim,
            out_dim=me_style_dim,
            motion_blocks=me_blocks,
            dtype=dtype,
            device=dev_ref,
        )
        me_module.load_state_dict(
            motion_encoder_weights,
            weight_alignment=1,
            strict=True,
        )
        with Graph(
            "wan_motion_encoder", input_types=me_input_types
        ) as me_graph:
            me_out = me_module(me_graph.inputs[0].tensor)
            me_graph.output(me_out)
        return self.session.load(
            me_graph, weights_registry=me_module.state_dict()
        )

    def compute_rope(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[Buffer, Buffer]:
        """Compute 3D RoPE cos/sin tensors for the given latent dimensions."""
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

    def encode_face(self, motion_vectors: Buffer) -> Buffer:
        """Run face encoder graph: motion_vectors -> face embeddings."""
        if self.face_encoder_model is None:
            raise RuntimeError("Face encoder model not compiled.")
        return self.face_encoder_model.execute(motion_vectors)[0]

    def encode_motion(self, face_pixels: Buffer) -> Buffer:
        """Run motion encoder graph: face_pixels [B, 3, size, size] -> [B, 512]."""
        if self.motion_encoder_model is None:
            raise RuntimeError("Motion encoder model not compiled.")
        return self.motion_encoder_model.execute(face_pixels)[0]

    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        clip_features: Buffer,
        pose_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        face_emb: Buffer,
        num_temporal_frames: Buffer,
    ) -> Buffer:
        if self.model is None:
            raise RuntimeError("Wan animate transformer model failed to load.")
        return self.model.execute(
            hidden_states,
            timestep,
            encoder_hidden_states,
            clip_features,
            pose_hidden_states,
            rope_cos,
            rope_sin,
            spatial_shape,
            face_emb,
            num_temporal_frames,
        )[0]
