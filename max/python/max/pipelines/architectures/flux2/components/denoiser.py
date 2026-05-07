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

"""Graph 3: Fused denoise step component for Flux2Executor."""

from __future__ import annotations

from typing import Any

from max.driver import Buffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.weights import load_weights
from max.nn.layer import Module
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced

from ..flux2 import Flux2Transformer2DModel
from ..model_config import Flux2Config
from ..weight_adapters import adapt_weights


class DenoiseStep(Module):
    """Fused concat + transformer + Euler scheduler step in a single graph.

    Accepts current latents, optional image latents (zero-seq-length for
    text-to-image), text embeddings, timestep, dt, guidance, and position
    IDs.  Produces updated latents after one denoising step.

    For text-to-image, ``image_latents`` has shape ``(B, 0, C)`` so the
    concat along the sequence axis is a no-op.

    Input:  9 tensors (see :meth:`input_types`)
    Output: ``(B, image_seq_len, C)`` model dtype
    """

    def __init__(
        self,
        transformer: Flux2Transformer2DModel,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self._dtype = dtype
        self._device = device

    def __call__(
        self,
        latents: TensorValue,
        image_latents: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        dt: TensorValue,
        guidance: TensorValue,
        latent_image_ids: TensorValue,
        image_latent_ids: TensorValue,
        txt_ids: TensorValue,
    ) -> TensorValue:
        # Concat image latents for img2img (no-op when img_seq=0).
        latents_concat = ops.concat([latents, image_latents], axis=1)
        latent_image_ids_concat = ops.concat(
            [latent_image_ids, image_latent_ids], axis=1
        )

        # Keep timestep and guidance as float32 here.  The transformer
        # multiplies by 1000.0 *before* casting to model dtype
        # (flux2.py:628-629), so leaving them in float32 avoids bf16
        # rounding on the raw sigma which shifts the x1000 result by
        # up to 4 ulps for smaller sigmas — matching the precision path
        # used by diffusers/torch.

        # Transformer forward.
        (noise_pred,) = self.transformer(
            latents_concat,
            encoder_hidden_states,
            timestep,
            latent_image_ids_concat,
            txt_ids,
            guidance,
        )

        # Slice noise_pred to latents.shape[1] tokens (discard image
        # predictions in img2img case).
        num_tokens = ops.shape_to_tensor([latents.shape[1]])
        noise_pred_sliced = ops.slice_tensor(
            noise_pred,
            [
                slice(None),
                (slice(0, num_tokens), "num_tokens"),
                slice(None),
            ],
        )

        # Euler step in float32 for numerical stability.
        latents_dtype = latents.dtype
        latents_f32 = ops.cast(latents, DType.float32)
        noise_pred_f32 = ops.cast(noise_pred_sliced, DType.float32)
        noise_pred_f32 = ops.rebind(noise_pred_f32, latents_f32.shape)
        updated = ops.cast(latents_f32 + dt * noise_pred_f32, latents_dtype)
        return updated

    def input_types(self) -> tuple[TensorType, ...]:
        in_channels = self.transformer.in_channels
        joint_attention_dim = self.transformer.joint_attention_dim
        return (
            # latents: (B, image_seq_len, C)
            TensorType(
                self._dtype,
                shape=["batch", "image_seq_len", in_channels],
                device=self._device,
            ),
            # image_latents: (B, img_seq, C) — zero-seq for t2i
            TensorType(
                self._dtype,
                shape=["batch", "img_seq", in_channels],
                device=self._device,
            ),
            # encoder_hidden_states: (B, text_seq_len, joint_attention_dim)
            TensorType(
                self._dtype,
                shape=["batch", "text_seq_len", joint_attention_dim],
                device=self._device,
            ),
            # timestep: (B,) float32 — transformer casts after x1000
            TensorType(DType.float32, shape=["batch"], device=self._device),
            # dt: (1,) float32
            TensorType(DType.float32, shape=[1], device=self._device),
            # guidance: (B,) float32 — transformer casts after x1000
            TensorType(DType.float32, shape=["batch"], device=self._device),
            # latent_image_ids: (B, image_seq_len, 4)
            TensorType(
                DType.int64,
                shape=["batch", "image_seq_len", 4],
                device=self._device,
            ),
            # image_latent_ids: (B, img_seq, 4) — zero-seq for t2i
            TensorType(
                DType.int64,
                shape=["batch", "img_seq", 4],
                device=self._device,
            ),
            # txt_ids: (B, text_seq_len, 4)
            TensorType(
                DType.int64,
                shape=["batch", "text_seq_len", 4],
                device=self._device,
            ),
        )


class Denoiser(CompiledComponent):
    """Graph 3: Fused concat + transformer + scheduler step.

    Encapsulates the full lifecycle: config extraction from the manifest,
    weight loading, Module construction, graph compilation, and runtime
    execution.

    Output shape: ``(B, image_seq_len, C)`` model dtype.
    """

    _model: Model

    @traced(message="Denoiser.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
    ) -> None:
        super().__init__(manifest, session)

        config = manifest["transformer"]
        config_dict = config.huggingface_config.to_dict()
        encoding = config.quantization_encoding or "bfloat16"
        devices = load_devices(config.device_specs)

        transformer_config = Flux2Config.initialize_from_config(
            config_dict, encoding, devices
        )

        dtype = transformer_config.dtype
        device = transformer_config.devices[0]

        # Load weights and adapt for NVFP4 / stacked-QKV checkpoints.
        paths = config.resolved_weight_paths()
        weights = load_weights(paths)
        raw_state_dict = {key: value.data() for key, value in weights.items()}
        raw_state_dict = adapt_weights(
            raw_state_dict, transformer_config.quant_config
        )

        has_guidance_embedder = any(
            "time_guidance_embed.guidance_embedder." in k
            for k in raw_state_dict
        )
        if not has_guidance_embedder and transformer_config.guidance_embeds:
            transformer_config = transformer_config.model_copy(
                update={"guidance_embeds": False}
            )

        # Build transformer and fused module.
        transformer = Flux2Transformer2DModel(transformer_config)
        fused = DenoiseStep(
            transformer=transformer,
            dtype=dtype,
            device=device,
        )

        # Prefix with "transformer." for the module hierarchy.
        state_dict: dict[str, Any] = {
            f"transformer.{key}": value for key, value in raw_state_dict.items()
        }
        fused.load_state_dict(state_dict, weight_alignment=1)

        # Build and compile graph.
        with Graph("denoise_step", input_types=fused.input_types()) as graph:
            outputs = fused(*(v.tensor for v in graph.inputs))
            graph.output(outputs)

        self._model = self._load_graph(
            graph, weights_registry=fused.state_dict()
        )

    @traced(message="Denoiser.__call__")
    def __call__(
        self,
        latents: Buffer,
        image_latents: Buffer,
        encoder_hidden_states: Buffer,
        timestep: Buffer,
        dt: Buffer,
        guidance: Buffer,
        latent_image_ids: Buffer,
        image_latent_ids: Buffer,
        txt_ids: Buffer,
    ) -> Buffer:
        """Execute one fused denoise step.

        Args:
            latents: Current latent state, shape ``(B, seq, C)``.
            image_latents: Packed image latents, shape ``(B, img_seq, C)``.
                Zero-seq-length for text-to-image.
            encoder_hidden_states: Text embeddings, shape
                ``(B, text_seq, D)``.
            timestep: Current sigma, shape ``(B,)``.
            dt: Step delta, shape ``(1,)`` float32.
            guidance: Guidance scale, shape ``(B,)``.
            latent_image_ids: Latent position IDs, shape
                ``(B, seq, 4)`` int64.
            image_latent_ids: Image latent position IDs, shape
                ``(B, img_seq, 4)`` int64.
            txt_ids: Text position IDs, shape ``(B, text_seq, 4)`` int64.

        Returns:
            Updated latents, shape ``(B, seq, C)``.
        """
        result = self._model.execute(
            latents,
            image_latents,
            encoder_hidden_states,
            timestep,
            dt,
            guidance,
            latent_image_ids,
            image_latent_ids,
            txt_ids,
        )
        return result[0] if isinstance(result, (list, tuple)) else result
