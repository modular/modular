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
"""Flux2 executor implementing the PipelineExecutor interface."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.tensor import Tensor
from max.pipelines.core import PixelContext
from max.pipelines.lib import float32_array_to_buffer
from max.pipelines.lib.config.config_enums import supported_encoding_dtype
from max.pipelines.lib.denoising_cache import TaylorSeerCache
from max.pipelines.lib.interfaces import TensorStruct
from max.pipelines.lib.interfaces.cache_mixin import DenoisingCacheConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_executor import PipelineExecutor
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.profiler import traced
from typing_extensions import Self

from .components import (
    DenoiseCompute,
    DenoisePredict,
    Denoiser,
    ImageEncoder,
    TextEncoder,
    VaeDecoder,
)

logger = logging.getLogger("max.pipelines")

# ---------------------------------------------------------------------------
# Input / Output structs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Flux2ExecutorInputs(TensorStruct):
    """Structured inputs for Flux2 execution.

    Core fields are always present. Optional feature fields are ``None``
    when the feature is disabled for a given request -- use
    ``is not None`` to gate feature-specific execution paths.

    Optional features are added via fluent ``.with_*()`` methods which
    return a new frozen instance (the original is not mutated)::

        inputs = Flux2ExecutorInputs(tokens=..., latents=..., ...)
        inputs = inputs.with_image(image_tensor)
    """

    # -- Core (always present) ------------------------------------------------

    tokens: Buffer
    """Token IDs for the text encoder, shape ``(S,)``."""

    text_ids: Buffer
    """Text position IDs for the transformer, shape ``(1, S, 4)`` int64."""

    latents: Buffer
    """Packed latent noise tensor, shape ``(B, seq, C*4)``."""

    latent_image_ids: Buffer
    """Latent positional identifiers, shape ``(B, seq, 4)`` int64."""

    timesteps: Buffer
    """Precomputed timesteps from the sigma schedule, shape ``(num_steps,)``
    (model dtype)."""

    dts: Buffer
    """Precomputed step deltas ``sigma[i+1] - sigma[i]``, shape
    ``(num_steps,)`` (float32)."""

    guidance: Buffer
    """Guidance scale broadcast tensor, shape ``(B,)``."""

    image_seq_len: Buffer
    """Packed image sequence length as a 1-element int64 tensor."""

    h_carrier: Buffer
    """Shape carrier of length ``packed_h``; content is never read."""

    w_carrier: Buffer
    """Shape carrier of length ``packed_w``; content is never read."""

    height: Buffer
    """Output image height in pixels as a 1-element int64 tensor."""

    width: Buffer
    """Output image width in pixels as a 1-element int64 tensor."""

    num_inference_steps: Buffer
    """Number of denoising steps as a 1-element int64 tensor."""

    num_images_per_prompt: Buffer
    """Number of images to generate per prompt as a 1-element int64 tensor."""

    # -- Optional features ----------------------------------------------------

    input_image: Buffer | None = None
    """Input image for image-to-image generation, shape ``(H, W, C)`` uint8.
    ``None`` when running in text-to-image mode."""

    # -- Device transfer -------------------------------------------------------

    _CPU_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "num_inference_steps",
            "num_images_per_prompt",
            "height",
            "width",
            "image_seq_len",
            "h_carrier",
            "w_carrier",
        }
    )

    def to(self, device: Device) -> Self:
        """Transfer GPU-bound tensors to *device*, keeping metadata on CPU."""
        updates: dict[str, Any] = {}
        for f in fields(self):
            if f.name in self._CPU_FIELDS:
                continue
            val = getattr(self, f.name)
            if isinstance(val, (Tensor, Buffer)):
                updates[f.name] = val.to(device)
        return replace(self, **updates)

    # -- Feature builders (return new frozen instance) ------------------------

    def with_image(self, input_image: Buffer) -> Self:
        """Enable image-to-image mode with the given input image."""
        return replace(self, input_image=input_image)


@dataclass(frozen=True)
class Flux2ExecutorOutputs(TensorStruct):
    """Structured outputs from Flux2 execution."""

    images: Buffer
    """Decoded images, shape ``(B, H, W, C)`` uint8.

    Buffer may be device-resident on return -- the caller is responsible
    for host transfer via ``.to(cpu)`` when needed.
    """


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Flux2Executor(
    PipelineExecutor[PixelContext, Flux2ExecutorInputs, Flux2ExecutorOutputs]
):
    """Flux2 pipeline executor.

    Implements the :class:`PipelineExecutor` interface for Flux2 image
    generation, wiring together the sub-components (text encoder, transformer
    denoiser, VAE) through the tensor-in/tensor-out executor contract.

    Graph structure (4 graphs, 3 for t2i):

    +---------+-------------------------------------------+------------------------+
    | Graph   | Purpose                                   | Called                 |
    +=========+===========================================+========================+
    | 1       | Text Encode: Mistral3 -> embeddings       | Once per request       |
    +---------+-------------------------------------------+------------------------+
    | 2       | Image Encode: VAE encode + BN-norm + pack | Once (img2img only)    |
    +---------+-------------------------------------------+------------------------+
    | 3       | Denoise Step: concat + transformer +      | Every denoising step   |
    |         | scheduler step                            |                        |
    +---------+-------------------------------------------+------------------------+
    | 4       | Decode: BN-denorm + unpatchify +          | Once per request       |
    |         | VAE decode -> uint8                       |                        |
    +---------+-------------------------------------------+------------------------+
    """

    # -- Compiled graphs (set during __init__) --------------------------------

    text_encoder: TextEncoder
    """Graph 1: Mistral3 text encoder -> prompt_embeds."""

    image_encoder: ImageEncoder
    """Graph 2: VAE encode + BN-normalize + patchify + pack."""

    denoiser: Denoiser | None
    """Graph 3: Fused concat + transformer + scheduler step.
    ``None`` when TaylorSeer is enabled (split into compute + predict)."""

    decoder: VaeDecoder
    """Graph 4: BN-denormalize + unpatchify + VAE decode -> uint8."""

    default_num_inference_steps: int = 28
    """Default number of denoising steps when the user does not specify one."""

    # Fallback VAE scale factor when not derivable from the manifest.
    _DEFAULT_VAE_SCALE_FACTOR: int = 8

    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        runtime_config: PipelineRuntimeConfig,
    ) -> None:
        self._manifest = manifest
        self._session = session
        self._runtime_config = runtime_config

        # Cache configuration (TaylorSeer).
        self._cache_config: DenoisingCacheConfig = (
            runtime_config.denoising_cache
        )
        self._resolve_cache_defaults()

        # Derive VAE scale factor from manifest config, falling back to 8.
        vae_config = (
            manifest["vae"].huggingface_config.to_dict()
            if "vae" in manifest
            else {}
        )
        block_out_channels = vae_config.get("block_out_channels", None)
        self._vae_scale_factor = (
            2 ** (len(block_out_channels) - 1)
            if block_out_channels
            else self._DEFAULT_VAE_SCALE_FACTOR
        )

        # Extract transformer config for helper methods.
        transformer_config = manifest["transformer"]
        encoding = transformer_config.quantization_encoding or "bfloat16"
        # For NVFP4, weights are stored as FP4 but compute stays bfloat16.
        self._model_dtype: DType = (
            DType.bfloat16
            if encoding == "float4_e2m1fnx2"
            else supported_encoding_dtype(encoding)
        )
        self._model_device: Device = load_devices(
            transformer_config.device_specs
        )[0]
        self._in_channels: int = 128  # Flux2 in_channels (pre-patchify: 32)

        # Build and store all compiled graphs.
        self.text_encoder = TextEncoder(manifest, session)
        self.image_encoder = ImageEncoder(manifest, session)
        self.decoder = VaeDecoder(manifest, session)

        # Conditionally build fused OR split denoise graphs.
        self._denoise_compute: DenoiseCompute | None
        self._denoise_predict: DenoisePredict | None
        self._taylor_cache: TaylorSeerCache | None

        if self._cache_config.taylorseer:
            logger.info(
                "TaylorSeer enabled: building split DenoiseCompute + "
                "DenoisePredict graphs (warmup=%d, interval=%d, order=%d).",
                self._cache_config.taylorseer_warmup_steps,
                self._cache_config.taylorseer_cache_interval,
                self._cache_config.taylorseer_max_order,
            )
            self.denoiser = None
            self._denoise_compute = DenoiseCompute(manifest, session)
            self._denoise_predict = DenoisePredict(
                manifest, session, self._model_dtype, self._model_device
            )
            self._taylor_cache = TaylorSeerCache(
                config=self._cache_config,
                dtype=self._model_dtype,
                device=self._model_device,
                session=session,
            )
        else:
            self.denoiser = Denoiser(manifest, session)
            self._denoise_compute = None
            self._denoise_predict = None
            self._taylor_cache = None

    # -- PipelineExecutor interface -------------------------------------------

    @traced(message="prepare_inputs")
    def prepare_inputs(
        self, contexts: list[PixelContext]
    ) -> Flux2ExecutorInputs:
        if len(contexts) != 1:
            raise ValueError(
                "Flux2Executor currently supports batch_size=1. "
                f"Got {len(contexts)} contexts."
            )
        context = contexts[0]

        if context.latents.size == 0:
            raise ValueError(
                "Flux2Executor requires non-empty latents in PixelContext"
            )
        if context.latent_image_ids.size == 0:
            raise ValueError(
                "Flux2Executor requires non-empty latent_image_ids "
                "in PixelContext"
            )
        if context.sigmas.size == 0:
            raise ValueError(
                "Flux2Executor requires non-empty sigmas in PixelContext"
            )

        latent_h = context.height // self._vae_scale_factor
        latent_w = context.width // self._vae_scale_factor
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = packed_h * packed_w

        tokens = Buffer.from_dlpack(context.tokens.array)
        text_ids = Buffer.from_dlpack(context.text_ids)
        latents = self._patchify_and_pack(context.latents)
        latent_image_ids = Buffer.from_dlpack(context.latent_image_ids)
        timesteps, dts = self._prepare_scheduler(context.sigmas)

        guidance = Buffer.from_dlpack(
            np.full(
                [context.num_images_per_prompt],
                context.guidance_scale,
                dtype=np.float32,
            )
        )

        h_carrier = Buffer.from_dlpack(np.empty(packed_h, dtype=np.float32))
        w_carrier = Buffer.from_dlpack(np.empty(packed_w, dtype=np.float32))

        height = Buffer.from_dlpack(np.array([context.height], dtype=np.int64))
        width = Buffer.from_dlpack(np.array([context.width], dtype=np.int64))
        num_inference_steps = Buffer.from_dlpack(
            np.array([context.num_inference_steps], dtype=np.int64)
        )
        num_images_per_prompt = Buffer.from_dlpack(
            np.array([context.num_images_per_prompt], dtype=np.int64)
        )
        image_seq_len_buf = Buffer.from_dlpack(
            np.array([image_seq_len], dtype=np.int64)
        )

        input_image: Buffer | None = None
        if context.input_image is not None:
            input_image = Buffer.from_dlpack(context.input_image)

        return Flux2ExecutorInputs(
            tokens=tokens,
            text_ids=text_ids,
            latents=latents,
            latent_image_ids=latent_image_ids,
            timesteps=timesteps,
            dts=dts,
            guidance=guidance,
            image_seq_len=image_seq_len_buf,
            h_carrier=h_carrier,
            w_carrier=w_carrier,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            input_image=input_image,
        )

    @traced(message="execute")
    def execute(self, inputs: Flux2ExecutorInputs) -> Flux2ExecutorOutputs:
        # Bulk device transfer -- CPU-only metadata fields are skipped by
        # Flux2ExecutorInputs.to().  Buffers already on the target device
        # are no-ops.
        inputs = inputs.to(self._model_device)

        # 1) Encode prompts (Graph 1).
        prompt_embeds = self._encode_prompts(
            inputs.tokens,
            inputs.num_images_per_prompt,
        )
        text_ids = inputs.text_ids

        # 2) Encode image if img2img (Graph 2).
        #    For text-to-image, pass zero-seq-length tensors so the fused
        #    denoise graph's concat is a no-op.
        if inputs.input_image is not None:
            image_latents, image_latent_ids = self._encode_image(
                inputs.input_image,
            )
        else:
            image_latents = self._empty_image_latents()
            image_latent_ids = self._empty_image_latent_ids()

        # 3) Denoising loop (Graph 3).
        latents = self._run_denoising_loop(
            latents=inputs.latents,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=inputs.latent_image_ids,
            timesteps=inputs.timesteps,
            dts=inputs.dts,
            guidance=inputs.guidance,
            num_inference_steps=inputs.num_inference_steps,
        )

        # 4) Decode final latents into images (Graph 4).
        images = self._decode_latents(
            latents,
            inputs.h_carrier,
            inputs.w_carrier,
        )

        return Flux2ExecutorOutputs(images=images)

    # -- Graph 1: Text Encode -------------------------------------------------

    @traced(message="encode_prompts")
    def _encode_prompts(
        self,
        tokens: Buffer,
        num_images_per_prompt: Buffer,
    ) -> Buffer:
        """Run Graph 1: encode text prompts into embeddings.

        Args:
            tokens: Token IDs, shape ``(S,)``.
            num_images_per_prompt: 1-element int64 tensor.

        Returns:
            ``prompt_embeds`` Buffer of shape ``(B, S, L*D)``.
        """
        return self.text_encoder(tokens)

    # -- Graph 2: Image Encode (img2img only) ---------------------------------

    @traced(message="encode_image")
    def _encode_image(
        self,
        input_image: Buffer,
    ) -> tuple[Buffer, Buffer]:
        """Run Graph 2: encode an input image into packed latents.

        VAE encode -> mode extraction -> patchify -> BN-normalize -> pack.

        Args:
            input_image: Raw input image, shape ``(H, W, C)`` uint8.

        Returns:
            A tuple of ``(image_latents, image_latent_ids)`` where:
            - ``image_latents`` has shape ``(B, img_seq, C*4)``
            - ``image_latent_ids`` has shape ``(B, img_seq, 4)`` int64
        """
        return self.image_encoder(input_image)

    # -- Graph 3: Denoise Step ------------------------------------------------

    @traced(message="denoise_step")
    def _denoise_step(
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
        """Run Graph 3: one fused denoise step.

        Executes concat + transformer forward + Euler scheduler step.
        Only used when TaylorSeer is disabled (fused path).

        Args:
            latents: Current latent state, shape ``(B, seq, C)``.
            image_latents: Packed image latents for img2img, shape
                ``(B, img_seq, C)``. Zero-seq-length ``(B, 0, C)``
                for text-to-image.
            encoder_hidden_states: Text embeddings, shape ``(B, S, D)``.
            timestep: Current sigma, shape ``(B,)``.
            dt: Step delta ``sigma[i+1] - sigma[i]``, shape ``(1,)``.
            guidance: Guidance scale, shape ``(B,)``.
            latent_image_ids: Latent position IDs, shape ``(B, seq, 4)``
                int64.
            image_latent_ids: Image latent position IDs, shape
                ``(B, img_seq, 4)`` int64.
            txt_ids: Text position IDs, shape ``(B, S, 4)`` int64.

        Returns:
            Updated latents, shape ``(B, seq, C)``.
        """
        assert self.denoiser is not None, (
            "_denoise_step called but fused Denoiser was not compiled. "
            "This is a bug — TaylorSeer path should use "
            "_denoise_compute + _denoise_predict instead."
        )
        return self.denoiser(
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

    # -- Graph 4: Decode ------------------------------------------------------

    @traced(message="decode_latents")
    def _decode_latents(
        self,
        latents: Buffer,
        h_carrier: Buffer,
        w_carrier: Buffer,
    ) -> Buffer:
        """Run Graph 4: decode denoised latents into an image tensor.

        BN-denormalize -> unpatchify -> VAE decode -> uint8.

        Args:
            latents: Denoised packed latents, shape ``(B, seq, C*4)``.
            h_carrier: Shape carrier of length ``packed_h``.
            w_carrier: Shape carrier of length ``packed_w``.

        Returns:
            Decoded images, shape ``(B, H, W, C)`` uint8.
        """
        return self.decoder(latents, h_carrier, w_carrier)

    # -- Denoising loop orchestration -----------------------------------------

    @traced(message="run_denoising_loop")
    def _run_denoising_loop(
        self,
        latents: Buffer,
        image_latents: Buffer,
        image_latent_ids: Buffer,
        prompt_embeds: Buffer,
        text_ids: Buffer,
        latent_image_ids: Buffer,
        timesteps: Buffer,
        dts: Buffer,
        guidance: Buffer,
        num_inference_steps: Buffer,
    ) -> Buffer:
        """Orchestrate the N-step denoising loop.

        When TaylorSeer is disabled, each step executes the fused denoise
        graph (Graph 3: concat + transformer + Euler step).

        When TaylorSeer is enabled, the loop alternates between:
        - **Compute steps** (warmup + every K-th step): run
          ``DenoiseCompute`` (transformer-only), then update Taylor
          factors.
        - **Predict steps** (all other steps): predict ``noise_pred``
          from cached Taylor factors, skipping the transformer entirely.
        Both paths feed into ``DenoisePredict`` (Euler step).

        Args:
            latents: Preprocessed packed latents, shape ``(B, seq, C)``.
            image_latents: Packed image latents for img2img, shape
                ``(B, img_seq, C)``. Zero-seq-length for text-to-image.
            image_latent_ids: Image latent position IDs, shape
                ``(B, img_seq, 4)`` int64. Zero-seq-length for
                text-to-image.
            prompt_embeds: Text encoder embeddings, shape ``(B, S, D)``.
            text_ids: Text position IDs, shape ``(B, S, 4)`` int64.
            latent_image_ids: Latent position IDs, shape ``(B, seq, 4)``
                int64.
            timesteps: Precomputed timesteps, shape ``(num_steps,)``
                (model dtype).
            dts: Precomputed step deltas, shape ``(num_steps,)`` (float32).
            guidance: Guidance scale, shape ``(B,)``.
            num_inference_steps: 1-element int64 tensor.

        Returns:
            Final denoised latents, shape ``(B, seq, C)``.
        """
        num_steps: int = np.from_dlpack(num_inference_steps).item()  # type: ignore[assignment]

        if self._taylor_cache is None:
            # Fused path (no TaylorSeer).
            for i in range(num_steps):
                timestep_i = timesteps[i : i + 1]
                dt_i = dts[i : i + 1]
                latents = self._denoise_step(
                    latents,
                    image_latents,
                    prompt_embeds,
                    timestep_i,
                    dt_i,
                    guidance,
                    latent_image_ids,
                    image_latent_ids,
                    text_ids,
                )
        else:
            # TaylorSeer path: split compute + predict.
            assert self._denoise_compute is not None
            assert self._denoise_predict is not None

            # Infer state dimensions from latents shape for factor
            # allocation.  latents is (B, seq, C).
            batch_size, seq_len, output_dim = latents.shape
            state = self._taylor_cache.create_state(
                batch_size, seq_len, output_dim
            )

            for i in range(num_steps):
                timestep_i = timesteps[i : i + 1]
                dt_i = dts[i : i + 1]

                if self._taylor_cache.should_skip(i):
                    # Predict step: skip the transformer.
                    noise_pred = self._taylor_cache.predict(state, i)
                else:
                    # Compute step: run full transformer.
                    noise_pred = self._denoise_compute(
                        latents,
                        image_latents,
                        prompt_embeds,
                        timestep_i,
                        guidance,
                        latent_image_ids,
                        image_latent_ids,
                        text_ids,
                    )
                    self._taylor_cache.update(state, noise_pred, i)

                latents = self._denoise_predict(latents, noise_pred, dt_i)
        return latents

    # -- Latent preprocessing -------------------------------------------------

    def _patchify_and_pack(
        self,
        latents: npt.NDArray[np.float32],
    ) -> Buffer:
        """Patchify and pack raw latents for the transformer.

        Reshapes ``(B, C, H, W)`` -> ``(B, H//2 * W//2, C*4)`` via
        2x2 patchification followed by sequence packing.

        Args:
            latents: Raw latent noise, shape ``(B, C, H, W)`` float32.

        Returns:
            Packed latents, shape ``(B, seq, C*4)`` in model dtype.
        """
        arr = latents  # (B, C, H, W) float32
        b, c, h, w = arr.shape
        h2, w2 = h // 2, w // 2
        # Patchify: (B, C, H, W) -> (B, C, H//2, 2, W//2, 2)
        arr = arr.reshape(b, c, h2, 2, w2, 2)
        # -> (B, C, 2, 2, H//2, W//2) -> (B, C*4, H//2, W//2)
        arr = arr.transpose(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h2, w2)
        # Pack: (B, C*4, H//2, W//2) -> (B, H//2*W//2, C*4)
        arr = arr.reshape(b, c * 4, h2 * w2).transpose(0, 2, 1)
        arr = np.ascontiguousarray(arr)
        return float32_array_to_buffer(
            arr, dtype=self._model_dtype, device=self._model_device
        )

    # -- Zero-seq-length image latents for t2i --------------------------------

    def _empty_image_latents(self) -> Buffer:
        """Create a zero-seq-length image latent buffer for text-to-image.

        Returns a ``(1, 0, C)`` buffer in model dtype so the fused
        denoise graph's concat is a no-op along the sequence dimension.

        Returns:
            Buffer with shape ``(1, 0, C)``.
        """
        return float32_array_to_buffer(
            np.zeros((1, 0, self._in_channels), dtype=np.float32),
            dtype=self._model_dtype,
            device=self._model_device,
        )

    def _empty_image_latent_ids(self) -> Buffer:
        """Create a zero-seq-length image latent ID buffer for text-to-image.

        Returns a ``(1, 0, 4)`` int64 buffer so the fused denoise
        graph's latent ID concat is a no-op along the sequence dimension.

        Returns:
            Buffer with shape ``(1, 0, 4)``.
        """
        return Buffer.from_dlpack(np.zeros((1, 0, 4), dtype=np.int64)).to(
            self._model_device
        )

    # -- Scheduler utilities --------------------------------------------------

    def _prepare_scheduler(
        self,
        sigmas: npt.NDArray[np.float32],
    ) -> tuple[Buffer, Buffer]:
        """Precompute timesteps and dt arrays from the sigma schedule.

        Timesteps are kept as float32 (the denoise graph casts to model
        dtype in-graph). Step deltas are float32 for numerical stability.

        Args:
            sigmas: Sigma schedule, shape ``(num_steps+1,)`` float32.

        Returns:
            A tuple of ``(timesteps, dts)`` where:
            - ``timesteps`` has shape ``(num_steps,)`` float32 on device
            - ``dts`` has shape ``(num_steps,)`` float32 on device
        """
        timesteps = np.ascontiguousarray(sigmas[:-1])
        dts = np.ascontiguousarray(sigmas[1:] - sigmas[:-1])
        return (
            Buffer.from_dlpack(timesteps),
            Buffer.from_dlpack(dts),
        )

    # -- Cache defaults -------------------------------------------------------

    # Flux2 model-specific defaults.
    _DEFAULT_TAYLORSEER_CACHE_INTERVAL: int = 5
    _DEFAULT_TAYLORSEER_WARMUP_STEPS: int = 9
    _DEFAULT_TAYLORSEER_MAX_ORDER: int = 1

    def _resolve_cache_defaults(self) -> None:
        """Fill nullable DenoisingCacheConfig fields with Flux2 defaults."""
        cc = self._cache_config
        if cc.taylorseer_cache_interval is None:
            cc.taylorseer_cache_interval = (
                self._DEFAULT_TAYLORSEER_CACHE_INTERVAL
            )
        if cc.taylorseer_warmup_steps is None:
            cc.taylorseer_warmup_steps = self._DEFAULT_TAYLORSEER_WARMUP_STEPS
        if cc.taylorseer_max_order is None:
            cc.taylorseer_max_order = self._DEFAULT_TAYLORSEER_MAX_ORDER
