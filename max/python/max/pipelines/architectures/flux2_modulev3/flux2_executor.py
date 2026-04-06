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

from dataclasses import dataclass, replace

import numpy as np
from max.driver import Buffer
from max.engine import InferenceSession, Model
from max.pipelines.core import PixelContext
from max.pipelines.lib.interfaces import TensorStruct
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_executor import PipelineExecutor
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.profiler import traced
from typing_extensions import Self

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
        inputs = inputs.with_residual_threshold(threshold_tensor)
    """

    # -- Core (always present) ------------------------------------------------

    tokens: Buffer
    """Token IDs for the text encoder, shape ``(S,)``."""

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

    residual_threshold: Buffer | None = None
    """Scalar float32 threshold for FBCache residual gating.
    ``None`` when first-block caching is not enabled."""

    # -- Feature builders (return new frozen instance) ------------------------

    def with_image(self, input_image: Buffer) -> Self:
        """Enable image-to-image mode with the given input image."""
        return replace(self, input_image=input_image)

    def with_residual_threshold(self, residual_threshold: Buffer) -> Self:
        """Enable FBCache with a per-request residual threshold."""
        return replace(self, residual_threshold=residual_threshold)


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
    """Flux2 pipeline executor (stub).

    Implements the :class:`PipelineExecutor` interface for Flux2 image
    generation. This class will eventually replace :class:`Flux2Pipeline`
    by wiring together the same sub-components (text encoder, transformer
    denoiser, VAE) through the tensor-in/tensor-out executor contract.

    Target graph structure (without TaylorSeer -- 4 graphs, 3 for t2i):

    +---------+-----------------------------------------+-----------------------+
    | Graph   | Purpose                                 | Called                |
    +---------+-----------------------------------------+-----------------------+
    | 1       | Text Encode: Mistral3 -> embeddings      | Once per request      |
    | 2       | Image Encode: VAE encode + BN-norm + pack | Once (img2img only)  |
    | 3       | Denoise Step: concat + transformer +     | Every denoising step  |
    |         |   scheduler step                         |                       |
    | 4       | Decode: BN-denorm + unpatchify +         | Once per request      |
    |         |   VAE decode -> uint8                    |                       |
    +---------+-----------------------------------------+-----------------------+

    With TaylorSeer enabled, Graph 3 splits into two variants (5 graphs):

    - **3a** (Compute): concat + transformer + scheduler step + Taylor
      factor update. Runs during warmup and every K-th step.
    - **3b** (Predict): Taylor predict + scheduler step. Contains zero
      model weights. Runs on all other steps.

    Python-level branching dispatches between 3a and 3b -- MAX graphs
    don't support true control flow, so both variants are independently
    captured as CUDA graphs.
    """

    # -- Compiled graphs (set during __init__) --------------------------------

    text_encoder: Model
    """Graph 1: Mistral3 text encoder -> prompt_embeds + text_ids."""

    image_encoder: Model
    """Graph 2: VAE encode + BN-normalize + patchify + pack."""

    denoise_compute: Model
    """Graph 3a: concat + transformer + scheduler step (+ Taylor update)."""

    denoise_predict: Model | None
    """Graph 3b: Taylor predict + scheduler step. ``None`` when TaylorSeer
    is disabled."""

    decoder: Model
    """Graph 4: BN-denormalize + unpatchify + VAE decode -> uint8."""

    # Default residual threshold when FBCache is enabled but the request
    # does not specify one.  Matches Flux2Pipeline.default_residual_threshold
    # (overrides the DiffusionPipeline base default of 0.05).
    _DEFAULT_RESIDUAL_THRESHOLD: float = 0.06

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

        # Derived config from the manifest.
        # TODO(GENAI-490): Derive vae_scale_factor from manifest VAE config
        # once ModelManifest exposes per-component configs.
        self._vae_scale_factor = self._DEFAULT_VAE_SCALE_FACTOR
        self._default_residual_threshold = self._DEFAULT_RESIDUAL_THRESHOLD

        # Build and store all compiled graphs.
        self.text_encoder = self._build_text_encode_graph()
        self.image_encoder = self._build_image_encode_graph()
        self.denoise_compute = self._build_denoise_compute_graph()
        self.denoise_predict = self._build_denoise_predict_graph()
        self.decoder = self._build_decode_graph()

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
        latents = self._patchify_and_pack(Buffer.from_dlpack(context.latents))
        latent_image_ids = Buffer.from_dlpack(context.latent_image_ids)
        timesteps, dts = self._prepare_scheduler(
            Buffer.from_dlpack(context.sigmas)
        )

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

        residual_threshold: Buffer | None = None
        if context.residual_threshold is not None:
            residual_threshold = Buffer.from_dlpack(
                np.array(context.residual_threshold, dtype=np.float32)
            )

        return Flux2ExecutorInputs(
            tokens=tokens,
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
            residual_threshold=residual_threshold,
        )

    @traced(message="execute")
    def execute(self, inputs: Flux2ExecutorInputs) -> Flux2ExecutorOutputs:
        # 1) Encode prompts (Graph 1).
        prompt_embeds, text_ids = self._encode_prompts(
            inputs.tokens,
            inputs.num_images_per_prompt,
        )

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

        # 3) Denoising loop (Graphs 3a/3b).
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
            residual_threshold=inputs.residual_threshold,
        )

        # 4) Decode final latents into images (Graph 4).
        images = self._decode_latents(
            latents,
            inputs.h_carrier,
            inputs.w_carrier,
        )

        return Flux2ExecutorOutputs(images=images)

    # -- Graph 1: Text Encode -------------------------------------------------

    @traced(message="build_text_encode_graph")
    def _build_text_encode_graph(self) -> Model:
        """Compile the Mistral3 text encoder graph.

        Produces ``prompt_embeds`` and ``text_ids`` from token IDs.

        Returns:
            Compiled :class:`Model` for Graph 1.
        """
        raise NotImplementedError

    @traced(message="encode_prompts")
    def _encode_prompts(
        self,
        tokens: Buffer,
        num_images_per_prompt: Buffer,
    ) -> tuple[Buffer, Buffer]:
        """Run Graph 1: encode text prompts into embeddings and text IDs.

        Args:
            tokens: Token IDs, shape ``(S,)``.
            num_images_per_prompt: 1-element int64 tensor.

        Returns:
            A tuple of ``(prompt_embeds, text_ids)`` where:
            - ``prompt_embeds`` has shape ``(B, S, L*D)``
            - ``text_ids`` has shape ``(B, S, 4)`` int64
        """
        raise NotImplementedError

    # -- Graph 2: Image Encode (img2img only) ---------------------------------

    @traced(message="build_image_encode_graph")
    def _build_image_encode_graph(self) -> Model:
        """Compile the VAE encoder + BN-normalize + patchify + pack graph.

        Produces ``image_latents`` and ``image_latent_ids`` from a raw
        input image.

        Returns:
            Compiled :class:`Model` for Graph 2.
        """
        raise NotImplementedError

    @traced(message="encode_image")
    def _encode_image(
        self,
        input_image: Buffer,
    ) -> tuple[Buffer, Buffer]:
        """Run Graph 2: encode an input image into packed latents.

        VAE encode -> BN-normalize -> patchify + pack.

        Args:
            input_image: Raw input image, shape ``(H, W, C)`` uint8.

        Returns:
            A tuple of ``(image_latents, image_latent_ids)`` where:
            - ``image_latents`` has shape ``(B, img_seq, C*4)``
            - ``image_latent_ids`` has shape ``(B, img_seq, 4)`` int64
        """
        raise NotImplementedError

    # -- Graph 3a: Denoise Compute Step ---------------------------------------

    @traced(message="build_denoise_compute_graph")
    def _build_denoise_compute_graph(self) -> Model:
        """Compile the fused denoise compute step graph.

        Fuses concat + transformer forward + scheduler step into a single
        compiled graph. For text-to-image, ``image_latents`` is passed as
        a zero-seq-length tensor ``(B, 0, C*4)``, making the concat a
        no-op.

        When TaylorSeer is enabled, this graph additionally updates
        Taylor factors via divided differences.

        Returns:
            Compiled :class:`Model` for Graph 3a.
        """
        raise NotImplementedError

    def _denoise_compute_step(
        self,
        latents: Buffer,
        image_latents: Buffer,
        image_latent_ids: Buffer,
        prompt_embeds: Buffer,
        text_ids: Buffer,
        latent_image_ids: Buffer,
        timestep: Buffer,
        dt: Buffer,
        guidance: Buffer,
        taylor_factors: tuple[Buffer, ...] | None,
        residual_threshold: Buffer | None,
    ) -> tuple[Buffer, tuple[Buffer, ...] | None]:
        """Run Graph 3a: full denoise compute step.

        Executes concat + transformer forward + scheduler step. When
        TaylorSeer is enabled, also updates Taylor factors via divided
        differences.

        Args:
            latents: Current latent state, shape ``(B, seq, C*4)``.
            image_latents: Packed image latents for img2img, shape
                ``(B, img_seq, C*4)``. Zero-seq-length ``(B, 0, C*4)``
                for text-to-image.
            image_latent_ids: Image latent position IDs, shape
                ``(B, img_seq, 4)`` int64. Zero-seq-length for
                text-to-image.
            prompt_embeds: Text encoder embeddings, shape ``(B, S, L*D)``.
            text_ids: Text position IDs, shape ``(B, S, 4)`` int64.
            latent_image_ids: Latent position IDs, shape ``(B, seq, 4)``
                int64.
            timestep: Current sigma value, shape ``(1,)``.
            dt: Step delta ``sigma[i+1] - sigma[i]``, shape ``(1,)``.
            guidance: Guidance scale, shape ``(B,)``.
            taylor_factors: Cached Taylor series factors from previous
                compute steps, or ``None`` when TaylorSeer is disabled.
            residual_threshold: FBCache residual threshold scalar, or
                ``None`` when first-block caching is disabled.

        Returns:
            A tuple of ``(updated_latents, updated_taylor_factors)`` where
            ``updated_taylor_factors`` is ``None`` when TaylorSeer is
            disabled.
        """
        raise NotImplementedError

    # -- Graph 3b: Denoise Predict Step (TaylorSeer only) ---------------------

    @traced(message="build_denoise_predict_graph")
    def _build_denoise_predict_graph(self) -> Model | None:
        """Compile the Taylor predict + scheduler step graph.

        This graph contains zero model weights -- it is purely arithmetic
        on cached Taylor factors and latents. Only built when TaylorSeer
        is enabled.

        Returns:
            Compiled :class:`Model` for Graph 3b, or ``None`` when
            TaylorSeer is disabled.
        """
        raise NotImplementedError

    def _denoise_predict_step(
        self,
        latents: Buffer,
        taylor_factors: tuple[Buffer, ...],
        step_offset: Buffer,
        dt: Buffer,
    ) -> Buffer:
        """Run Graph 3b: Taylor series prediction step.

        Skips the full transformer and estimates noise_pred via Taylor
        approximation: ``factor_0 + factor_1 * d + factor_2 * d^2 / 2``.

        Args:
            latents: Current latent state, shape ``(B, seq, C*4)``.
            taylor_factors: Cached Taylor series factors from the most
                recent compute step.
            step_offset: Offset from the last compute step, shape ``(1,)``.
            dt: Step delta ``sigma[i+1] - sigma[i]``, shape ``(1,)``.

        Returns:
            Updated latents after the predicted scheduler step, shape
            ``(B, seq, C*4)``.
        """
        raise NotImplementedError

    # -- Graph 4: Decode ------------------------------------------------------

    @traced(message="build_decode_graph")
    def _build_decode_graph(self) -> Model:
        """Compile the BN-denormalize + unpatchify + VAE decode graph.

        Produces uint8 images from denoised packed latents.

        Returns:
            Compiled :class:`Model` for Graph 4.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        residual_threshold: Buffer | None,
    ) -> Buffer:
        """Orchestrate the N-step denoising loop with Python-level branching.

        Dispatches between :meth:`_denoise_compute_step` (Graph 3a) and
        :meth:`_denoise_predict_step` (Graph 3b) based on the step index.
        When TaylorSeer is disabled, every step is a compute step.

        The branching decision is simple modular arithmetic on the step
        index -- no GPU sync or dynamic condition required.

        Args:
            latents: Preprocessed packed latents, shape ``(B, seq, C*4)``.
            image_latents: Packed image latents for img2img, shape
                ``(B, img_seq, C*4)``. Zero-seq-length for text-to-image.
            image_latent_ids: Image latent position IDs, shape
                ``(B, img_seq, 4)`` int64. Zero-seq-length for
                text-to-image.
            prompt_embeds: Text encoder embeddings, shape ``(B, S, L*D)``.
            text_ids: Text position IDs, shape ``(B, S, 4)`` int64.
            latent_image_ids: Latent position IDs, shape ``(B, seq, 4)``
                int64.
            timesteps: Precomputed timesteps, shape ``(num_steps,)``
                (model dtype).
            dts: Precomputed step deltas, shape ``(num_steps,)`` (float32).
            guidance: Guidance scale, shape ``(B,)``.
            num_inference_steps: 1-element int64 tensor.
            residual_threshold: FBCache residual threshold scalar, or
                ``None`` when first-block caching is disabled.

        Returns:
            Final denoised latents, shape ``(B, seq, C*4)``.
        """
        raise NotImplementedError

    # -- Latent preprocessing -------------------------------------------------

    def _patchify_and_pack(
        self,
        latents: Buffer,
    ) -> Buffer:
        """Patchify and pack raw latents for the transformer.

        Reshapes ``(B, C, H, W)`` -> ``(B, H//2 * W//2, C*4)`` via
        2x2 patchification followed by sequence packing.

        Args:
            latents: Raw latent noise, shape ``(B, C, H, W)``.

        Returns:
            Packed latents, shape ``(B, seq, C*4)``.
        """
        raise NotImplementedError

    # -- Zero-seq-length image latents for t2i --------------------------------

    def _empty_image_latents(self) -> Buffer:
        """Create a zero-seq-length image latent buffer for text-to-image.

        Returns a ``(1, 0, C*4)`` float32 buffer so the fused denoise
        graph's concat is a no-op along the sequence dimension.

        Returns:
            Buffer with shape ``(1, 0, C*4)``.
        """
        raise NotImplementedError

    def _empty_image_latent_ids(self) -> Buffer:
        """Create a zero-seq-length image latent ID buffer for text-to-image.

        Returns a ``(1, 0, 4)`` int64 buffer so the fused denoise
        graph's latent ID concat is a no-op along the sequence dimension.

        Returns:
            Buffer with shape ``(1, 0, 4)``.
        """
        raise NotImplementedError

    # -- Scheduler utilities --------------------------------------------------

    def _prepare_scheduler(
        self,
        sigmas: Buffer,
    ) -> tuple[Buffer, Buffer]:
        """Precompute timesteps and dt arrays from the sigma schedule.

        Args:
            sigmas: Sigma schedule, shape ``(num_steps+1,)``.

        Returns:
            A tuple of ``(timesteps, dts)`` where:
            - ``timesteps`` has shape ``(num_steps,)`` (model dtype)
            - ``dts`` has shape ``(num_steps,)`` (float32)
        """
        raise NotImplementedError
