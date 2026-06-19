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
"""Flux2 single-``Module`` model (ModuleV3 scaffold)."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, load_devices
from max.dtype import DType
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.pipelines.architectures.flux2.flux2_executor import (
    Flux2ExecutorOutputs,
)
from max.pipelines.architectures.flux_modulev3 import Vae
from max.pipelines.context import PixelContext
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.profiler import traced

from .components import TextEncoder
from .flux2_inputs import Flux2ModuleV3Inputs

logger = logging.getLogger("max.pipelines")


class FLUXModule(Module[[Tensor, Tensor], tuple[Tensor, Tensor]]):
    """Single-``Module`` implementation of the FLUX.2 pipeline.

    Expresses the entire FLUX.2 forward pass -- text encode, optional
    image encode, denoising loop, and VAE decode -- inside a single
    :class:`~max.experimental.nn.Module` tree, with sub-Modules placed
    on the device specified by their corresponding manifest entry (for
    example, ``text_encoder`` on
    ``manifest["text_encoder"].device_specs[0]``).

    No graph construction, weight binding, or session compilation lives
    on this class.  The caller (typically
    :class:`~max.pipelines.diffusion.pipeline.PixelGenerationPipeline`)
    is responsible for:

    1. Wrapping construction in :func:`max.experimental.functional.lazy`.
    2. Composing ``manifest.loader()`` through
       :func:`~max.pipelines.lib.weight_loader.adapt_module_loader` and
       materialising the Module's declared parameters into a
       compile-ready ``weights`` mapping (the walker descends into every
       ``HasLoaderAdapter`` sub-Module automatically).
    3. Calling :meth:`~max.experimental.nn.Module.compile` with
       :meth:`input_types` and the materialised weights.
    4. Driving the resulting ``CompiledModel`` against
       :meth:`prepare_inputs` and :meth:`from_outputs` for each batch.
    """

    # Fallback VAE scale factor when not derivable from the manifest.
    _DEFAULT_VAE_SCALE_FACTOR: int = 8

    def __init__(self, manifest: ModelManifest) -> None:
        self._manifest = manifest

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

        # Extract transformer config for input-staging dtype.
        transformer_config = manifest["transformer"]
        encoding = transformer_config.quantization_encoding or "bfloat16"
        # For NVFP4, weights are stored as FP4 but compute stays bfloat16.
        self._model_dtype: DType = (
            DType.bfloat16
            if encoding == "float4_e2m1fnx2"
            else supported_encoding_dtype(encoding)
        )

        # Sub-Modules: built up component-by-component.  Each component
        # receives its own ``device_specs`` from the corresponding
        # manifest entry and is responsible for placing itself on the
        # appropriate device inside its own constructor.  Their
        # devices are also cached on this Module so :meth:`prepare_inputs`
        # can stage Buffers onto the matching device before the
        # compiled graph runs.
        text_encoder_config = manifest["text_encoder"]
        self._text_encoder_device: Device = load_devices(
            text_encoder_config.device_specs
        )[0]
        self.text_encoder = TextEncoder(
            huggingface_config=text_encoder_config.huggingface_config.to_dict(),
            quantization_encoding=text_encoder_config.quantization_encoding,
            device_specs=text_encoder_config.device_specs,
        )

        vae_manifest_config = manifest["vae"]
        self._vae_device: Device = load_devices(
            vae_manifest_config.device_specs
        )[0]
        self.vae = Vae(
            huggingface_config=vae_manifest_config.huggingface_config.to_dict(),
            quantization_encoding=vae_manifest_config.quantization_encoding,
            device_specs=vae_manifest_config.device_specs,
        )

    # -- Module forward + compile surface -------------------------------------

    def forward(
        self, tokens: Tensor, input_image: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through the Module tree.

        Runs both the text encoder and the VAE image encoder
        unconditionally: ``input_image`` is always passed (a
        ``(0, 0, 3)`` placeholder for text-to-image requests), and the
        zero spatial dims propagate through to a ``(1, 0, num_channels)``
        latent that the downstream denoiser concatenates as a no-op.
        Additional components (denoiser, VAE decoder) will plug in here
        as they are ported.

        Returns:
            ``(text_embeddings, image_latents)``.
        """
        text_embeddings = self.text_encoder(tokens)
        image_latents = self.vae.encode(input_image)
        return text_embeddings, image_latents

    def input_types(self) -> tuple[TensorType, ...]:
        """Input tensor types for compilation, sourced from sub-Modules."""
        return (
            *self.text_encoder.input_types(),
            *self.vae.input_types(),
        )

    # -- Pipeline I/O contract -----------------------------------------------

    @traced(message="prepare_inputs")
    def prepare_inputs(
        self, contexts: list[PixelContext]
    ) -> Flux2ModuleV3Inputs:
        if len(contexts) != 1:
            raise ValueError(
                "FLUXModule currently supports batch_size=1. "
                f"Got {len(contexts)} contexts."
            )
        context = contexts[0]

        if context.latent_image_ids.size == 0:
            raise ValueError(
                "FLUXModule requires non-empty latent_image_ids in PixelContext"
            )
        if context.sigmas.size == 0:
            raise ValueError(
                "FLUXModule requires non-empty sigmas in PixelContext"
            )
        if context.seed is None:
            raise ValueError(
                "FLUXModule requires a seed on PixelContext; "
                "pass `seed` on the request body so the graph can sample "
                "initial noise deterministically."
            )

        latent_h = context.height // self._vae_scale_factor
        latent_w = context.width // self._vae_scale_factor
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = packed_h * packed_w

        tokens = Buffer.from_dlpack(context.tokens.array).to(
            self._text_encoder_device
        )
        text_ids = Buffer.from_dlpack(context.text_ids)
        seed = Buffer.from_dlpack(np.array([context.seed], dtype=np.int64))
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

        # Always stage an input_image buffer on the VAE's device.  When
        # the request is text-only the placeholder has shape ``(0, 0, 3)``
        # so the encoder runs without contributing to the denoiser.
        if context.input_image is not None:
            input_image_array = context.input_image
        else:
            input_image_array = np.empty((0, 0, 3), dtype=np.uint8)
        input_image = Buffer.from_dlpack(input_image_array).to(self._vae_device)

        return Flux2ModuleV3Inputs(
            tokens=tokens,
            text_ids=text_ids,
            seed=seed,
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

    def from_outputs(self, tensors: list[Tensor]) -> Flux2ExecutorOutputs:
        """Translate compiled-call output tensors into pipeline-facing outputs.

        The pipeline invokes the compiled :class:`~max.experimental.nn.Module`
        and receives a flat list of output tensors; this method packs them
        into a :class:`Flux2ExecutorOutputs` so the
        :class:`~max.pipelines.lib.pipeline_variants.pixel_generation.PixelGenerationPipeline`
        can keep its post-processing logic uniform across the legacy
        executor and ModuleV3 paths.

        Stubbed until :meth:`forward` produces a decoded image; today
        ``forward`` only runs the text encoder.
        """
        raise NotImplementedError(
            "FLUXModule.from_outputs() is not implemented yet."
        )

    # -- Input staging helpers ------------------------------------------------

    def _prepare_scheduler(
        self,
        sigmas: npt.NDArray[np.float32],
    ) -> tuple[Buffer, Buffer]:
        """Precompute ``(timesteps, dts)`` buffers from the sigma schedule."""
        timesteps = np.ascontiguousarray(sigmas[:-1])
        dts = np.ascontiguousarray(sigmas[1:] - sigmas[:-1])
        return (
            Buffer.from_dlpack(timesteps),
            Buffer.from_dlpack(dts),
        )
