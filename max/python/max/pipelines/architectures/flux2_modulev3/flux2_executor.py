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
from max.engine import InferenceSession
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.pipelines.architectures.flux2.flux2_executor import (
    Flux2ExecutorOutputs,
)
from max.pipelines.context import PixelContext
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_executor import PipelineExecutor
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.profiler import traced

from .flux2_inputs import Flux2ModuleV3Inputs

logger = logging.getLogger("max.pipelines")


class FLUXModule(
    Module[..., Tensor],
    PipelineExecutor[PixelContext, Flux2ModuleV3Inputs, Flux2ExecutorOutputs],
):
    """Single-``Module`` implementation of the FLUX.2 pipeline.

    This is the ModuleV3 counterpart to
    :class:`max.pipelines.architectures.flux2.Flux2Executor`.  The intent
    is to express the entire FLUX.2 forward pass -- text encode, optional
    image encode, denoising loop, and VAE decode -- inside a single
    :class:`~max.experimental.nn.Module`, rather than as four separately
    compiled graphs glued together by an executor.

    The :class:`PipelineExecutor` mix-in is temporary: it lets
    :class:`~max.pipelines.lib.pipeline_variants.pixel_generation.PixelGenerationPipeline`
    discover and instantiate this class through the existing
    ``manifest``/``session``/``runtime_config`` entry point.  Once the
    serving layer can drive a bare ``Module``, the mix-in can be dropped.
    """

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

        # Extract transformer config for input-staging helpers.
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

    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        raise NotImplementedError(
            "FLUXModule (ModuleV3) forward() is not implemented yet."
        )

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
                "FLUXModule (ModuleV3) requires a seed on PixelContext; "
                "pass `seed` on the request body so the graph can sample "
                "initial noise deterministically."
            )

        latent_h = context.height // self._vae_scale_factor
        latent_w = context.width // self._vae_scale_factor
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        image_seq_len = packed_h * packed_w

        tokens = Buffer.from_dlpack(context.tokens.array)
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

        input_image: Buffer | None = None
        if context.input_image is not None:
            input_image = Buffer.from_dlpack(context.input_image)

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

    def execute(self, inputs: Flux2ModuleV3Inputs) -> Flux2ExecutorOutputs:
        raise NotImplementedError(
            "FLUXModule (ModuleV3) execute() is not implemented yet."
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
