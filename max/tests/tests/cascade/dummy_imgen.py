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
"""Worker-based dummy image components for cascade pipeline selection tests."""

import asyncio
import io
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.experimental.cascade import (
    CascadePipeline,
    ImageGenInterface,
    ImageGenRequest,
    Worker,
    pipeline_method,
    worker_method,
)
from PIL import Image
from scipy import ndimage

Int32Array = npt.NDArray[np.int32]
UInt8Array = npt.NDArray[np.uint8]


class DummyTokenizer(Worker):
    """Build a small image-generation context from request parameters."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["cpu"])

    @worker_method()
    async def encode(self, prompt: str) -> Int32Array:
        """Return a deterministic token array for the prompt."""
        return np.array([ord(char) for char in prompt], dtype=np.int32)

    @worker_method()
    async def prepare_sigmas(
        self, height: int, width: int, num_steps: int
    ) -> Int32Array:
        """Return a small fake schedule tensor."""
        del height, width
        return np.arange(num_steps + 1, dtype=np.int32)

    @worker_method()
    async def prepare_latents(
        self, height: int, width: int, seed: int
    ) -> Int32Array:
        """Return a deterministic latent tensor seed carrier."""
        return np.array([height, width, seed], dtype=np.int32)

    @worker_method()
    async def prepare_latent_image_ids(
        self, height: int, width: int
    ) -> Int32Array:
        """Return fake latent position ids."""
        return np.array([height, width], dtype=np.int32)


class DummyTextEncoder(Worker):
    """Produce a deterministic pseudo-embedding for a prompt."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["cpu"])

    @worker_method()
    async def encode(self, tokens: Int32Array) -> Int32Array:
        """Map a prompt context to a reproducible pseudo-random embedding."""
        rng = np.random.default_rng(tokens)
        return rng.integers(low=0, high=65536, size=1024, dtype=np.int32)


class DummyDenoiser(Worker):
    """Package prompt embeddings and context into fake latent state."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def denoise(
        self,
        prompt_embeds: Int32Array,
        tokens: Int32Array,
        latents: Int32Array,
        latent_image_ids: Int32Array,
        sigmas: Int32Array,
        guidance_scale: float,
    ) -> dict[str, object]:
        """Return fake latent state for the dummy image pipeline."""
        await asyncio.sleep(0.01)
        return {
            "prompt_embeds": prompt_embeds,
            "tokens": tokens,
            "latents": latents,
            "latent_image_ids": latent_image_ids,
            "sigmas": sigmas,
            "guidance_scale": guidance_scale,
        }

    @worker_method()
    async def denoise_streaming(
        self,
        prompt_embeds: Int32Array,
        tokens: Int32Array,
        latents: Int32Array,
        latent_image_ids: Int32Array,
        sigmas: Int32Array,
        guidance_scale: float,
    ) -> AsyncIterator[dict[str, object]]:
        """Yield fake latent state once per denoising step.

        Each emitted frame carries a progressively longer prefix of the
        sigma schedule so downstream stages observe an evolving step count.
        """
        num_steps = int(sigmas.shape[0]) - 1
        for i in range(num_steps):
            await asyncio.sleep(0.01)
            yield {
                "prompt_embeds": prompt_embeds,
                "tokens": tokens,
                "latents": latents,
                "latent_image_ids": latent_image_ids,
                "sigmas": sigmas[: i + 2],
                "guidance_scale": guidance_scale,
            }


class DummyVAEDecoder(Worker):
    """Decode fake latent state into a deterministic image array."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def decode(
        self, latents: dict[str, object], height: int, width: int
    ) -> UInt8Array:
        """Generate a deterministic image array from the context seed."""
        return _decode_latents(latents, height, width)

    @worker_method()
    async def decode_streaming(
        self,
        latents_iter: AsyncIterable[dict[str, object]],
        height: int,
        width: int,
    ) -> AsyncIterator[UInt8Array]:
        """Forward a stream of latents into a stream of image arrays."""
        async for latents in latents_iter:
            yield _decode_latents(latents, height, width)


class DummyImageSerializer(Worker):
    """Serialize dummy image arrays into the requested output format."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["cpu"])

    @worker_method()
    async def serialize(self, img: UInt8Array, output_format: str) -> bytes:
        """Serialize an image array into an encoded byte buffer."""
        return _serialize_image(img, output_format)

    @worker_method()
    async def serialize_streaming(
        self,
        img_iter: AsyncIterable[UInt8Array],
        output_format: str,
    ) -> AsyncIterator[bytes]:
        """Forward a stream of image arrays into a stream of encoded bytes."""
        async for img in img_iter:
            yield _serialize_image(img, output_format)


def _decode_latents(
    latents: dict[str, object], height: int, width: int
) -> UInt8Array:
    """Build a deterministic image array from a fake latent state."""
    _ = latents["prompt_embeds"]
    sigmas = latents["sigmas"]
    seed = int(cast(Int32Array, latents["latents"])[2])
    num_steps = int(cast(Int32Array, sigmas).shape[0] - 1)
    rng = np.random.default_rng(seed)
    img = rng.random((height, width, 3), dtype=np.float32)
    img = cast(Any, ndimage).gaussian_filter(img, sigma=max(1.0, num_steps / 4))
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _serialize_image(img: UInt8Array, output_format: str) -> bytes:
    """Encode an image array as bytes using the requested format."""
    pil_image = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=output_format.upper())
    buffer.seek(0)
    return buffer.getvalue()


@dataclass
class DummyImageGenPipeline(CascadePipeline, ImageGenInterface):
    """Wire dummy image-generation workers into an end-to-end pipeline."""

    tokenizer: DummyTokenizer
    text_encoder: DummyTextEncoder
    denoiser: DummyDenoiser
    vae_decoder: DummyVAEDecoder
    image_serializer: DummyImageSerializer

    @pipeline_method
    async def generate_image(
        self, req: ImageGenRequest, prompt: str
    ) -> Awaitable[bytes]:
        """Generate an image from a text prompt."""
        tokens = await self.tokenizer.encode(prompt)
        sigmas = await self.tokenizer.prepare_sigmas(
            req.height, req.width, req.num_steps
        )
        latents = await self.tokenizer.prepare_latents(
            req.height, req.width, req.seed
        )
        latent_image_ids = await self.tokenizer.prepare_latent_image_ids(
            req.height, req.width
        )
        prompt_embeds = await self.text_encoder.encode(tokens)
        denoised = await self.denoiser.denoise(
            prompt_embeds,
            tokens,
            latents,
            latent_image_ids,
            sigmas,
            req.guidance_scale,
        )
        image = await self.vae_decoder.decode(denoised, req.height, req.width)
        return await self.image_serializer.serialize(image, req.output_format)

    @pipeline_method
    async def generate_image_streaming(
        self, req: ImageGenRequest, prompt: str
    ) -> AsyncIterator[bytes]:
        """Stream encoded images, emitting one frame per denoising step.

        Wires the streaming variants of the denoiser, VAE decoder, and image
        serializer into an end-to-end async pipeline. Each downstream worker
        consumes the upstream stream and forwards a transformed frame, so the
        caller observes ``num_steps`` byte buffers without any intermediate
        materialization.
        """
        tokens = await self.tokenizer.encode(prompt)
        sigmas = await self.tokenizer.prepare_sigmas(
            req.height, req.width, req.num_steps
        )
        latents = await self.tokenizer.prepare_latents(
            req.height, req.width, req.seed
        )
        latent_image_ids = await self.tokenizer.prepare_latent_image_ids(
            req.height, req.width
        )
        prompt_embeds = await self.text_encoder.encode(tokens)
        denoised_stream = await self.denoiser.denoise_streaming(
            prompt_embeds,
            tokens,
            latents,
            latent_image_ids,
            sigmas,
            req.guidance_scale,
        )
        image_stream = await self.vae_decoder.decode_streaming(
            denoised_stream, req.height, req.width
        )
        async for frame in await self.image_serializer.serialize_streaming(
            image_stream, req.output_format
        ):
            yield frame


async def build_dummy_imgen_pipeline() -> DummyImageGenPipeline:
    """Build the dummy image pipeline."""
    return DummyImageGenPipeline(
        DummyTokenizer(),
        DummyTextEncoder(),
        DummyDenoiser(),
        DummyVAEDecoder(),
        DummyImageSerializer(),
    )
