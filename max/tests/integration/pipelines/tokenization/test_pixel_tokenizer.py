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
"""Tests for PixelGenerationTokenizer.

These tests require network access to HuggingFace and are marked as manual
in the BUILD.bazel file. They are not run in CI by default.

To run these tests manually:
    ./bazelw test //max/tests/integration/pipelines/tokenization:test_pixel_tokenizer
"""

from __future__ import annotations

import numpy as np
import pytest
from max.interfaces import RequestID
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import OpenResponsesRequestBody
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.lib.config import PipelineConfig


class TestPixelGenerationTokenizer:
    """Test suite for PixelGenerationTokenizer.

    These tests use the real Flux 2.0 model from HuggingFace and require
    network access. They are marked as manual to prevent running in CI.
    """

    @pytest.fixture
    def flux_model_path(self) -> str:
        """Flux 2.0 model path from HuggingFace."""
        return "black-forest-labs/FLUX.2-dev"

    @pytest.fixture
    def flux_pipeline_config(self, flux_model_path: str) -> PipelineConfig:
        """Pipeline config for Flux model."""
        return PipelineConfig(model_path=flux_model_path, defer_resolve=True)

    @pytest.fixture
    def zimage_model_path(self) -> str:
        """Z-Image model path from HuggingFace."""
        return "Tongyi-MAI/Z-Image-Turbo"

    @pytest.fixture
    def zimage_pipeline_config(self, zimage_model_path: str) -> PipelineConfig:
        """Pipeline config for Z-Image model."""
        return PipelineConfig(model_path=zimage_model_path, defer_resolve=True)

    def test_initialization_basic(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test basic initialization of PixelGenerationTokenizer.

        Uses FLUX.2 which has a single text encoder (Mistral-Small-3.2-24B),
        verifying that delegate_2 is None when no secondary tokenizer is specified.
        """
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=77,
        )

        assert tokenizer.model_path == flux_model_path
        assert tokenizer.max_length == 77
        assert tokenizer.delegate is not None
        assert tokenizer.delegate_2 is None
        assert tokenizer._pipeline_class_name == "Flux2Pipeline"

    def test_initialization_without_diffusers_config(
        self, flux_model_path: str
    ) -> None:
        """Test that initialization fails without diffusers_config."""
        # Use a non-diffusion model (text-generation model) which won't have diffusers_config
        non_diffusion_model = "gpt2"
        config = PipelineConfig(
            model_path=non_diffusion_model, defer_resolve=True
        )

        with pytest.raises(ValueError, match="diffusers_config cannot be None"):
            PixelGenerationTokenizer(
                model_path=flux_model_path,
                pipeline_config=config,
                subfolder="tokenizer",
                max_length=2048,
            )

    def test_static_config_caching(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test that static configuration values are cached during initialization."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        # Verify cached values
        assert tokenizer._vae_scale_factor == 8
        assert tokenizer._default_sample_size == 128
        assert (
            tokenizer._num_channels_latents == 32
        )  # Flux2 has in_channels=128, so 128//4=32
        assert tokenizer._scheduler.base_image_seq_len == 256
        assert tokenizer._scheduler.max_image_seq_len == 4096
        assert tokenizer._scheduler.base_shift == 0.5
        assert tokenizer._scheduler.max_shift == 1.15
        assert (
            tokenizer._scheduler._use_empirical_mu is False
        )  # Flux2 model doesn't have guidance_embeds in transformer config

    @pytest.mark.asyncio
    async def test_encode_primary_tokenizer(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test encoding with the primary tokenizer."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=77,
        )

        prompt = "A beautiful sunset over the ocean"
        token_ids, attention_mask = await tokenizer.encode(prompt)

        assert isinstance(token_ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert token_ids.dtype == np.int64
        assert attention_mask.dtype == np.bool_
        assert len(token_ids) == 77  # max_length
        assert len(attention_mask) == 77

    @pytest.mark.asyncio
    async def test_new_context_flux(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test creating a PixelContext for Flux model."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=77,
        )

        body = OpenResponsesRequestBody(
            model="flux-dev",
            input="A majestic mountain landscape",
            seed=42,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    height=1024,
                    width=1024,
                    steps=50,
                    guidance_scale=3.5,
                )
            ),
        )
        request = OpenResponsesRequest(
            request_id=RequestID("test-request-1"), body=body
        )

        context = await tokenizer.new_context(request)

        # Verify context properties
        assert context.request_id == request.request_id
        assert context.height == 1024
        assert context.width == 1024
        assert context.num_inference_steps == 50
        assert context.guidance_scale == 3.5
        assert context.tokens is not None
        assert context.latents is not None
        assert context.timesteps is not None

        # Verify timesteps are normalized correctly for Flux (standard)
        assert np.all(context.timesteps >= 0.0)
        assert np.all(context.timesteps <= 1.0)

    @pytest.mark.asyncio
    async def test_new_context_zimage(
        self, zimage_model_path: str, zimage_pipeline_config: PipelineConfig
    ) -> None:
        """Test creating a PixelContext for Z-Image model with inverted timesteps."""
        tokenizer = PixelGenerationTokenizer(
            model_path=zimage_model_path,
            pipeline_config=zimage_pipeline_config,
            subfolder="tokenizer",
            max_length=77,
        )

        body = OpenResponsesRequestBody(
            model="z-image-turbo",
            input="A futuristic cityscape",
            seed=123,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    height=1024,
                    width=1024,
                    steps=8,
                    guidance_scale=3.5,
                )
            ),
        )
        request = OpenResponsesRequest(
            request_id=RequestID("test-request-2"), body=body
        )

        context = await tokenizer.new_context(request)

        # Verify Z-Image uses inverted timestep normalization
        assert context.timesteps is not None
        assert np.all(context.timesteps >= 0.0)
        assert np.all(context.timesteps <= 1.0)

        # Z-Image should have different timestep values than Flux
        # due to inverted normalization: (1000 - t) / 1000 vs t / 1000

    @pytest.mark.asyncio
    async def test_new_context_default_dimensions(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test that default dimensions are computed correctly."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        body = OpenResponsesRequestBody(
            model="flux-dev",
            input="Test prompt",
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    steps=10,
                    # No height/width specified
                )
            ),
        )
        request = OpenResponsesRequest(
            request_id=RequestID("test-request-4"), body=body
        )

        context = await tokenizer.new_context(request)

        # Default: 128 * vae_scale_factor (8) = 1024
        assert context.height == 1024
        assert context.width == 1024

    @pytest.mark.asyncio
    async def test_postprocess(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test pixel data postprocessing."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        # Create mock pixel data (NCHW format, normalized to [-1, 1])
        pixel_data = np.random.randn(1, 3, 64, 64).astype(np.float32)

        processed = await tokenizer.postprocess(pixel_data)

        # Verify output is NHWC format and normalized to [0, 1]
        assert processed.shape == (1, 64, 64, 3)
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)

    def test_calculate_shift(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test the shift calculation for timestep scheduling."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        # Test with different image sequence lengths
        mu_small = tokenizer._scheduler._calculate_mu(256, 50)
        mu_large = tokenizer._scheduler._calculate_mu(4096, 50)

        # Mu should increase with sequence length
        assert mu_small == 0.5  # At base
        assert mu_large == 1.15  # At max
        assert mu_small < mu_large

    def test_prepare_latents(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test latent preparation."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        latents, latent_image_ids = tokenizer._prepare_latents(
            batch_size=2,
            num_channels_latents=16,
            latent_height=128,
            latent_width=128,
            seed=42,
        )

        # Verify latents shape
        assert latents.shape == (2, 16, 128, 128)
        assert latents.dtype == np.float32

        # Verify latent_image_ids shape
        assert latent_image_ids.shape == (64 * 64, 3)  # (h//2 * w//2, 3)
        assert latent_image_ids.dtype == np.float32

    def test_prepare_latents_deterministic(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test that latent preparation is deterministic with the same seed."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        latents1, _ = tokenizer._prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            latent_height=64,
            latent_width=64,
            seed=12345,
        )

        latents2, _ = tokenizer._prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            latent_height=64,
            latent_width=64,
            seed=12345,
        )

        # Same seed should produce identical latents
        np.testing.assert_array_equal(latents1, latents2)

    @pytest.mark.asyncio
    async def test_decode_not_implemented(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test that decode raises NotImplementedError."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        dummy_encoded = (np.array([1, 2, 3]), np.array([True, True, True]))

        with pytest.raises(NotImplementedError):
            await tokenizer.decode(dummy_encoded)

    def test_properties(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test tokenizer properties."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        # Test eos property
        assert isinstance(tokenizer.eos, int)

        # Test expects_content_wrapping property
        assert tokenizer.expects_content_wrapping is False
