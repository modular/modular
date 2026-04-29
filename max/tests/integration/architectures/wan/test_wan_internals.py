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

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from max.driver import DeviceSpec
from max.interfaces import RequestID, TokenBuffer
from max.pipelines import PipelineConfig
from max.pipelines.architectures.wan.arch import WanArchConfig
from max.pipelines.architectures.wan.components import (
    vae_wrapper as wan_vae_wrapper,
)
from max.pipelines.architectures.wan.tokenizer import WanTokenizer
from max.pipelines.core import PixelContext
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pixel_tokenizer import PixelGenerationTokenizer


def test_wan_arch_config_initialize_uses_transformer_component() -> None:
    pipeline_config = cast(
        PipelineConfig,
        SimpleNamespace(
            models=ModelManifest(
                {
                    "transformer": MAXModelConfig(
                        model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                        device_specs=[DeviceSpec.cpu()],
                    )
                }
            )
        ),
    )

    config = WanArchConfig.initialize(pipeline_config=pipeline_config)

    assert config.pipeline_config is pipeline_config


def test_wan_vae_decode_preserves_5d_video_shape() -> None:
    decoded_video = np.arange(1 * 4 * 2 * 2 * 3, dtype=np.uint8).reshape(
        1, 4, 2, 2, 3
    )

    wrapper = cast(Any, object.__new__(wan_vae_wrapper.VaeWrapper))
    wrapper._denorm_model = cast(
        Any, SimpleNamespace(execute=lambda *args: [object()])
    )
    wrapper._vae = cast(
        Any, SimpleNamespace(decode_5d=lambda _latents: object())
    )
    wrapper._postprocess_model = cast(
        Any, SimpleNamespace(execute=lambda _decoded_video: [decoded_video])
    )
    wrapper._vae_std_buf = cast(Any, object())
    wrapper._vae_mean_buf = cast(Any, object())

    output = wrapper.decode(
        latents=cast(Any, object()),
        num_frames=4,
        height=2,
        width=2,
    )

    assert output.dtype == np.uint8
    assert output.shape == (1, 3, 4, 2, 2)


def test_wan_vae_decode_preserves_single_frame_image_shape() -> None:
    decoded_video = np.arange(1 * 1 * 2 * 2 * 3, dtype=np.uint8).reshape(
        1, 1, 2, 2, 3
    )

    wrapper = cast(Any, object.__new__(wan_vae_wrapper.VaeWrapper))
    wrapper._denorm_model = cast(
        Any, SimpleNamespace(execute=lambda *args: [object()])
    )
    wrapper._vae = cast(
        Any, SimpleNamespace(decode_5d=lambda _latents: object())
    )
    wrapper._postprocess_model = cast(
        Any, SimpleNamespace(execute=lambda _decoded_video: [decoded_video])
    )
    wrapper._vae_std_buf = cast(Any, object())
    wrapper._vae_mean_buf = cast(Any, object())

    output = wrapper.decode(
        latents=cast(Any, object()),
        num_frames=1,
        height=2,
        width=2,
    )

    assert output.dtype == np.uint8
    assert output.shape == (1, 2, 2, 3)


def test_wan_tokenizer_uses_single_frame_video_latents_for_images(
    monkeypatch: Any,
) -> None:
    async def _mock_base_new_context(
        self: PixelGenerationTokenizer,
        request: Any,
        input_image: Any = None,
    ) -> PixelContext:
        return PixelContext(
            tokens=TokenBuffer(np.array([0], dtype=np.int64)),
            request_id=RequestID(),
            latents=np.zeros((1, 16, 104, 60), dtype=np.float32),
            height=832,
            width=480,
            num_inference_steps=4,
            num_images_per_prompt=1,
        )

    monkeypatch.setattr(
        PixelGenerationTokenizer,
        "new_context",
        _mock_base_new_context,
    )

    tokenizer = cast(Any, object.__new__(WanTokenizer))
    tokenizer._scheduler = SimpleNamespace(
        use_flow_sigmas=False,
        order=1,
        retrieve_timesteps_and_sigmas=lambda image_seq_len,
        num_inference_steps: (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ),
        build_step_coefficients=lambda: np.zeros((2, 9), dtype=np.float32),
    )
    tokenizer._vae_scale_factor = 8
    tokenizer._num_channels_latents = 16
    tokenizer._manifest_metadata = {}
    tokenizer._randn_tensor = lambda shape, seed: np.zeros(
        shape, dtype=np.float32
    )

    request = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(video=None),
            seed=42,
        )
    )

    context = asyncio.run(tokenizer.new_context(request))

    assert context.num_frames == 1
    assert context.latents.shape == (1, 16, 1, 104, 60)
