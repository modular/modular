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
from max.pipelines import PipelineConfig
from max.pipelines.architectures.wan.arch import WanArchConfig
from max.pipelines.architectures.wan.components import (
    vae_wrapper as wan_vae_wrapper,
)
from max.pipelines.architectures.wan.tokenizer import WanTokenizer
from max.pipelines.context import PixelContext, TokenBuffer
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pixel_tokenizer import PixelGenerationTokenizer
from max.pipelines.modeling.types import RequestID
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    VideoProviderOptions,
)


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
            provider_options=SimpleNamespace(video=None, image=None),
            seed=42,
        )
    )

    context = asyncio.run(tokenizer.new_context(request))

    assert context.num_frames == 1
    assert context.latents.shape == (1, 16, 1, 104, 60)


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


def _build_flow_shift_tokenizer(captured: list[float | None]) -> Any:
    """WanTokenizer whose scheduler records the flow_shift passed to
    retrieve_timesteps_and_sigmas."""

    def _retrieve(
        image_seq_len: int,
        num_inference_steps: int,
        flow_shift: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        captured.append(flow_shift)
        return (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        )

    tokenizer = cast(Any, object.__new__(WanTokenizer))
    tokenizer._scheduler = SimpleNamespace(
        use_flow_sigmas=True,
        flow_shift=3.0,  # model-config default
        order=1,
        retrieve_timesteps_and_sigmas=_retrieve,
        build_step_coefficients=lambda: np.zeros((2, 9), dtype=np.float32),
    )
    tokenizer._vae_scale_factor = 8
    tokenizer._num_channels_latents = 16
    tokenizer._manifest_metadata = {}
    tokenizer._randn_tensor = lambda shape, seed: np.zeros(
        shape, dtype=np.float32
    )
    return tokenizer


def test_wan_tokenizer_honors_request_flow_shift_override(
    monkeypatch: Any,
) -> None:
    """video_options.flow_shift is passed through to the scheduler."""
    monkeypatch.setattr(
        PixelGenerationTokenizer, "new_context", _mock_base_new_context
    )

    captured: list[float | None] = []
    tokenizer = _build_flow_shift_tokenizer(captured)

    request = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(
                video=VideoProviderOptions(flow_shift=5.5), image=None
            ),
            seed=42,
        )
    )
    asyncio.run(tokenizer.new_context(request))

    assert captured == [5.5]
    # The shared scheduler default is left untouched.
    assert tokenizer._scheduler.flow_shift == 3.0


def test_wan_tokenizer_honors_image_block_flow_shift(
    monkeypatch: Any,
) -> None:
    """flow_shift on the image block is honored when no video block is set."""
    monkeypatch.setattr(
        PixelGenerationTokenizer, "new_context", _mock_base_new_context
    )

    captured: list[float | None] = []
    tokenizer = _build_flow_shift_tokenizer(captured)

    request = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(
                video=None, image=ImageProviderOptions(flow_shift=6.0)
            ),
            seed=42,
        )
    )
    asyncio.run(tokenizer.new_context(request))

    assert captured == [6.0]


def test_wan_tokenizer_flow_shift_override_does_not_leak_across_requests(
    monkeypatch: Any,
) -> None:
    """A per-request override must not bleed into a later request that omits
    the field — the second request resolves the model-config default."""
    monkeypatch.setattr(
        PixelGenerationTokenizer, "new_context", _mock_base_new_context
    )

    captured: list[float | None] = []
    tokenizer = _build_flow_shift_tokenizer(captured)

    req_override = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(
                video=VideoProviderOptions(flow_shift=7.0), image=None
            ),
            seed=42,
        )
    )
    asyncio.run(tokenizer.new_context(req_override))

    req_no_override = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(video=None, image=None),
            seed=42,
        )
    )
    asyncio.run(tokenizer.new_context(req_no_override))

    # Request 1 passes 7.0; request 2 falls back to the config default (3.0),
    # not the 7.0 from request 1.
    assert captured == [7.0, 3.0]
