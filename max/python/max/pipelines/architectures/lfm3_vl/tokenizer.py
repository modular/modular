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
"""The LFM3-VL tokenizer, which patchifies media with numpy.

The checkpoint's processor needs torch. Its image processing is Qwen3-VL's,
so this reuses MAX's numpy port of it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    GrammarEnforcementState,
    ImageMetadata,
    TextAndVisionContext,
    TokenBuffer,
    probe_video,
    validate_video,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    max_tokens_to_generate,
    resolve_eos_token_ids,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import (
    checked_token_ids,
    encode_dkv_cache_hint,
    open_image,
    resolve_token_id_bound,
    run_with_default_executor,
)
from max.pipelines.modeling.types import TextGenerationRequest
from max.support.image import hash_image
from PIL import Image
from transformers import AutoTokenizer
from transformers.utils import cached_file

from ..qwen3vl_moe.nn.data_processing import QWEN3VL_MAX_PIXELS
from ..qwen3vl_moe.nn.video_processing import (
    QWEN3VL_VIDEO_MAX_FRAMES,
    preprocess_clip,
)
from ..qwen3vl_moe.tokenizer import qwen3vl_image_preprocessing

IMAGE_TOKEN = "<|image|>"
VIDEO_TOKEN = "<|video|>"

Frame = tuple[npt.NDArray[np.float32], int, int]
"""An encoder input, its hash and its placeholder token count."""

Media = dict[str, list[tuple[str, list[Frame]]]]
"""Per placeholder token, each item's replacement text and frames."""


def _frame_hash(frame: npt.NDArray[np.float32]) -> int:
    """Hashes a frame's pixels with its block grid, which the bytes omit."""
    return hash_image(np.array([hash_image(frame), *frame.shape], np.int64))


class Lfm3VlTokenizer(TextAndVisionTokenizer):
    """Expands each media placeholder into one token per merged patch block.

    Each entry of the context's ``images`` holds one frame's patches as
    ``[blocks_h, blocks_w, merge**2, patch_dim]``, so the frame's grid
    travels with its pixels. A video contributes one frame per temporal
    patch, each in its own timestamped placeholder run. Subclasses add media
    kinds through :meth:`_media`.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        chat_template: str | None = None,
        **unused_kwargs: object,
    ) -> None:
        self.model_path = model_path
        self._chat_template = chat_template
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.processor = self.delegate
        self.max_length = max_length or self.delegate.model_max_length
        self._eos_token_ids = resolve_eos_token_ids(
            self.delegate.eos_token_id, pipeline_config
        )
        self._token_id_bound = resolve_token_id_bound(
            self.delegate, pipeline_config
        )
        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        config = pipeline_config.model.huggingface_config
        assert config is not None
        self.config = config
        self.vision_token_ids: list[int] = []
        vision = getattr(config, "vision_config", None)
        self.has_vision = vision is not None
        if vision is not None:
            self.vision_token_ids += [
                config.image_token_id,
                config.video_token_id,
            ]
            self.patch_size = vision.patch_size
            self.temporal_patch_size = vision.temporal_patch_size
            self.merge_size = vision.spatial_merge_size
            processor = {}
            try:
                path = cached_file(
                    model_path, "processor_config.json", revision=revision
                )
            except OSError:
                path = None
            if path:
                with open(path) as f:
                    processor = json.load(f)
            size = processor.get("image_processor", {}).get("size", {})
            self.min_pixels = int(size.get("shortest_edge", 65536))
            self.max_pixels = int(size.get("longest_edge", QWEN3VL_MAX_PIXELS))
            self.max_frames = int(
                processor.get("video_processor", {}).get(
                    "max_frames", QWEN3VL_VIDEO_MAX_FRAMES
                )
            )

    def _as_frame(
        self, pixels: npt.NDArray[np.float32], h: int, w: int
    ) -> npt.NDArray[np.float32]:
        m = self.merge_size
        return pixels.reshape(h // m, w // m, m * m, -1)

    def _preprocess_images(
        self, images: Sequence[bytes | Image.Image]
    ) -> list[npt.NDArray[np.float32]]:
        frames = []
        for image in images:
            pixels, (_, h, w) = qwen3vl_image_preprocessing(
                open_image(image).convert("RGB"),
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            frames.append(self._as_frame(pixels, h, w))
        return frames

    def _preprocess_videos(
        self, videos: Sequence[bytes], max_decoded_bytes: int | None
    ) -> list[list[tuple[npt.NDArray[np.float32], float]]]:
        clips = []
        for video in videos:
            max_frames = self.max_frames
            if max_decoded_bytes is not None:
                validate_video(video, max_decoded_bytes)
                max_frames = min(
                    max_frames,
                    max_decoded_bytes // probe_video(video).frame_bytes,
                )
            pixels, (t, h, w), times = preprocess_clip(
                video,
                patch_size=self.patch_size,
                merge_size=self.merge_size,
                temporal_patch_size=self.temporal_patch_size,
                max_frames=max_frames,
            )
            frames = [self._as_frame(p, h, w) for p in np.split(pixels, t)]
            clips.append(list(zip(frames, times, strict=True)))
        return clips

    async def _media(self, request: TextGenerationRequest) -> Media:
        """Preprocesses the request's media for :meth:`_expand`."""
        if not self.has_vision:
            if request.images or request.videos:
                raise ValueError("This model doesn't take images or video.")
            return {}
        images = await run_with_default_executor(
            self._preprocess_images,
            request.images_for_processing() if request.images else [],
        )
        clips = await run_with_default_executor(
            self._preprocess_videos, request.videos, request.max_decoded_bytes
        )
        media: Media = {IMAGE_TOKEN: [], VIDEO_TOKEN: []}
        for frame, raw in zip(images, request.images, strict=True):
            n = frame.shape[0] * frame.shape[1]
            frame_hash = hash_image(raw, self.max_pixels)
            media[IMAGE_TOKEN].append(
                (IMAGE_TOKEN * n, [(frame, frame_hash, n)])
            )
        for clip in clips:
            runs, frames = [], []
            for frame, time in clip:
                n = frame.shape[0] * frame.shape[1]
                runs.append(
                    f"<{time:.1f} seconds><|vision_start|>{VIDEO_TOKEN * n}"
                    "<|vision_end|>"
                )
                frames.append((frame, _frame_hash(frame), n))
            media[VIDEO_TOKEN].append(("".join(runs), frames))
        return media

    @staticmethod
    def _expand(prompt: str, media: Media) -> tuple[str, list[Frame]]:
        """Expands the media placeholders, and returns the frames in order."""
        for token, items in media.items():
            if prompt.count(token) != len(items):
                raise ValueError(
                    f"The prompt has {prompt.count(token)} {token} tokens, "
                    f"but the request has {len(items)} of that media."
                )
        if not media:
            return prompt, []
        queues = {token: iter(entries) for token, entries in media.items()}
        pattern = "(" + "|".join(map(re.escape, media)) + ")"
        pieces = re.split(pattern, prompt)
        expanded, frames = [pieces[0]], []
        for token, text in zip(pieces[1::2], pieces[2::2], strict=True):
            replacement, item_frames = next(queues[token])
            expanded += [replacement, text]
            frames += item_frames
        return "".join(expanded), frames

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Builds a context, expanding each media placeholder in the prompt."""
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.messages:
            prompt = await run_with_default_executor(
                self.apply_chat_template,
                request.messages,
                request.tools,
                **(request.chat_template_options or {}),
            )
            add_special_tokens = False
        elif request.prompt is not None:
            prompt = request.prompt
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        media = await self._media(request)
        frames: list[Frame] = []
        if isinstance(prompt, str):
            expanded, frames = self._expand(prompt, media)
            encoded = np.array(
                await run_with_default_executor(
                    self.delegate.encode,
                    expanded,
                    add_special_tokens=add_special_tokens,
                ),
                dtype=np.int64,
            )
        elif any(media.values()):
            raise ValueError(
                "A prompt given as token ids can't carry media. Send the "
                "media with messages instead."
            )
        else:
            encoded = checked_token_ids(prompt, self._token_id_bound).astype(
                np.int64
            )
        if self.max_length and len(encoded) > self.max_length:
            raise PromptTooLongError(len(encoded), self.max_length)

        # Frames claim placeholders in order, so adjacent runs stay apart.
        positions = np.flatnonzero(np.isin(encoded, self.vision_token_ids))
        metadata, cursor = [], 0
        for frame, frame_hash, n in frames:
            run = positions[cursor : cursor + n]
            if len(run) != n or run[-1] - run[0] != n - 1:
                raise ValueError(
                    "The prompt's media placeholders don't match the "
                    "request's media."
                )
            metadata.append(
                ImageMetadata(
                    start_idx=int(run[0]),
                    end_idx=int(run[0]) + n,
                    pixel_values=frame,
                    image_hash=frame_hash,
                )
            )
            cursor += n
        if cursor != len(positions):
            raise ValueError(
                "The prompt has media placeholders the request's media "
                "don't fill."
            )

        max_gen_tokens = max_tokens_to_generate(
            len(encoded),
            self.max_length,
            request.sampling_params.max_new_tokens,
        )
        response_format = request.response_format
        return TextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            tokens=TokenBuffer(array=encoded),
            vocab_size=self.tokenizer_vocab_size,
            max_length=len(encoded) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json.dumps(response_format.json_schema)
            if response_format and response_format.json_schema is not None
            else None,
            grammar=response_format.grammar if response_format else None,
            grammar_state=GrammarEnforcementState.from_response_format(
                response_format
            ),
            sampling_params=request.sampling_params,
            model_name=request.model_name,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            images=metadata,
            vision_token_ids=self.vision_token_ids,
            cache_salt=request.cache_salt,
        )
