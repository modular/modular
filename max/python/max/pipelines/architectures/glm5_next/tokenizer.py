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
"""GLM-5.3-Flash tokenizer: the GLM-5.x text tokenizer plus images and video.

The chat template wraps each image as
``<|begin_of_image|><|image|><|end_of_image|>``, and the reference processor
(``Glm5NextProcessor.replace_image_token``) widens the one ``<|image|>`` to
``t * h * w / merge**2`` copies, one per merged vision token. A video renders as
``<|begin_of_video|><|video|><|end_of_video|>``, and ``replace_video_token``
turns the ``<|video|>`` into one
``<|begin_of_image|><|image|>...<|end_of_image|>{t:.1f} seconds`` run per
temporal patch: the same placeholder as an image, told apart only by the video
span around it. Each run is an entry of its own with ``grid_t = 1``, which is
how the reference tower sees a clip as well, so a clip is encoded and cached
frame pair by frame pair. The tower's rows are scattered onto exactly those
positions; the decoder is NoPE, so there are no position ids to adjust.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.glm5_1.tokenizer import GlmTokenizer
from max.pipelines.context import (
    GrammarEnforcementState,
    ImageMetadata,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import (
    PipelineConfig,
    VisionPreprocessCache,
    max_tokens_to_generate,
)
from max.pipelines.lib.tokenizer import (
    encode_dkv_cache_hint,
    open_image,
    run_with_default_executor,
)
from max.pipelines.modeling.types import (
    TextGenerationRequest,
    TextGenerationRequestMessage,
    VisionPreprocessCacheStats,
)
from max.support.image import hash_image
from PIL import Image
from transformers.utils import cached_file

from .context import Glm5NextTextAndVisionContext
from .image_processor import Glm5NextImageProcessor
from .video_processor import Glm5NextVideoProcessor

logger = logging.getLogger("max.pipelines")

__all__ = ["Glm5NextTokenizer"]

_PROCESSOR_CONFIG = "processor_config.json"

# An image's patch rows and its `(t, h, w)` grid.
_PreprocessedImage = tuple[npt.NDArray[np.float32], tuple[int, int, int]]


def _load_processor_config(
    model_path: str, revision: str | None
) -> dict[str, Any]:
    """Reads ``processor_config.json``, or ``{}`` if the checkpoint has none."""
    try:
        path = cached_file(model_path, _PROCESSOR_CONFIG, revision=revision)
    except OSError:
        path = None
    if path is None:
        logger.warning(
            "%s has no %s; preprocessing images with GLM-5.3-Flash's "
            "published defaults.",
            model_path,
            _PROCESSOR_CONFIG,
        )
        return {}
    with open(path) as f:
        return json.load(f)


class _Video(NamedTuple):
    """One decoded and patchified clip."""

    patches: npt.NDArray[np.float32]
    """``(grid_t * h * w, patch_dim)`` rows, temporal patch major."""

    grid: tuple[int, int, int]
    indices: list[int]
    """Source frame of each sampled frame, two per temporal patch."""

    fps: float
    content_hash: int | None


class _Entry(NamedTuple):
    """One tower input inside a placeholder's replacement."""

    offset: int
    """Index of the entry's first placeholder within the replacement."""

    num_tokens: int
    """How many placeholder tokens the entry covers."""
    pixel_values: npt.NDArray[np.float32]
    image_hash: int | None
    grid: tuple[int, int, int]


class Glm5NextTokenizer(GlmTokenizer):
    """Builds :class:`.Glm5NextTextAndVisionContext` for text, images and video.

    Text handling -- the chat template, reasoning effort and tool-result
    ordering -- is :class:`GlmTokenizer`'s, unchanged.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path, pipeline_config, revision=revision, **kwargs
        )
        hf_config = pipeline_config.model.huggingface_config
        self.image_token_id = int(hf_config.image_token_id)
        self._image_start_token_id = int(hf_config.image_start_token_id)
        self._image_end_token_id = int(hf_config.image_end_token_id)
        self._video_token_id = int(hf_config.video_token_id)
        self._video_start_token_id = int(hf_config.video_start_token_id)
        self._video_end_token_id = int(hf_config.video_end_token_id)
        processor_config = _load_processor_config(model_path, revision)
        self.image_processor = Glm5NextImageProcessor.from_processor_config(
            processor_config
        )
        self.video_processor = Glm5NextVideoProcessor.from_processor_config(
            processor_config,
            # The model encodes a temporal patch in one pass, and the memory
            # planner reserves for passes of this many rows.
            max_entry_tokens=pipeline_config.runtime.max_batch_input_tokens,
        )
        # Every image expands to at least `min_image_tokens` placeholders, so
        # more images than this cannot fit; the router rejects them before
        # downloading any.
        self.max_images_per_request = (
            self.max_length // max(1, self.image_processor.min_image_tokens)
            if self.max_length
            else None
        )
        self._preprocess_cache: VisionPreprocessCache[_PreprocessedImage] = (
            VisionPreprocessCache.for_images(pipeline_config.runtime)
        )
        self._video_preprocess_cache: VisionPreprocessCache[_Video] = (
            VisionPreprocessCache.for_videos(pipeline_config.runtime)
        )
        caches_by_hash = (
            pipeline_config.model.kv_cache.enable_prefix_caching
            or pipeline_config.runtime.vision_cache_utilization != 0
        )
        self._hash_images = caches_by_hash or self._preprocess_cache.enabled
        self._hash_videos = (
            caches_by_hash or self._video_preprocess_cache.enabled
        )

    @property
    def expects_content_wrapping(self) -> bool:
        """Keeps image and video parts as placeholders for the chat template."""
        return True

    def preprocess_cache_stats(
        self,
    ) -> Mapping[str, VisionPreprocessCacheStats]:
        """This tokenizer's :class:`PreprocessCacheStatsProbe` snapshot."""
        return {
            "image": self._preprocess_cache.stats(),
            "video": self._video_preprocess_cache.stats(),
        }

    def _flatten_message(
        self, message: TextGenerationRequestMessage
    ) -> dict[str, Any]:
        """Leaves a message with an image or a video as a list of parts.

        GLM's template renders each media part in place. Text-only messages
        are flattened as before, which joins multiple text parts with
        newlines.
        """
        if message.number_of_images == 0 and message.number_of_videos == 0:
            return super()._flatten_message(message)
        return message.model_dump(exclude_none=True)

    def _image_placeholders(
        self, token_ids: npt.NDArray[np.integer[Any]]
    ) -> npt.NDArray[np.intp]:
        """Positions of image placeholders outside any video span.

        A video frame uses the same token; the reference tells them apart by
        whether a ``<|begin_of_video|>`` is open, as done here. Frames are
        expanded from ``<|video|>``, so a rendered prompt has no ``<|image|>``
        in a video span until then.
        """
        is_placeholder = token_ids == self.image_token_id
        in_video = np.cumsum(token_ids == self._video_start_token_id) > (
            np.cumsum(token_ids == self._video_end_token_id)
        )
        if np.any(is_placeholder & in_video):
            raise ValueError(
                "The prompt has an image placeholder inside a video span, "
                "which no image or video fills."
            )
        return np.flatnonzero(is_placeholder)

    def video_replacement(
        self, grid: tuple[int, int, int], timestamps: Sequence[float]
    ) -> tuple[npt.NDArray[np.int64], list[int]]:
        """Returns the ids one ``<|video|>`` becomes, and each frame's offset.

        ``Glm5NextProcessor.replace_video_token`` on token ids rather than
        text. The timestamp text always sits between two special tokens, which
        the tokenizer splits on first, so encoding it alone yields the ids it
        gets inside the whole prompt.

        Args:
            grid: The clip's ``(t, h, w)`` patch grid.
            timestamps: Seconds at the first frame of each temporal patch.

        Returns:
            The replacement ids, and the offset of each temporal patch's first
            placeholder within them.
        """
        grid_t, grid_h, grid_w = grid
        per_frame = grid_h * grid_w // self.video_processor.merge_size**2
        texts = [f"{timestamp:.1f} seconds" for timestamp in timestamps]
        text_ids = {
            text: np.asarray(
                self.delegate.encode(text, add_special_tokens=False),
                dtype=np.int64,
            )
            for text in dict.fromkeys(texts)
        }
        placeholders = np.full(per_frame, self.image_token_id, dtype=np.int64)
        begin = np.array([self._image_start_token_id], dtype=np.int64)
        end = np.array([self._image_end_token_id], dtype=np.int64)
        pieces: list[npt.NDArray[np.int64]] = []
        offsets: list[int] = []
        offset = 0
        for _, text in zip(range(grid_t), texts, strict=True):
            offsets.append(offset + 1)
            pieces += [begin, placeholders, end, text_ids[text]]
            offset += per_frame + 2 + len(text_ids[text])
        return np.concatenate(pieces), offsets

    def _video_entries(
        self, video: _Video
    ) -> tuple[npt.NDArray[np.int64], list[_Entry]]:
        """Splits a clip into one ``grid_t = 1`` entry per temporal patch.

        The reference tower attends within a temporal patch only, so an entry
        encodes to exactly its rows of the whole clip. Its cache key is the
        clip's bytes plus the two source frames and the canvas, which together
        fix its pixels: the canvas depends on the clip's frame count, so the
        same frames sampled from a longer clip key apart.
        """
        grid_t, grid_h, grid_w = video.grid
        rows = grid_h * grid_w
        # `[::2]`, not `[::temporal_patch_size]`: the reference's own slice.
        timestamps = [index / video.fps for index in video.indices[::2]]
        replacement, offsets = self.video_replacement(
            video.grid, timestamps[:grid_t]
        )
        count = rows // self.video_processor.merge_size**2
        entries = [
            _Entry(
                offset=offset,
                num_tokens=count,
                pixel_values=video.patches[t * rows : (t + 1) * rows],
                image_hash=hash_image(
                    np.array(
                        [
                            video.content_hash,
                            *video.indices[2 * t : 2 * t + 2],
                            *video.grid,
                        ],
                        dtype=np.int64,
                    )
                )
                if video.content_hash is not None
                else None,
                grid=(1, grid_h, grid_w),
            )
            for t, offset in enumerate(offsets)
        ]
        return replacement, entries

    def _preprocess_video(
        self, raw: bytes, max_decoded_bytes: int | None
    ) -> _Video:
        """Hashes, decodes, samples and patchifies one clip, or reuses it.

        Runs in the executor with the decode: an encoded clip can run to
        gigabytes, so hashing it on the event loop would stall every request.
        """
        # Keyed on the encoded bytes alone: each temporal patch's key adds
        # the frames and the canvas it was sampled at.
        content_hash = (
            hash_image(raw, self.video_processor.max_image_tokens)
            if self._hash_videos
            else None
        )

        def preprocess() -> _Video:
            frames, indices, fps = self.video_processor.decode(
                raw, max_decoded_bytes
            )
            patches, grid = self.video_processor.preprocess(frames)
            return _Video(patches, grid, indices, fps, content_hash)

        # The decode budget can lower the frame cap, so the sample depends on
        # it as well as on the bytes.
        budget = -1 if max_decoded_bytes is None else max_decoded_bytes
        key = (
            None
            if content_hash is None
            else hash_image(np.array([content_hash, budget], dtype=np.int64))
        )
        return self._video_preprocess_cache.get_or_preprocess(key, preprocess)

    def _preprocess_videos(
        self, videos: Sequence[bytes], max_decoded_bytes: int | None
    ) -> list[_Video]:
        """Preprocesses each clip, for the executor."""
        return [
            self._preprocess_video(raw, max_decoded_bytes) for raw in videos
        ]

    def expand_image_placeholders(
        self,
        token_ids: npt.NDArray[np.integer[Any]],
        images: Sequence[bytes | Image.Image],
        image_hashes: Sequence[int | None],
        videos: Sequence[_Video] = (),
    ) -> tuple[
        npt.NDArray[np.integer[Any]], list[ImageMetadata], npt.NDArray[np.int64]
    ]:
        """Preprocesses the images and widens each placeholder to its rows.

        A ``<|video|>`` widens to its clip's frame runs, and every temporal
        patch becomes an entry of its own, in prompt order with the images. A
        request without media passes through unchanged, so a prompt that
        quotes ``<|image|>`` or ``<|video|>`` keeps it as text, as the base GLM
        tokenizer did.

        Args:
            token_ids: The encoded prompt, one placeholder per image or video.
            images: The request's images, in prompt order.
            image_hashes: One content hash per image, or ``None`` per image
                when no cache needs one.
            videos: The request's preprocessed clips, in prompt order.

        Returns:
            The expanded token ids, one :class:`ImageMetadata` per image or
            temporal patch whose span covers exactly its placeholders, and
            their ``(t, h, w)`` grids.

        Raises:
            ValueError: If a placeholder count differs from its media count.
            PromptTooLongError: If the expanded prompt exceeds ``max_length``.
        """
        if not images and not videos:
            return token_ids, [], np.empty((0, 3), dtype=np.int64)
        image_positions = self._image_placeholders(token_ids)
        if len(image_positions) != len(images):
            raise ValueError(
                f"The prompt has {len(image_positions)} image placeholders but "
                f"the request carries {len(images)} images."
            )
        video_positions = np.flatnonzero(token_ids == self._video_token_id)
        if len(video_positions) != len(videos):
            raise ValueError(
                f"The prompt has {len(video_positions)} video placeholders but "
                f"the request carries {len(videos)} videos."
            )

        opened = [open_image(image) for image in images]
        counts = [
            self.image_processor.num_image_tokens(
                self.image_processor.grid_thw(image)
            )
            for image in opened
        ]
        edits: list[tuple[int, npt.NDArray[np.int64], list[_Entry]]] = [
            (int(position), *self._video_entries(video))
            for position, video in zip(video_positions, videos, strict=True)
        ]
        # Checked before patchifying: images below the token floor are
        # upscaled, so many tiny ones would take gigabytes first.
        length = (
            len(token_ids)
            + sum(counts)
            - len(counts)
            + sum(len(replacement) - 1 for _, replacement, _ in edits)
        )
        if self.max_length and length > self.max_length:
            raise PromptTooLongError(length, self.max_length)
        for position, image, count, image_hash in zip(
            image_positions, opened, counts, image_hashes, strict=True
        ):
            # Keyed on the hash the encoder cache uses: the bytes and the
            # token budget fix the patches, so a hit skips the resize and
            # patchify.
            patches, grid = self._preprocess_cache.get_or_preprocess(
                image_hash, partial(self.image_processor.preprocess, image)
            )
            edits.append(
                (
                    int(position),
                    np.full(count, self.image_token_id, dtype=np.int64),
                    [_Entry(0, count, patches, image_hash, grid)],
                )
            )
        edits.sort(key=lambda edit: edit[0])

        pieces: list[npt.NDArray[np.integer[Any]]] = []
        metadata: list[ImageMetadata] = []
        grids: list[tuple[int, int, int]] = []
        previous = 0
        shift = 0
        for position, replacement, entries in edits:
            pieces += [token_ids[previous:position], replacement]
            for entry in entries:
                start = position + shift + entry.offset
                metadata.append(
                    ImageMetadata(
                        start_idx=start,
                        end_idx=start + entry.num_tokens,
                        pixel_values=entry.pixel_values,
                        image_hash=entry.image_hash,
                    )
                )
                grids.append(entry.grid)
            shift += len(replacement) - 1
            previous = position + 1
        pieces.append(token_ids[previous:])
        return (
            np.concatenate(pieces),
            metadata,
            np.array(grids, dtype=np.int64),
        )

    async def new_context(
        self, request: TextGenerationRequest
    ) -> Glm5NextTextAndVisionContext:
        """Creates a context, expanding image and video placeholders."""
        _prompt, token_ids = await self._generate_prompt_and_token_ids(
            prompt=request.prompt,
            messages=request.messages,
            tools=request.tools,
            **(request.chat_template_options or {}),
        )
        # Before any clip is decoded, which can take seconds.
        if request.videos:
            placeholders = int(
                np.count_nonzero(np.asarray(token_ids) == self._video_token_id)
            )
            if placeholders != len(request.videos):
                raise ValueError(
                    f"The prompt has {placeholders} video placeholders but "
                    f"the request carries {len(request.videos)} videos."
                )

        # Keyed on the encoded bytes and the token budget, which together fix
        # the padded canvas the tower sees.
        image_hashes = [
            hash_image(raw, self.image_processor.max_image_tokens)
            if self._hash_images
            else None
            for raw in request.images
        ]
        videos = (
            await run_with_default_executor(
                self._preprocess_videos,
                request.videos,
                request.max_decoded_bytes,
            )
            if request.videos
            else []
        )
        token_ids, images, grid_thws = await run_with_default_executor(
            self.expand_image_placeholders,
            token_ids,
            request.images_for_processing(),
            image_hashes,
            videos,
        )
        if self.max_length and len(token_ids) > self.max_length:
            raise PromptTooLongError(len(token_ids), self.max_length)

        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )
        grammar = (
            request.response_format.grammar if request.response_format else None
        )
        max_gen_tokens = max_tokens_to_generate(
            len(token_ids),
            self.max_length,
            request.sampling_params.max_new_tokens,
        )
        return Glm5NextTextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            max_length=len(token_ids) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            tokens=TokenBuffer(array=token_ids.astype(np.int64, copy=False)),
            vocab_size=self.tokenizer_vocab_size,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
            grammar=grammar,
            grammar_state=GrammarEnforcementState.from_response_format(
                request.response_format
            ),
            sampling_params=request.sampling_params,
            model_name=request.model_name,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            cache_salt=request.cache_salt,
            images=images,
            vision_token_ids=[self.image_token_id],
            grid_thws=grid_thws,
        )
