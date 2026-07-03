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

"""Gemma4 tokenizer for the gemma4 architecture."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import to_rgb
from max.pipelines.context import (
    ImageMetadata,
    TokenBuffer,
)
from max.pipelines.context.context import GrammarEnforcementState
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import TextAndVisionTokenizer, max_tokens_to_generate
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import open_image, resolve_single_special_token
from max.pipelines.modeling.types import (
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import find_contiguous_ranges, hash_image
from transformers import AutoTokenizer, GenerationConfig

from .context import Gemma4Context
from .image_processor import Gemma4ImageProcessor
from .processing_utils import load_processor_config
from .video_processor import Gemma4VideoProcessor, VideoMetadata


class SpecialToken(str, Enum):
    """Gemma4 special tokens for tool calls."""

    TOOL_CALL_START = "<|tool_call>"
    TOOL_CALL_END = "<tool_call|>"
    TOOL_START = "<|tool>"
    TOOL_END = "<tool|>"
    TOOL_RESPONSE_START = "<|tool_response>"
    TOOL_RESPONSE_END = "<tool_response|>"
    STRING_DELIM = '<|"|>'
    TURN_END = "<turn|>"


# Reasoning-block opener Gemma 4 prefills on the generation turn (see
# apply_chat_template). Single source of truth — the reasoning parser derives
# its prefix from this too.
REASONING_OPEN = "<|channel>thought\n"

# Generation-turn header the chat template emits before the reasoning
# channel; reused to re-open a turn after a tool result (apply_chat_template).
MODEL_TURN_OPEN = "<|turn>model\n"


class Gemma4Tokenizer(TextAndVisionTokenizer):
    """Gemma4-specific tokenizer handling text and vision inputs.

    Uses a custom ``Gemma4ImageProcessor`` (numpy/PIL only) instead of
    HuggingFace's ``AutoProcessor`` to avoid pulling in torch.
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
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )

        if chat_template is not None:
            self.delegate.chat_template = chat_template
        self.max_length = max_length or self.delegate.model_max_length

        config = pipeline_config.model.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_path}'"
            )

        # EOS token IDs
        self._default_eos_token_ids = set([self.eos])
        if eos_token_id := getattr(config, "eos_token_id", None):
            if isinstance(eos_token_id, int):
                self._default_eos_token_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                self._default_eos_token_ids.update(eos_token_id)

        # Gemma 4 ships an ``eos_token_id`` list in ``generation_config.json``
        # that extends what ``config.json`` declares — for the 31B-IT release
        # it adds ``<|tool_response>`` (id 50) alongside ``<eos>`` and
        # ``<turn|>``. Google uses ``<|tool_response>`` as the assistant's
        # tool-call-turn terminator, so without picking it up the model can
        # emit token 50 and keep generating past the tool call. Mirror
        # vLLM's ``update_from_generation_config`` behavior by reading
        # ``generation_config.json`` here.
        try:
            gen_config = GenerationConfig.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            # ``generation_config.json`` is optional and HF may raise for a
            # variety of reasons (missing file, malformed JSON, hub
            # connection error). None of those should fail tokenizer init.
            gen_config = None
        if gen_config is not None:
            gen_eos = getattr(gen_config, "eos_token_id", None)
            if isinstance(gen_eos, int):
                self._default_eos_token_ids.add(gen_eos)
            elif isinstance(gen_eos, list):
                self._default_eos_token_ids.update(gen_eos)

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.enable_vision_caching = (
            pipeline_config.runtime.max_vision_cache_entries > 0
        )
        # Image token IDs — try both naming conventions
        self.image_token_id: int = _require_attr(
            config, "image_token_id", "image_token_index"
        )
        self.boi_token_id: int = _require_attr(config, "boi_token_id")
        self.eoi_token_id: int = _require_attr(config, "eoi_token_id")

        self.vision_token_ids = [self.image_token_id]

        # Token strings — prefer tokenizer attributes, fall back to decode
        self.image_token: str = getattr(
            self.delegate, "image_token", None
        ) or self.delegate.decode([self.image_token_id])
        self.boi_token: str = getattr(
            self.delegate, "boi_token", None
        ) or self.delegate.decode([self.boi_token_id])
        self.eoi_token: str = getattr(
            self.delegate, "eoi_token", None
        ) or self.delegate.decode([self.eoi_token_id])

        # TODO: Replace processors with HF native processors.
        proc_cfg = load_processor_config(model_path, revision=revision)
        self.img_processor = Gemma4ImageProcessor(
            **proc_cfg.get("image_processor", {}),
        )

        # Video token — the upstream tokenizer_config.json doesn't include
        # <|video|> yet (the HF Processor adds it dynamically).  Mirror that
        # here so the token is in the vocabulary for tokenization.
        self.video_token = "<|video|>"
        self.delegate.add_special_tokens(
            {"additional_special_tokens": [self.video_token]}
        )
        self.video_token_id: int = self.delegate.convert_tokens_to_ids(
            self.video_token
        )
        self.vision_token_ids.append(self.video_token_id)
        self.video_processor = Gemma4VideoProcessor(
            **proc_cfg.get("video_processor", {}),
        )

        self._patch_chat_template_for_video()

        # Pre-compute special token IDs that should be skipped during decoding.
        # Gemma4 marks tool tokens as special, but we need to preserve them so
        # downstream tool parsing can match ``<|tool_call>...<tool_call|>``.
        # Only preserve tokens the model legitimately emits as part of a
        # tool-call *body* — ``<|tool_call>``, ``<tool_call|>``, and the
        # ``<|"|>`` string delimiter — so downstream tool parsing can match
        # ``<|tool_call>call:NAME{...}<tool_call|>``.
        #
        # ``<|tool>``/``<tool|>`` (tool declarations) and
        # ``<|tool_response>``/``<tool_response|>`` (tool result feedback)
        # are prompt-only: they're injected by the chat template, never by
        # the model in well-formed output. Token 50 (``<|tool_response>``)
        # additionally acts as EOS via Google's generation_config, so if
        # the model emits it we want it silenced rather than leaked as
        # text. Strip them.
        tool_token_strings = [
            SpecialToken.TOOL_CALL_START,
            SpecialToken.TOOL_CALL_END,
            SpecialToken.STRING_DELIM,
        ]
        tool_token_ids = {
            self.delegate.convert_tokens_to_ids(token)
            for token in tool_token_strings
        }
        # Skip all special tokens except tool-related ones.
        # Exposed publicly so the streaming detokenizer can apply the same
        # filter — the HuggingFace ``DecodeStream`` API only supports an all-
        # or-nothing ``skip_special_tokens`` flag, so the streaming path
        # consults this set to preserve tool tokens while still stripping
        # other specials like ``<|im_end|>``.
        self.skipped_special_token_ids: set[int] = (
            set(self.delegate.all_special_ids) - tool_token_ids
        )

        # ReasoningPipelineTokenizer surface — Gemma 4 wraps reasoning in
        # ``<|channel>thought\n...<channel|>`` blocks; expose the delimiter
        # ids so the overlap pipeline's thinking-mode temperature scaling
        # can find them without hardcoding ``<think>``/``</think>``.
        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, "<|channel>"
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, "<channel|>"
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<|channel>`` (opens a Gemma 4 reasoning span)."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``<channel|>`` (closes a Gemma 4 reasoning span)."""
        return self._reasoning_end_token_id

    def _patch_chat_template_for_video(self) -> None:
        """Patch the chat template to handle ``type == 'video'`` if missing.

        Some upstream ``tokenizer_config.json`` ship a Jinja chat template
        that inserts ``<|image|>`` for image content parts but has no
        corresponding branch for video.  When that's the case we splice in
        a ``video`` handler right after the ``image`` handler so that
        ``apply_chat_template`` emits ``<|video|>`` placeholders.

        We can delete this when the upstream tokenizer_config.json is fixed.
        """
        ct = self.delegate.chat_template
        if ct is None or "<|video|>" in ct:
            return

        # The image branch looks like:
        #   {%- elif item['type'] == 'image' -%}
        #       {{- '\n\n<|image|>\n\n' -}}
        #       {%- set ns.prev_message_type = 'image' -%}
        # We insert an analogous video branch right after it.
        image_block = "{%- set ns.prev_message_type = 'image' -%}"
        video_branch = (
            "{%- set ns.prev_message_type = 'image' -%}\n"
            "                    {%- elif item['type'] == 'video' -%}\n"
            "                        {{- '\\n\\n<|video|>\\n\\n' -}}\n"
            "                        {%- set ns.prev_message_type = 'video' -%}"
        )

        if image_block in ct:
            self.delegate.chat_template = ct.replace(
                image_block, video_branch, 1
            )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        chat_template_options = {
            "add_generation_prompt": True,
            **chat_template_options,
        }
        templated_message = self.delegate.apply_chat_template(
            [msg.model_dump(exclude_none=True) for msg in messages],
            tokenize=False,
            tools=tools,
            **chat_template_options,
        )
        assert isinstance(templated_message, str)

        # When thinking is on, force the reasoning channel open on the
        # generation turn so the model reasons on *every* assistant turn,
        # including after a tool result. Gemma otherwise only hints via
        # <|think|> and skips thinking post-tool, which fails OpenRouter's
        # reasoning+tool-call test and makes OR auto-disable tools.
        # Match the chat template, which only reads ``enable_thinking``.
        thinking_enabled = bool(chat_template_options.get("enable_thinking"))
        if (
            thinking_enabled
            and chat_template_options.get("add_generation_prompt")
            and not templated_message.rstrip("\n").endswith(
                REASONING_OPEN.rstrip("\n")
            )
        ):
            # After a tool result the template leaves the model mid-turn (no
            # <|turn>model header), so REASONING_OPEN alone has no turn
            # boundary and Gemma -- which only reasons at the start of a fresh
            # model turn -- closes the channel empty. Re-open a turn first,
            # matching the user-turn structure that does reason.
            stripped = templated_message.rstrip("\n")
            if stripped.endswith(SpecialToken.TOOL_RESPONSE_END.value):
                templated_message = stripped + SpecialToken.TURN_END.value
                templated_message += "\n" + MODEL_TURN_OPEN
            templated_message += REASONING_OPEN

        return templated_message

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]] | int, **kwargs
    ) -> str:
        """Decode tokens, preserving tool-related special tokens.

        Gemma4 marks tool tokens as special (unlike Kimi), so we need
        to selectively preserve them when skip_special_tokens=True by filtering
        unwanted special tokens before decoding.
        """
        # Log-probability responses decode one token id (a plain int) at a
        # time; match the text tokenizer's handling.
        if isinstance(encoded, int):
            encoded = np.array(encoded)
        skip_special_tokens = kwargs.get("skip_special_tokens", True)

        if not skip_special_tokens:
            # No filtering needed
            return await super().decode(encoded, **kwargs)

        # Filter out special tokens that should be skipped (all except tool tokens)
        filtered_ids = [
            token_id
            for token_id in encoded.tolist()
            if token_id not in self.skipped_special_token_ids
        ]

        # Decode with skip_special_tokens=False since we already filtered
        kwargs_no_skip = {**kwargs, "skip_special_tokens": False}
        return await super().decode(np.array(filtered_ids), **kwargs_no_skip)

    async def new_context(
        self, request: TextGenerationRequest
    ) -> Gemma4Context:
        """Create a new context for text + optional vision/video input."""
        # Extract prompt
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages:
            prompt = self.apply_chat_template(
                request.messages,
                request.tools,
                **(request.chat_template_options or {}),
            )
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        # Load and process images
        pixel_values_list: list[npt.NDArray[np.float32]] = []
        pixel_position_ids_list: list[npt.NDArray[np.int32]] = []
        num_soft_tokens: list[int] | None = None

        if request.images:
            images = [
                to_rgb(open_image(image))
                for image in request.images_for_processing()
            ]
            pixel_values_list, pixel_position_ids_list, num_soft_tokens = (
                self.img_processor(images)
            )

        # Process videos — unpack padded per-video arrays into flat
        # per-frame lists so the model doesn't redo this every batch.
        video_frame_patches: list[npt.NDArray[np.float32]] = []
        video_frame_pos_ids: list[npt.NDArray[np.int32]] = []
        video_frame_patch_counts: list[int] = []
        video_frame_soft_token_counts: list[int] = []
        video_num_soft_tokens: list[int] = []

        video_metadata_list: list[VideoMetadata] = []
        video_hashes: list[int] = []
        if request.videos:
            (
                padded_pvs,
                padded_pos,
                video_num_soft_tokens,
                video_metadata_list,
            ) = self.video_processor(request.videos)
            k = self.video_processor.pooling_kernel_size
            for pv, pos in zip(padded_pvs, padded_pos, strict=True):
                real_mask = pos[:, :, 0] >= 0
                for f in range(pv.shape[0]):
                    n_real = int(real_mask[f].sum())
                    video_frame_patches.append(pv[f, :n_real, :])
                    video_frame_pos_ids.append(pos[f, :n_real, :])
                    video_frame_patch_counts.append(n_real)
                    video_frame_soft_token_counts.append(n_real // (k * k))

            if self.enable_vision_caching:
                # Content hash per video: hash all real pixels, then fold in
                # (n_frames, patches_per_frame) so clips with identical pixels
                # but different sizes don't collide (same pattern as M3).
                for pv, pos in zip(padded_pvs, padded_pos, strict=True):
                    n_frames = pv.shape[0]
                    real_mask = pos[:, :, 0] >= 0
                    real_patches = np.concatenate(
                        [pv[f, real_mask[f]] for f in range(n_frames)], axis=0
                    )
                    pixel_hash = hash_image(real_patches)
                    video_hashes.append(
                        hash_image(
                            np.array(
                                [pixel_hash, n_frames, int(real_mask[0].sum())],
                                dtype=np.int64,
                            )
                        )
                    )

        # Expand image placeholders
        if isinstance(prompt, str):
            text_list = [prompt]
        else:
            text_list = None

        if text_list is not None and num_soft_tokens is not None:
            replacements = [
                f"{self.boi_token}{self.image_token * n}{self.eoi_token}"
                for n in num_soft_tokens
            ]
            replacements_iter = iter(replacements)
            pattern = re.escape(self.image_token)
            text_list = [
                re.sub(pattern, lambda _: next(replacements_iter), t)
                for t in text_list
            ]

        # Expand video placeholders with per-frame timestamps
        if text_list is not None and video_metadata_list:
            video_replacements: list[str] = []
            for metadata, n_tokens in zip(
                video_metadata_list, video_num_soft_tokens, strict=True
            ):
                timestamp_strs = [
                    f"{int(s // 60):02d}:{int(s % 60):02d}"
                    for s in metadata.timestamps
                ]
                video_replacements.append(
                    " ".join(
                        f"{t} {self.boi_token}"
                        f"{self.video_token * n_tokens}"
                        f"{self.eoi_token}"
                        for t in timestamp_strs
                    )
                )
            replacements_iter = iter(video_replacements)
            pattern = re.escape(self.video_token)
            text_list = [
                re.sub(pattern, lambda _: next(replacements_iter), t)
                for t in text_list
            ]

        # Tokenize
        if text_list is not None:
            tokenizer_out = self.delegate(
                text_list,
                add_special_tokens=add_special_tokens,
                padding=False,
                return_token_type_ids=False,
            )
            if isinstance(tokenizer_out["input_ids"][0], int):
                encoded_prompt = np.array(
                    tokenizer_out["input_ids"], dtype=np.int64
                )
            else:
                encoded_prompt = np.array(
                    tokenizer_out["input_ids"][0], dtype=np.int64
                )
        else:
            encoded_prompt = np.array(list(prompt), dtype=np.int64)

        # Compute token type IDs (0=text, 1=image, 2=video)
        mm_token_type_ids = np.zeros_like(encoded_prompt, dtype=np.int64)
        mm_token_type_ids[encoded_prompt == self.image_token_id] = 1
        mm_token_type_ids[encoded_prompt == self.video_token_id] = 2

        # Compute generation budget
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        response_format_schema = (
            request.response_format.json_schema
            if request.response_format
            else None
        )
        json_schema = (
            json.dumps(response_format_schema)
            if response_format_schema
            else None
        )

        grammar = (
            request.response_format.grammar if request.response_format else None
        )

        grammar_state = GrammarEnforcementState.from_response_format(
            request.response_format
        )

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise PromptTooLongError(encoded_prompt.shape[0], self.max_length)

        # Build ImageMetadata for images only (not videos).
        # Find contiguous ranges of *image* tokens only.
        image_token_ranges = find_contiguous_ranges(
            encoded_prompt, [self.image_token_id]
        )
        image_metadata = [
            ImageMetadata(
                start_idx=start_idx,
                end_idx=end_idx,
                pixel_values=pixels,
                image_hash=hash_image(pixels)
                if self.enable_prefix_caching or self.enable_vision_caching
                else None,
            )
            for (start_idx, end_idx), pixels in zip(
                image_token_ranges, pixel_values_list, strict=True
            )
        ]

        # Build video token ranges
        video_token_ranges = [
            (int(s), int(e))
            for s, e in find_contiguous_ranges(
                encoded_prompt, [self.video_token_id]
            )
        ]

        eos_tracker = await self.create_eos_tracker(request)
        context = Gemma4Context(
            request_id=request.request_id,
            eos_tracker=eos_tracker,
            target_endpoint=request.target_endpoint,
            mm_token_type_ids=mm_token_type_ids.astype(np.int64, copy=False),
            pixel_position_ids=pixel_position_ids_list,
            video_frame_patches=video_frame_patches,
            video_frame_pos_ids=video_frame_pos_ids,
            video_frame_patch_counts=video_frame_patch_counts,
            video_frame_soft_token_counts=video_frame_soft_token_counts,
            video_token_ranges=video_token_ranges,
            video_hashes=video_hashes,
            tokens=TokenBuffer(
                array=encoded_prompt.astype(np.int64, copy=False),
            ),
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            grammar=grammar,
            grammar_state=grammar_state,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
            images=image_metadata,
            vision_token_ids=self.vision_token_ids,
            vocab_size=self.tokenizer_vocab_size,
        )

        return context


def _require_attr(config: Any, *names: str) -> int:
    """Return the first found attribute from *config*, or raise."""
    for name in names:
        val = getattr(config, name, None)
        if val is not None:
            return val
    raise ValueError(
        f"None of {names} found in config; available attributes: "
        f"{[a for a in dir(config) if not a.startswith('_')]}"
    )
