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
"""Tokenizer for Lfm2VlForConditionalGeneration.

`TextAndVisionTokenizer` does the prompt/token/span work; this subclass
adds what the stock path cannot carry: the LFM2-VL processor's NA-FLEX
outputs. The processor emits, per request,

- `pixel_values` `[num_tiles, max_num_patches, 16*16*3]` (tiles of every
  image, padded to the largest tile's patch count),
- `spatial_shapes` `[num_tiles, 2]` — each tile grid in PATCHES,
- `pixel_attention_mask` `[num_tiles, max_num_patches]` — 1 on real
  patches,

and the stock tokenizer expects one pixel array per image and keeps none
of that. So the processor is wrapped (not replaced): the call is
recorded, the padded canvas is unpacked per tile with the mask, the
tiles of the (single) image are concatenated in processor tile order —
tiles first, thumbnail last — while `spatial_shapes` travels as the
processor left it: the stock path forwards it on the context's
`extra_model_args`, which is where the batch processor reads the
per-tile grids.

Multi-image requests are refused rather than mis-served: one image per
request is what the parity gate measures, and the stock tokenizer's
strict image-span bookkeeping is not worth reproducing for more.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    PipelineConfig,
    TextAndVisionTokenizer,
    resolve_eos_token_ids,
)
from max.pipelines.lib.tokenizer import resolve_token_id_bound
from max.pipelines.modeling.types import TextGenerationRequest

# The per-tile patch grids, forwarded by the stock tokenizer from the
# processor's `spatial_shapes` output.
TILE_GRIDS_KEY = "spatial_shapes"


class _RecordingProcessor:
    """Delegates to the HF processor and records the NaFlex outputs."""

    def __init__(self, processor: Any) -> None:
        self._processor = processor

    def __getattr__(self, name: str) -> Any:
        # `__dict__`, NOT `self._processor`: a copy/unpickle probe asks
        # for dunder attributes BEFORE `__init__` has run, and reading
        # `self._processor` there recurses. Read the instance dict
        # directly and refuse anything before then.
        processor = self.__dict__.get("_processor")
        if processor is None:
            raise AttributeError(name)
        return getattr(processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # A raw token-id prompt (the parity walk's id form —
        # /v1/completions takes prompt: str | list[int]) never reaches the
        # HF processor: the ids are already ids and an image-free request
        # has no NaFlex canvas to unpack. The processor would .count() its
        # image token over the ints and 500 (AttributeError), which made
        # 63 of the walk's 64 positions unmeasurable. Bypass, don't wrap:
        # the lfm3-vl lane's recorded fix, same processor family.
        text = kwargs.get("text")
        if isinstance(text, (list, tuple)) and text and type(text[0]) is int:
            if not kwargs.get("images"):
                return {"input_ids": list(text)}
        out = self._processor(*args, **kwargs)
        grids = out.get("spatial_shapes") if hasattr(out, "get") else None
        rows = out.get("pixel_values") if hasattr(out, "get") else None
        mask = out.get("pixel_attention_mask") if hasattr(out, "get") else None
        if grids is None or rows is None or mask is None:
            return out
        grids = np.asarray(grids, dtype=np.int64).reshape(-1, 2)
        rows = np.asarray(rows)
        mask = np.asarray(mask)
        if rows.shape[0] != len(grids) or mask.shape[0] != len(grids):
            raise ValueError(
                f"{rows.shape[0]} pixel rows and a {len(grids)}-row grid: "
                "the processor's NaFlex layout changed"
            )
        # Pad-aware unpick: keep only each tile's real patches, in tile
        # order, as one [patches_total, 16*16*3] matrix per REQUEST (2-D,
        # not the stock tokenizer's per-image list), plus the per-tile
        # grids for the batch processor.
        pieces = []
        for i, (gh, gw) in enumerate(grids):
            valid = mask[i] > 0
            if int(valid.sum()) != int(gh) * int(gw):
                raise ValueError(
                    f"tile {i} has {int(valid.sum())} real patches for a "
                    f"{int(gh)}x{int(gw)} grid"
                )
            pieces.append(rows[i][valid])
        patches = (
            np.concatenate(pieces, axis=0)
            if pieces
            else rows.reshape(-1, rows.shape[-1])
        )
        out["pixel_values"] = [patches]
        return out


class _TextOnlyProcessor:
    """Serves text where the vision processor cannot be constructed.

    This family's image processor needs torchvision, which some serve
    environments do not ship; AutoProcessor then raises and the stock
    tokenizer cannot be built at all. This shim covers the two members
    the stock path uses for text requests - the call (delegated to the
    text tokenizer) and apply_chat_template (the checkpoint ships one
    chat_template.jinja, which the text tokenizer loads as well, so the
    render is identical to the processor's). Image requests are refused
    with the reason rather than mis-served.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __call__(
        self,
        *,
        text: str,
        images: Any = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if images:
            raise ValueError(
                "an image request reached a serve built without the vision "
                "processor (torchvision is missing); images are not served"
            )
        ids = self._delegate(text, add_special_tokens=add_special_tokens)[
            "input_ids"
        ]
        return {"input_ids": [ids]}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        tools: Any = None,
        **options: Any,
    ) -> str:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") != "text":
                    raise ValueError(
                        "an image request reached a serve built without "
                        "the vision processor (torchvision is missing); "
                        "images are not served"
                    )
        rendered = self._delegate.apply_chat_template(
            messages, tokenize=tokenize, tools=tools, **options
        )
        assert isinstance(rendered, str)
        return rendered


class Lfm2VlTokenizer(TextAndVisionTokenizer):
    """Stock text+vision tokenizer, with the one-image rule checked here."""

    def __init__(
        self, model_path: str, pipeline_config: PipelineConfig, **kwargs: Any
    ) -> None:
        try:
            super().__init__(model_path, pipeline_config, **kwargs)
        except ValueError:
            if "delegate" not in self.__dict__ or "processor" in self.__dict__:
                raise
            # Only the AutoProcessor construction can raise here with the
            # delegate already built (missing torchvision). Finish the
            # stock init's remaining steps; the shim below covers text.
            self._finish_init_text_only(pipeline_config)
        processor = getattr(self, "processor", None)
        self.processor = (
            _RecordingProcessor(processor)
            if processor is not None
            else _TextOnlyProcessor(self.delegate)
        )

    def _finish_init_text_only(self, pipeline_config: PipelineConfig) -> None:
        """Completes what the stock init does after building the processor."""
        self.processor = None
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
        vision_token_ids: list[int] = []
        for vision_token_id_name in ("image_token_id", "image_token_index"):
            if vision_token_id := getattr(config, vision_token_id_name, None):
                vision_token_ids.append(vision_token_id)
        if not vision_token_ids:
            raise ValueError("vision_token_id not found in model_config config")
        self.vision_token_ids = vision_token_ids

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Check the one-image rule; the grids ride `extra_model_args`."""
        context = await super().new_context(request)
        images = list(getattr(context, "next_images", []))
        if not images:
            return context
        if len(images) > 1:
            raise ValueError(
                f"a request carried {len(images)} images. This port serves "
                "ONE image per request; the stock tokenizer's span "
                "bookkeeping would have to be reproduced for more."
            )
        grids = (context.extra_model_args or {}).get(TILE_GRIDS_KEY)
        if grids is None:
            raise ValueError(
                "the processor returned an image but no `spatial_shapes`: "
                "the tower cannot place patches without them"
            )
        return context
