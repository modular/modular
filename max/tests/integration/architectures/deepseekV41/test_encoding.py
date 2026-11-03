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
# ruff: noqa: RUF001
"""Tests flattening request messages and rendering the V4.1 prompt."""

from __future__ import annotations

import asyncio
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from max.pipelines.architectures.deepseekV41.encoding_dsv41 import (
    ASSISTANT_SP_TOKEN,
    IMAGE_PLACEHOLDER,
    ImageSlot,
    encode_messages,
)
from max.pipelines.architectures.deepseekV41.image_processor import (
    DeepseekV41ImageProcessor,
    ImageGrid,
    expand_image_spans,
)
from max.pipelines.architectures.deepseekV41.tokenizer import (
    DeepseekV41Tokenizer,
    _flatten_message,
    encode_literal_text,
    render_prompt,
    render_prompt_with_image_slots,
)
from max.pipelines.architectures.deepseekV41_modulev3.tokenizer import (
    DeepseekV41Tokenizer as DeepseekV41ModuleV3Tokenizer,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.tokenizer import _render_custom_chat_template
from max.pipelines.modeling.types import (
    ImageContentPart,
    LabelScoringTokenizer,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from PIL import Image
from test_common.mocks import DummyPipelineConfig
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerFast,
)


def _make_tokenizer_with_image_added_token() -> tuple[
    DeepseekV41Tokenizer, PreTrainedTokenizerFast, int
]:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    delegate = Tokenizer(
        models.BPE(
            vocab={char: index for index, char in enumerate(alphabet)},
            merges=[],
        )
    )
    # These Rust-backed properties are writable despite their read-only stubs.
    setattr(  # noqa: B010
        delegate,
        "pre_tokenizer",
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    )
    setattr(  # noqa: B010
        delegate, "decoder", decoders.ByteLevel()
    )
    delegate.add_tokens(
        [AddedToken(IMAGE_PLACEHOLDER, normalized=False, special=False)]
    )
    image_token_id = delegate.token_to_id(IMAGE_PLACEHOLDER)
    assert image_token_id is not None
    delegate = PreTrainedTokenizerFast(tokenizer_object=delegate)

    class ImageProcessor(DeepseekV41ImageProcessor):
        grid = ImageGrid(n_vit_h=1, n_vit_w=1, n_llm_h=1, n_llm_w=1)

        def __init__(self, vision_config: Any) -> None:
            pass

        def plan(self, width: int, height: int) -> tuple[ImageGrid, int, int]:
            return self.grid, width, height

        def __call__(self, image: Image.Image) -> tuple[np.ndarray, ImageGrid]:
            return np.full((1, 3), image.width, dtype=np.uint8), self.grid

    tokenizer = object.__new__(DeepseekV41Tokenizer)
    tokenizer.delegate = delegate
    tokenizer.image_token_id = image_token_id
    tokenizer.image_processor = ImageProcessor(None)
    tokenizer.max_length = 4096
    tokenizer._token_id_bound = len(delegate)
    tokenizer._eos_token_ids = set()
    tokenizer._chat_template = None
    return tokenizer, delegate, image_token_id


_MULTI_PART_MESSAGE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "first part"},
        {"type": "text", "text": "second part"},
    ],
}


def test_flatten_message_joins_multi_part_text_with_base_separator() -> None:
    """Checks multi-part text joins with the blank line the encoder uses."""
    message = TextGenerationRequestMessage.model_validate(_MULTI_PART_MESSAGE)
    flattened = _flatten_message(message)
    assert flattened["content"] == "first part\n\nsecond part"


def test_render_prompt_renders_flattened_multi_part_message() -> None:
    """Checks flattened text renders as the encoder renders the text blocks."""
    message = TextGenerationRequestMessage.model_validate(_MULTI_PART_MESSAGE)
    prompt = render_prompt([_flatten_message(message)], None, {})
    assert "first part\n\nsecond part" in prompt
    assert prompt == render_prompt([_MULTI_PART_MESSAGE], None, {})


_TOOL_TURN: list[dict[str, Any]] = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "r2",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "value"},
]


def test_render_prompt_keeps_current_turn_reasoning_across_tool_results() -> (
    None
):
    """Checks a tool result does not start a turn that drops its reasoning."""
    prompt = render_prompt(_TOOL_TURN, None, {})
    assert "r1" not in prompt
    assert "<think>r2</think>" in prompt
    assert prompt.endswith(
        f"<tool_result>value</tool_result>{ASSISTANT_SP_TOKEN}<think>"
    )


def test_render_prompt_drops_tool_turn_reasoning_after_next_user() -> None:
    prompt = render_prompt(
        [*_TOOL_TURN, {"role": "user", "content": "q3"}], None, {}
    )
    assert "r2" not in prompt
    assert prompt.endswith(f"q3{ASSISTANT_SP_TOKEN}<think>")


def test_render_prompt_accepts_image_placeholder_quoted_in_tool_output() -> (
    None
):
    """The placeholder is ``special: false`` in the V4.1 tokenizer (round-trips
    as text token 129264). A user prompt and tool result that quote it must
    encode without error: the model sees it as text either way.
    """
    quoted_source = (
        "fn main() {\n"
        "    let id = encoder.encode('<｜deepseek_image｜>', None);\n"
        "    assert_eq!(id, vec![129264]);\n"
        "}\n"
    )
    messages = [
        {
            "role": "user",
            "content": "Run the test that quotes the placeholder:\n\n"
            + quoted_source,
        },
        {"role": "tool", "tool_call_id": "call_0", "content": quoted_source},
        {"role": "user", "content": "continue"},
    ]
    prompt, slots = render_prompt_with_image_slots(messages, None, {})
    assert "<｜deepseek_image｜>" in prompt
    assert "<tool_result>" in prompt
    assert slots == []  # no image blocks: the encoder placed none


def test_expand_image_spans_uses_encoder_supplied_token_positions() -> None:
    """Only the token position supplied for the inserted image expands."""
    image_token_id = 129264
    # The user token stays plain; the image token position is explicit.
    token_ids = [100, image_token_id, 101, image_token_id, 102]
    grid = ImageGrid(n_vit_h=2, n_vit_w=2, n_llm_h=1, n_llm_w=1)
    expanded, starts = expand_image_spans(
        token_ids, [grid], image_token_id, image_positions=[3]
    )

    # The encoder's token at original position 3 expands; user text stays.
    # Expanded stream: [100, 129264(text), 101, four span tokens, 102].
    assert starts == [3]
    assert expanded[1] == image_token_id  # user-typed: bare text token
    assert expanded[3:7].tolist() == [image_token_id] * 4  # span
    assert expanded[7] == 102


def test_expand_image_spans_no_images_all_placeholders_stay_text() -> None:
    """With no images, every placeholder stays as a bare text token."""
    image_token_id = 129264
    token_ids = [100, image_token_id, 101, image_token_id, 102]
    expanded, starts = expand_image_spans(
        token_ids, [], image_token_id, image_positions=[]
    )
    assert starts == []
    assert expanded.tolist() == token_ids  # unchanged


def test_expand_image_spans_requires_one_position_per_image() -> None:
    image_token_id = 129264
    token_ids = [100, image_token_id, 101]
    grid = ImageGrid(n_vit_h=2, n_vit_w=2, n_llm_h=1, n_llm_w=1)
    with pytest.raises(ValueError):
        expand_image_spans(
            token_ids, [grid, grid], image_token_id, image_positions=[1]
        )


def test_render_prompt_returns_inserted_image_offsets() -> None:
    """Text literals do not affect the renderer's image provenance."""
    messages: list[dict[str, Any]] = [
        # User text mentioning the placeholder (user-typed)
        {"role": "user", "content": "before <｜deepseek_image｜> after"},
        # Assistant whose reasoning quotes the placeholder (rendered because
        # it is the current turn in thinking mode with drop_thinking=False)
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "the token is <｜deepseek_image｜>",
        },
        # The image block has an independent offset in the rendered prompt.
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "img"}}],
        },
    ]
    prompt, slots = render_prompt_with_image_slots(
        messages, None, {"drop_thinking": False}
    )
    assert len(slots) == 1
    assert slots[0].image_index == 0
    offset = slots[0].offset
    assert prompt[offset : offset + len(IMAGE_PLACEHOLDER)] == IMAGE_PLACEHOLDER
    assert prompt.count("<｜deepseek_image｜>") == 3


def test_chat_encoding_defuses_literals_and_inserts_image_ids_at_offsets() -> (
    None
):
    """User literals stay text while the renderer-owned offset inserts an ID."""
    tokenizer, delegate, image_token_id = (
        _make_tokenizer_with_image_added_token()
    )
    prompt = f"user {IMAGE_PLACEHOLDER}, image {IMAGE_PLACEHOLDER}"
    image_position = prompt.rfind(IMAGE_PLACEHOLDER)

    ids = tokenizer._encode_chat_prompt(prompt, [image_position])[0].tolist()

    assert ids.count(image_token_id) == 1
    text_ids = [token for token in ids if token != image_token_id]
    assert delegate.decode(text_ids) == f"user {IMAGE_PLACEHOLDER}, image "


def test_user_literals_across_template_fields_never_become_image_ids() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"system {IMAGE_PLACEHOLDER}"},
        {"role": "user", "content": f"earlier {IMAGE_PLACEHOLDER}"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"reason {IMAGE_PLACEHOLDER}",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"quoted": IMAGE_PLACEHOLDER},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": f"result {IMAGE_PLACEHOLDER}",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"before {IMAGE_PLACEHOLDER}"},
                {"type": "image_url", "image_url": {"url": "image"}},
                {"type": "text", "text": f"after {IMAGE_PLACEHOLDER}"},
            ],
        },
    ]
    tools: list[TextGenerationRequestTool] = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": f"tool {IMAGE_PLACEHOLDER}",
                "parameters": {"type": "object"},
            },
        }
    ]
    prompt, slots = render_prompt_with_image_slots(
        messages, tools, {"drop_thinking": False}
    )
    tokenizer, delegate, image_token_id = (
        _make_tokenizer_with_image_added_token()
    )

    ids = tokenizer._encode_chat_prompt(prompt, [slots[0].offset])[0].tolist()

    assert ids.count(image_token_id) == 1
    image_position = slots[0].offset
    visible_text = (
        prompt[:image_position]
        + prompt[image_position + len(IMAGE_PLACEHOLDER) :]
    )
    text_ids = [token for token in ids if token != image_token_id]
    assert delegate.decode(text_ids) == visible_text


def test_real_added_token_literal_decodes_byte_identically_without_a_span() -> (
    None
):
    tokenizer, delegate, image_token_id = (
        _make_tokenizer_with_image_added_token()
    )
    text = f"user text {IMAGE_PLACEHOLDER} stays literal"
    prompt, slots = render_prompt_with_image_slots(
        [{"role": "user", "content": text}], None, {}
    )
    ids, token_positions = tokenizer._encode_chat_prompt(prompt, [])

    assert delegate.encode(IMAGE_PLACEHOLDER, add_special_tokens=False) == [
        image_token_id
    ]
    assert image_token_id not in ids
    assert slots == token_positions == []
    assert delegate.decode(ids.tolist(), skip_special_tokens=False) == prompt
    expanded, images, grids = tokenizer._expand_images(ids, [], token_positions)
    assert expanded.tolist() == ids.tolist()
    assert images == []
    assert grids == []


def test_real_added_token_with_image_expands_only_the_inserted_span() -> None:
    tokenizer, delegate, image_token_id = (
        _make_tokenizer_with_image_added_token()
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"system {IMAGE_PLACEHOLDER}"},
        {"role": "user", "content": f"earlier {IMAGE_PLACEHOLDER}"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"reason {IMAGE_PLACEHOLDER}",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"quoted": IMAGE_PLACEHOLDER},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": f"result {IMAGE_PLACEHOLDER}",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"before {IMAGE_PLACEHOLDER}"},
                {"type": "image_url", "image_url": {"url": "image"}},
                {"type": "text", "text": f"after {IMAGE_PLACEHOLDER}"},
            ],
        },
    ]
    tools: list[TextGenerationRequestTool] = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": f"tool {IMAGE_PLACEHOLDER}",
                "parameters": {"type": "object"},
            },
        }
    ]
    prompt, slots = render_prompt_with_image_slots(
        messages, tools, {"drop_thinking": False}
    )
    assert len(slots) == 1
    ids, token_positions = tokenizer._encode_chat_prompt(
        prompt, [slots[0].offset]
    )
    assert len(token_positions) == 1
    assert ids.tolist().count(image_token_id) == 1
    assert delegate.decode(ids.tolist(), skip_special_tokens=False) == prompt

    expanded, metadata, grids = tokenizer._expand_images(
        ids, [Image.new("RGB", (14, 14), color="black")], token_positions
    )
    assert len(metadata) == len(grids) == 1
    assert metadata[0].start_idx == token_positions[0]
    assert metadata[0].end_idx - metadata[0].start_idx == grids[0].num_tokens
    assert expanded.tolist().count(image_token_id) == grids[0].num_tokens


def _image_message(*parts: dict[str, Any]) -> TextGenerationRequestMessage:
    return TextGenerationRequestMessage.model_validate(
        {"role": "user", "content": list(parts)}
    )


def _render_custom(
    tokenizer: DeepseekV41Tokenizer,
    template: str,
    messages: list[TextGenerationRequestMessage],
    **options: Any,
) -> tuple[str, list[ImageSlot]]:
    tokenizer._chat_template = template
    return tokenizer._render_chat(messages, None, **options)


def _render_plain(
    delegate: PreTrainedTokenizerFast,
    template: str,
    messages: list[TextGenerationRequestMessage],
    **options: Any,
) -> str:
    """Renders the template as the base did."""
    return _render_custom_chat_template(
        delegate,
        template,
        [message.flatten_content() for message in messages],
        None,
        **{"add_generation_prompt": True, **options},
    )


def test_custom_template_rejects_image_parts_before_rendering() -> None:
    tokenizer, _, _ = _make_tokenizer_with_image_added_token()
    # This template fails whenever it renders, so the image error shows the
    # check ran first.
    tokenizer._chat_template = "{{ raise_exception('rendered') }}"
    messages = [
        _image_message({"type": "text", "text": "look"}, {"type": "image"})
    ]

    with pytest.raises(InputError, match="does not support image inputs"):
        tokenizer._render_chat(messages, None)
    with pytest.raises(InputError, match="does not support image inputs"):
        tokenizer.apply_chat_template(messages, tools=None)


def test_custom_template_render_failure_is_an_input_error() -> None:
    tokenizer, _, _ = _make_tokenizer_with_image_added_token()
    message = TextGenerationRequestMessage(role="user", content="hi")

    with pytest.raises(InputError, match="rendered"):
        _render_custom(
            tokenizer, "{{ raise_exception('rendered') }}", [message]
        )


_P = IMAGE_PLACEHOLDER
_CONTENT = "{% for message in messages %}{{ message.content }}{% endfor %}"


@pytest.mark.parametrize(
    ("template", "content", "options"),
    [
        pytest.param(_CONTENT, f"a {_P} b", {}, id="content"),
        pytest.param(
            "{{ messages[0].content == '" + _P + "' }}" + _CONTENT,
            _P,
            {},
            id="content_equality",
        ),
        pytest.param(
            "{{ left }}{{ right }}{{ whole }}" + _CONTENT,
            "hi",
            {"left": _P[:12], "right": _P[12:], "whole": _P},
            id="kwargs",
        ),
        pytest.param(
            "{% set framed %}[" + _P + "]{% endset %}"
            "{{ framed }}{{ framed == '["
            + _P
            + "]' }}{{ framed | tojson }}"
            + _CONTENT,
            "hi",
            {},
            id="set_block",
        ),
        pytest.param(
            "{% macro image() %}"
            + _P
            + "{% endmacro %}{{ image() }}"
            + _CONTENT,
            "hi",
            {},
            id="macro",
        ),
        pytest.param(
            "{% raw %}" + _P + "{{ x }}{% endraw %}" + _CONTENT,
            "hi",
            {},
            id="raw_block",
        ),
        pytest.param(_P + _CONTENT + _P, "hi", {}, id="template_text"),
        pytest.param(
            "{{ '" + _P + "' }}"
            "{{ messages[0].content | replace('\uff5c', '') }}",
            f"<\uff5c{_P}\uff5c>",
            {},
            id="expressions",
        ),
    ],
)
def test_custom_template_placeholders_from_any_source_are_text(
    template: str, content: str, options: dict[str, Any]
) -> None:
    tokenizer, delegate, image_token_id = (
        _make_tokenizer_with_image_added_token()
    )
    message = TextGenerationRequestMessage(role="user", content=content)

    first = _render_custom(tokenizer, template, [message], **options)
    second = _render_custom(tokenizer, template, [message], **options)
    prompt, slots = first
    ids, token_positions = tokenizer._encode_chat_prompt(prompt, [])

    assert first == second
    assert prompt == _render_plain(delegate, template, [message], **options)
    assert _P in prompt
    assert slots == token_positions == []
    assert image_token_id not in ids.tolist()
    assert delegate.decode(ids.tolist(), skip_special_tokens=False) == prompt


def test_custom_template_prompt_matches_base_without_literals() -> None:
    tokenizer, delegate, _ = _make_tokenizer_with_image_added_token()
    template = (
        "{% for message in messages %}<{{ message.role }}>"
        "{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    messages = [
        TextGenerationRequestMessage(role="system", content="be brief"),
        TextGenerationRequestMessage.model_validate(_MULTI_PART_MESSAGE),
    ]

    prompt, _ = _render_custom(tokenizer, template, messages)
    ids, _ = tokenizer._encode_chat_prompt(prompt, [])

    base = _render_plain(delegate, template, messages)
    assert prompt == base
    # The custom path flattens text parts with flatten_content's single
    # newline, as base did; only the built-in encoder uses a blank line.
    assert prompt == (
        "<system>be brief<user>first part\nsecond part<assistant>"
    )
    assert ids.tolist() == delegate.encode(base, add_special_tokens=False)


def test_literal_free_multi_part_text_ids_match_base() -> None:
    tokenizer, delegate, _ = _make_tokenizer_with_image_added_token()
    message = TextGenerationRequestMessage.model_validate(_MULTI_PART_MESSAGE)

    prompt, slots = tokenizer._render_chat([message], None)
    ids, _ = tokenizer._encode_chat_prompt(prompt, [])
    assert slots == []
    base_prompt = render_prompt(
        [{"role": "user", "content": "first part\n\nsecond part"}], None, {}
    )
    assert ids.tolist() == delegate.encode(
        base_prompt, add_special_tokens=False
    )


def test_custom_template_kwargs_image_count_remains_available() -> None:
    tokenizer, _, _ = _make_tokenizer_with_image_added_token()
    tokenizer._chat_template = "{{ image_count }}"
    message = TextGenerationRequestMessage.model_validate(
        {"role": "user", "content": "hello"}
    )

    prompt, _ = tokenizer._render_chat([message], None, image_count=7)

    assert prompt == "7"


def test_encoder_returns_images_in_prompt_order_with_message_order_slots() -> (
    None
):
    """Tool results move into call order; each slot keeps its block's number."""
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {}},
                }
                for call_id in ("a", "b")
            ],
            "content": "",
        },
        {
            "role": "tool",
            "tool_call_id": "b",
            "content": [{"type": "image", "data": "source-b"}],
        },
        {
            "role": "tool",
            "tool_call_id": "a",
            "content": [{"type": "image", "data": "source-a"}],
        },
    ]

    prompt, media = encode_messages(
        messages, thinking_mode="chat", return_multi_modal_data=True
    )

    assert [slot.image_index for slot in media["image_slots"]] == [1, 0]
    assert [image["data"] for image in media["images"]] == [
        "source-a",
        "source-b",
    ]
    assert prompt.count(IMAGE_PLACEHOLDER) == 2
    for slot in media["image_slots"]:
        end = slot.offset + len(IMAGE_PLACEHOLDER)
        assert prompt[slot.offset : end] == IMAGE_PLACEHOLDER


def test_text_only_encode_messages_returns_the_prompt_string() -> None:
    prompt = encode_messages(
        [{"role": "user", "content": f"a {IMAGE_PLACEHOLDER}"}],
        thinking_mode="chat",
    )
    assert isinstance(prompt, str)
    assert IMAGE_PLACEHOLDER in prompt


def test_image_marker_text_in_caller_text_stays_text() -> None:
    """Text shaped like the encoder's marker is not taken for an image."""
    lookalike = "\ufdd0DSV41_IMAGE_00000000000000000000000000000000_0\ufdd1"
    prompt, media = encode_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": lookalike},
                    {"type": "image", "data": "x"},
                ],
            }
        ],
        thinking_mode="chat",
        return_multi_modal_data=True,
    )
    assert lookalike in prompt
    assert len(media["image_slots"]) == 1


def test_apply_chat_template_satisfies_label_scoring_protocol() -> None:
    tokenizer, delegate, _ = _make_tokenizer_with_image_added_token()
    message = TextGenerationRequestMessage.model_validate(
        {"role": "user", "content": "Hi"}
    )

    assert isinstance(tokenizer, LabelScoringTokenizer)
    prompt = tokenizer.apply_chat_template(
        [message], tools=None, thinking=False
    )
    assert type(prompt) is str

    ids = asyncio.run(tokenizer.encode(prompt, add_special_tokens=False))
    assert ids.tolist() == delegate.encode(prompt, add_special_tokens=False)


def test_chat_encoding_replaces_unpaired_surrogates() -> None:
    tokenizer, delegate, _ = _make_tokenizer_with_image_added_token()
    ids, positions = tokenizer._encode_chat_prompt("x\ud800", [])

    assert positions == []
    assert ids.tolist() == delegate.encode("x\ufffd", add_special_tokens=False)


def test_checkpoint_fast_tokenizer_uses_list_encode_api() -> None:
    """Checks real checkpoint tokenization and context creation when cached."""
    checkpoint_cache = (
        Path(
            os.environ.get(
                "HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")
            )
        )
        / "models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots"
    )
    snapshots = sorted(checkpoint_cache.glob("*/tokenizer.json"))
    if not snapshots:
        pytest.skip("DeepSeek-V4.1 checkpoint tokenizer is not cached")

    delegate = AutoTokenizer.from_pretrained(
        snapshots[-1].parent,
        local_files_only=True,
        trust_remote_code=False,
    )
    assert isinstance(delegate, PreTrainedTokenizerFast)
    config = PretrainedConfig.from_pretrained(
        snapshots[-1].parent, local_files_only=True
    )
    assert isinstance(config.vision_config, dict)
    config.vision_config = PretrainedConfig.from_dict(config.vision_config)
    pipeline_config = DummyPipelineConfig(
        model_path=str(snapshots[-1].parent),
        quantization_encoding="float32",
        max_batch_size=None,
        max_length=4096,
        huggingface_config=config,
    )
    tokenizer = DeepseekV41Tokenizer(
        str(snapshots[-1].parent), pipeline_config, max_length=4096
    )
    assert isinstance(tokenizer.delegate, PreTrainedTokenizerFast)

    plain = asyncio.run(
        tokenizer.new_context(
            TextGenerationRequest(
                request_id=RequestID(),
                model_name="checkpoint-test",
                messages=[
                    TextGenerationRequestMessage(role="user", content="Hi")
                ],
            )
        )
    )
    literal = asyncio.run(
        tokenizer.new_context(
            TextGenerationRequest(
                request_id=RequestID(),
                model_name="checkpoint-test",
                messages=[
                    TextGenerationRequestMessage(
                        role="user", content=f"Hi {IMAGE_PLACEHOLDER}"
                    )
                ],
            )
        )
    )
    image_bytes = BytesIO()
    Image.new("RGB", (32, 32), color="black").save(image_bytes, format="PNG")
    image_request = TextGenerationRequest(
        request_id=RequestID(),
        model_name="checkpoint-test",
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    TextContentPart(text=IMAGE_PLACEHOLDER),
                    ImageContentPart(),
                    TextContentPart(text=IMAGE_PLACEHOLDER),
                ],
            )
        ],
        images=[image_bytes.getvalue()],
    )
    with_image = asyncio.run(tokenizer.new_context(image_request))

    image_token_id = tokenizer.image_token_id
    assert image_token_id not in literal.tokens.array.tolist()
    assert literal.extra_model_args == {}
    assert len(with_image.images) == 1
    image_span = with_image.images[0]
    image_token_indices = with_image.extra_model_args["image_token_indices"]
    assert image_span.start_idx == int(image_token_indices[0])
    assert image_span.end_idx == int(image_token_indices[-1]) + 1
    assert image_token_id not in plain.tokens.array.tolist()


def _png(width: int) -> bytes:
    data = BytesIO()
    Image.new("RGB", (width, 14), color="black").save(data, format="PNG")
    return data.getvalue()


def _request(
    messages: list[TextGenerationRequestMessage] | None = None,
    *,
    prompt: str | list[int] | None = None,
    images: list[bytes] | None = None,
) -> TextGenerationRequest:
    return TextGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        messages=messages or [],
        prompt=prompt,
        images=images or [],
    )


def test_token_prompt_with_the_image_token_is_an_input_error() -> None:
    """A prompt request carries no images, so it cannot hold the image token."""
    tokenizer, delegate, _ = _make_tokenizer_with_image_added_token()
    ids = delegate.encode(
        f"first {IMAGE_PLACEHOLDER}", add_special_tokens=False
    )

    with pytest.raises(InputError, match="image token"):
        asyncio.run(tokenizer.new_context(_request(prompt=ids)))


def test_text_prompt_placeholders_stay_text() -> None:
    """A completions prompt keeps a typed placeholder as text, as chat does."""
    tokenizer, delegate, image_token_id = (
        _make_tokenizer_with_image_added_token()
    )
    prompt = f"first {IMAGE_PLACEHOLDER}, then {IMAGE_PLACEHOLDER}"

    context = asyncio.run(tokenizer.new_context(_request(prompt=prompt)))

    ids = context.tokens.array.tolist()
    assert image_token_id not in ids
    assert delegate.decode(ids) == prompt
    assert context.images == []


def test_encode_keeps_placeholders_in_rendered_text_as_text() -> None:
    """Decision scoring encodes a rendered prompt string with ``encode``."""
    tokenizer, delegate, image_token_id = (
        _make_tokenizer_with_image_added_token()
    )
    prompt = tokenizer.apply_chat_template(
        [
            TextGenerationRequestMessage(
                role="user", content=f"what is {IMAGE_PLACEHOLDER}?"
            )
        ]
    )

    ids = asyncio.run(tokenizer.encode(prompt, add_special_tokens=False))

    assert image_token_id not in ids.tolist()
    assert delegate.decode(ids.tolist()) == prompt


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain",
        IMAGE_PLACEHOLDER,
        f"{IMAGE_PLACEHOLDER}{IMAGE_PLACEHOLDER}",
        f"a {IMAGE_PLACEHOLDER} b\n{IMAGE_PLACEHOLDER}c",
        f"{IMAGE_PLACEHOLDER[:-1]}{IMAGE_PLACEHOLDER}>",
    ],
)
def test_literal_text_ids_match_encoding_each_piece(text: str) -> None:
    """One batched call gives the ids of encoding each piece on its own."""
    _, delegate, image_token_id = _make_tokenizer_with_image_added_token()
    *placeholder_ends, rest = text.split(IMAGE_PLACEHOLDER)
    expected: list[int] = []
    for piece in placeholder_ends:
        expected += delegate.encode(
            piece + IMAGE_PLACEHOLDER[:-1], add_special_tokens=False
        )
        expected += delegate.encode(
            IMAGE_PLACEHOLDER[-1], add_special_tokens=False
        )
    expected += delegate.encode(rest, add_special_tokens=False)

    ids = encode_literal_text(delegate, text, add_special_tokens=False)

    assert ids == expected
    assert image_token_id not in ids
    assert delegate.decode(ids) == text


def test_modulev3_text_placeholders_stay_text() -> None:
    """The text-only ModuleV3 tokenizer never emits the image token."""
    _, delegate, image_token_id = _make_tokenizer_with_image_added_token()
    tokenizer = object.__new__(DeepseekV41ModuleV3Tokenizer)
    tokenizer.delegate = delegate
    tokenizer.max_length = 4096
    tokenizer._token_id_bound = len(delegate)
    text = f"what is {IMAGE_PLACEHOLDER}?"

    ids = asyncio.run(tokenizer.encode(text, add_special_tokens=False))

    assert image_token_id not in ids.tolist()
    assert delegate.decode(ids.tolist()) == text


def test_image_parts_without_request_images_are_an_input_error() -> None:
    tokenizer, _, _ = _make_tokenizer_with_image_added_token()
    message = _image_message(
        {"type": "text", "text": "look"}, {"type": "image"}
    )

    with pytest.raises(InputError, match="1 image part"):
        asyncio.run(tokenizer.new_context(_request([message])))


def test_chat_images_follow_tool_results_into_call_order() -> None:
    """Each image binds to its own slot after tool results are re-sorted."""
    tokenizer, _, image_token_id = _make_tokenizer_with_image_added_token()
    messages = [
        TextGenerationRequestMessage.model_validate(message)
        for message in [
            {"role": "user", "content": f"quote {IMAGE_PLACEHOLDER}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                    for call_id in ("a", "b")
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "b",
                "content": [{"type": "image"}],
            },
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": [{"type": "image"}],
            },
        ]
    ]

    context = asyncio.run(
        tokenizer.new_context(_request(messages, images=[_png(14), _png(28)]))
    )

    # Request image 1 (tool "a") renders first.
    assert [image.pixel_values[0, 0] for image in context.images] == [28, 14]
    spans = context.extra_model_args["image_token_indices"].tolist()
    assert spans == [
        position
        for image in context.images
        for position in range(image.start_idx, image.end_idx)
    ]
    tokens = context.tokens.array.tolist()
    assert tokens.count(image_token_id) == len(spans)
