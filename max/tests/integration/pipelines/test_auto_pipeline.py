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
"""Tests for AutoPipeline against a real small CPU model."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import pytest
from max.pipelines import (
    PIPELINE_REGISTRY,
    AutoPipeline,
    AutoProcessor,
    PipelineConfig,
)
from max.pipelines.lib import (
    TextGenerationPipeline,
    auto_pipeline,
    generate_local_model_path,
)
from max.pipelines.lib.generate import GenerateResult
from max.pipelines.modeling.types import (
    ImageContentPart,
    TextContentPart,
    TextGenerationRequestMessage,
)
from PIL import Image

REPO_ID = "modularai/SmolLM-135M-Instruct-FP32"
NUM_NEW_TOKENS = 8
TEXT = "The capital of France is"
# Decoding with skip_special_tokens drops the marker, so this prompt does not
# come back from its decoded ids.
SPECIAL_PROMPT = "<|im_start|>2 + 2 ="
PROMPTS = [TEXT, SPECIAL_PROMPT]
CONVERSATION: list[dict[str, object]] = [{"role": "user", "content": TEXT}]
OTHER_CONVERSATION: list[dict[str, object]] = [
    {"role": "system", "content": "Answer with a number."},
    {"role": "user", "content": "2 + 2 ="},
]


@pytest.fixture(scope="module")
def model_path() -> str:
    # A cached snapshot skips the Hub probe, which CI runs without network.
    try:
        return generate_local_model_path(REPO_ID)
    except FileNotFoundError:
        return REPO_ID


def _load_cpu(model_path: str, **overrides: object) -> AutoPipeline:
    return AutoPipeline(
        model_path,
        max_batch_size=len(PROMPTS),
        device_graph_capture=False,
        prefer_module_v3=False,
        devices="cpu",
        quantization_encoding="float32",
        max_length=512,
        **overrides,
    )


@pytest.fixture(scope="module")
def pipe(model_path: str) -> Iterator[AutoPipeline]:
    with _load_cpu(model_path) as pipe:
        yield pipe


def test_greedy_matches_generate_then_batch_decode(pipe: AutoPipeline) -> None:
    texts = pipe(PROMPTS, max_new_tokens=NUM_NEW_TOKENS, top_k=1)

    inputs = pipe.tokenizer(PROMPTS)
    sequences = pipe.model.generate(
        **inputs, max_new_tokens=NUM_NEW_TOKENS, top_k=1
    ).sequences
    decoded = pipe.tokenizer.batch_decode(sequences, skip_special_tokens=True)
    decoded_prompts = pipe.tokenizer.batch_decode(
        inputs["input_ids"], skip_special_tokens=True
    )
    assert decoded_prompts[-1] != SPECIAL_PROMPT
    for text, prompt, full, decoded_prompt in zip(
        texts, PROMPTS, decoded, decoded_prompts, strict=True
    ):
        assert full.startswith(decoded_prompt)
        assert text == prompt + full[len(decoded_prompt) :]
        assert len(text) > len(prompt)


def test_outputs_start_with_their_prompts(pipe: AutoPipeline) -> None:
    texts = pipe(PROMPTS, max_new_tokens=NUM_NEW_TOKENS, top_k=1)
    assert isinstance(texts, list)
    for text, prompt in zip(texts, PROMPTS, strict=True):
        assert text.startswith(prompt)
        assert len(text) > len(prompt)


def test_a_tuple_of_prompts_returns_a_list(pipe: AutoPipeline) -> None:
    texts = pipe(tuple(PROMPTS), max_new_tokens=NUM_NEW_TOKENS, top_k=1)
    assert texts == pipe(PROMPTS, max_new_tokens=NUM_NEW_TOKENS, top_k=1)


def test_a_string_returns_a_string(pipe: AutoPipeline) -> None:
    text = pipe(TEXT, max_new_tokens=NUM_NEW_TOKENS, top_k=1)
    assert isinstance(text, str)
    assert text == pipe([TEXT], max_new_tokens=NUM_NEW_TOKENS, top_k=1)[0]


@pytest.mark.parametrize(
    ("text", "error", "match"),
    [
        ([], ValueError, "text must"),
        ((), ValueError, "text must"),
        ([TEXT, 1], TypeError, "text must"),
        ([TEXT, CONVERSATION], TypeError, "text must"),
        ([[1, 2, 3]], TypeError, "Row 0 has a message of type int"),
        (b"bytes", TypeError, "text must"),
        (5, TypeError, "text must"),
        ({"role": "user", "content": TEXT}, TypeError, "text must be a str"),
        ((prompt for prompt in PROMPTS), TypeError, "text must"),
    ],
)
def test_rejects_text_that_is_not_prompts_or_conversations(
    pipe: AutoPipeline, text: object, error: type[Exception], match: str
) -> None:
    # Typed loosely so the call can pass what the overloads reject.
    call: Callable[..., object] = pipe
    with pytest.raises(error, match=match):
        call(text, max_new_tokens=1)


def test_conversations_return_only_the_new_text(
    pipe: AutoPipeline, model_path: str
) -> None:
    conversations = [CONVERSATION, OTHER_CONVERSATION]
    texts = pipe(conversations, max_new_tokens=NUM_NEW_TOKENS, top_k=1)

    processor = AutoProcessor(model_path)
    outputs = pipe.model.generate(
        **processor(conversations), max_new_tokens=NUM_NEW_TOKENS, top_k=1
    )
    assert texts == processor.batch_decode(
        [
            sequence[length:]
            for sequence, length in zip(
                outputs.sequences, outputs.prompt_lengths, strict=True
            )
        ],
        skip_special_tokens=True,
    )
    assert all(texts)


def test_a_conversation_returns_a_string(pipe: AutoPipeline) -> None:
    text = pipe(CONVERSATION, max_new_tokens=NUM_NEW_TOKENS, top_k=1)
    assert isinstance(text, str)
    assert (
        text == pipe([CONVERSATION], max_new_tokens=NUM_NEW_TOKENS, top_k=1)[0]
    )


def test_a_conversation_passes_messages_images_and_sampling(
    pipe: AutoPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    png = buffer.getvalue()
    [prompt_ids] = pipe.tokenizer([TEXT])["input_ids"]
    [new_ids] = pipe.tokenizer([" Paris"])["input_ids"]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(*args: object, **kwargs: object) -> GenerateResult:
        calls.append((args, kwargs))
        return GenerateResult(
            sequences=[[*prompt_ids, *new_ids]],
            prompt_lengths=[len(prompt_ids)],
        )

    monkeypatch.setattr(auto_pipeline, "generate", record)
    text = pipe(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": png},
                    {"type": "text", "text": TEXT},
                ],
            }
        ],
        chat_template_kwargs={"enable_thinking": False},
        max_new_tokens=3,
        top_k=4,
        seed=7,
    )
    assert text == pipe.tokenizer.decode(new_ids, skip_special_tokens=True)
    assert calls == [
        (
            (pipe.model.pipeline,),
            {
                "messages": [
                    [
                        TextGenerationRequestMessage(
                            role="user",
                            content=[
                                ImageContentPart(),
                                TextContentPart(text=TEXT),
                            ],
                        )
                    ]
                ],
                "images": [[png]],
                "chat_template_kwargs": {"enable_thinking": False},
                "max_new_tokens": 3,
                "min_new_tokens": None,
                "temperature": None,
                "top_k": 4,
                "top_p": None,
                "min_p": None,
                "frequency_penalty": None,
                "presence_penalty": None,
                "repetition_penalty": None,
                "ignore_eos": False,
                "stop_token_ids": None,
                "seed": 7,
                "caller": "AutoPipeline",
            },
        )
    ]


def test_a_conversation_drops_special_tokens(
    pipe: AutoPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    [prompt_ids] = pipe.tokenizer([TEXT])["input_ids"]
    [new_ids] = pipe.tokenizer([" Paris<|im_end|>"])["input_ids"]

    def record(*args: object, **kwargs: object) -> GenerateResult:
        return GenerateResult(
            sequences=[[*prompt_ids, *new_ids]],
            prompt_lengths=[len(prompt_ids)],
        )

    monkeypatch.setattr(auto_pipeline, "generate", record)
    text = pipe(CONVERSATION, max_new_tokens=3)
    assert "<|im_end|>" in pipe.tokenizer.decode(new_ids)
    assert "<|im_end|>" not in text
    assert text == pipe.tokenizer.decode(new_ids, skip_special_tokens=True)


def test_chat_template_kwargs_with_string_prompts_raise(
    pipe: AutoPipeline,
) -> None:
    with pytest.raises(
        ValueError, match="Pass chat_template_kwargs with conversations"
    ):
        pipe(
            TEXT,
            chat_template_kwargs={"enable_thinking": False},
            max_new_tokens=1,
        )


def test_images_on_a_text_only_model_raise(pipe: AutoPipeline) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    with pytest.raises(ValueError, match=r"Row 0 has 1 image, .* text only"):
        pipe(
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": buffer.getvalue()}],
                }
            ],
            max_new_tokens=1,
        )


def test_video_parts_on_a_text_only_model_raise(pipe: AutoPipeline) -> None:
    video: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "video"}]}
    ]
    with pytest.raises(
        ValueError, match=r"^Row 1 has a video part, but .* takes text only\.$"
    ):
        pipe([CONVERSATION, video], max_new_tokens=1)


def test_passes_every_sampling_argument(
    pipe: AutoPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    [prompt_ids] = pipe.tokenizer([TEXT])["input_ids"]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(*args: object, **kwargs: object) -> GenerateResult:
        calls.append((args, kwargs))
        return GenerateResult(
            sequences=[[*prompt_ids, prompt_ids[-1]]],
            prompt_lengths=[len(prompt_ids)],
        )

    monkeypatch.setattr(auto_pipeline, "generate", record)
    pipe(
        TEXT,
        max_new_tokens=3,
        min_new_tokens=2,
        temperature=0.5,
        top_k=4,
        top_p=0.9,
        min_p=0.1,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        repetition_penalty=1.1,
        ignore_eos=True,
        stop_token_ids=[9],
        seed=7,
    )
    assert calls == [
        (
            (pipe.model.pipeline, [prompt_ids]),
            {
                "max_new_tokens": 3,
                "min_new_tokens": 2,
                "temperature": 0.5,
                "top_k": 4,
                "top_p": 0.9,
                "min_p": 0.1,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.1,
                "ignore_eos": True,
                "stop_token_ids": [9],
                "seed": 7,
                "caller": "AutoPipeline",
            },
        )
    ]


def test_image_arguments_on_a_text_model_raise(
    pipe: AutoPipeline, model_path: str
) -> None:
    untyped: Callable[..., object] = pipe
    with pytest.raises(ValueError) as error:
        untyped(TEXT, max_new_tokens=1, height=256)
    assert str(error.value) == (
        "height is for image generation models, and AutoModel loaded "
        f"{model_path}, a text generation model."
    )
    with pytest.raises(TypeError, match="'max_new_tokens'"):
        untyped(TEXT)


def test_batch_errors_name_auto_pipeline(pipe: AutoPipeline) -> None:
    with pytest.raises(ValueError) as error:
        pipe([*PROMPTS, TEXT], max_new_tokens=1)
    assert str(error.value).splitlines()[0] == (
        "AutoPipeline does not support scheduling across batches. 3 prompts "
        "exceed the pipeline's max batch size of 2, and AutoPipeline never "
        "splits its prompts."
    )


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stubs the registry's build so the config resolves without a model."""
    pipeline = MagicMock(spec=TextGenerationPipeline)

    def retrieve(pipeline_config: PipelineConfig) -> tuple[object, MagicMock]:
        return pipeline.tokenizer, pipeline

    monkeypatch.setattr(PIPELINE_REGISTRY, "retrieve", retrieve)
    return pipeline


def test_tokenizer_loads_from_the_resolved_config(
    model_path: str,
    stub_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def record(path: str, **kwargs: object) -> MagicMock:
        calls.append((path, kwargs))
        return MagicMock()

    monkeypatch.setattr(auto_pipeline, "AutoTokenizer", record)
    with _load_cpu(model_path, trust_remote_code=True) as pipe:
        model_config = pipe.model.pipeline_config.model
    assert calls == [
        (
            model_config.model_path,
            {
                "revision": model_config.huggingface_model_revision,
                "trust_remote_code": True,
            },
        )
    ]


def test_closed_pipeline_raises(
    model_path: str,
    stub_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_pipeline, "AutoTokenizer", MagicMock())
    pipe = _load_cpu(model_path)
    pipe.close()
    with pytest.raises(RuntimeError, match="AutoModel is closed"):
        pipe(TEXT, max_new_tokens=1)


def test_closed_pipeline_checks_the_input_first(
    model_path: str,
    stub_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_pipeline, "AutoTokenizer", MagicMock())
    pipe = _load_cpu(model_path)
    pipe.close()
    call: Callable[..., object] = pipe
    with pytest.raises(TypeError, match="text must be"):
        call([TEXT, CONVERSATION], max_new_tokens=1)


def test_tokenizer_failure_closes_the_model(
    model_path: str,
    stub_pipeline: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(path: str, **kwargs: object) -> MagicMock:
        raise OSError("no tokenizer")

    monkeypatch.setattr(auto_pipeline, "AutoTokenizer", fail)
    with pytest.raises(OSError, match="no tokenizer"):
        _load_cpu(model_path)
    stub_pipeline.kv_manager.shutdown.assert_called_once_with()


def test_prefer_module_v3_passes_through(
    model_path: str, stub_pipeline: MagicMock
) -> None:
    with AutoPipeline(model_path, devices="cpu", prefer_module_v3=True) as pipe:
        assert pipe.model.pipeline_config.runtime.prefer_module_v3 is True
