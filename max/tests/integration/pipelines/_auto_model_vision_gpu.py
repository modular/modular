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
"""Checks AutoModel's text and image rows on an image-capable model.

Each ``test_auto_model_vision_*_gpu.py`` file loads one model, so each test
process compiles one. The oracle is ``max generate``'s row path, the
pipeline's tokenizer building each row, run through
``GenerateIterator.from_contexts``.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Iterator

import huggingface_hub
import numpy as np
import pytest
from max.driver import DeviceSpec
from max.pipelines import AutoModel, AutoTokenizer
from max.pipelines.context import (
    SamplingParams,
    SamplingParamsInput,
    TextContext,
)
from max.pipelines.lib.generate import GenerateIterator
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from PIL import Image

PROMPTS = ["The capital of France is", "Water is made of"]
MAX_NEW_TOKENS = 16


def load(repo_id: str, prefer_module_v3: bool = False) -> Iterator[AutoModel]:
    """Yields ``repo_id`` loaded for greedy two-row batches.

    Skips when the runner's offline Hugging Face cache lacks the model.
    """
    if huggingface_hub.constants.HF_HUB_OFFLINE:
        try:
            huggingface_hub.snapshot_download(repo_id, local_files_only=True)
        except huggingface_hub.errors.LocalEntryNotFoundError:
            pytest.skip(f"{repo_id} is not in this runner's cache.")
    with AutoModel(
        repo_id,
        max_batch_size=2,
        device_specs=[DeviceSpec.accelerator()],
        prefer_module_v3=prefer_module_v3,
        trust_remote_code=True,
        max_length=1024,
    ) as model:
        yield model


def _greedy_sampling_params(model: AutoModel) -> SamplingParams:
    return SamplingParams.from_input_and_generation_config(
        SamplingParamsInput(max_new_tokens=MAX_NEW_TOKENS, top_k=1),
        sampling_params_defaults=model.pipeline_config.model.sampling_params_defaults,
    )


def _from_contexts(
    model: AutoModel, requests: list[TextGenerationRequest]
) -> list[list[int]]:
    """Runs ``requests`` as ``max generate`` builds and runs a row."""
    pipeline = model.pipeline

    async def new_contexts() -> list[TextContext]:
        return [
            await pipeline.tokenizer.new_context(request)
            for request in requests
        ]

    with GenerateIterator.from_contexts(
        pipeline, asyncio.run(new_contexts())
    ) as batch:
        while not batch.done:
            batch.step()
    return batch.sequences


def _red_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (448, 448), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def check_text_rows(model: AutoModel) -> None:
    """Checks token-id rows against the tokenizer-built rows."""
    repo_id = model.pipeline_config.model.model_path
    tokenizer = AutoTokenizer(repo_id, trust_remote_code=True)
    inputs = tokenizer(PROMPTS)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, top_k=1)

    sampling_params = _greedy_sampling_params(model)
    expected = _from_contexts(
        model,
        [
            TextGenerationRequest(
                request_id=RequestID(),
                model_name=repo_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            for prompt in PROMPTS
        ],
    )

    assert outputs.sequences == expected
    assert outputs.prompt_lengths == [len(ids) for ids in inputs["input_ids"]]
    for sequence, prompt_ids in zip(
        outputs.sequences, inputs["input_ids"], strict=True
    ):
        assert sequence[: len(prompt_ids)] == prompt_ids
        assert len(sequence) > len(prompt_ids)
    new_text = tokenizer.decode(
        outputs.sequences[0][len(inputs["input_ids"][0]) :]
    )
    assert "Paris" in new_text


def check_image_rows(model: AutoModel) -> None:
    """Checks a red image row and a text row against the oracle."""
    repo_id = model.pipeline_config.model.model_path
    messages = [
        [
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(
                        text="What color is this image? Answer in one word."
                    ),
                ],
            )
        ],
        [TextGenerationRequestMessage(role="user", content="What is 2 + 2?")],
    ]
    images = [[_red_png()], []]
    outputs = model.generate(
        messages=messages,
        images=images,
        max_new_tokens=MAX_NEW_TOKENS,
        top_k=1,
        output_logits=True,
    )

    sampling_params = _greedy_sampling_params(model)
    expected = _from_contexts(
        model,
        [
            TextGenerationRequest(
                request_id=RequestID(),
                model_name=repo_id,
                messages=conversation,
                images=row_images,
                sampling_params=sampling_params,
            )
            for conversation, row_images in zip(messages, images, strict=True)
        ],
    )
    assert outputs.sequences == expected

    new_tokens = [
        sequence[length:]
        for sequence, length in zip(
            outputs.sequences, outputs.prompt_lengths, strict=True
        )
    ]
    assert all(new_tokens)
    assert outputs.logits is not None
    for tokens, logits in zip(new_tokens, outputs.logits, strict=True):
        assert logits.shape[0] == len(tokens)
        assert np.argmax(logits, axis=-1).tolist() == tokens
    answer = AutoTokenizer(repo_id, trust_remote_code=True).decode(
        new_tokens[0], skip_special_tokens=True
    )
    assert "red" in answer.lower()

    # These models do not compute log probabilities.
    with pytest.raises(ValueError, match="does not compute log probabilities"):
        model.generate(
            messages=messages,
            images=images,
            max_new_tokens=MAX_NEW_TOKENS,
            output_logprobs=1,
        )
