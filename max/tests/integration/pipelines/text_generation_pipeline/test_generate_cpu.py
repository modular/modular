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
"""``generate`` against a real small model on CPU."""

from __future__ import annotations

import asyncio
import shlex
from unittest.mock import patch

import pytest
from max.driver import DeviceSpec
from max.pipelines.architectures import register_all_models
from max.pipelines.context import (
    SamplingParams,
    SamplingParamsInput,
    TextContext,
)
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
    TextGenerationPipeline,
    generate_local_model_path,
)
from max.pipelines.lib.generate import GenerateIterator, generate
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)

REPO_ID = "modularai/SmolLM-135M-Instruct-FP32"
MAX_NEW_TOKENS = 12
PROMPTS = [
    "The capital of France is",
    "Here is a short list of prime numbers: 2, 3, 5,",
]
# Hugging Face greedy decoding on CPU gives these tokens. To regenerate them:
#   tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
#   model = AutoModelForCausalLM.from_pretrained(REPO_ID)
#   for prompt in PROMPTS:
#       ids = tokenizer(prompt, return_tensors="pt").input_ids
#       out = model.generate(ids, do_sample=False, max_new_tokens=12)
#       print(out[0, ids.shape[1] :].tolist())
GOLDEN_SEQUENCES = [
    [7042, 28, 260, 3995, 2240, 281, 4649, 284, 260, 2642, 284, 2536],
    [216, 39, 28, 216, 33, 33, 28, 216, 33, 35, 28, 216],
]


@pytest.fixture(scope="module")
def pipeline() -> TextGenerationPipeline[TextContext]:
    register_all_models()
    try:
        model_path = generate_local_model_path(REPO_ID)
    except FileNotFoundError:
        model_path = REPO_ID
    config = PipelineConfig.from_args(
        PipelineArgs(
            model_path=model_path,
            device_specs=[DeviceSpec.cpu()],
            max_length=512,
            quantization_encoding="float32",
            runtime=PipelineRuntimeConfig(max_batch_size=4),
        )
    )
    _, retrieved = PIPELINE_REGISTRY.retrieve(config)
    assert isinstance(retrieved, TextGenerationPipeline)
    return retrieved


@pytest.fixture(scope="module")
def input_ids(pipeline: TextGenerationPipeline[TextContext]) -> list[list[int]]:
    async def encode(prompt: str) -> list[int]:
        encoded = await pipeline.tokenizer.encode(
            prompt, add_special_tokens=True
        )
        return [int(token) for token in encoded]

    return [asyncio.run(encode(prompt)) for prompt in PROMPTS]


def test_greedy_tokens_match_golden(
    pipeline: TextGenerationPipeline[TextContext], input_ids: list[list[int]]
) -> None:
    sampling = SamplingParamsInput(
        max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, top_k=1, seed=0
    )
    requests = [
        TextGenerationRequest(
            request_id=RequestID(),
            model_name=REPO_ID,
            prompt=prompt,
            sampling_params=SamplingParams.from_input_and_generation_config(
                sampling,
                sampling_params_defaults=pipeline.pipeline_config.model.sampling_params_defaults,
            ),
        )
        for prompt in input_ids
    ]
    contexts = [
        asyncio.run(pipeline.tokenizer.new_context(request))
        for request in requests
    ]
    with GenerateIterator.from_contexts(pipeline, contexts) as batch:
        while not batch.done:
            batch.step()

    output = generate(
        pipeline,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_k=1,
    )

    for sequences in (batch.sequences, output.sequences):
        new_tokens = [
            sequence[len(prompt) :]
            for sequence, prompt in zip(sequences, input_ids, strict=True)
        ]
        assert new_tokens == GOLDEN_SEQUENCES


def test_messages_row_matches_the_tokenizer_built_row(
    pipeline: TextGenerationPipeline[TextContext],
) -> None:
    messages = [
        TextGenerationRequestMessage(
            role="user", content="What is the capital of France?"
        )
    ]
    sampling = SamplingParamsInput(max_new_tokens=MAX_NEW_TOKENS, top_k=1)
    context = asyncio.run(
        pipeline.tokenizer.new_context(
            TextGenerationRequest(
                request_id=RequestID(),
                model_name=REPO_ID,
                messages=messages,
                sampling_params=SamplingParams.from_input_and_generation_config(
                    sampling,
                    sampling_params_defaults=pipeline.pipeline_config.model.sampling_params_defaults,
                ),
            )
        )
    )
    with GenerateIterator.from_contexts(pipeline, [context]) as batch:
        while not batch.done:
            batch.step()

    output = generate(
        pipeline, messages=[messages], max_new_tokens=MAX_NEW_TOKENS, top_k=1
    )

    assert output.sequences == batch.sequences
    assert output.prompt_lengths == batch.prompt_lengths
    assert len(output.sequences[0]) > output.prompt_lengths[0]


def test_serve_command_rebuilds_the_cpu_pipeline(
    pipeline: TextGenerationPipeline[TextContext], input_ids: list[list[int]]
) -> None:
    max_batch_size = pipeline.max_batch_size
    with pytest.raises(ValueError) as error:
        GenerateIterator(
            pipeline,
            [input_ids[0]] * (max_batch_size + 1),
            max_new_tokens=MAX_NEW_TOKENS,
        )
    command = shlex.join(
        [
            "max",
            "serve",
            "--model-path",
            pipeline.pipeline_config.model.model_path,
            "--devices",
            "cpu",
            "--quantization-encoding",
            "float32",
            "--max-length",
            "512",
            "--max-batch-size",
            str(max_batch_size),
        ]
    )
    assert str(error.value).splitlines()[2] == f"  {command}"


def test_interrupted_batch_leaves_pipeline_reusable(
    pipeline: TextGenerationPipeline[TextContext], input_ids: list[list[int]]
) -> None:
    def run() -> list[list[int]]:
        return generate(
            pipeline,
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            top_k=1,
        ).sequences

    expected = run()
    kv_manager = pipeline.kv_manager
    with patch.object(kv_manager, "claim", wraps=kv_manager.claim) as claim:
        with pytest.raises(KeyboardInterrupt):
            with GenerateIterator(
                pipeline, input_ids, max_new_tokens=MAX_NEW_TOKENS, top_k=1
            ) as batch:
                batch.step()
                raise KeyboardInterrupt

    contexts = [call.args[0] for call in claim.call_args_list]
    assert len(contexts) == len(input_ids)
    assert not any(kv_manager.contains(context) for context in contexts)
    assert run() == expected


def _sample(
    pipeline: TextGenerationPipeline[TextContext],
    input_ids: list[list[int]],
    seed: int,
) -> list[list[int]]:
    return generate(
        pipeline,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_k=50,
        seed=seed,
    ).sequences


def test_seed_reproduces_sampled_tokens(
    pipeline: TextGenerationPipeline[TextContext], input_ids: list[list[int]]
) -> None:
    first = _sample(pipeline, input_ids, seed=1)

    assert _sample(pipeline, input_ids, seed=1) == first
    assert _sample(pipeline, input_ids, seed=2) != first


def test_duplicate_prompts_draw_different_samples(
    pipeline: TextGenerationPipeline[TextContext], input_ids: list[list[int]]
) -> None:
    duplicated = [input_ids[0], input_ids[0]]
    first = _sample(pipeline, duplicated, seed=1)

    assert first[0] != first[1]
    assert _sample(pipeline, duplicated, seed=1) == first
