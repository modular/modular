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
"""Test Suite for Unit Testing the TextGenerationPipeline with structured output."""

import asyncio
import json
from typing import cast

import hf_repo_lock
from max.driver import DeviceSpec
from max.interfaces import (
    RequestID,
    SamplingParams,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationResponseFormat,
)
from max.pipelines import (
    PipelineConfig,
    TextGenerationPipeline,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import MAXModelConfig, SamplingConfig
from max.pipelines.lib.registry import PipelineRegistry

pytest_plugins = "test_common.registry"


def test_smollm_with_structured_output_gpu(
    pipeline_registry: PipelineRegistry,
) -> None:
    revision = hf_repo_lock.revision_for_hf_repo(
        "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    assert revision is not None
    pipeline_config = PipelineConfig(
        model=MAXModelConfig(
            model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
            quantization_encoding="bfloat16",
            device_specs=[DeviceSpec.accelerator()],
            huggingface_model_revision=revision,
            max_length=8192,
        ),
        sampling=SamplingConfig(enable_structured_output=True),
        max_batch_size=1,
    )

    tokenizer, pipeline_factory = pipeline_registry.retrieve_factory(
        pipeline_config
    )

    prompt = """
    Please provide a json response, with the person's name and age extracted from the excerpt.
    For example, provided an excerpt 'Bob Dylan is 83 years old.' return with {"badnamey": "Bob Dylan", "badagey": 83}.
    Please extract the person's name and age from the following excerpt:
    'John Mayer is 47 years old.'
    """

    request_id = RequestID("request_0")
    sampling_params = SamplingParams(max_new_tokens=50, top_k=1)
    request = TextGenerationRequest(
        model_name=pipeline_config.model.model_path,
        request_id=request_id,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        sampling_params=sampling_params,
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "age": {
                        "type": "integer",
                    },
                },
                "required": ["name", "age"],
            },
        ),
    )

    # Get Context
    context: TextContext = asyncio.run(tokenizer.new_context(request))

    pipeline = pipeline_factory()
    assert isinstance(pipeline, TextGenerationPipeline)
    # SpeechTokenGenerationPipeline subclasses TextGenerationPipeline, so at
    # this point MyPy thinks pipeline could be TextGenerationPipeline[Any] or
    # SpeechTokenGenerationPipeline.  Unfortunately 'assert not isinstance' is
    # not recognized by MyPy and it's not clear how else to get it out of that
    # union without some force.  So we cast it off.  This is bad, ideally we
    # wouldn't have to do this, but we boxed ourselves in here.
    pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
    kv_manager = pipeline.kv_manager
    kv_manager.claim(context.request_id, replica_idx=0)

    tokens = []
    while True:
        inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
            batches=[[context]], num_steps=1
        )
        kv_manager.alloc(context, replica_idx=0, num_steps=1)
        response = pipeline.execute(inputs)

        for token in response[request_id].tokens:
            tokens.append(token)

        if response[request_id].is_done:
            break

    response_content = asyncio.run(
        tokenizer.decode(tokens, skip_special_tokens=True)
    )

    result = json.loads(response_content)

    assert result["name"] == "John Mayer"
    assert result["age"] == 47
