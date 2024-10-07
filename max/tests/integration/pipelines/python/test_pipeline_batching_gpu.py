# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from dataclasses import dataclass
from typing import Literal

import pytest
from evaluate_llama import (
    PROMPTS,
    NumpyDecoder,
    compare_values,
    find_runtime_path,
    golden_data_fname,
    next_token_with_logits,
)
from huggingface_hub import hf_hub_download
from llama3.config import InferenceConfig, SupportedEncodings, SupportedVersions
from llama3.llama3 import Llama3
from max.driver import CPU, CUDA


@dataclass(frozen=True)
class PipelineModelParams:
    name: Literal["tinyllama", "llama3_1"]
    encoding: SupportedEncodings
    max_length: int
    max_new_tokens: int = -1
    max_batch_size: int = 1
    version: SupportedVersions = SupportedVersions.llama3_1

    """Whether to include a print hook. This is generally for debugging
    purposes, but it's also helping to avoid a segfault in the heterogeneous
    test."""

    def __str__(self):
        return f"{self.name}-{self.encoding}-{self.max_length}-{self.max_batch_size}"


@pytest.fixture(scope="session")
def pipeline_model(testdata_directory, request):
    model_params: PipelineModelParams = request.param
    print(f"\nPipelineModel: {model_params}")
    model_encoding = model_params.encoding
    if model_params.name == "tinyllama":
        weight_path = testdata_directory / "tiny_llama.gguf"
    else:
        weights_repo_id = f"modularai/llama-{model_params.version}"
        weights_encoding_file = model_encoding.hf_model_name(
            model_params.version
        )
        weight_path = hf_hub_download(
            repo_id=weights_repo_id,
            filename=weights_encoding_file,
        )
        print(f"- Downloaded: {weight_path}")

    config = InferenceConfig(
        weight_path=weight_path,
        version=model_params.version,
        quantization_encoding=model_encoding,
        max_length=model_params.max_length,
        max_new_tokens=model_params.max_new_tokens,
        max_cache_batch_size=model_params.max_batch_size,
        device=CUDA() if model_encoding == "bfloat16" else CPU(),
    )
    print(
        f"- Using config: {config.version}, MaxLength={config.max_length},"
        f" MaxNewTokens={config.max_new_tokens},"
        f" BatchSize={config.max_cache_batch_size}"
    )
    model = Llama3(config)

    return model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_model",
    [
        PipelineModelParams(
            "llama3_1",
            SupportedEncodings.bfloat16,
            512,
            10,
            4,
        ),
    ],
    ids=PipelineModelParams.__str__,
    indirect=True,
)
async def test_pipeline_heterogeneous_batch_logits(
    pipeline_model, testdata_directory
):
    """Execute a batch with prompts with different lengths and validates the
    logits.
    """
    golden_data_path = find_runtime_path(
        golden_data_fname("llama3_1", "bfloat16"), testdata_directory
    )
    expected_results = NumpyDecoder().decode(golden_data_path.read_text())

    prompt_a = PROMPTS[0]
    prompt_b = PROMPTS[1]
    prompt_c = PROMPTS[2]

    stored_logits = {"A": [], "B": [], "C": []}

    # TODO(MSDK-1093): `reset_cache` must be manually called since we're not
    # using `next_token`.
    await pipeline_model.reset_cache()
    # Send in A for context encoding.
    context_a = await pipeline_model.new_context(prompt_a)
    next_token_with_logits(pipeline_model, {"A": context_a}, stored_logits)

    # Send in B for context encoding
    context_b = await pipeline_model.new_context(prompt_b)
    next_token_with_logits(pipeline_model, {"B": context_b}, stored_logits)

    # Send in both A and B for token generation
    next_token_with_logits(
        pipeline_model, {"A": context_a, "B": context_b}, stored_logits
    )

    # Send in C for context encoding
    context_c = await pipeline_model.new_context(prompt_c)
    next_token_with_logits(pipeline_model, {"C": context_c}, stored_logits)

    # Send in both B and C for token generation
    next_token_with_logits(
        pipeline_model, {"B": context_b, "C": context_c}, stored_logits
    )

    with pytest.raises(AssertionError):
        compare_values(
            [
                {"prompt": prompt_a, "values": stored_logits["A"]},
                {"prompt": prompt_b, "values": stored_logits["B"]},
                {"prompt": prompt_c, "values": stored_logits["C"]},
            ],
            expected_results,
            rtol=1e-4,
        )

    await pipeline_model.reset_cache()
