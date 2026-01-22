# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Utilities for running vLLM models for testing."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from test_common.test_data import MockTextGenerationRequest


def run_text_generation(
    model_path: str,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int = 10,
    print_outputs: bool = False,
    encoding_name: str | None = None,
    trust_remote_code: bool = False,
    gpu_memory_utilization: float = 0.9,
    max_batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Run text generation using vLLM.

    NOTE: We import vLLM inside this function to avoid triggering any
    CUDA initialization or multiprocessing side-effects at module-import time.
    """

    # vLLM V1 defaults to `FLASHINFER`, which requires `ninja` to JIT compile
    # kernels at runtime. We don't currently support `ninja`, so we force
    # `FLASH_ATTN` (Flash Attention 2) which uses pre-compiled kernels.
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    try:
        from vllm import (  # type: ignore[import-not-found, unused-ignore]
            LLM,
            SamplingParams,
        )
    except ImportError:
        raise SystemExit(
            "Attempted to import vLLM, which is only supported for nvidia GPUs."
        ) from None

    # Map encoding_name to vLLM dtype/quantization
    dtype = "auto"
    quantization = None

    if encoding_name:
        if encoding_name in ["float32", "float16", "bfloat16"]:
            dtype = encoding_name
        elif encoding_name == "float8_e4m3fn":
            # vLLM often runs FP8 models automatically if hardware supports it,
            # but usually setting dtype to float16/bfloat16 is safer for the container
            dtype = "float16"
        elif encoding_name in ["awq", "gptq", "squeezellm", "fp8"]:
            quantization = encoding_name
        else:
            raise ValueError(f"Unrecognized encoding: {encoding_name}")

    # Handle batch size limit if provided
    # vLLM uses max_num_seqs to control how many sequences are processed at once
    max_num_seqs = max_batch_size if max_batch_size is not None else 256

    # Initialize vLLM
    # We set gpu_memory_utilization explicitly to avoid OOM if the runner
    # has some overhead, though vLLM usually dominates.
    llm: Any = LLM(
        model=model_path,
        dtype=dtype,  # type: ignore[arg-type, unused-ignore]
        quantization=quantization,  # type: ignore[arg-type, unused-ignore]
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        # Default max_logprobs is 20. We increase this to support full logits
        # retrieval. 262144 covers large vocabs (e.g. Qwen2.5 is ~152k).
        max_logprobs=262144,
        # Force eager mode if needed for debugging, but V1 prefers cuda graphs
        enforce_eager=False,
    )

    tokenizer = llm.get_tokenizer()
    vocab_size = tokenizer.vocab_size

    prompts = []
    sampling_params_list = []

    for request in textgen_requests:
        prompts.append(request.prompt)

        # We use logprobs=vocab_size to get the full distribution. This is the
        # closest approximation to logits we can get via the vLLM API.
        sp: Any = SamplingParams(
            max_tokens=num_steps,
            temperature=0,
            logprobs=vocab_size,
        )
        sampling_params_list.append(sp)

    outputs = llm.generate(prompts, sampling_params_list)

    results = []

    # Process outputs to match the format of torch_utils.py
    for request, output in zip(textgen_requests, outputs, strict=False):
        saved_logits = []

        # `output.outputs[0].logprobs` is a list of dicts (one per step). vLLM
        # may return `None` for the first step (prompt) depending on version,
        # but it usually returns generation steps.
        generated_data = output.outputs[0]

        if generated_data.logprobs:
            for step_logprobs in generated_data.logprobs:
                # Initialize with a proxy for -inf
                logits_np = np.full((vocab_size,), -100.0, dtype=np.float32)

                # Fill in the values returned by vLLM
                # vLLM returns {token_id: LogprobObject}
                for token_id, logprob_obj in step_logprobs.items():
                    val = getattr(logprob_obj, "logprob", logprob_obj)
                    if token_id < vocab_size:
                        logits_np[token_id] = val

                next_token = logits_np.argmax()

                saved_logits.append(
                    {
                        "next_token": next_token,
                        "next_token_logprobs": float(logits_np[next_token]),
                        "logprobs": logits_np,
                    }
                )

        if print_outputs:
            print(
                "Prompt:",
                f"{request.prompt[:100]}...{request.prompt[-100:]}"
                if len(request.prompt) > 200
                else request.prompt,
            )
            print("Output:", request.prompt + output.outputs[0].text)

        results.append({"prompt": request.prompt, "values": saved_logits})

    return results
