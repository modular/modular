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
"""Utilities for running vLLM models for testing."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from test_common.test_data import MockTextGenerationRequest


def _setup_ninja_path() -> None:
    """Add ninja binary to PATH for FlashInfer JIT compilation.

    FlashInfer relies on ninja to JIT-compile kernels. In Bazel's
    pycross_wheel_library environment, ninja.BIN_DIR can be empty, so we
    locate the binary relative to the installed ninja package and prepend
    it to PATH. This must run before FlashInfer is imported or initialized.
    """
    try:
        import ninja  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        # ninja not available: let flashinfer import fail separately.
        return

    ninja_bin_dir = ninja.BIN_DIR
    if not ninja_bin_dir:
        # In Bazel pycross_wheel_library, bin is at ../../bin relative to
        # the package location.
        ninja_bin_dir = os.path.normpath(
            os.path.join(os.path.dirname(ninja.__file__), "..", "..", "bin")
        )
    if ninja_bin_dir and os.path.isdir(ninja_bin_dir):
        if ninja_bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = (
                ninja_bin_dir + os.pathsep + os.environ.get("PATH", "")
            )


def _resolve_vocab_size(*, llm: Any, tokenizer: Any, model_path: str) -> int:
    """Resolve vocab size for dense logit reconstruction.

    vLLM returns logprobs as a sparse mapping of {token_id: logprob}. For
    verification we rebuild a dense vector of shape (vocab_size,) and place
    each logprob at its token ID index. This requires a vocab size that matches
    the model's output head, not just the tokenizer's current size.

    Why this matters:
    - Token IDs are produced by the tokenizer, so we must index by tokenizer
      IDs.
    - The dense vector must be sized to the model's true vocab size (the logits
      dimension) so it aligns with MAX/Torch outputs during verification.
    - Some models (e.g. DeepSeek R1 NVFP4) report a vocab size in the model
      config that differs from len(tokenizer) or tokenizer.vocab_size. Using
      the tokenizer size yields vectors with the wrong length (e.g. 128000 or
      128815 vs 129280), which breaks element-wise comparison and tolerance
      checks.

    Resolution order:
    1) vLLM model config (hf_config.vocab_size) if available.
    2) Local HF config.json (no network) via huggingface_hub cache.
    3) Tokenizer length or vocab_size as a fallback.
    """
    vocab_size: int | None = None
    try:
        model_config = llm.llm_engine.model_config
        hf_config = getattr(model_config, "hf_config", None)
        vocab_size = getattr(hf_config, "vocab_size", None)
    except Exception:
        vocab_size = None

    if vocab_size is None:
        try:
            import json

            from huggingface_hub import (  # type: ignore[import-not-found, unused-ignore]
                hf_hub_download,
            )

            config_path = hf_hub_download(
                repo_id=model_path,
                filename="config.json",
                local_files_only=True,
            )
            with open(config_path) as f:
                vocab_size = json.load(f).get("vocab_size")
        except Exception:
            vocab_size = None

    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)
        except TypeError:
            vocab_size = tokenizer.vocab_size

    return int(vocab_size)


def run_text_generation(
    model_path: str,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int = 10,
    print_outputs: bool = False,
    encoding_name: str | None = None,
    trust_remote_code: bool = False,
    gpu_memory_utilization: float = 0.9,
    max_batch_size: int | None = None,
    tensor_parallel_size: int = 1,
) -> list[dict[str, Any]]:
    """Run text generation using vLLM.

    NOTE: We import vLLM inside this function to avoid triggering any
    CUDA initialization or multiprocessing side-effects at module-import time.
    """

    # Set `ninja` path since vLLM V1 defaults to `FLASHINFER`, which may
    # require `ninja` to JIT compile kernels at runtime.
    _setup_ninja_path()

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
        elif encoding_name == "float4_e2m1fnx2":
            # NVFP4 models - vLLM loads these natively with auto dtype detection
            dtype = "auto"
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
        # Force eager mode for stability.
        enforce_eager=True,
        # Avoid vLLM custom all-reduce path for stability.
        disable_custom_all_reduce=True,
        # Tensor parallelism for multi-GPU models
        tensor_parallel_size=tensor_parallel_size,
    )

    tokenizer = llm.get_tokenizer()
    vocab_size = _resolve_vocab_size(
        llm=llm, tokenizer=tokenizer, model_path=model_path
    )

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
                # Initialize with a proxy for -inf logprob
                logits_np = np.full((vocab_size,), -100.0, dtype=np.float32)

                # Fill in the values returned by vLLM
                # vLLM returns {token_id: LogprobObject}
                for token_id, logprob_obj in step_logprobs.items():
                    val = getattr(logprob_obj, "logprob", logprob_obj)
                    if token_id < vocab_size:
                        logits_np[token_id] = val

                next_token = logits_np.argmax()

                # Save vLLM logprobs explicitly; verification normalizes as needed.
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
