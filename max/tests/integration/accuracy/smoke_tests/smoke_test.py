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
from __future__ import annotations

"""
This script is used for the CI "Max Serve Smoke Test" workflow.
It does two things:
    1. Starts the MAX/SGLang/VLLM inference server for the given model
    2. Runs a tiny evaluation task using against the chat/completions API

Currently there is a hard dependency that two virtualenvs are already created:
    - .venv-serve (not needed for max-ci, which uses bazel)
    - .venv-eval

Where the serve environment should already have either MAX/VLLM/SGLang installed.
The eval environment should already have lm-eval installed.
These dependencies are to be removed once this script
has been integrated into bazel.

Note that if you're running this script inside bazel, only available for max-ci,
then the virtualenvs are not needed.
"""

import csv
import logging
import os
import shlex
import sys
from functools import cache
from pathlib import Path
from pprint import pformat

import click
from eval_runner import (
    TEXT_TASK,
    VISION_TASK,
    build_eval_summary,
    call_eval,
    get_gpu_name_and_count,
    print_samples,
    resolve_canonical_repo_id,
    safe_model_name,
    test_single_request,
    validate_hf_token,
    write_github_output,
    write_results,
)
from inference_server_harness import start_server
from requests.structures import CaseInsensitiveDict

URL = "http://127.0.0.1:8000/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maps alias model names to their real HuggingFace model path and extra
# MAX serve args. Aliases let the same weights be tested under different
# configurations while keeping results separate in dashboards.
# max_serve_args are only applied to MAX frameworks, not vllm/sglang.
# fmt: off
MODEL_ALIASES = CaseInsensitiveDict({
    "google/gemma-4-26B-A4B-it__no_dgc": {
        "max_serve_args": "--max-num-steps 1 --no-device-graph-capture",
    },
    "meta-llama/Llama-3.1-8B-Instruct__modulev3": {
        "max_serve_args": "--prefer-module-v3",
    },
    "meta-llama/Llama-3.2-1B-Instruct__modulev3": {
        "max_serve_args": "--prefer-module-v3",
    },
    "unsloth/gpt-oss-20b-BF16__modulev3": {
        "max_serve_args": "--prefer-module-v3",
    },
    "microsoft/Phi-3.5-mini-instruct__modulev3": {
        "max_serve_args": "--prefer-module-v3",
    },
    "microsoft/phi-4__modulev3": {
        "max_serve_args": "--prefer-module-v3",
    },
    "google/gemma-3-27b-it__modulev3": {
        # TODO(MXF-332): Investigate extra memory usage in multi-GPU ModuleV3.
        "max_serve_args": "--prefer-module-v3 --device-memory-utilization 0.7",
    },
    "nvidia/DeepSeek-V3.1-NVFP4__fp8kv": {
        "max_serve_args": "--kv-cache-format float8_e4m3fn",
    },
    "nvidia/DeepSeek-V3.1-NVFP4__tpep": {
        "max_serve_args": "--data-parallel-degree 1",
    },
    "nvidia/DeepSeek-V3.1-NVFP4__tpep_ar": {
        "max_serve_args": "--data-parallel-degree 1 --ep-use-allreduce",
    },
    "nvidia/DeepSeek-V3.1-NVFP4__tptp": {
        "max_serve_args": "--ep-size 1 --data-parallel-degree 1",
    },
    "meta-llama/Llama-3.1-8B-Instruct__eagle": {
        "max_serve_args": (
            "--draft-model-path atomicapple0/EAGLE-LLaMA3.1-Instruct-8B "
            "--devices gpu:0 "
            "--speculative-method eagle"
        ),
    },
    "nvidia/DeepSeek-V3.1-NVFP4__mtp": {
        "max_serve_args": (
            "--speculative-method eagle "
            "--kv-cache-format float8_e4m3fn "
            "--num-speculative-tokens 3"
        ),
    },
    "nvidia/DeepSeek-V3.1-NVFP4__mtp_tpep": {
        "max_serve_args": (
            "--data-parallel-degree 1 "
            "--speculative-method eagle "
            "--kv-cache-format float8_e4m3fn "
            "--num-speculative-tokens 3"
        ),
    },
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__eagle": {
        "max_serve_args": (
            "--draft-model-path nvidia/Kimi-K2.5-Thinking-Eagle3 "
            "--speculative-method eagle "
            "--num-speculative-tokens 3 "
            "--kv-cache-format float8_e4m3fn "
            "--device-memory-utilization 0.75 "
            "--max-batch-input-tokens 4096"
        ),
    },
    "meta-llama/Llama-3.1-8B-Instruct__local_kvconnector": {
        "max_serve_args": "--kv-connector local",
    },
    "meta-llama/Llama-3.1-8B-Instruct__eagle_local_kvconnector": {
        "max_serve_args": (
            "--draft-model-path atomicapple0/EAGLE-LLaMA3.1-Instruct-8B "
            "--devices gpu:0 "
            "--speculative-method eagle "
            "--kv-connector local"
        )
    },
    "meta-llama/Llama-3.1-8B-Instruct__tiered_kvconnector": {
        "max_serve_args": "--kv-connector tiered",
    },
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__local_kvconnector_tpep": {
        "max_serve_args": (
            "--data-parallel-degree 1 "
            "--kv-cache-format float8_e4m3fn "
            "--device-memory-utilization 0.75 "
            "--max-batch-input-tokens 4096 "
            "--kv-connector local"
        ),
    },
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__tiered_kvconnector_tpep": {
        "max_serve_args": (
            "--data-parallel-degree 1 "
            "--kv-cache-format float8_e4m3fn "
            "--device-memory-utilization 0.75 "
            "--max-batch-input-tokens 4096 "
            "--kv-connector tiered"
        ),
    },
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__eagle_tiered_kvconnector_tpep": {
        "max_serve_args": (
            "--data-parallel-degree 1 "
            "--draft-model-path nvidia/Kimi-K2.5-Thinking-Eagle3 "
            "--speculative-method eagle "
            "--num-speculative-tokens 3 "
            "--kv-cache-format float8_e4m3fn "
            "--device-memory-utilization 0.75 "
            "--max-batch-input-tokens 4096 "
            "--kv-connector tiered"
        ),
    },
})
# fmt: on


# TODO Refactor this to a model list/matrix specifying type of model
def is_vision_model(model: str) -> bool:
    """Check if the model supports vision tasks."""
    model = model.casefold()
    if any(
        kw in model
        for kw in (
            "no_vision",
            "__eagle",
            "__mtp",
            "_kvconnector",
            "gemma-3-1b",
        )
    ):
        return False
    return any(
        kw in model
        for kw in (
            "gemma-3",
            "gemma-4",
            "idefics",
            "internvl",
            "kimi-k2",
            "kimi-vl",
            "olmocr",
            "pixtral",
            "qwen2.5-vl",
            "qwen3-vl",
            "vision",
        )
    )


def is_huge_moe(model: str) -> bool:
    """Large MoE models that need expert parallelism instead of tensor parallelism."""
    model = model.casefold()
    if "deepseek" in model and "lite" not in model:
        return True
    return any(x in model for x in ["minimax-m", "kimi-k", "qwen3-235b"])


def _inside_bazel() -> bool:
    return os.getenv("BUILD_WORKSPACE_DIRECTORY") is not None


@cache
def _load_hf_repo_lock() -> dict[str, str]:
    """Read hf-repo-lock.tsv, return {lowercase_repo: revision} mapping."""
    tsv = Path(__file__).resolve().parent.parent.parent / "hf-repo-lock.tsv"
    if not tsv.exists():
        logger.warning("hf-repo-lock.tsv not found, skipping revision pinning")
        return {}
    db = {}
    with open(tsv) as f:
        for row in csv.DictReader(f, dialect="excel-tab"):
            db[row["hf_repo"].lower()] = row["revision"]
    return db


def get_server_cmd(
    framework: str,
    model: str,
    *,
    serve_extra_args: str = "",
) -> list[str]:
    gpu_model, gpu_count = get_gpu_name_and_count()
    sglang_backend = "triton" if "b200" in gpu_model.lower() else "fa3"
    SGLANG = f"sglang.launch_server --attention-backend {sglang_backend} --mem-fraction-static 0.8"
    # limit-mm-per-prompt.video is for InternVL3 on B200
    VLLM = "vllm.entrypoints.openai.api_server --max-model-len auto --limit-mm-per-prompt.video 0"
    MAX = "max.entrypoints.pipelines serve"

    is_huge_model = is_huge_moe(model)
    if is_huge_model and framework != "sglang":
        MAX += f" --device-memory-utilization 0.8 --devices gpu:{','.join(str(i) for i in range(gpu_count))} --ep-size {gpu_count} --max-batch-input-tokens 1024"
        VLLM += " --enable-chunked-prefill --gpu-memory-utilization 0.8 --enable-expert-parallel"
        # resolve attention parallelism strategy
        if "--data-parallel-degree 1" not in serve_extra_args:
            # default to DP Attn + EP MoE strategy
            MAX += f" --data-parallel-degree {gpu_count}"
            VLLM += f" --data-parallel-size={gpu_count}"
        else:
            # TP Attn + EP MoE strategy
            VLLM += f" --tensor-parallel-size={gpu_count}"

        # Remove once vLLM >= 0.17 (which includes vllm-project/vllm#34673).
        if "minimax-m2" in model.casefold():
            os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] = "0"
            VLLM += " --attention-backend FLASH_ATTN"
        # Have not been successful in getting SGLang to work with R1 yet
    elif gpu_count > 1:
        MAX += f" --devices gpu:{','.join(str(i) for i in range(gpu_count))}"
        VLLM += f" --tensor-parallel-size={gpu_count}"
        SGLANG += f" --tp-size={gpu_count}"

    # Force MAX to rely solely on the KVConnector for prefix cache hits to test
    # cpu/disk KV offload code paths.
    if framework in ("max", "max-ci") and "--kv-connector" in serve_extra_args:
        os.environ["MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE"] = "1"

    if _inside_bazel():
        assert framework == "max-ci", "bazel invocation only supports max-ci"
        cmd = [sys.executable, "-m", *MAX.split()]
    else:
        assert framework != "max-ci", "max-ci must be run through bazel"
        interpreter = [".venv-serve/bin/python", "-m"]
        commands = {
            "sglang": [*interpreter, *SGLANG.split()],
            "vllm": [*interpreter, *VLLM.split()],
            "max": [*interpreter, *MAX.split()],
        }
        cmd = commands[framework]

    cmd = cmd + ["--port", "8000", "--trust-remote-code", "--model", model]

    # GPT-OSS uses repetition_penalty in lm_eval to prevent reasoning loops,
    # so we need to enable penalties on the server
    if "gpt-oss" in model.casefold() and framework in ["max-ci", "max"]:
        cmd += ["--enable-penalties"]

    revision = _load_hf_repo_lock().get(model.casefold())
    if revision:
        if framework in ("max", "max-ci"):
            cmd += [
                "--huggingface-model-revision",
                revision,
                "--huggingface-weight-revision",
                revision,
            ]
        else:  # vllm, sglang
            cmd += ["--revision", revision]
        logger.info(f"Pinned to revision {revision[:12]}")
    else:
        logger.warning(f"No locked revision for {model}")

    if serve_extra_args:
        if framework in ["max-ci", "max"]:
            cmd += shlex.split(serve_extra_args)
        else:
            logger.warning(
                "Ignoring --serve-extra-args for framework %s", framework
            )
    return cmd


@click.command()
@click.argument(
    "hf_model_path",
    type=str,
    required=True,
)
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "max", "max-ci"]),
    default="max-ci",
    required=False,
    help="Framework to use for the smoke test. Only max-ci is supported when running in bazel.",
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help="If provided, a summary json file and the eval result are written here",
)
@click.option(
    "--print-responses",
    is_flag=True,
    default=False,
    help="Print question/response pairs from eval samples after the run finishes",
)
@click.option(
    "--print-cot",
    is_flag=True,
    default=False,
    help="Print the model's chain-of-thought reasoning for each sample. Must be used with --print-responses",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=64,
    help="Maximum concurrent requests to send to the server",
)
@click.option(
    "--num-questions",
    type=int,
    default=320,
    help="Number of questions to ask the model",
)
@click.option(
    "--serve-extra-args",
    type=str,
    default="",
    help=(
        "Extra args appended to MAX serve command, for example: "
        '"--device-graph-capture --max-batch-size=16"'
    ),
)
@click.option(
    "--disable-timeouts",
    is_flag=True,
    default=False,
    help="Disable all timeouts. Useful when debugging hangs.",
)
def smoke_test(
    hf_model_path: str,
    framework: str,
    output_path: Path | None,
    print_responses: bool,
    print_cot: bool,
    max_concurrent: int,
    num_questions: int,
    serve_extra_args: str,
    disable_timeouts: bool,
) -> None:
    """
    Example usage: ./bazelw run smoke-test -- meta-llama/Llama-3.2-1B-Instruct

    This command asks 320 questions against the model behind the given hf_model_path.
    It runs the provided framework (defaulting to MAX serve) in the background,
    and fires off HTTP requests to chat/completions API.
    Note: Only models with a chat template (typically -instruct, -it, -chat, etc.) are supported.

    Accuracy is reported at the end, with higher values being better.
    A 1.0 value means 100% accuracy.

    """
    validate_hf_token()

    if print_cot and not print_responses:
        raise ValueError("--print-cot must be used with --print-responses")

    build_workspace = os.getenv("BUILD_WORKSPACE_DIRECTORY")
    if output_path and build_workspace and not output_path.is_absolute():
        output_path = Path(build_workspace) / output_path

    model = hf_model_path.strip()
    alias = MODEL_ALIASES.get(model)
    hf_model_path = model.rsplit("__", 1)[0] if alias else model
    hf_model_path = resolve_canonical_repo_id(hf_model_path)
    if alias and framework in ["max-ci", "max"]:
        serve_extra_args = (
            f"{serve_extra_args} {alias['max_serve_args']}".strip()
        )
    cmd = get_server_cmd(
        framework,
        hf_model_path,
        serve_extra_args=serve_extra_args,
    )

    tasks = [TEXT_TASK]
    if is_vision_model(model):
        tasks = [VISION_TASK] + tasks

    logger.info(f"Starting server with command:\n {' '.join(cmd)}")
    results = []
    all_samples = []
    if disable_timeouts:
        timeout = sys.maxsize
    elif is_huge_moe(hf_model_path) or "step-3.5" in hf_model_path.casefold():
        # TODO(GEX-3508): Reduce timeout once model build time is optimized
        timeout = 2700
    else:
        timeout = 900

    with start_server(cmd, timeout) as server:
        logger.info(f"Server started in {server.startup_time:.2f} seconds")
        write_github_output("startup_time", f"{server.startup_time:.2f}")

        for task in tasks:
            test_single_request(
                URL, hf_model_path, task, disable_timeouts=disable_timeouts
            )
            result, samples = call_eval(
                URL,
                hf_model_path,
                task,
                max_concurrent=max_concurrent,
                num_questions=num_questions,
                disable_timeouts=disable_timeouts,
            )
            if print_responses:
                print_samples(samples, print_cot)

            results.append(result)
            all_samples.append(samples)

    if results:
        summary = build_eval_summary(
            results, startup_time_seconds=server.startup_time
        )

        if output_path is not None:
            path = output_path / safe_model_name(model)
            path.mkdir(parents=True, exist_ok=True)
            write_results(path, summary, results, all_samples, tasks)

        logger.info(pformat(summary, indent=2))


if __name__ == "__main__":
    smoke_test()
