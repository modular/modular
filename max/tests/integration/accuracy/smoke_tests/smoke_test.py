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
import yaml
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
from pydantic import BaseModel, ConfigDict, Field
from requests.structures import CaseInsensitiveDict

URL = "http://127.0.0.1:8000/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Maps alias model names to reusable MAX recipe configs. Aliases let the same
# weights be tested under different configurations while keeping results
# separate in dashboards. Paths use the portable ``max/pipelines/architectures/``
# prefix (same convention as the MAX CLI config loader). The smoke driver
# resolves them from the checkout for YAML parsing; ``max serve`` still loads
# the same paths from the installed package in ``.venv-serve``.
#
# Values are fully spelled-out so they can be copy-pasted into a CLI invocation:
#   max serve --config-file max/pipelines/architectures/deepseekV3/recipes/nvfp4_8x_b200.yaml
# fmt: off
MODEL_RECIPES = CaseInsensitiveDict({
    "deepseek-ai/DeepSeek-R1-0528": "max/pipelines/architectures/deepseekV3/recipes/r1_0528_8x_b200.yaml",
    "deepseek-ai/DeepSeek-V3.1-Terminus": "max/pipelines/architectures/deepseekV3/recipes/terminus_8x_b200.yaml",
    "google/gemma-4-26B-A4B-it__no_dgc": "max/pipelines/architectures/gemma4/recipes/gemma4_26b_a4b_no_dgc.yaml",
    "nvidia/Gemma-4-26B-A4B-NVFP4__no_dgc": "max/pipelines/architectures/gemma4/recipes/gemma4_26b_a4b_nvfp4_no_dgc.yaml",
    "google/gemma-3-27b-it__modulev3": "max/pipelines/architectures/gemma3_modulev3/recipes/gemma3_27b.yaml",
    "MiniMaxAI/MiniMax-M2.7": "max/pipelines/architectures/minimax_m2/recipes/minimax_m2_8x_b200.yaml",
    "amd/MiniMax-M2.7-MXFP4": "max/pipelines/architectures/minimax_m2/recipes/minimax_m2_mxfp4_8x_mi355.yaml",
    "lukealonso/MiniMax-M2.7-NVFP4": "max/pipelines/architectures/minimax_m2/recipes/minimax_m2_nvfp4_8x_b200.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__eagle": "max/pipelines/architectures/llama3/recipes/llama31_8b_eagle.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__eagle_local_kvconnector": "max/pipelines/architectures/llama3/recipes/llama31_8b_eagle_local_kvconnector.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__local_kvconnector": "max/pipelines/architectures/llama3/recipes/llama31_8b_local_kvconnector.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__modulev3": "max/pipelines/architectures/llama3_modulev3/recipes/llama31_8b.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__tiered_kvconnector": "max/pipelines/architectures/llama3/recipes/llama31_8b_tiered_kvconnector.yaml",
    "microsoft/Phi-3.5-mini-instruct__modulev3": "max/pipelines/architectures/phi3_modulev3/recipes/phi35_mini.yaml",
    "microsoft/phi-4__modulev3": "max/pipelines/architectures/phi3_modulev3/recipes/phi4.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__fp8kv": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_fp8kv_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__mtp": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_mtp_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__mtp_tpep": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_mtp_tpep_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__tpep": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_tpep_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__tpep_ar": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_tpep_ar_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__tptp": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_tptp_8x_b200.yaml",
    "amd/Kimi-K2.5-MXFP4": "max/pipelines/architectures/kimik2_5/recipes/mxfp4_8x_mi355.yaml",
    "nvidia/Kimi-K2.5-NVFP4": "max/pipelines/architectures/kimik2_5/recipes/nvfp4_with_vision_8x_b200.yaml",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "max/pipelines/architectures/qwen3/recipes/qwen3_235b_a22b_8x_b200.yaml",
    "unsloth/gpt-oss-20b-BF16__modulev3": "max/pipelines/architectures/gpt_oss_modulev3/recipes/gpt_oss_20b.yaml",
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__eagle": "max/pipelines/architectures/kimik2_5/recipes/nvfp4_eagle_8x_b200.yaml",
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__eagle_tiered_kvconnector_tpep": "max/pipelines/architectures/kimik2_5/recipes/nvfp4_eagle_tiered_kvconnector_tpep_8x_b200.yaml",
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__local_kvconnector_tpep": "max/pipelines/architectures/kimik2_5/recipes/nvfp4_local_kvconnector_tpep_8x_b200.yaml",
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__tiered_kvconnector_tpep": "max/pipelines/architectures/kimik2_5/recipes/nvfp4_tiered_kvconnector_tpep_8x_b200.yaml",
})
# fmt: on

# Aliases whose recipe may not be present in every checkout. Register
# only when the YAML exists on disk so unit tests that iterate
# ``MODEL_RECIPES`` don't try to open a file that isn't there.
_OPTIONAL_MODEL_RECIPES = {
    "nvidia/Kimi-K2.5-NVFP4__internal": "max/pipelines/architectures/kimik2_5/recipes/internal/nvfp4_8x_b200.yaml",
}
_max_dir = Path(__file__).resolve().parents[4]
for _alias, _path in _OPTIONAL_MODEL_RECIPES.items():
    if (_max_dir / "python" / _path).is_file():
        MODEL_RECIPES[_alias] = _path


class RecipeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    class KVCache(BaseModel):
        model_config = ConfigDict(extra="ignore")

        device_memory_utilization: float | None = None
        kv_connector: str | None = None

    class Model(BaseModel):
        model_config = ConfigDict(extra="ignore")

        model_path: str | None = None
        device_specs: list[int] | None = None
        data_parallel_degree: int = 1
        kv_cache: RecipeConfig.KVCache = Field(
            default_factory=lambda: RecipeConfig.KVCache()
        )

    class Runtime(BaseModel):
        model_config = ConfigDict(extra="ignore")

        ep_size: int | None = None
        enable_chunked_prefill: bool | None = None

    class Speculative(BaseModel):
        model_config = ConfigDict(extra="ignore")

        num_speculative_tokens: int | None = None

    model: Model = Field(default_factory=Model)
    draft_model: Model | None = None
    runtime: Runtime = Field(default_factory=Runtime)
    speculative: Speculative | None = None


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
            "__internal",
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
            "qwen3.5",
            "vision",
        )
    )


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


def _resolve_recipe_path(recipe_path: str) -> str:
    """Resolve a recipe path to an absolute file path.
    Recipe paths use the ``max/pipelines/architectures/`` prefix and are
    resolved by the shared config resolver against the installed package.
    """
    if not recipe_path.startswith("max/pipelines/architectures/"):
        return recipe_path
    max_dir = Path(__file__).resolve().parents[4]
    resolved = max_dir / "python" / recipe_path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Built-in recipe not found: {recipe_path} (resolved to {resolved})"
        )
    return str(resolved)


@cache
def _load_recipe(recipe_path: str) -> RecipeConfig:
    with open(_resolve_recipe_path(recipe_path), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RecipeConfig.model_validate(data)


def _recipe_gpu_overrides(recipe: RecipeConfig, gpu_count: int) -> list[str]:
    """Builds smoke-test GPU overrides for a fixed-GPU recipe.

    Recipes are reusable CLI presets with explicit device IDs and parallelism.
    Smoke tests run on several GPU counts, so fields that equal the recipe's
    device count are treated as scalable and overridden to match the local
    machine. Fixed one-GPU recipes and intentional non-scaled parallelism values
    are preserved.

    Note that this may be removed in the future when we implement
    flexible "auto" gpu args for max parallelism CLI and config.
    """
    if gpu_count <= 0:
        return []

    model_gpu_count = (
        len(recipe.model.device_specs)
        if recipe.model.device_specs is not None
        else None
    )
    draft_gpu_count = (
        len(recipe.draft_model.device_specs)
        if recipe.draft_model is not None
        and recipe.draft_model.device_specs is not None
        else None
    )
    devices_arg = f"gpu:{','.join(str(i) for i in range(gpu_count))}"

    args = []
    if model_gpu_count != 1:
        args += ["--devices", devices_arg]
    if model_gpu_count is not None and model_gpu_count != 1:
        if recipe.model.data_parallel_degree == model_gpu_count:
            args += ["--data-parallel-degree", str(gpu_count)]
        if recipe.runtime.ep_size == model_gpu_count:
            args += ["--ep-size", str(gpu_count)]

    if recipe.draft_model is not None:
        if draft_gpu_count != 1:
            args += ["--draft-devices", devices_arg]
        if (
            draft_gpu_count is not None
            and draft_gpu_count != 1
            and recipe.draft_model.data_parallel_degree == draft_gpu_count
        ):
            args += ["--draft-data-parallel-degree", str(gpu_count)]

    return args


def _revision_args(
    framework: str,
    model: str,
    recipe: RecipeConfig | None = None,
) -> list[str]:
    revision = _load_hf_repo_lock().get(model.casefold())
    args: list[str] = []
    if revision:
        if framework in ("max", "max-ci"):
            args += [
                "--model-override",
                f"main.huggingface_model_revision={revision}",
                "--model-override",
                f"main.huggingface_weight_revision={revision}",
            ]
        else:  # vllm, sglang
            args += ["--revision", revision]
        logger.info(f"Pinned to revision {revision[:12]}")
    else:
        logger.warning(f"No locked revision for {model}")

    if (
        recipe is not None
        and framework in ("max", "max-ci")
        and recipe.draft_model is not None
        and recipe.draft_model.model_path is not None
        and (
            draft_revision := _load_hf_repo_lock().get(
                recipe.draft_model.model_path.casefold()
            )
        )
    ):
        args += [
            "--model-override",
            f"draft.huggingface_model_revision={draft_revision}",
            "--model-override",
            f"draft.huggingface_weight_revision={draft_revision}",
        ]
        logger.info(f"Pinned draft model to revision {draft_revision[:12]}")

    return args


def get_server_cmd(
    framework: str,
    model: str,
    *,
    serve_extra_args: str = "",
    recipe_path: str | None = None,
) -> list[str]:
    gpu_model, gpu_count = get_gpu_name_and_count()
    if recipe_path is None:
        recipe_path = MODEL_RECIPES.get(model)
    recipe = _load_recipe(recipe_path) if recipe_path else None
    recipe_config: tuple[str, RecipeConfig] | None = None
    if (
        recipe is not None
        and recipe_path is not None
        and framework in ["max-ci", "max"]
    ):
        recipe_config = (recipe_path, recipe)

    sglang_backend = "triton" if "b200" in gpu_model.lower() else "fa3"
    SGLANG = [
        "sglang.launch_server",
        "--attention-backend",
        sglang_backend,
    ]
    # limit-mm-per-prompt.video is for InternVL3 on B200
    VLLM = [
        "vllm.entrypoints.openai.api_server",
        "--max-model-len",
        "auto",
        "--limit-mm-per-prompt.video",
        "0",
    ]
    MAX = ["max.entrypoints.pipelines", "serve", "--pretty-print-config"]

    if gpu_count > 1:
        if recipe is not None:
            if (
                recipe.runtime.ep_size is not None
                and recipe.runtime.ep_size > 1
            ):
                VLLM += ["--enable-expert-parallel"]
                SGLANG += ["--expert-parallel-size", str(gpu_count)]

            if recipe.runtime.enable_chunked_prefill is not False:
                VLLM += ["--enable-chunked-prefill"]
            else:
                SGLANG += ["--chunked-prefill-size", "-1"]

            if recipe.model.kv_cache.device_memory_utilization is not None:
                mem_cap = recipe.model.kv_cache.device_memory_utilization
                VLLM += [
                    "--gpu-memory-utilization",
                    f"{mem_cap:g}",
                ]
                SGLANG += ["--mem-fraction-static", f"{mem_cap:g}"]

            if recipe.model.data_parallel_degree > 1:
                VLLM += [f"--data-parallel-size={gpu_count}"]
                SGLANG += [
                    f"--data-parallel-size={gpu_count}",
                    "--enable-dp-attention",
                ]
            else:
                VLLM += [f"--tensor-parallel-size={gpu_count}"]
                SGLANG += [f"--tp-size={gpu_count}"]

            # Remove once vLLM >= 0.17 (which includes vllm-project/vllm#34673).
            if "minimax-m2" in model.casefold():
                os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] = "0"
                VLLM += ["--attention-backend", "FLASH_ATTN"]

        else:  # gpu_count > 1 and recipe is None
            MAX += [
                "--devices",
                f"gpu:{','.join(str(i) for i in range(gpu_count))}",
            ]
            SGLANG += [f"--tp-size={gpu_count}"]
            VLLM += [f"--tensor-parallel-size={gpu_count}"]

    # Force MAX to rely solely on the KVConnector for prefix cache hits to test
    # cpu/disk KV offload code paths.
    if framework in ("max", "max-ci") and (
        "--kv-connector" in serve_extra_args
        or (
            recipe is not None
            and recipe.model.kv_cache.kv_connector is not None
        )
    ):
        os.environ["MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE"] = "1"

    if _inside_bazel():
        assert framework == "max-ci", "bazel invocation only supports max-ci"
        cmd = [sys.executable, "-m", *MAX]
    else:
        assert framework != "max-ci", "max-ci must be run through bazel"
        interpreter = [".venv-serve/bin/python", "-m"]
        commands = {
            "sglang": [*interpreter, *SGLANG],
            "vllm": [*interpreter, *VLLM],
            "max": [*interpreter, *MAX],
        }
        cmd = commands[framework]

    cmd = cmd + ["--port", "8000"]
    if recipe_config is not None:
        config_file_path, recipe = recipe_config
        cmd += [
            "--config-file",
            config_file_path,
            *_recipe_gpu_overrides(recipe, gpu_count),
        ]
    else:
        cmd += ["--trust-remote-code", "--model", model]

    # GPT-OSS uses repetition_penalty in lm_eval to prevent reasoning loops,
    # so we need to enable penalties on the server
    if "gpt-oss" in model.casefold() and framework in ["max-ci", "max"]:
        cmd += ["--enable-penalties"]

    recipe = recipe_config[1] if recipe_config is not None else None
    cmd += _revision_args(framework, model, recipe)

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
    recipe_path = MODEL_RECIPES.get(model)
    if recipe_path:
        recipe_model_path = _load_recipe(recipe_path).model.model_path
        if recipe_model_path is None:
            raise ValueError("Recipe model section must contain model_path.")
        hf_model_path = recipe_model_path
    else:
        hf_model_path = model
    hf_model_path = resolve_canonical_repo_id(hf_model_path)
    cmd = get_server_cmd(
        framework,
        hf_model_path,
        serve_extra_args=serve_extra_args,
        recipe_path=recipe_path,
    )

    tasks = [TEXT_TASK]
    if is_vision_model(model):
        tasks = [VISION_TASK] + tasks

    logger.info(f"Starting server with command:\n {' '.join(cmd)}")
    results = []
    all_samples = []
    if disable_timeouts:
        timeout = sys.maxsize
    else:
        # TODO(GEX-3508): Reduce timeout once model build time is optimized
        timeout = 2700

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
