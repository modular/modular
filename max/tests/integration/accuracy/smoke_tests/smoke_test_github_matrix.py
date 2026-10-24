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

# /// script
# dependencies = ["click>=8,<9"]
# ///

import json
import re
from collections.abc import Mapping

import click

RUNNERS = {
    "B200": "modrunner-b200-efa",
    "MI355": "modrunner-mi355",
    "2xB200": "modrunner-b200-efa-2x",
    "2xMI355": "modrunner-mi355-2x",
    "4xB200": "modrunner-b200-4x",
    "4xMI355": "modrunner-mi355-4x",
    "8xB200": "modrunner-b200-efa-8x",
    "8xMI355": "modrunner-mi355-8x",
}

_1xB200 = {"B200"}
_1xMI355 = {"MI355"}
_2xB200 = {"2xB200"}
_2xMI355 = {"2xMI355"}
_4xB200 = {"4xB200"}
_4xMI355 = {"4xMI355"}
_8xB200 = {"8xB200"}
_8xMI355 = {"8xMI355"}
DISABLE: set[str] = set()
NIGHTLY = {"NIGHTLY"}
VARIANT = {"VARIANT"}
PRIVATE = {"PRIVATE"}

# Model → HW it runs on. DISABLE runs nowhere. NIGHTLY adds the model to the
# scheduled nightly run (`--tier nightly`) on the HW it lists; without it the
# model runs only on manual dispatch. VARIANT marks a `<hf-repo>__<variant>` key
# that runs a recipe from MODEL_RECIPES in smoke_test.py; PRIVATE marks one
# whose recipe ships with a private arch and resolves from private hw_recipes.
#
# To add a model, trigger the smoke test with it first:
# https://github.com/modularml/modular/actions/workflows/serveSmokeTest.yaml
# then list it below with the HW it passed on. VLMs also go in is_vision_model
# in smoke_test.py.
# Every entry is exactly one line, sorted by key, with tags in the order
# NIGHTLY, VARIANT, PRIVATE, HW. The only comment allowed is an existing
# Linear ticket at the end of the line, as `# TODO(PROJ-1234)`.
# fmt: off
MODELS: Mapping[str, set[str]] = {
    "allenai/Olmo-3-7B-Instruct": _1xB200 | _1xMI355,
    "allenai/olmOCR-2-7B-1025-FP8": _1xB200 | _1xMI355,
    "amd/Kimi-K2.7-Code-MXFP4": NIGHTLY | _4xMI355,
    "amd/MiniMax-M3-MXFP4": NIGHTLY | _4xMI355,
    "ByteDance-Seed/academic-ds-9B": DISABLE,  # TODO(SERVOPT-1120)
    "deepseek-ai/DeepSeek-V2-Lite-Chat": DISABLE,  # TODO(SERVOPT-1120)
    "deepseek-ai/DeepSeek-V2-Lite-Chat__modulev3": VARIANT | _1xB200 | _1xMI355,
    "deepseek-ai/DeepSeek-V3.1-Terminus": _8xB200,
    "deepseek-ai/DeepSeek-V4-Flash-0731__tp2": NIGHTLY | VARIANT | _2xB200,
    "google/diffusiongemma-26B-A4B-it": DISABLE,
    "google/gemma-3-1b-it": _1xB200,  # TODO(KERN-3014)
    "google/gemma-3-27b-it": _1xB200 | _1xMI355,
    "google/gemma-3-27b-it__modulev3": VARIANT | _1xB200 | _1xMI355 | _2xB200 | _2xMI355,
    "google/gemma-4-12B-it__device_graph_synthesis": VARIANT | _1xB200 | _1xMI355,
    "google/gemma-4-12B-it__dspark": NIGHTLY | VARIANT | _1xB200 | _1xMI355,
    "google/gemma-4-26B-A4B-it": NIGHTLY | _1xB200 | _1xMI355,
    "google/gemma-4-26B-A4B-it__tuned": NIGHTLY | VARIANT | _1xB200,
    "google/gemma-4-31B-it": NIGHTLY | _1xB200 | _1xMI355,
    "google/gemma-4-31B-it__cascade": NIGHTLY | VARIANT | _1xB200,
    "google/gemma-4-31B-it__modulev3": VARIANT | _1xB200 | _1xMI355,
    "google/gemma-4-31B-it__tuned": NIGHTLY | VARIANT | _1xB200,
    "meta-llama/Llama-3.1-8B-Instruct": _1xB200 | _1xMI355,
    "meta-llama/Llama-3.1-8B-Instruct__dflash": VARIANT | _1xB200 | _1xMI355,  # TODO(SERVOPT-1168)
    "meta-llama/Llama-3.1-8B-Instruct__eagle": VARIANT | _1xB200 | _1xMI355,  # TODO(SERVOPT-1168)
    "meta-llama/Llama-3.1-8B-Instruct__modulev3": VARIANT | _1xB200 | _1xMI355,
    "meta-llama/Llama-3.1-8B-Instruct__rust_tiered_kvconnector": VARIANT | _1xB200,
    "meta-models/Muse-Glimmer-30B": NIGHTLY | _1xB200,
    "microsoft/Phi-3.5-mini-instruct": _1xB200 | _1xMI355,
    "microsoft/Phi-3.5-mini-instruct__modulev3": VARIANT | _1xB200 | _1xMI355,
    "microsoft/phi-4": _1xB200 | _1xMI355,
    "microsoft/phi-4__modulev3": VARIANT | _1xB200 | _1xMI355,
    "MiniMaxAI/MiniMax-M3-MXFP8": NIGHTLY | _8xMI355,  # TODO(MODELS-1611)
    "MiniMaxAI/MiniMax-M3-MXFP8__mtp": NIGHTLY | VARIANT | PRIVATE | _8xB200,
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": _1xB200 | _1xMI355,
    "modularai/Llama-3.1-405B-Instruct-autofp8": _4xMI355 | _8xB200,
    "modularai/MiniMax-M3-MXFP6": NIGHTLY | _4xMI355 | _8xMI355,
    "nvidia/DeepSeek-V3.1-NVFP4": _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__fp8kv": VARIANT | _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__mtp": VARIANT | _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__mtp_tpep": VARIANT | _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__tpep": VARIANT | _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__tpep_ar": VARIANT | _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__tptp": VARIANT | _8xB200,
    "nvidia/DeepSeek-V4.1-Flash-NVFP4": _4xB200,
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4": DISABLE,
    "nvidia/Gemma-4-26B-A4B-NVFP4": NIGHTLY | _1xB200,
    "nvidia/Gemma-4-26B-A4B-NVFP4__tuned": NIGHTLY | VARIANT | _1xB200,
    "nvidia/Gemma-4-31B-IT-NVFP4": NIGHTLY | _1xB200 | _2xB200,
    "nvidia/Gemma-4-31B-IT-NVFP4__tuned": NIGHTLY | VARIANT | _1xB200,
    "nvidia/GLM-5.2-NVFP4__mtp_tpep": NIGHTLY | VARIANT | _8xB200,
    "nvidia/Kimi-K2.7-Code-NVFP4": NIGHTLY | _8xB200,
    "nvidia/Kimi-K2.7-Code-NVFP4__modulev3": VARIANT | _8xB200,
    "nvidia/Llama-3.1-405B-Instruct-NVFP4": _8xB200,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16__modulev3": NIGHTLY | VARIANT | _1xMI355 | _2xMI355,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__modulev3": NIGHTLY | VARIANT | _1xB200 | _2xB200,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__modulev3_dspark": NIGHTLY | VARIANT | _1xB200 | _2xB200,
    "openai/gpt-oss-20b": _1xB200 | _1xMI355 | _2xB200,
    "OpenGVLab/InternVL3_5-8B-Instruct": _1xB200 | _1xMI355,
    "poolside/Laguna-S-2.1-NVFP4": NIGHTLY | _1xB200,
    "poolside/Laguna-S-2.1-NVFP4__modulev3": NIGHTLY | VARIANT | _1xB200,
    "Qwen/Qwen2.5-7B-Instruct": _1xB200 | _1xMI355,
    "Qwen/Qwen2.5-VL-7B-Instruct": _1xB200 | _1xMI355,
    "Qwen/Qwen3-8B": _1xB200 | _1xMI355,
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8": _1xB200 | _2xB200,
    "Qwen/Qwen3-VL-4B-Instruct-FP8": _1xB200 | _2xB200,
    "Qwen/Qwen3.5-9B": _1xB200,
    "Qwen/Qwen3.6-27B": _1xB200,
    "Qwen/Qwen3.8-27B": NIGHTLY | _1xB200 | _1xMI355 | _2xB200 | _2xMI355,
    "RadixArk/GLM-5.3-NVFP4__mtp_tpep": NIGHTLY | VARIANT | _8xB200,
    "RadixArk/Qwen3.8-27B-NVFP4": NIGHTLY | _1xB200,
    "RadixArk/Qwen3.8-27B-NVFP4__dflash2": NIGHTLY | VARIANT | _1xB200,
    "RadixArk/Qwen3.8-27B-NVFP4__mtp": NIGHTLY | VARIANT | _1xB200,
    "RedHatAI/gemma-3-27b-it-FP8-dynamic": _1xB200 | _1xMI355,  # TODO(MODELS-1021)
    "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic": _4xMI355 | _8xB200,
    "tencent/Hy4-preview-FP8": NIGHTLY | _8xB200,
    "tencent/Hy4-preview-FP8__mtp": NIGHTLY | VARIANT | _8xB200,
    "thinkingmachines/Inkling-Small-NVFP4": NIGHTLY | _2xB200,
    "thinkingmachines/Inkling-Small-NVFP4__modulev3": VARIANT | _2xB200,
    "thinkingmachines/Inkling-Small-NVFP4__mtp": NIGHTLY | VARIANT | _2xB200,
    "zai-org/GLM-5.3-Flash__tpep": NIGHTLY | VARIANT | _8xB200,
    "zai-org/GLM-5.3-Flash__tpep_mi355": NIGHTLY | VARIANT | _2xMI355,
    "zai-org/GLM-5.3-Flash__tpep_mi355_modulev3": NIGHTLY | VARIANT | _2xMI355,
    "zai-org/GLM-5.3__fp8_tpep_mi355": NIGHTLY | VARIANT | _8xMI355,
}
# fmt: on

NIGHTLY_MODELS = frozenset(
    model for model, tags in MODELS.items() if "NIGHTLY" in tags
)

TIERS = ("nightly", "all")


def tier_models(tier: str) -> list[str]:
    """Lists the models a tier schedules, before framework/GPU filtering.

    Args:
        tier: Either ``"nightly"`` for the deployed families or ``"all"`` for
            every entry in ``MODELS``.

    Returns:
        The model keys the tier covers.
    """
    if tier == "all":
        return list(MODELS)
    return [model for model in MODELS if model in NIGHTLY_MODELS]


def excluded(framework: str, gpu: str, model: str) -> bool:
    """Check if a model is excluded from a given framework and/or GPU.

    A model only runs on the HW listed for it in ``MODELS``.
    """
    tags = MODELS.get(model, set())
    # Variants run MAX recipes; vLLM, SGLang and Mach have no equivalent.
    if "VARIANT" in tags and framework in {"vllm", "sglang", "mach-ci"}:
        return True
    return gpu not in tags


def parse_override(raw: str | None) -> list[str]:
    """Parse a comma-separated list of models from the command line."""
    if not raw:
        return []
    parts = re.split(r"[, \n\r]+", raw)
    return [p.strip() for p in parts if p.strip()]


@click.command()
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "max-ci", "max", "mach-ci"]),
    required=True,
)
@click.option(
    "--models-override",
    default=None,
    help="Comma list of models; ignores the per-model HW list.",
)
@click.option(
    "--tier",
    type=click.Choice(TIERS),
    default="all",
    show_default=True,
    help="Model set: 'nightly' for the deployed families, 'all' for every model.",
)
@click.option("--run-on-b200", is_flag=True)
@click.option("--run-on-mi355", is_flag=True)
@click.option("--run-on-2xb200", is_flag=True)
@click.option("--run-on-2xmi355", is_flag=True)
@click.option("--run-on-4xb200", is_flag=True)
@click.option("--run-on-4xmi355", is_flag=True)
@click.option("--run-on-8xb200", is_flag=True)
@click.option("--run-on-8xmi355", is_flag=True)
def main(
    framework: str,
    models_override: str | None,
    tier: str,
    run_on_b200: bool,
    run_on_mi355: bool,
    run_on_2xb200: bool,
    run_on_2xmi355: bool,
    run_on_4xb200: bool,
    run_on_4xmi355: bool,
    run_on_8xb200: bool,
    run_on_8xmi355: bool,
) -> None:
    flags = {
        "B200": run_on_b200,
        "MI355": run_on_mi355,
        "2xB200": run_on_2xb200,
        "2xMI355": run_on_2xmi355,
        "4xB200": run_on_4xb200,
        "4xMI355": run_on_4xmi355,
        "8xB200": run_on_8xb200,
        "8xMI355": run_on_8xmi355,
    }
    gpus = [gpu for gpu, ok in flags.items() if ok]
    models = parse_override(models_override) or tier_models(tier)
    ignore_exclusions = models_override is not None

    job = []
    for gpu in sorted(gpus):
        for model in sorted(models):
            if ignore_exclusions or not excluded(framework, gpu, model):
                job.append(
                    {
                        "model": model,
                        "runs_on": RUNNERS[gpu],
                        "display_name": f"{gpu} - {model}",
                    }
                )

    print(json.dumps({"include": job}, indent=2))


if __name__ == "__main__":
    main()
