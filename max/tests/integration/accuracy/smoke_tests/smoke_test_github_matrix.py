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
    "4xMI355": "modrunner-mi355-4x",
    "8xB200": "modrunner-b200-efa-8x",
    "8xMI355": "modrunner-mi355-8x",
}

_1xB200 = {"B200"}
_1xMI355 = {"MI355"}
_2xB200 = {"2xB200"}
_2xMI355 = {"2xMI355"}
_4xMI355 = {"4xMI355"}
_8xB200 = {"8xB200"}
_8xMI355 = {"8xMI355"}
DISABLE: set[str] = set()

# Model → HW it runs on. DISABLE runs nowhere.
#
# To add a model, trigger the smoke test with it first:
# https://github.com/modularml/modular/actions/workflows/serveSmokeTest.yaml
# then list it below with the HW it passed on. VLMs also go in is_vision_model
# and reasoning models in is_reasoning_model, both in smoke_test.py.
# fmt: off
HF_MODELS: Mapping[str, set[str]] = {
    "allenai/Olmo-3-7B-Instruct": _1xB200 | _1xMI355,
    "allenai/olmOCR-2-7B-1025-FP8": _1xB200 | _1xMI355,
    "amd/Kimi-K2.7-Code-MXFP4": _4xMI355,
    "nvidia/Kimi-K2.7-Code-NVFP4": _8xB200,
    "amd/MiniMax-M3-MXFP4": _4xMI355,
    "ByteDance-Seed/academic-ds-9B": DISABLE,  # SERVOPT-1120
    "deepseek-ai/DeepSeek-V2-Lite-Chat": DISABLE,  # SERVOPT-1120
    "deepseek-ai/DeepSeek-V3.1-Terminus": _8xB200,
    "google/diffusiongemma-26B-A4B-it": DISABLE,
    "google/gemma-3-1b-it": _1xB200,  # TODO(KERN-3014): MI355
    "google/gemma-3-27b-it": _1xB200 | _1xMI355,
    "google/gemma-4-26B-A4B-it": _1xB200 | _1xMI355,
    "google/gemma-4-31B-it": _1xB200 | _1xMI355,
    "nvidia/Gemma-4-26B-A4B-NVFP4": _1xB200,
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4": DISABLE,
    "nvidia/Gemma-4-31B-IT-NVFP4": _1xB200 | _2xB200,
    "meta-llama/Llama-3.1-8B-Instruct": _1xB200 | _1xMI355,
    "microsoft/Phi-3.5-mini-instruct": _1xB200 | _1xMI355,
    "microsoft/phi-4": _1xB200 | _1xMI355,
    "MiniMaxAI/MiniMax-M3-MXFP8": _8xMI355,  # MODELS-1611: 8xB200 runs __mtp
    "modularai/MiniMax-M3-MXFP6": _4xMI355 | _8xMI355,
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": _1xB200 | _1xMI355,
    "modularai/Llama-3.1-405B-Instruct-autofp8": _4xMI355 | _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4": _8xB200,
    "OpenGVLab/InternVL3_5-8B-Instruct": _1xB200 | _1xMI355,
    "Qwen/Qwen2.5-7B-Instruct": _1xB200 | _1xMI355,
    "Qwen/Qwen2.5-VL-7B-Instruct": _1xB200 | _1xMI355,
    "Qwen/Qwen3-8B": _1xB200 | _1xMI355,
    "Qwen/Qwen3-VL-4B-Instruct-FP8": _1xB200 | _2xB200,  # MI355: no FP8
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8": _1xB200 | _2xB200,  # MI355: no FP8
    "Qwen/Qwen3.5-9B": _1xB200,
    "Qwen/Qwen3.6-27B": _1xB200,
    # TODO(MODELS-1021)
    "RedHatAI/gemma-3-27b-it-FP8-dynamic": _1xB200 | _1xMI355,
    "nvidia/Llama-3.1-405B-Instruct-NVFP4": _8xB200,
    "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic": _4xMI355 | _8xB200,
    "openai/gpt-oss-20b": _1xB200 | _1xMI355 | _2xB200,
    "thinkingmachines/Inkling-Small-NVFP4": _2xB200,
}

# Models tested with custom MAX recipe presets. MODEL_RECIPES in
# smoke_test.py maps each alias to its reusable recipe config.
CUSTOM_MODELS: Mapping[str, set[str]] = {
    "meta-llama/Llama-3.1-8B-Instruct__modulev3": _1xB200 | _1xMI355,
    "google/gemma-3-27b-it__modulev3": _1xB200 | _1xMI355 | _2xB200 | _2xMI355,
    "google/gemma-4-31B-it__modulev3": _1xB200 | _1xMI355,
    "microsoft/Phi-3.5-mini-instruct__modulev3": _1xB200 | _1xMI355,
    "microsoft/phi-4__modulev3": _1xB200 | _1xMI355,
    "deepseek-ai/DeepSeek-V2-Lite-Chat__modulev3": _1xB200 | _1xMI355,
    "nvidia/DeepSeek-V3.1-NVFP4__fp8kv": _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__tpep": _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__tpep_ar": _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__tptp": _8xB200,
    # TODO(SERVOPT-1168): multi-GPU
    "meta-llama/Llama-3.1-8B-Instruct__eagle": _1xB200 | _1xMI355,
    "meta-llama/Llama-3.1-8B-Instruct__dflash": _1xB200 | _1xMI355,
    "nvidia/DeepSeek-V3.1-NVFP4__mtp": _8xB200,
    "nvidia/DeepSeek-V3.1-NVFP4__mtp_tpep": _8xB200,
    # Synthesis is single-device for now.
    "google/gemma-4-12B-it__device_graph_synthesis": _1xB200 | _1xMI355,
    "google/gemma-4-12B-it__dspark": _1xB200 | _1xMI355,
    # Tuned recipes use an FP8 KV cache, which MI355 does not support.
    "google/gemma-4-26B-A4B-it__tuned": _1xB200,
    "google/gemma-4-31B-it__tuned": _1xB200,
    "nvidia/Gemma-4-26B-A4B-NVFP4__tuned": _1xB200,
    "nvidia/Gemma-4-31B-IT-NVFP4__tuned": _1xB200,
    "nvidia/Kimi-K2.7-Code-NVFP4__modulev3": _8xB200,
    "meta-llama/Llama-3.1-8B-Instruct__rust_tiered_kvconnector": _1xB200,
    "nvidia/GLM-5.2-NVFP4__mtp_tpep": _8xB200,
    "RadixArk/GLM-5.3-NVFP4__mtp_tpep": _8xB200,
    "thinkingmachines/Inkling-Small-NVFP4__mtp": _2xB200,
    "MiniMaxAI/MiniMax-M3-MXFP8__mtp": _8xB200,
}

# Aliases whose recipe ships with a private arch, so it cannot appear in
# MODEL_RECIPES: the private entrypoint resolves it from its own hw_recipes
# table instead.
PRIVATE_RECIPE_MODELS = frozenset({"MiniMaxAI/MiniMax-M3-MXFP8__mtp"})

MODELS: Mapping[str, set[str]] = {**HF_MODELS, **CUSTOM_MODELS}
# fmt: on

NIGHTLY_MODELS = frozenset(
    {
        "google/gemma-4-12B-it__dspark",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-26B-A4B-it__tuned",
        "google/gemma-4-31B-it",
        "google/gemma-4-31B-it__tuned",
        "nvidia/Gemma-4-26B-A4B-NVFP4",
        "nvidia/Gemma-4-26B-A4B-NVFP4__tuned",
        "nvidia/Gemma-4-31B-IT-NVFP4",
        "nvidia/Gemma-4-31B-IT-NVFP4__tuned",
        "MiniMaxAI/MiniMax-M3-MXFP8",
        "MiniMaxAI/MiniMax-M3-MXFP8__mtp",
        "amd/MiniMax-M3-MXFP4",
        "modularai/MiniMax-M3-MXFP6",
        "nvidia/GLM-5.2-NVFP4__mtp_tpep",
        "amd/Kimi-K2.7-Code-MXFP4",
        "nvidia/Kimi-K2.7-Code-NVFP4",
        "thinkingmachines/Inkling-Small-NVFP4",
        "thinkingmachines/Inkling-Small-NVFP4__mtp",
    }
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
    # Custom MAX recipe variants are MAX only; no vLLM/SGLang equivalent.
    if model in CUSTOM_MODELS and framework in {"vllm", "sglang"}:
        return True
    return gpu not in MODELS.get(model, set())


def parse_override(raw: str | None) -> list[str]:
    """Parse a comma-separated list of models from the command line."""
    if not raw:
        return []
    parts = re.split(r"[, \n\r]+", raw)
    return [p.strip() for p in parts if p.strip()]


@click.command()
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "max-ci", "max"]),
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
    run_on_4xmi355: bool,
    run_on_8xb200: bool,
    run_on_8xmi355: bool,
) -> None:
    flags = {
        "B200": run_on_b200,
        "MI355": run_on_mi355,
        "2xB200": run_on_2xb200,
        "2xMI355": run_on_2xmi355,
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
