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

import click

RUNNERS = {
    "B200": "modrunner-b200",
    "MI355": "modrunner-mi355",
    "2xB200": "modrunner-b200-2x",
    "2xMI355": "modrunner-mi355-2x",
    "8xB200": "modrunner-b200-8x",
    "8xMI355": "modrunner-mi355-8x",
    "8xB200_internal": "modrunner-prod-2-b200-8x",
}

# Framework → GPUs that framework cannot run on.
HW_EX = {
    "vllm": {"MI355", "2xMI355", "8xMI355"},
    "sglang": {"MI355", "2xMI355", "8xMI355"},
}

# Tags: skip model on multi-GPU runners.
XL = {"8xB200", "8xMI355", "8xB200_internal"}
MULTI = {"2xB200", "2xMI355"} | XL
NON_XL = (set(RUNNERS) - XL) | {"8xB200_internal"}
DISABLE = set(RUNNERS)
# Runs only on the dedicated internal 8xB200 runner; everything else excluded.
INTERNAL_ONLY = set(RUNNERS) - {"8xB200_internal"}

# Model → set of exclusion tags:
#   - framework        (e.g. "max")
#   - gpu              (e.g. "MI355")
#   - framework@gpu    (e.g. "sglang@B200")
#   - use XL           to skip on 8xB200 and 8xMI355
#   - use MULTI        to skip on all multi-GPU runners
#   - use NON_XL       to skip on everything except 8xB200 and 8xMI355
#   - use DISABLE      to skip on all runners (temporarily disable a model)
#
# If you want to add a model to the smoke test:
#   1. Trigger the smoke test job with the model name you want to add:
#   https://github.com/modularml/modular/actions/workflows/pipelineVerification.yaml
#   2. Review the results, and the need for framework/GPU exclusions (if any)
#   3. Add the model to the dictionary below, with the appropriate exclusions
#    3a) For VLMs, add it to the is_vision_model check in smoke_test.py
#    3b) For reasoning models, add it to the is_reasoning_model check in smoke_test.py
# fmt: off
HF_MODELS: dict[str, set[str]] = {
    "allenai/Olmo-3-7B-Instruct": MULTI | {"max"},
    "allenai/olmOCR-2-7B-1025-FP8": MULTI | {"sglang"},
    "amd/Kimi-K2.5-MXFP4": NON_XL | {"8xB200"},
    "amd/MiniMax-M2.7-MXFP4": NON_XL | {"8xB200"},
    "ByteDance-Seed/academic-ds-9B": MULTI | {"max", "max-ci", "sglang@B200", "vllm@B200"},  # SERVOPT-1120
    "deepseek-ai/DeepSeek-R1-0528": NON_XL | {"max", "sglang", "8xMI355"},  # 8xMI355: needs nvshmem
    "deepseek-ai/DeepSeek-V2-Lite-Chat": MULTI | {"max", "max-ci", "vllm@B200"},  # SERVOPT-1120
    "deepseek-ai/DeepSeek-V3.1-Terminus": NON_XL | {"8xMI355"},
    "google/gemma-3-1b-it": MULTI | {"vllm@B200"},
    "google/gemma-3-27b-it": MULTI,
    "google/gemma-4-26B-A4B-it": MULTI | {"max", "max-ci"},  # TODO(SERVOPT-1292)
    "google/gemma-4-31B-it": MULTI,
    "nvidia/Gemma-4-26B-A4B-NVFP4": MULTI | {"max", "max-ci", "MI355"},  # TODO(SERVOPT-1292)
    "nvidia/Gemma-4-31B-IT-NVFP4": MULTI | {"MI355"},
    "meta-llama/Llama-3.1-8B-Instruct": MULTI,
    "microsoft/Phi-3.5-mini-instruct": MULTI,
    "microsoft/phi-4": MULTI,
    "MiniMaxAI/MiniMax-M2.7": NON_XL | {"8xMI355", "sglang"},
    "lukealonso/MiniMax-M2.7-NVFP4": NON_XL | {"8xMI355", "sglang"},
    "mistralai/Mistral-Nemo-Instruct-2407": MULTI | {"vllm"},
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": MULTI | {"vllm"},
    "modularai/Llama-3.1-405B-Instruct-autofp8": NON_XL | {"max"},
    "nvidia/DeepSeek-V3.1-NVFP4": NON_XL | {"8xMI355"},
    "nvidia/Kimi-K2.5-NVFP4": NON_XL | {"8xMI355"},
    "OpenGVLab/InternVL3_5-8B-Instruct": MULTI | {"max", "sglang"},
    "Qwen/Qwen2.5-7B-Instruct": MULTI,
    "Qwen/Qwen2.5-VL-7B-Instruct": MULTI,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": NON_XL | {"max", "8xMI355"},
    "Qwen/Qwen3-30B-A3B-Instruct-2507": MULTI,
    "Qwen/Qwen3-8B": MULTI,
    "Qwen/Qwen3-VL-4B-Instruct": XL | {"vllm@B200"},  # MODELS-1020
    "Qwen/Qwen3-VL-4B-Instruct-FP8": XL | {"MI355", "2xMI355"},  # MI355: no FP8
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8": DISABLE,  # MODELS-1442
    "Qwen/Qwen3-VL-30B-A3B-Thinking": XL | {"max"},
    "Qwen/Qwen3.5-9B": MULTI | {"max", "max-ci@MI355"},
    "Qwen/Qwen3.6-27B": MULTI | {"max", "max-ci@MI355"},
    "RedHatAI/gemma-3-27b-it-FP8-dynamic": MULTI,  # TODO(MODELS-1021)
    "nvidia/Llama-3.1-405B-Instruct-NVFP4": NON_XL | {"max", "8xMI355"},
    "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic": NON_XL,
    "openai/gpt-oss-20b": XL | {"2xMI355"},
    "stepfun-ai/Step-3.5-Flash": NON_XL | {"8xMI355"},
    "unsloth/gpt-oss-20b-BF16": XL | {"2xMI355"},
}

# Models tested with custom MAX recipe presets. MODEL_RECIPES in
# smoke_test.py maps each alias to its reusable recipe config.
CUSTOM_MODELS: dict[str, set[str]] = {
    "meta-llama/Llama-3.1-8B-Instruct__modulev3": MULTI,
    "google/gemma-3-27b-it__modulev3": XL,
    "unsloth/gpt-oss-20b-BF16__modulev3": DISABLE,  # TODO(MXF-121)
    "microsoft/Phi-3.5-mini-instruct__modulev3": MULTI,
    "microsoft/phi-4__modulev3": MULTI,
    "nvidia/DeepSeek-V3.1-NVFP4__fp8kv": NON_XL | {"8xMI355"},
    "nvidia/DeepSeek-V3.1-NVFP4__tpep": NON_XL | {"8xMI355"},
    "nvidia/DeepSeek-V3.1-NVFP4__tpep_ar": NON_XL | {"8xMI355"},
    "nvidia/DeepSeek-V3.1-NVFP4__tptp": NON_XL | {"8xMI355"},
    # TODO(SERVOPT-1168): Support multi-GPU eagle llama
    "meta-llama/Llama-3.1-8B-Instruct__eagle": MULTI | {"vllm", "sglang"},
    "nvidia/DeepSeek-V3.1-NVFP4__mtp": NON_XL | {"8xMI355"},
    "nvidia/DeepSeek-V3.1-NVFP4__mtp_tpep": NON_XL | {"8xMI355"},
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__eagle": DISABLE,  # SERVSYS-1248
    "google/gemma-4-26B-A4B-it__no_dgc": MULTI,
    "nvidia/Gemma-4-26B-A4B-NVFP4__no_dgc": MULTI | {"MI355"},
    "meta-llama/Llama-3.1-8B-Instruct__local_kvconnector": MULTI | {"vllm", "sglang", "MI355"},
    "meta-llama/Llama-3.1-8B-Instruct__eagle_local_kvconnector": MULTI | {"vllm", "sglang", "MI355"},
    "meta-llama/Llama-3.1-8B-Instruct__tiered_kvconnector": MULTI | {"vllm", "sglang", "MI355"},
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__local_kvconnector_tpep": NON_XL | {"8xMI355"},
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__tiered_kvconnector_tpep": NON_XL | {"8xMI355"},
    "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3__eagle_tiered_kvconnector_tpep": NON_XL | {"8xMI355"},
    # Runs exclusively on the dedicated internal 8xB200 runner. The recipe
    # loads weights pre-staged on the runner; see the matching MODEL_RECIPES
    # entry in smoke_test.py.
    "nvidia/Kimi-K2.5-NVFP4__internal": INTERNAL_ONLY,
}

MODELS = {**HF_MODELS, **CUSTOM_MODELS}
# fmt: on


def excluded(framework: str, gpu: str, model: str) -> bool:
    """Check if a model is excluded from a given framework and/or GPU."""
    if gpu in HW_EX.get(framework, set()):
        return True
    tags = MODELS.get(model, set())
    return framework in tags or gpu in tags or f"{framework}@{gpu}" in tags


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
    help="Comma list of models; skips exclusions.",
)
@click.option("--run-on-b200", is_flag=True)
@click.option("--run-on-mi355", is_flag=True)
@click.option("--run-on-2xb200", is_flag=True)
@click.option("--run-on-2xmi355", is_flag=True)
@click.option("--run-on-8xb200", is_flag=True)
@click.option("--run-on-8xmi355", is_flag=True)
@click.option("--run-on-8xb200-internal", is_flag=True)
def main(
    framework: str,
    models_override: str | None,
    run_on_b200: bool,
    run_on_mi355: bool,
    run_on_2xb200: bool,
    run_on_2xmi355: bool,
    run_on_8xb200: bool,
    run_on_8xmi355: bool,
    run_on_8xb200_internal: bool,
) -> None:
    flags = {
        "B200": run_on_b200,
        "MI355": run_on_mi355,
        "2xB200": run_on_2xb200,
        "2xMI355": run_on_2xmi355,
        "8xB200": run_on_8xb200,
        "8xMI355": run_on_8xmi355,
        "8xB200_internal": run_on_8xb200_internal,
    }
    gpus = [gpu for gpu, ok in flags.items() if ok]
    models = parse_override(models_override) or list(MODELS)
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
