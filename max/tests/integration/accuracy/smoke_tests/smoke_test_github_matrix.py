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
    "H100": "modrunner-h100",
    "B200": "modrunner-b200",
    "MI355": "modrunner-mi355",
    "2xH100": "modrunner-h100-2x",
    "8xB200": "modrunner-b200-8x",
    "8xMI355": "modrunner-mi355-8x",
}

# Framework → GPUs that framework cannot run on.
HW_EX = {"vllm": {"MI355", "8xMI355"}, "sglang": {"MI355", "8xMI355"}}

# Tags: skip model on multi-GPU runners.
XL = {"8xB200", "8xMI355"}
MULTI = {"2xH100"} | XL
NON_XL = set(RUNNERS) - XL
DISABLE = set(RUNNERS)

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
MODELS: dict[str, set[str]] = {
    "allenai/olmo-3-7b-instruct": MULTI | {"max"},
    "allenai/olmocr-2-7b-1025-fp8": MULTI | {"sglang"},
    "bytedance-seed/academic-ds-9b": MULTI | {"max", "max-ci@MI355", "sglang@B200", "vllm@B200"},
    "deepseek-ai/deepseek-r1-0528": NON_XL | {"max", "sglang", "8xMI355"},  # 8xMI355: needs nvshmem
    "deepseek-ai/deepseek-v2-lite-chat": MULTI | {"max", "vllm@B200"},
    "google/gemma-3-1b-it": MULTI | {"vllm@B200"},
    "google/gemma-3-12b-it": MULTI,
    "google/gemma-3-27b-it": MULTI,  # TODO(MODELS-1021)
    "meta-llama/llama-3.1-8b-instruct": MULTI,
    "meta-llama/llama-3.2-1b-instruct": MULTI,
    "microsoft/phi-3.5-mini-instruct": MULTI,
    "microsoft/phi-4": MULTI,
    "mistralai/mistral-nemo-instruct-2407": MULTI | {"vllm"},
    "mistralai/mistral-small-3.1-24b-instruct-2503": MULTI | {"vllm"},
    "opengvlab/internvl3-8b-instruct": MULTI | {"sglang"},
    "opengvlab/internvl3_5-8b-instruct": MULTI | {"max", "sglang"},
    "qwen/qwen2.5-7b-instruct": MULTI,
    "qwen/qwen2.5-vl-3b-instruct": MULTI,
    "qwen/qwen2.5-vl-7b-instruct": MULTI,
    "qwen/qwen3-30b-a3b-instruct-2507": MULTI | {"max-ci@H100"},  # MODELS-1020
    "qwen/qwen3-8b": MULTI,
    "qwen/qwen3-vl-4b-instruct": XL | {"max-ci@H100", "vllm@B200"},  # MODELS-1020
    "qwen/qwen3-vl-4b-instruct-fp8": XL | {"max", "MI355", "max-ci@2xH100", "max-ci@H100"},  # max: 26.2, MI355: no FP8
    "qwen/qwen3-vl-30b-a3b-instruct": XL | {"max-ci@2xH100", "max-ci@H100", "max@2xH100", "max@H100"},
    "qwen/qwen3-vl-30b-a3b-instruct-fp8": XL | {"max", "MI355", "max-ci@2xH100", "max-ci@B200", "max-ci@H100", "sglang@B200"},  # max: 26.2, MI355: no FP8, B200: MODELS-1020
    "qwen/qwen3-vl-30b-a3b-thinking": XL | {"max", "max-ci@2xH100", "max-ci@H100"},
    "redhatai/gemma-3-27b-it-fp8-dynamic": MULTI,  # TODO(MODELS-1021)
    "nvidia/llama-3.1-405b-instruct-nvfp4": NON_XL | {"max", "8xMI355"},
    "redhatai/meta-llama-3.1-405b-instruct-fp8-dynamic": NON_XL,
    "unsloth/gpt-oss-20b-bf16": XL | {"max@H100"},
}

# These models are hardcoded to run on module-V3. They're duplicates
# of the model from the org listed in the comment next to the model name.
V3_MODELS: dict[str, set[str]] = {
    "tbmod/meta-llama-3.1-8b-instruct": MULTI, # unsloth/
    "tbmod/llama-3.2-1b-instruct": MULTI, # unsloth/
    "tbmod/gpt-oss-20b-bf16": DISABLE,  # unsloth/ TODO(MXF-121)
    "tbmod/phi-3.5-mini-instruct": MULTI, # microsoft/
    "tbmod/phi-4": MULTI, # microsoft/
}

MODELS = {**MODELS, **V3_MODELS}
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
    return [p.strip().lower() for p in parts if p.strip()]


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
@click.option("--run-on-h100", is_flag=True)
@click.option("--run-on-b200", is_flag=True)
@click.option("--run-on-mi355", is_flag=True)
@click.option("--run-on-2xh100", is_flag=True)
@click.option("--run-on-8xb200", is_flag=True)
@click.option("--run-on-8xmi355", is_flag=True)
def main(
    framework: str,
    models_override: str | None,
    run_on_h100: bool,
    run_on_b200: bool,
    run_on_mi355: bool,
    run_on_2xh100: bool,
    run_on_8xb200: bool,
    run_on_8xmi355: bool,
) -> None:
    flags = {
        "H100": run_on_h100,
        "B200": run_on_b200,
        "MI355": run_on_mi355,
        "2xH100": run_on_2xh100,
        "8xB200": run_on_8xb200,
        "8xMI355": run_on_8xmi355,
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
                        "model": model.lower(),
                        "runs_on": RUNNERS[gpu],
                        "display_name": f"{gpu} - {model}",
                    }
                )

    print(json.dumps({"include": job}, indent=2))


if __name__ == "__main__":
    main()
