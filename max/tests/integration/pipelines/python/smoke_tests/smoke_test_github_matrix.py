# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
}

# Framework → GPUs that framework cannot run on.
HW_EX = {"vllm": {"MI355"}, "sglang": {"MI355"}}

# Models tagged "multi" are skipped on these GPUs.
MULTI_GPUS = {"2xH100"}

# Model → list of exclusions:
#   - framework (e.g. "max")
#   - gpu (e.g. "MI355")
#   - framework@gpu (e.g. "sglang@B200")
#   - "multi" (skip model on MULTI_GPUS)
#
# If you want to add a model to the smoke test:
#   1. Trigger the smoke test job with the model name you want to add:
#   https://github.com/modularml/modular/actions/workflows/pipelineVerification.yaml
#   2. Review the results, and the need for framework/GPU exclusions (if any)
#   3. Add the model to the dictionary above, with the appropriate exclusions
MODELS = {
    "allenai/olmOCR-2-7B-1025-FP8": [
        "max",  # Wait for 25.7
        "sglang",
        "multi",
        "max-ci@MI355",  # TODO(GEX-2814)
    ],
    # E2EOPT-571: DeepSeek v2 lite chat not working on MAX
    "deepseek-ai/deepseek-v2-lite-chat": ["max-ci", "max", "multi"],
    "google/gemma-3-12b-it": ["multi"],
    "google/gemma-3-27b-it": [],
    "meta-llama/llama-3.1-8b-instruct": ["multi"],
    "meta-llama/llama-3.2-1b-instruct": ["multi"],
    "microsoft/phi-3.5-mini-instruct": ["multi"],
    "microsoft/phi-4": ["multi"],
    "mistralai/mistral-nemo-instruct-2407": ["multi"],
    "opengvlab/internvl3-8b-instruct": [
        "sglang@B200",
        "multi",
        "MI355",  # TODO(KERN-2196)
    ],
    "opengvlab/internvl3_5-8b-instruct": [
        "multi",
        "sglang@B200",  # FA3 vision enc not supported on B200
        "vllm@B200",
        "max",
    ],
    "qwen/qwen2.5-7b-instruct": ["multi"],
    "qwen/qwen2.5-vl-7b-instruct": ["multi", "MI355"],  # TODO(KERN-2196)
    "qwen/qwen3-8b": ["multi"],
    "redhatai/gemma-3-27b-it-fp8-dynamic": ["MI355"],
    "unsloth/gpt-oss-20b-bf16": [
        "max-ci@H100",
        "max@H100",
        "max@MI355",
        "multi",
    ],
}


def excluded(framework: str, gpu: str, model: str) -> bool:
    """Check if a model is excluded from a given framework and/or GPU."""
    if gpu in HW_EX.get(framework, set()):
        return True
    ex = set(MODELS.get(model, []))
    if "multi" in ex and gpu in MULTI_GPUS:
        return True
    return framework in ex or gpu in ex or f"{framework}@{gpu}" in ex


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
def main(
    framework: str,
    models_override: str | None,
    run_on_h100: bool,
    run_on_b200: bool,
    run_on_mi355: bool,
    run_on_2xh100: bool,
) -> None:
    flags = {
        "H100": run_on_h100,
        "B200": run_on_b200,
        "MI355": run_on_mi355,
        "2xH100": run_on_2xh100,
    }
    gpus = [gpu for gpu, ok in flags.items() if ok]
    models = parse_override(models_override) or list(MODELS)
    ignore_exclusions = models_override is not None

    job = []
    for model in sorted(models):
        for gpu in sorted(gpus):
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
