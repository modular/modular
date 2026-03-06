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
"""Generate the GitHub Actions matrix for Pipeline Dataset Evaluation."""

# /// script
# dependencies = ["click>=8,<9"]
# ///

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click

CONFIGS_DIR = Path("max/tests/integration/accuracy/dataset_eval_configs")
# Switch to llama once llama is fixed in dataset eval.
SMOKE_TEST_PIPELINE = "sentence-transformers/all-mpnet-base-v2"

PIPELINES = [
    {
        "pipeline": "meta-llama/Meta-Llama-3-8B-Instruct",
        "runner": "modrunner-h100",
        "gpu_flag": "--devices gpu:0",
        "instance_type": "bm.gpu.h100.1",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "OpenGVLab/InternVL3-8B-Instruct",
        "runner": "modrunner-h100",
        "gpu_flag": "--devices gpu:0",
        "instance_type": "bm.gpu.h100.1",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "modularai/Llama-3.1-8B-Instruct-GGUF",
        "runner": "modrunner-h100",
        "gpu_flag": "--devices gpu:0",
        "instance_type": "bm.gpu.h100.1",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "sentence-transformers/all-mpnet-base-v2",
        "runner": "modrunner-h100",
        "gpu_flag": "--devices gpu:0",
        "instance_type": "bm.gpu.h100.1",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "Qwen/Qwen2.5-VL-3B-Instruct",
        "runner": "modrunner-h100",
        "gpu_flag": "--devices gpu:0",
        "instance_type": "bm.gpu.h100.1",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "Qwen/Qwen2.5-VL-7B-Instruct",
        "runner": "modrunner-h100",
        "gpu_flag": "--devices gpu:0",
        "instance_type": "bm.gpu.h100.1",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "OpenGVLab/InternVL3-38B-Instruct",
        "runner": "modrunner-h100-2x",
        "gpu_flag": "--devices gpu:0,1",
        "instance_type": "bm.gpu.h100.2",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "Qwen/Qwen2.5-VL-32B-Instruct",
        "runner": "modrunner-h100-2x",
        "gpu_flag": "--devices gpu:0,1",
        "instance_type": "bm.gpu.h100.2",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "RedHatAI/gemma-3-27b-it-FP8-dynamic",
        "runner": "modrunner-b200",
        "gpu_flag": "--devices gpu:0",
        "instance_type": "bm.gpu.b200.1",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "deepseek-ai/DeepSeek-R1",
        "runner": "modrunner-b200-8x",
        "gpu_flag": "--devices gpu:0,1,2,3,4,5,6,7",
        "instance_type": "bm.gpu.b200.8",
        "timeout": 90,  # 1.5 hours
    },
    {
        "pipeline": "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",
        "runner": "modrunner-b200-4x",
        "gpu_flag": "--devices gpu:0,1,2,3",
        "instance_type": "bm.gpu.b200.4",
        "timeout": 120,  # 2 hours
    },
    {
        "pipeline": "deepseek-ai/DeepSeek-R1-longbench-v2",
        "runner": "modrunner-b200-8x",
        "gpu_flag": "--devices gpu:0,1,2,3,4,5,6,7",
        "instance_type": "bm.gpu.b200.8",
        "timeout": 390,  # 6.5 hours
    },
    {
        "pipeline": "nvidia/DeepSeek-V3.1-NVFP4-longbench-v2",
        "runner": "modrunner-b200-8x",
        "gpu_flag": "--devices gpu:0,1,2,3,4,5,6,7",
        "instance_type": "bm.gpu.b200.8",
        "timeout": 390,  # 6.5 hours
    },
]


def _changed_pipelines(base_ref: str) -> set[str]:
    """Return pipeline names whose config .sh files changed vs base_ref."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    changed: set[str] = set()
    for line in result.stdout.strip().splitlines():
        p = Path(line)
        if p.suffix == ".sh":
            try:
                relative = p.relative_to(CONFIGS_DIR)
            except ValueError:
                continue
            changed.add(str(relative.with_suffix("")))
    return changed


def generate_matrix(
    event_name: str,
    selected_pipeline: str,
    base_ref: str | None,
) -> list[dict]:
    """Return the filtered list of pipeline dicts for the GH Actions matrix."""
    if event_name == "pull_request":
        assert base_ref is not None
        changed = _changed_pipelines(base_ref)
        if changed:
            final = [p for p in PIPELINES if p["pipeline"] in changed]
            if not final:
                print(
                    f"::warning::Changed configs {changed} not found in"
                    " matrix, running smoke test",
                    file=sys.stderr,
                )
                final = [
                    p for p in PIPELINES if p["pipeline"] == SMOKE_TEST_PIPELINE
                ]
        else:
            print(
                f"::notice::No pipeline configs changed, running smoke test"
                f" only ({SMOKE_TEST_PIPELINE})",
                file=sys.stderr,
            )
            final = [
                p for p in PIPELINES if p["pipeline"] == SMOKE_TEST_PIPELINE
            ]

    elif selected_pipeline and selected_pipeline != "all":
        final = [p for p in PIPELINES if p["pipeline"] == selected_pipeline]
        if not final:
            print(
                f"::error::Pipeline '{selected_pipeline}' not found!",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # schedule or workflow_dispatch with "all"
        final = list(PIPELINES)

    return final


@click.command()
@click.option(
    "--event-name",
    required=True,
    type=click.Choice(["pull_request", "schedule", "workflow_dispatch"]),
)
@click.option("--selected-pipeline", default="")
@click.option("--base-ref", default=None)
def main(event_name: str, selected_pipeline: str, base_ref: str | None) -> None:
    """Generate the GitHub Actions matrix for Pipeline Dataset Evaluation."""
    final = generate_matrix(event_name, selected_pipeline, base_ref)
    matrix = {"include": final}
    click.echo(json.dumps(matrix))


if __name__ == "__main__":
    main()
