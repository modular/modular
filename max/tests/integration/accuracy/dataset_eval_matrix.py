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

import dataclasses
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

CONFIGS_DIR = Path("max/tests/integration/accuracy/dataset_eval_configs")
# Switch to llama once llama is fixed in dataset eval.
SMOKE_TEST_PIPELINE = "sentence-transformers/all-mpnet-base-v2"


@dataclass(frozen=True)
class PipelineEntry:
    """A pipeline-eval matrix entry consumed by the GH Actions workflow."""

    pipeline: str
    runner: str
    gpu_flag: str
    instance_type: str
    timeout: int  # minutes


PIPELINES: list[PipelineEntry] = [
    PipelineEntry(
        pipeline="meta-llama/Meta-Llama-3-8B-Instruct",
        runner="modrunner-h100",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.h100.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="OpenGVLab/InternVL3-8B-Instruct",
        runner="modrunner-h100",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.h100.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="modularai/Llama-3.1-8B-Instruct-GGUF",
        runner="modrunner-h100",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.h100.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="sentence-transformers/all-mpnet-base-v2",
        runner="modrunner-h100",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.h100.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="LiquidAI/LFM2.5-1.2B-Instruct",
        runner="modrunner-h100",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.h100.1",
        # LFM2 requires batch_size=1 (SSM/conv state can't be batched across
        # sequences in the same way as pure-attention models), so all evaluation
        # tasks run sequentially rather than in parallel batches.  3 hours
        # instead of the usual 2 hours accounts for that serialization overhead.
        timeout=180,
    ),
    PipelineEntry(
        pipeline="Qwen/Qwen2.5-VL-3B-Instruct",
        runner="modrunner-h100",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.h100.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="Qwen/Qwen2.5-VL-7B-Instruct",
        runner="modrunner-h100",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.h100.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="OpenGVLab/InternVL3-38B-Instruct",
        runner="modrunner-h100-2x",
        gpu_flag="--devices gpu:0,1",
        instance_type="bm.gpu.h100.2",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="Qwen/Qwen2.5-VL-32B-Instruct",
        runner="modrunner-h100-2x",
        gpu_flag="--devices gpu:0,1",
        instance_type="bm.gpu.h100.2",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="RedHatAI/gemma-3-27b-it-FP8-dynamic",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="google/gemma-4-26B-A4B-it",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="google/gemma-4-31B-it",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="nvidia/Gemma-4-26B-A4B-NVFP4",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="nvidia/Gemma-4-31B-IT-NVFP4",
        runner="modrunner-b200",
        gpu_flag="--devices gpu:0",
        instance_type="bm.gpu.b200.1",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="deepseek-ai/DeepSeek-R1",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=90,  # 1.5 hours
    ),
    PipelineEntry(
        pipeline="RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",
        runner="modrunner-b200-4x",
        gpu_flag="--devices gpu:0,1,2,3",
        instance_type="bm.gpu.b200.4",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="deepseek-ai/DeepSeek-R1-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
    PipelineEntry(
        pipeline="deepseek-ai/DeepSeek-V3.1-Terminus",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=90,  # 1.5 hours
    ),
    PipelineEntry(
        pipeline="MiniMaxAI/MiniMax-M2.7",
        runner="modrunner-b200-4x",
        gpu_flag="--devices gpu:0,1,2,3",
        instance_type="bm.gpu.b200.4",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="lukealonso/MiniMax-M2.7-NVFP4",
        runner="modrunner-b200-4x",
        gpu_flag="--devices gpu:0,1,2,3",
        instance_type="bm.gpu.b200.4",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="amd/MiniMax-M2.7-MXFP4",
        runner="modrunner-mi355-4x",
        gpu_flag="--devices gpu:0,1,2,3",
        instance_type="bm.gpu.mi355x.4",
        timeout=120,  # 2 hours
    ),
    PipelineEntry(
        pipeline="nvidia/DeepSeek-V3.1-NVFP4-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
    PipelineEntry(
        pipeline="nvidia/DeepSeek-V3.1-NVFP4-fp8kv-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
    PipelineEntry(
        pipeline="nvidia/Kimi-K2.5-NVFP4-ep-dp-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
    PipelineEntry(
        pipeline="nvidia/Kimi-K2.5-NVFP4-ep-tp-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
    PipelineEntry(
        pipeline="nvidia/Kimi-K2.5-NVFP4-ep-dp-eagle3-longbench-v2",
        runner="modrunner-b200-8x",
        gpu_flag="--devices gpu:0,1,2,3,4,5,6,7",
        instance_type="bm.gpu.b200.8",
        timeout=390,  # 6.5 hours
    ),
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
) -> list[PipelineEntry]:
    """Return the filtered list of pipeline entries for the GH Actions matrix."""
    if event_name == "pull_request":
        assert base_ref is not None
        changed = _changed_pipelines(base_ref)
        if changed:
            final = [p for p in PIPELINES if p.pipeline in changed]
            if not final:
                print(
                    f"::warning::Changed configs {changed} not found in"
                    " matrix, running smoke test",
                    file=sys.stderr,
                )
                final = [
                    p for p in PIPELINES if p.pipeline == SMOKE_TEST_PIPELINE
                ]
        else:
            print(
                f"::notice::No pipeline configs changed, running smoke test"
                f" only ({SMOKE_TEST_PIPELINE})",
                file=sys.stderr,
            )
            final = [p for p in PIPELINES if p.pipeline == SMOKE_TEST_PIPELINE]

    elif selected_pipeline and selected_pipeline != "all":
        final = [p for p in PIPELINES if p.pipeline == selected_pipeline]
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
    matrix = {"include": [dataclasses.asdict(p) for p in final]}
    click.echo(json.dumps(matrix))


if __name__ == "__main__":
    main()
