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
"""Shared scaffolding for per-pipeline GitHub Actions matrix generators.

Concrete generators (``dataset_eval_matrix.py``, ``llm_fuzz_matrix.py``)
own the ``PIPELINES`` list, the configs directory, and the smoke-test
key; this module provides the dataclass shape, the PR-mode change
detector, and the event-name → entries filter that they share.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineEntry:
    """A pipeline-eval matrix entry consumed by a GH Actions workflow."""

    pipeline: str
    runner: str
    gpu_flag: str
    instance_type: str
    timeout: int  # minutes
    # Model path passed to MAX Serve. When None, callers should treat
    # `pipeline` as the model path. Keeping this separate from
    # `pipeline` lets a workflow override the served weights (e.g.
    # swap to a local mount) without changing the pipeline's
    # config-selector key.
    model_path: str | None = None


def changed_pipelines(base_ref: str, configs_dir: Path) -> set[str]:
    """Return pipeline names whose config .sh files changed vs ``base_ref``."""
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
                relative = p.relative_to(configs_dir)
            except ValueError:
                continue
            changed.add(str(relative.with_suffix("")))
    return changed


def filter_entries(
    pipelines: list[PipelineEntry],
    *,
    event_name: str,
    selected_pipeline: str,
    base_ref: str | None,
    configs_dir: Path,
    smoke_test_pipeline: str,
) -> list[PipelineEntry]:
    """Return the entries to run for this event.

    On ``pull_request``, runs only entries whose config .sh files
    changed (falls back to a single smoke-test entry when nothing
    matched). On ``workflow_dispatch`` with a specific
    ``--selected-pipeline``, returns just that entry; ``all`` or an
    empty selection returns every entry. ``schedule`` events behave
    like ``all``.
    """
    if event_name == "pull_request":
        assert base_ref is not None
        changed = changed_pipelines(base_ref, configs_dir)
        if changed:
            final = [p for p in pipelines if p.pipeline in changed]
            if not final:
                print(
                    f"::warning::Changed configs {changed} not found in"
                    " matrix, running smoke test",
                    file=sys.stderr,
                )
                final = [
                    p for p in pipelines if p.pipeline == smoke_test_pipeline
                ]
        else:
            print(
                f"::notice::No pipeline configs changed, running smoke test"
                f" only ({smoke_test_pipeline})",
                file=sys.stderr,
            )
            final = [p for p in pipelines if p.pipeline == smoke_test_pipeline]
        return final

    if selected_pipeline and selected_pipeline != "all":
        final = [p for p in pipelines if p.pipeline == selected_pipeline]
        if not final:
            print(
                f"::error::Pipeline '{selected_pipeline}' not found!",
                file=sys.stderr,
            )
            sys.exit(1)
        return final

    # schedule or workflow_dispatch with "all"
    return list(pipelines)


def entries_to_matrix(entries: list[PipelineEntry]) -> dict[str, Any]:
    """Render entries as the ``{"include": [...]}`` matrix dict.

    Unset optional fields (None values) are stripped so consumers
    aren't exposed to keys their workflow YAML doesn't reference.
    """
    return {
        "include": [
            {k: v for k, v in asdict(e).items() if v is not None}
            for e in entries
        ],
    }
