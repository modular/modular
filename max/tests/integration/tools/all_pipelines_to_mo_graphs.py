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

"""Dump build-only graphs for every model in the CI smoke-test matrix.

Enumerates the same (GPU config, model) pairs the Serve Smoke Test CI runs
for the max-ci framework, derives each entry's server command via
get_server_cmd, translates it into a pipeline_to_mo_graph invocation, and runs
pipeline_to_mo_graph --build-only in a subprocess. Compilation targets virtual
devices matching each GPU config, so no GPU is required. Graphs land in
<output-dir>/<gpu>/<model>. Failures are recorded and summarized at the
end instead of aborting the sweep.

Models in SKIP_MODELS are dropped from the sweep before it starts. Use it for
models that cannot dump today, either because they break the orchestrator
itself (the AMD Kimi MXFP4 model balloons host memory during virtual compile
and OOM-kills the pod, taking every concurrent dump down with it) or because
they always fail their own dump and would otherwise re-fail every run.

Run with:
    ./bazelw run //max/tests/integration/tools:all_pipelines_to_mo_graphs -- \\
        --output-dir /tmp/graphs
"""

from __future__ import annotations

import fnmatch
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import click
import python.runfiles
from eval_runner import safe_model_name
from smoke_test import get_server_cmd
from smoke_test_github_matrix import MODELS, RUNNERS, excluded

logger = logging.getLogger(__name__)

# CI matrix GPU configs, taken from the smoke-test runner set so a new CI
# runner is swept automatically. 8xB200_internal is dropped: it resolves to the
# same family, count and target as 8xB200, so it would only dump duplicates.
GPU_CONFIGS = tuple(r for r in RUNNERS if r != "8xB200_internal")

# GPU family -> virtual device target to compile for.
_GPU_TARGETS = {"B200": "cuda:sm_100", "MI355": "hip:gfx950"}

# Case-insensitive globs (matched against the CI model name) dropped from the
# sweep on every GPU. Use this for models that fail on every GPU they run on; a
# model that fails only on some GPUs belongs in the per-GPU exclusion tags in
# smoke_test_github_matrix instead.
SKIP_MODELS: set[str] = {
    # Balloons host memory during virtual compile and OOM-kills the pod, taking
    # every concurrent dump down with it.
    "amd/kimi*mxfp4",
    # Eager device-buffer allocation aborts under the build-only virtual device
    # (VirtualDeviceContext::memAlloc not implemented). Covers the speculative
    # eagle/dflash recipes and the new-arch base models that hit the same path.
    "*__*eagle*",
    "*__dflash",
    "nvidia/deepseek-v3.1-nvfp4__mtp",
    "nvidia/deepseek-v3.1-nvfp4__mtp_tpep",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.6-27b",
    "*minimax-m2.7*",
    # Checkpoint not supported by the installed transformers version.
    "google/gemma-4-31b-it__mtp",
}


class _GpuSpec(NamedTuple):
    name: str
    num_gpus: int
    target: str


def _gpu_spec(config: str) -> _GpuSpec:
    """Parses a matrix GPU config into its name, count and compile target.

    Handles bare families ("B200") and counted configs ("2xMI355").
    """
    count, sep, name = config.partition("x")
    if not sep:  # bare family, e.g. "B200"
        count, name = "1", count
    return _GpuSpec(name, int(count), _GPU_TARGETS[name])


@dataclass
class _Result:
    gpu: str
    model: str
    ok: bool
    num_graphs: int = 0
    error: str = ""


@dataclass(frozen=True)
class _Sweep:
    """Settings shared by every entry in a single dump sweep."""

    binary: str
    root: Path
    target: str | None
    timeout: int
    total: int


def _pipeline_to_mo_graph_binary() -> str:
    """Resolves the pipeline_to_mo_graph binary from the runfiles tree.

    Each dump runs this binary in its own subprocess, so its max.pipelines
    import and any per-model crash or timeout stay out of this orchestrator
    process. Running the built binary gives the subprocess its own bazel import
    path.
    """
    runfiles = python.runfiles.Create()
    assert runfiles is not None, "Unable to find runfiles tree"
    loc = runfiles.Rlocation(
        "_main/max/tests/integration/tools/pipeline_to_mo_graph"
    )
    assert loc is not None, "Unable to find pipeline_to_mo_graph binary"
    return loc


def _serve_args(server_cmd: list[str]) -> list[str]:
    """Returns the serve command's arguments, ready to pass to the dump tool.

    Keeps everything after the serve subcommand; pipeline_to_mo_graph ignores
    the serve-only flags (``--port``, ``--pretty-print-config``, ...) itself.
    """
    return server_cmd[server_cmd.index("serve") + 1 :]


def _matches(name: str, globs: list[str] | None) -> bool:
    return globs is None or any(fnmatch.fnmatch(name.lower(), g) for g in globs)


def _is_skipped(model: str) -> bool:
    """Returns whether `model` matches a SKIP_MODELS glob."""
    return any(fnmatch.fnmatch(model.lower(), g) for g in SKIP_MODELS)


def _dump_one(
    sweep: _Sweep, index: int, gpu: str, model: str, server_cmd: list[str]
) -> _Result:
    """Runs pipeline_to_mo_graph for one matrix entry in a subprocess."""
    prefix = f"[{index}/{sweep.total}] {gpu} - {model}"
    logger.info(prefix)

    model_dir = sweep.root / gpu / safe_model_name(model)
    cmd = [
        sweep.binary,
        *_serve_args(server_cmd),
        "--build-only",
        "--output-dir",
        str(model_dir),
        "--target",
        sweep.target or _gpu_spec(gpu).target,
    ]
    logger.info("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=sweep.timeout
        )
    except subprocess.TimeoutExpired:
        logger.error("%s: timed out after %ds", prefix, sweep.timeout)
        return _Result(
            gpu, model, ok=False, error=f"timed out after {sweep.timeout}s"
        )
    except Exception as e:
        logger.error("%s: failed to launch dump: %s", prefix, e)
        return _Result(gpu, model, ok=False, error=str(e))

    num_graphs = len(list(model_dir.iterdir())) if model_dir.is_dir() else 0
    if proc.returncode == 0:
        logger.info(
            "%s: dumped %d graph files to %s", prefix, num_graphs, model_dir
        )
        return _Result(gpu, model, ok=True, num_graphs=num_graphs)
    tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
    logger.error(
        "%s: pipeline_to_mo_graph exited with %d:\n%s",
        prefix,
        proc.returncode,
        tail,
    )
    return _Result(gpu, model, ok=False, num_graphs=num_graphs, error=tail)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    default="max-graphs",
    show_default=True,
    help="Root directory; graphs land in <output-dir>/<gpu>/<model>.",
)
@click.option(
    "--gpus",
    type=str,
    default=",".join(GPU_CONFIGS),
    show_default=True,
    help="Comma-separated CI GPU configs to cover.",
)
@click.option(
    "--target",
    type=str,
    default=None,
    help=(
        "Override the per-GPU virtual device target (e.g., cuda:sm_90, "
        "hip:gfx942) for all entries."
    ),
)
@click.option(
    "--models",
    type=str,
    default=None,
    help=(
        "Comma-separated case-insensitive globs matched against the CI model "
        "names (e.g. 'gemma*,deepseek*'). Default is all models."
    ),
)
@click.option(
    "--timeout",
    type=int,
    default=600,
    show_default=True,
    help="Per-model timeout in seconds.",
)
@click.option(
    "--jobs",
    type=int,
    default=8,
    show_default=True,
    help="Number of matrix entries to dump concurrently.",
)
def main(
    output_dir: str,
    gpus: str,
    target: str | None,
    models: str | None,
    timeout: int,
    jobs: int,
) -> None:
    """Dump build-only graphs for every CI smoke-test matrix entry."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    selected_gpus = [g.strip() for g in gpus.split(",") if g.strip()]
    unknown = sorted(set(selected_gpus) - set(GPU_CONFIGS))
    if unknown:
        raise click.UsageError(
            f"Unknown GPU config(s): {', '.join(unknown)}. Choose from: "
            f"{', '.join(GPU_CONFIGS)}."
        )

    globs = None
    if models:
        globs = [p.strip().lower() for p in models.split(",") if p.strip()]

    matrix: list[tuple[str, str]] = []
    skipped: set[str] = set()
    for gpu in selected_gpus:
        for model in sorted(MODELS):
            if not _matches(model, globs) or excluded("max-ci", gpu, model):
                continue
            if _is_skipped(model):
                skipped.add(model)
                continue
            matrix.append((gpu, model))

    for model in sorted(skipped):
        logger.info("skipping %s", model)

    if not matrix:
        raise click.UsageError(
            f"No matrix entries match gpus={gpus!r} models={models!r}."
        )

    root = Path(output_dir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    sweep = _Sweep(
        binary=_pipeline_to_mo_graph_binary(),
        root=root,
        target=target,
        timeout=timeout,
        total=len(matrix),
    )

    # Build every server command up front, single-threaded: get_server_cmd
    # mutates os.environ, which is not safe to do from the worker threads.
    # Entries that fail here are recorded directly and never reach the pool.
    results: list[_Result] = []
    tasks: list[tuple[int, str, str, list[str]]] = []
    for index, (gpu, model) in enumerate(matrix, 1):
        spec = _gpu_spec(gpu)
        try:
            server_cmd = get_server_cmd(
                "max-ci", model, gpu_spec=(spec.name, spec.num_gpus)
            )
        except Exception as e:
            logger.error(
                "[%d/%d] %s - %s: failed to build server command: %s",
                index,
                sweep.total,
                gpu,
                model,
                e,
            )
            results.append(_Result(gpu, model, ok=False, error=str(e)))
        else:
            tasks.append((index, gpu, model, server_cmd))

    # map preserves input order, so dumped results stay in matrix order; the
    # command-build failures above are grouped ahead of them in the summary.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results.extend(pool.map(lambda t: _dump_one(sweep, *t), tasks))

    failed = [r for r in results if not r.ok]
    click.echo(
        f"\nDumped graphs for {len(results) - len(failed)}/{len(results)} "
        f"matrix entries into {root}"
    )
    gpu_width = max(len(r.gpu) for r in results)
    model_width = max(len(r.model) for r in results)
    for r in results:
        status = "ok" if r.ok else "FAIL"
        line = (
            f"  {r.gpu:<{gpu_width}}  {r.model:<{model_width}}  "
            f"{status:<4}  {r.num_graphs} graphs"
        )
        if r.error:
            line += f"  {r.error.splitlines()[-1]}"
        click.echo(line)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
