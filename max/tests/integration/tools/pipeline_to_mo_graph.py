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

"""Dump the graphs a pipeline generates, without serving it.

Accepts the same model and pipeline flags as max serve. Instead of starting a
server, compiles the pipeline with IR dumping enabled and writes the generated
graphs as per-stage MLIR files into --output-dir. Each MAX graph as built by
the pipeline lands as <graph-names>.mo.mlir, where the stem is the graph's
top-level graph names joined by +; later compiler stages land alongside
it. With --build-only, graphs are dumped as soon as they are built and the
graph compiler never runs, so only the *.mo.mlir files are produced.

Graphs are emitted with debug info, and every op carries the Python source
location it was built from, so the dumps trace each op back to the pipeline
code that created it.

Compilation always targets virtual devices (``--target`` is required), so no
physical hardware is needed. Unrecognized serve-only flags are ignored, so a
``max serve`` command line can be passed through verbatim.

Run with:
    ./bazelw run //max/tests/integration/tools:pipeline_to_mo_graph -- \\
        --model <model> --target cuda:sm_90 --output-dir /tmp/graphs
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

# Capture the Python call stack on each op during graph construction so the
# dumped graphs carry source locations back to the building Python code. The
# flag is read once when max.graph is imported, so set it before that import.
os.environ.setdefault("MODULAR_MAX_DEBUG_SOURCE_TRACEBACKS", "true")

import click
from max.engine import CompiledModel, CustomExtensionsType, InferenceSession
from max.entrypoints.cli import pipeline_config_options
from max.entrypoints.cli.entrypoint import configure_cli_logging
from max.graph import Graph, Module
from max.pipelines import PIPELINE_REGISTRY, PipelineConfig


class _DumpedArtifact:
    """Stands in for a `CompiledModel` when build-only skips compilation.

    Exposes the same `_graph_names` attribute `CompiledModel` provides so the
    engine's real `InferenceSession.init_all` virtual-device path can build one
    mock `Model` per graph from it, letting pipeline construction proceed.
    """

    def __init__(self, graph_names: tuple[str, ...]) -> None:
        self._graph_names = graph_names


@contextlib.contextmanager
def _intercept_graph_compilation(output_dir: Path) -> Iterator[None]:
    """Patches `InferenceSession.compile` to dump graphs instead of compiling.

    While active, `compile` writes each `Graph` or `Module` it receives to
    `output_dir` as `<graph-names>.mo.mlir`, where the stem is the module's
    top-level graph names joined by + (or `graph` when there are none), and
    returns a stub artifact without invoking the
    graph compiler. The tool runs in virtual-device mode, so the engine's own
    `init_all` resolves the stub to one mock `Model` per graph; pipeline
    construction proceeds and graphs built after the main model (samplers,
    logprobs processors) are still dumped.
    """
    original_compile = InferenceSession.compile

    def compile_to_dump(
        self: InferenceSession,
        model: str | Path | Module | Graph,
        *,
        custom_extensions: CustomExtensionsType | None = None,
    ) -> CompiledModel | _DumpedArtifact:
        if isinstance(model, Graph):
            module = model.module
        elif isinstance(model, Module):
            module = model
        else:
            # A path to a pre-compiled artifact has no MO graph to dump.
            return original_compile(
                self, model, custom_extensions=custom_extensions
            )
        graph_names = tuple(module.top_level_graph_names())
        base_name = "+".join(graph_names) or "graph"
        path = output_dir / f"{base_name}.mo.mlir"
        # Emit with source locations so each op shows where it was built in the
        # pipeline's Python code, which the graph compiler would otherwise add.
        path.write_text(module._to_mlir_str(source_locations=True))
        logging.info("Dumped %s without compiling", path)
        return _DumpedArtifact(graph_names)

    with mock.patch.object(InferenceSession, "compile", compile_to_dump):
        yield


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    default="max-graphs",
    show_default=True,
    help="Directory into which the generated graph files are written.",
)
@click.option(
    "--target",
    type=str,
    required=True,
    help=(
        "Target API and architecture to compile for (e.g., cuda, cuda:sm_90, "
        "hip:gfx942). Compilation uses virtual devices, so no physical "
        "hardware is required."
    ),
)
@click.option(
    "--build-only",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Dump only the MAX graphs as built by the pipeline (*.mo.mlir), "
        "skipping the graph compiler and weight loading entirely. Much "
        "faster, but produces no later-stage dumps."
    ),
)
@click.argument("ignored_serve_args", nargs=-1, type=click.UNPROCESSED)
@pipeline_config_options
def main(
    output_dir: str,
    target: str,
    build_only: bool,
    ignored_serve_args: tuple[str, ...],
    **config_kwargs: Any,
) -> None:
    """Build and compile the model as `serve` would, dumping its graphs."""
    configure_cli_logging(level="INFO")

    if ignored_serve_args:
        logging.info(
            "Ignoring serve-only arguments: %s", " ".join(ignored_serve_args)
        )

    # Force the compiled-model cache off: a cache hit would skip graph
    # compilation entirely and nothing would be dumped.
    os.environ["MODULAR_MAX_ENABLE_MODEL_IR_CACHE"] = "false"

    # Virtual devices are enabled by pipeline_config_options when --target is
    # set; report the target here for context.
    logging.info("Compiling for target %s using virtual devices", target)

    out_dir = Path(output_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = PipelineConfig(**config_kwargs)
    if build_only:
        with _intercept_graph_compilation(out_dir):
            PIPELINE_REGISTRY.retrieve(pipeline_config)
    else:
        InferenceSession.debug.ir_output_dir = str(out_dir)
        PIPELINE_REGISTRY.retrieve(pipeline_config)

    dumped = sorted(p.name for p in out_dir.iterdir())
    if not dumped:
        raise click.ClickException(
            f"Pipeline was built but no graph files were written to {out_dir}."
        )
    click.echo(f"Wrote {len(dumped)} graph files to {out_dir}:")
    for name in dumped:
        click.echo(f"  {name}")


if __name__ == "__main__":
    main()
