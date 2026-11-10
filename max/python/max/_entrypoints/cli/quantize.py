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
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import requests
from max import tree
from max.engine import InferenceSession, Model
from max.engine.api import PrintStyle
from max.nn import Linear, Module
from max.nn._weight_utils import (
    _ExternalWeight,
    name_weights,
    shard_checkpoint,
)
from max.nn.common_layers.linear import QKVLinear
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.moe import MoE
from max.nn.common_layers.quant_linear import (
    QuantizedLinear,
    QuantizedMLP,
    QuantizedQKVLinear,
)
from max.nn.common_layers.quant_moe import QuantizedMoE
from max.nn.common_layers.quant_tensor import NVFP4TensorQuantizer
from max.nn.quant_config import QuantConfig, QuantFormat, WeightSource
from max.pipelines import PIPELINE_REGISTRY, TextGenerationPipelineInterface
from max.pipelines.context import (
    SamplingParams,
)
from max.pipelines.lib import (
    ModuleV3MultiGraphPipelineModelWithKVCache,
    ModuleV3PipelineModelWithKVCache,
    PipelineArgs,
    PipelineConfig,
)
from max.pipelines.logging_utils import log_basic_config
from max.pipelines.modeling.types import (
    Pipeline,
)
from max.pipelines.weights.checkpoint_files import (
    _host_tensor,
    _SavedTensor,
    copy_auxiliary_files,
    write_quantization_config,
    write_shards,
)
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    nvfp4_quantization_config,
)
from max.profiler import OneShotCapture, Tracer
from max.tensor import Tensor, compilation, defaults

from .generate import stream_text_to_console
from .metrics import TextGenerationMetrics
from .quant_error_report import JSON_NAME, clear_error_dumps, write_report

logger = logging.getLogger("max._entrypoints")

MODEL_NAME = "model"

_FUSED_GROUPS = (
    (MLP, ("gate_proj", "up_proj")),
    (MoE, ("gate_proj", "up_proj")),
    (QKVLinear, ("q_proj", "k_proj", "v_proj")),
)


def _child_path(parent: str, name: str) -> str:
    return f"{parent}.{name}" if parent else name


def _projection_hits(
    paths: list[str], patterns: list[re.Pattern[str]]
) -> list[str]:
    return [
        child
        for child in paths
        if any(pattern.search(child) for pattern in patterns)
    ]


def _require_full_group(
    names: tuple[str, ...], paths: list[str], hits: list[str]
) -> None:
    """Rejects a fused matmul whose members were only partly targeted."""
    if len(hits) == len(paths):
        return
    missing = [child for child in paths if child not in hits]
    joined = ", ".join(names)
    raise ValueError(
        f"{joined} are one matmul, so quantize all of them or "
        f"none of them. Matched {hits}; left out {missing}."
    )


def _routed_experts_match(
    path: str,
    moe: MoE,
    names: tuple[str, ...],
    patterns: list[re.Pattern[str]],
) -> bool:
    """Whether every routed expert's fused projections match ``patterns``.

    Experts live at ``experts.N.gate_proj``, not on the MoE itself. A MoE
    becomes one :class:`QuantizedMoE`, so a pattern that hits only some
    experts is rejected. No hits leaves the MoE unchanged.
    """
    matched = 0
    for index in range(len(moe.experts)):
        expert_path = _child_path(_child_path(path, "experts"), str(index))
        paths = [_child_path(expert_path, name) for name in names]
        hits = _projection_hits(paths, patterns)
        if not hits:
            continue
        _require_full_group(names, paths, hits)
        matched += 1
    if matched == 0:
        return False
    if matched != len(moe.experts):
        raise ValueError(
            "every expert's gate_proj and up_proj are one group, so "
            "quantize all experts or none of them. "
            f"Matched {matched} of {len(moe.experts)}."
        )
    return True


def _fused_modules_to_swap(
    module: Module[..., Any], patterns: Iterable[re.Pattern[str]]
) -> list[MLP | QKVLinear | MoE]:
    """Returns fusers whose members all match ``patterns``.

    Gate/up and q/k/v are each one matmul. A pattern that hits only part of
    a group has no shared global scale, so that is rejected before compile.
    A stacked ``QKVLinear`` is already one weight and is left alone. A MoE
    is swapped only when every routed expert's gate and up match; the MoE
    has no projections of its own.
    """
    compiled = list(patterns)
    swaps: list[MLP | QKVLinear | MoE] = []
    for cls, names in _FUSED_GROUPS:
        for path, node in tree.nodes(module, cls).items():
            if type(node) is not cls:
                continue
            if isinstance(node, QKVLinear) and not node.name_transparent:
                continue
            if cls is MoE:
                moe = cast(MoE, node)
                if _routed_experts_match(path, moe, names, compiled):
                    swaps.append(moe)
                continue
            paths = [_child_path(path, name) for name in names]
            hits = _projection_hits(paths, compiled)
            if not hits:
                continue
            _require_full_group(names, paths, hits)
            swaps.append(node)
    return swaps


def _finish_quantized_moe(moe: MoE, quant_config: QuantConfig) -> None:
    """Turns a built MoE into a QuantizedMoE without rebuilding its weights.

    The class swap skips ``QuantizedMoE.__init__``, which is where the mesh
    and the routed-expert config are stored. Forward reads both.
    """
    quantized = cast(QuantizedMoE, moe)
    quantized.__class__ = QuantizedMoE
    _, quantized.mesh = defaults()
    quantized.quant_config = quant_config
    quantized.swiglu_limit = 0.0


#: Formats whose on-the-fly quantizer can decode in the graph to measure
#: its own error.
_ERROR_REPORT_FORMATS = frozenset(
    {QuantFormat.NVFP4, QuantFormat.MXFP8, QuantFormat.MXFP4}
)


def insert_quantized_linears(
    module: Module[..., Any],
    targets: Iterable[str],
    quant_config: QuantConfig,
    *,
    verbose: bool = False,
) -> None:
    """Replaces every ``Linear`` matching ``targets`` with a quantized one.

    The config is supplied rather than built here: it is the caller's
    statement of intent. Gate/up and unstacked q/k/v are one matmul, so a
    full match swaps the parent onto the quant fuser. A MoE whose routed
    experts all match becomes a ``QuantizedMoE``. A partial match is
    rejected.

    With ``quant_config.report_error``, each quantizer is built to print its
    weight's quantization error from inside the graph, named after the
    weight. Only the on-the-fly block-scaled formats in
    :data:`_ERROR_REPORT_FORMATS` can measure their error.

    Raises:
        ValueError: If ``quant_config.report_error`` is set for a format that
            cannot measure its error.
    """
    if quant_config.report_error and (
        quant_config.format not in _ERROR_REPORT_FORMATS
    ):
        known = ", ".join(sorted(f.value for f in _ERROR_REPORT_FORMATS))
        raise ValueError(
            "error reporting is only implemented for "
            f"{known}; got {quant_config.format.value}"
        )
    patterns = [re.compile(pattern) for pattern in targets]
    swaps = _fused_modules_to_swap(module, patterns)
    count = 0
    for key, value in tree.nodes(module, Linear).items():
        if any(pattern.search(key) for pattern in patterns):
            wrapped = QuantizedLinear(
                int(value.in_dim),
                int(value.out_dim),
                bias=isinstance(value.bias, Tensor),
                quant_config=quant_config,
            )
            tree.update(
                module,
                {key: wrapped},
                leaf=lambda node, value=value: node is value,
            )
            count += 1
    for fused in swaps:
        if type(fused) is MLP:
            # QuantizedMLP does not subclass MLP, so this is not a type[MLP].
            fused.__class__ = cast(type[MLP], QuantizedMLP)
        elif type(fused) is QKVLinear:
            fused.__class__ = QuantizedQKVLinear
        elif type(fused) is MoE:
            _finish_quantized_moe(fused, quant_config)
    if verbose:
        print(module)
    logger.info(f"Quantizing {count} weight tensors.")


def nvfp4_host_tensors(
    path: str, leaves: Mapping[str, Tensor]
) -> list[tuple[str, _SavedTensor]]:
    """Packed weight and the three scale tensors, under ``path``."""
    return [
        (f"{path}.weight", _host_tensor(leaves["weight"])),
        (f"{path}.weight_scale", _host_tensor(leaves["weight_scale"])),
        (f"{path}.weight_scale_2", _host_tensor(leaves["weight_scale_2"])),
        (f"{path}.input_scale", _host_tensor(leaves["input_scale"])),
    ]


def _materialize_quantizers(
    module: Module[..., Any], state_dict: Mapping[str, Any]
) -> dict[str, dict[str, Tensor]]:
    """Runs the quantizers and returns packed weights and scales.

    The generate trace's quant tensors are graph values with no host bytes,
    so this graph produces the buffers that get written.
    """
    declared = tree.map(_ExternalWeight.like, module, leaf=Tensor, shared=True)
    name_weights(declared)
    registry = shard_checkpoint(declared, state_dict)

    def export_quants() -> dict[str, dict[str, Tensor]]:
        exported: dict[str, dict[str, Tensor]] = {}
        for path, quantizer in tree.nodes(
            declared, NVFP4TensorQuantizer
        ).items():
            quantized = quantizer.quant_tensor
            exported[path] = {
                "weight": quantized.weight,
                "weight_scale": quantized.weight_scale,
                "weight_scale_2": quantized.weight_scale_2,
                "input_scale": quantized.input_scale,
            }
        return exported

    # ``compile`` returns a function of input types. This graph takes none.
    compiled = compilation.compile(export_quants, weights=registry)()
    return compiled()


def export_quantized_checkpoint(model: Any, dest: Path) -> None:
    """Writes a new sharded checkpoint from the quantizers on ``model``.

    Quantized tensors and the weights left beside them are written under
    the source checkpoint's names. Sidecars are copied and the index
    describes the new files.
    """
    module = model.nn_module
    if not isinstance(module, Module):
        raise TypeError(
            f"{type(module).__name__} is not a ModuleV3 module, so its "
            "quantizers cannot be exported"
        )
    state_dict = model.final_state_dict
    ready = _materialize_quantizers(module, state_dict)
    if not ready:
        logger.info("No quantizers inserted, aborting export.")
        return

    saved: list[tuple[str, _SavedTensor]] = []
    for path, leaves in ready.items():
        checkpoint_path = state_dict[f"{path}.weight"].name.removesuffix(
            ".weight"
        )
        saved.extend(nvfp4_host_tensors(checkpoint_path, leaves))

    ignore = set()
    for name, value in state_dict.items():
        if name.removesuffix(".weight") not in ready:
            checkpoint_name = state_dict[f"{name}"].name
            saved.append((checkpoint_name, _host_tensor(value)))
            if "layers" in checkpoint_name:
                _, suffix = re.split(r"\.layers\.\d+\.", checkpoint_name)
                suffix = suffix.split(".")[0]
                prefix = checkpoint_name[: checkpoint_name.find(suffix)]
                ignore.add(f"{prefix}{suffix}*")

    src = Path(model.pipeline_config.model.resolved_weight_paths()[0]).parent
    dest.mkdir(parents=True, exist_ok=True)
    write_shards(dest, saved)
    write_quantization_config(
        src, dest, nvfp4_quantization_config(FP4Format.NVFP4, list(ignore))
    )
    copy_auxiliary_files(src, dest)
    logger.info("Exported %d quantized weights to %s", len(ready), dest)


#: The config each ``--quantization-format`` value selects.
#:
#: ``source`` is set here rather than by the caller: the command exists to
#: quantize a native-precision checkpoint, so CHECKPOINT would describe
#: bytes that are not there.
#:
#: The layer tuples are empty because ``QuantConfig`` selects layers by
#: index and this command selects them by name instead:
#: ``insert_quantized_linears`` wraps whatever matches ``--targets``.
#:
#: A row here is necessary but not sufficient -- ``quant_ops`` must have a
#: quantizer registered for the format too, or building the weight raises.
_ON_THE_FLY: dict[str, QuantConfig] = {
    "nvfp4": dataclasses.replace(
        QuantConfig.nvfp4(mlp_quantized_layers=(), attn_quantized_layers=()),
        source=WeightSource.QUANTIZE,
    ),
    "mxfp8": dataclasses.replace(
        QuantConfig.mxfp8(mlp_quantized_layers=(), attn_quantized_layers=()),
        source=WeightSource.QUANTIZE,
    ),
    "mxfp4": dataclasses.replace(
        QuantConfig.mxfp4(mlp_quantized_layers=(), attn_quantized_layers=()),
        source=WeightSource.QUANTIZE,
    ),
}


def quant_formats() -> tuple[str, ...]:
    """The values ``--quantization-format`` accepts."""
    return tuple(sorted(_ON_THE_FLY))


def on_the_fly_config(name: str, *, report_error: bool = False) -> QuantConfig:
    """The config ``--quantization-format`` selects.

    With ``report_error``, the quantizers built from it print their
    quantization error from inside the graph.

    Raises:
        ValueError: if ``name`` is not a format this command can produce.
    """
    try:
        config = _ON_THE_FLY[name]
    except KeyError:
        raise ValueError(
            f"unknown quantization format {name!r}; this command can "
            f"produce {', '.join(quant_formats())}"
        ) from None
    # A copy, because QuantConfig is mutable and the table is shared.
    return dataclasses.replace(config, report_error=report_error)


def _enable_error_prints(
    session: InferenceSession, directory: Path | None
) -> None:
    """Has ``session`` dump the graph's prints, which carry the error stats.

    Must run before the graphs compile. Dumps from an earlier run are
    removed, so the report holds only this run's weights. A no-op without a
    directory.
    """
    if directory is None:
        return
    clear_error_dumps(directory)
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=directory
    )


def _single_graph_quantizing_model(
    model_cls: type[Any],
    targets: Iterable[str],
    quant_config: QuantConfig,
    verbose: bool,
    error_report_dir: Path | None,
) -> type[Any]:
    """Returns a subclass that quantizes what ``_instantiate_module`` builds.

    Single-graph ``graph_specs`` calls ``_instantiate_module`` and never
    ``_language_spec``.
    """
    # mypy only accepts a class as a base. The runtime base is the concrete
    # pipeline model retrieved for this checkpoint.
    if TYPE_CHECKING:
        base = ModuleV3PipelineModelWithKVCache[Any, Any]
    else:
        base = model_cls

    class QuantizingPipelineModel(base):
        def _instantiate_module(self, model_config: Any) -> Any:
            nn_model = super()._instantiate_module(model_config)
            insert_quantized_linears(
                nn_model,
                targets,
                quant_config,
                verbose=verbose,
            )
            self.nn_module = nn_model
            return nn_model

        @property
        def final_state_dict(self) -> dict[str, Any]:
            state_dict = self._load_state_dict()
            model_config = self._create_model_config(state_dict)
            return self._prepare_state_dict(state_dict, model_config)

        def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
            _enable_error_prints(session, error_report_dir)
            return super().load_graphs(session)

    return QuantizingPipelineModel


def _multi_graph_quantizing_model(
    model_cls: type[Any],
    targets: Iterable[str],
    quant_config: QuantConfig,
    verbose: bool,
    error_report_dir: Path | None,
) -> type[Any]:
    """Returns a subclass that quantizes what ``_language_spec`` builds.

    Multi-graph ``graph_specs`` calls ``_language_spec`` and never
    ``_instantiate_module``.
    """
    if TYPE_CHECKING:
        base = ModuleV3MultiGraphPipelineModelWithKVCache[Any, Any]
    else:
        base = model_cls

    class QuantizingPipelineModel(base):
        def _language_spec(
            self, model_config: Any, state_dict: dict[str, Any]
        ) -> Any:
            spec = super()._language_spec(model_config, state_dict)
            language = spec.module
            if not isinstance(language, Module):
                raise TypeError(
                    f"{type(language).__name__} is not a ModuleV3 module, "
                    "so layer quantizers cannot be inserted"
                )
            insert_quantized_linears(
                language,
                targets,
                quant_config,
                verbose=verbose,
            )
            self.nn_module = language
            return spec

        @property
        def final_state_dict(self) -> dict[str, Any]:
            _ = self._load_state_dict()
            return self._language_weights_dict

        def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
            _enable_error_prints(session, error_report_dir)
            return super().load_graphs(session)

    return QuantizingPipelineModel


def _factory_with_layer_quantizers(
    factory: Callable[[], Any],
    targets: Iterable[str],
    quant_config: QuantConfig,
    *,
    verbose: bool = False,
    error_report_dir: Path | None = None,
) -> Callable[[], Any]:
    """Returns a factory that quantizes layers before the module is compiled.

    ``retrieve_factory`` binds the pipeline-model class into the factory.
    The subclass overrides the one hook ``graph_specs`` calls for that
    class: ``_instantiate_module`` for a single graph, ``_language_spec``
    for a multi-graph model.
    """
    if not isinstance(factory, functools.partial):
        raise TypeError(
            "pipeline factory must be a functools.partial so its "
            "pipeline_model class can be replaced"
        )
    keywords = dict(factory.keywords or {})
    model_cls = keywords["pipeline_model"]
    if issubclass(model_cls, ModuleV3MultiGraphPipelineModelWithKVCache):
        pipeline_model = _multi_graph_quantizing_model(
            model_cls, targets, quant_config, verbose, error_report_dir
        )
    elif issubclass(model_cls, ModuleV3PipelineModelWithKVCache):
        pipeline_model = _single_graph_quantizing_model(
            model_cls, targets, quant_config, verbose, error_report_dir
        )
    else:
        raise TypeError(
            f"{model_cls.__name__} is not a ModuleV3 pipeline model, so "
            "layer quantizers cannot be inserted"
        )
    pipeline_model.__name__ = f"Quantizing{model_cls.__name__}"
    pipeline_model.__qualname__ = pipeline_model.__name__
    keywords["pipeline_model"] = pipeline_model
    return functools.partial(factory.func, *factory.args, **keywords)


def quantize_pipeline(
    pipeline_args: PipelineArgs,
    sampling_params: SamplingParams,
    targets: Iterable[str],
    prompt: str,
    quantization_format: str,
    verbose: bool = False,
    image_urls: Iterable[str] = (),
    num_warmups: int = 0,
    profile: bool = False,
    profile_top_n: int = 15,
    error_report_dir: Path | None = None,
    export_path: Path | None = None,
) -> None:
    """Quantizes the targeted layers on load, then generates text.

    With ``error_report_dir``, every quantized weight's error against its
    native-precision source is measured inside the compiled graph and
    printed as the graph runs. The report is built from the files that run
    leaves behind.
    """
    quant_config = on_the_fly_config(
        quantization_format, report_error=error_report_dir is not None
    )
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    retrieved = PIPELINE_REGISTRY.retrieve_factory(pipeline_config)
    tokenizer = retrieved.tokenizer
    pipeline = _factory_with_layer_quantizers(
        retrieved.factory,
        targets,
        quant_config,
        verbose=verbose,
        error_report_dir=error_report_dir,
    )()
    assert isinstance(pipeline, Pipeline)
    log_basic_config(pipeline_config, memory_plan=retrieved.memory_plan)
    logger.info("max_batch_size: %d", pipeline.max_batch_size)

    capture = OneShotCapture(top_n=profile_top_n) if profile else None
    try:
        # Run timed run & print results.
        with TextGenerationMetrics(print_report=True) as metrics:
            if image_urls:
                logger.info("Downloading images")
                images = [requests.get(url).content for url in image_urls]
            else:
                images = []

            if num_warmups > 0:
                logger.info("Running warmup")
                warmup_params = dataclasses.replace(
                    sampling_params, max_new_tokens=num_warmups
                )
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        sampling_params=warmup_params,
                        metrics=None,
                        print_tokens=False,
                    )
                )

            # Run and print results.
            logger.info("Beginning text generation")

            with contextlib.ExitStack() as exit_stack:
                if capture is not None:
                    capture.start()
                    # ``Tracer`` adds an NVTX "inference" label visible in
                    # nsys-ui. It must be entered after ``capture.start``
                    # so its NVTX range sits inside the cuda profiler
                    # window.
                    exit_stack.enter_context(
                        Tracer("inference", color="modular_purple")
                    )
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        sampling_params=sampling_params,
                        metrics=metrics,
                        print_tokens=True,
                    )
                )
        # ``TextGenerationMetrics.__exit__`` has printed the report here.
        # End the capture *now*, so nsys's file-writing output follows.
    finally:
        if capture is not None:
            capture.end_and_finalize()
    if error_report_dir is not None:
        write_report(error_report_dir)
        logger.info(f"Wrote {error_report_dir / JSON_NAME}")
    if export_path is not None:
        assert isinstance(pipeline, TextGenerationPipelineInterface)
        export_quantized_checkpoint(pipeline._pipeline_model, export_path)
