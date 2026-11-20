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
import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from max import tree
from max.driver import CPU, Accelerator, accelerator_count
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
from max.nn.common_layers.quant_tensor import (
    MXFP4Quantizer,
    MXFP8Quantizer,
    NVFP4TensorQuantizer,
)
from max.nn.quant_config import QuantConfig, QuantFormat, WeightSource
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.context import (
    SamplingParams,
)
from max.pipelines.lib import (
    GraphRole,
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
    ShardWriter,
    _host_tensor,
    _SavedTensor,
    copy_auxiliary_files,
    write_quantization_config,
)
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    nvfp4_quantization_config,
)
from max.profiler import OneShotCapture, Tracer
from max.tensor import Tensor, compilation
from max.tensor.support import _session

from .generate import stream_text_to_console
from .metrics import TextGenerationMetrics
from .quant_error_report import JSON_NAME, clear_error_dumps, write_report

logger = logging.getLogger("max._entrypoints")

MODEL_NAME = "model"

# MoE before MLP. Rebuilding a MoE constructs new experts, so the MLP walk
# must run after that and see those experts rather than the ones discarded.
_FUSED_GROUPS = (
    (MoE, ("gate_proj", "up_proj")),
    (MLP, ("gate_proj", "up_proj")),
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


def _can_swap_fused_module(
    cls: type[Any],
    names: tuple[str, ...],
    path: str,
    node: Any,
    patterns: list[re.Pattern[str]],
) -> bool:
    """Whether ``node`` is an exact ``cls`` whose fused members all match.

    Gate/up and q/k/v are each one matmul. A pattern that hits only part of
    a group has no shared global scale, so that is rejected. A stacked
    ``QKVLinear`` is already one weight. A MoE is swapped only when every
    routed expert's gate and up match.
    """
    if type(node) is not cls:
        return False
    if isinstance(node, QKVLinear) and not node.name_transparent:
        return False
    if cls is MoE and isinstance(node, MoE):
        return _routed_experts_match(path, node, names, patterns)
    child_paths = [_child_path(path, name) for name in names]
    hits = _projection_hits(child_paths, patterns)
    if not hits:
        return False
    _require_full_group(names, child_paths, hits)
    return True


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
) -> Module[Any, Any]:
    """Quantizes every weight matching ``targets``.

    A full gate/up, q/k/v, or expert group is rebuilt as the quant container
    first, so its forward fuses those members. Each matching projection,
    including one the model built as its own module around a plain ``[out,
    in]`` tensor, is then constructed as a :class:`QuantizedLinear`. A
    projection that is not a common layer logs a warning. Weights-only
    export still writes those weights, and on-the-fly quantization is
    likely to break. A partial fused group is rejected. The constructor's
    weight is the parameter the checkpoint fills. Its values stay the
    constructor's zeros, and it takes the replaced linear's device map.

    With ``quant_config.report_error``, each quantizer is built to print its
    weight's quantization error from inside the graph, named after the
    weight. Only the on-the-fly block-scaled formats in
    :data:`_ERROR_REPORT_FORMATS` can measure their error.

    Returns:
        ``module``. A new object when ``module`` itself was the container:
        a constructor cannot change the object the caller already holds.

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
    count = 0
    # The container is built first, with plain linear children. The loop
    # below is what gives a matching child its quantizer, so an untargeted
    # sibling (down_proj, a shared expert) stays a plain tensor.
    for cls, names in _FUSED_GROUPS:
        containers: dict[str, Module[Any, Any]] = {}
        container_ids: set[int] = set()
        for path, node in tree.nodes(module, cls).items():
            if not _can_swap_fused_module(cls, names, path, node, patterns):
                continue
            built: Module[Any, Any]
            if type(node) is MLP:
                built = QuantizedMLP(
                    int(node.hidden_dim),
                    int(node.feed_forward_length),
                    bias=bool(node.has_bias),
                    activation_function=node.activation_function,
                )
            elif type(node) is QKVLinear:
                built = QuantizedQKVLinear(
                    int(node.q_proj.in_dim),
                    node.q_dim,
                    node.kv_dim,
                    bias=node._has_bias,
                )
            elif type(node) is MoE:
                shared = node.shared_experts
                built = QuantizedMoE(
                    hidden_dim=int(node.hidden_dim),
                    num_experts=node.num_experts,
                    num_experts_per_token=node.num_experts_per_token,
                    moe_dim=int(node.moe_dim),
                    has_shared_experts=shared is not None,
                    shared_experts_dim=(
                        int(shared.feed_forward_length)
                        if shared is not None
                        else 0
                    ),
                )
            else:
                raise TypeError(
                    f"{type(node).__name__} is not a fused container this "
                    "command replaces"
                )
            containers[path] = built
            container_ids.add(id(node))
        # The root is not a child, so a container built in its place
        # comes back as the return value. One update per container type,
        # after MoE, so an MLP walk does not rebuild experts the new MoE
        # already replaced.
        if not containers:
            continue
        updated = tree.update(
            module,
            containers,
            leaf=lambda found, ids=container_ids: id(found) in ids,
        )
        if not isinstance(updated, Module):
            raise TypeError(f"rebuilt {type(module).__name__} is not a Module")
        module = updated
    # One walk for a Linear, a QuantizedLinear still holding a plain
    # tensor, and a projection the model built as its own module. The
    # container updates above have already landed, so this sees those new
    # children. A weight that is already a quantizer is not a Tensor and
    # is skipped. The tree cannot be edited while it is walked, so the
    # replacements are applied once afterwards.
    replacements: dict[str, QuantizedLinear] = {}
    replacing: set[int] = set()
    for path, node in tree.nodes(module, Module).items():
        weight = getattr(node, "weight", None)
        if type(weight) is not Tensor or weight.rank != 2:
            continue
        if not any(pattern.search(path) for pattern in patterns):
            continue
        if type(node) not in (Linear, QuantizedLinear):
            logger.warning(
                "Replaced a custom %s with a common QuantizedLinear. "
                "This should be ok for weights-only export but is likely "
                "to break on-the-fly quantization.",
                type(node).__name__,
            )
        if hasattr(node, "in_dim") and hasattr(node, "out_dim"):
            in_dim, out_dim = int(node.in_dim), int(node.out_dim)
        else:
            in_dim, out_dim = int(weight.shape[1]), int(weight.shape[0])
        bias = getattr(node, "bias", None)
        wrapped = QuantizedLinear(
            in_dim,
            out_dim,
            bias=isinstance(bias, Tensor),
            quant_config=quant_config,
        )
        # The zeros are the parameter. The map is the one the model already
        # gave this weight, including a tensor-parallel shard or an expert's
        # device. Bias is sharded on its own map, and QuantizedLinear.to
        # leaves a plain bias where it was constructed.
        wrapped.weight = wrapped.weight.to(weight.mapping)
        if isinstance(bias, Tensor) and isinstance(wrapped.bias, Tensor):
            wrapped.bias = wrapped.bias.to(bias.mapping)
        replacements[path] = wrapped
        replacing.add(id(node))
        count += 1
    if replacements:
        updated = tree.update(
            module,
            replacements,
            leaf=lambda found: id(found) in replacing,
        )
        if not isinstance(updated, Module):
            raise TypeError(f"rebuilt {type(module).__name__} is not a Module")
        module = updated
    if verbose:
        print(module)
    logger.info(f"Quantizing {count} weight tensors.")
    return module


# Leaves each on-the-fly quantizer writes, in checkpoint order. The name is
# the safetensors suffix: ``weight`` becomes ``{path}.weight``.
_QUANTIZER_LEAVES: tuple[tuple[type[Any], tuple[str, ...]], ...] = (
    (
        NVFP4TensorQuantizer,
        ("weight", "weight_scale", "weight_scale_2", "input_scale"),
    ),
    (MXFP8Quantizer, ("weight", "weight_scale")),
    (MXFP4Quantizer, ("weight", "weight_scale")),
)


def quantized_host_tensors(
    path: str, leaves: Mapping[str, Tensor]
) -> list[tuple[str, _SavedTensor]]:
    """The quantized weight and its scale tensors, under ``path``."""
    return [
        (f"{path}.{name}", _host_tensor(tensor))
        for name, tensor in leaves.items()
    ]


def checkpoint_quantization_config(
    fmt: str, ignored: Sequence[str]
) -> dict[str, object]:
    """The ``quantization_config`` block ``fmt``'s loader already reads.

    MXFP8 is modelopt: encoding detection reads ``quantization.quant_algo``,
    and the layer parser reads ``ignore`` from that same nested section.
    MXFP4's parser keys only on ``quant_method`` and quantizes every MLP,
    leaving attention in bf16.
    """
    if fmt == "nvfp4":
        return nvfp4_quantization_config(FP4Format.NVFP4, ignored)
    if fmt == "mxfp8":
        patterns = list(ignored)
        return {
            "quant_method": "modelopt",
            "quantization": {"quant_algo": "MXFP8", "ignore": patterns},
        }
    if fmt == "mxfp4":
        return {"quant_method": "mxfp4"}
    raise ValueError(
        f"unknown quantization format {fmt!r}; this command can "
        f"produce {', '.join(quant_formats())}"
    )


def _ignore_pattern(name: str, quantized_stems: Iterable[str]) -> str | None:
    """The ignore glob for one tensor that was copied unchanged.

    A layer subtree that also holds a quantized weight keeps this tensor's
    own module, so ``mlp.gate*`` does not unquantize the experts beside it.
    A subtree with nothing quantized is ignored whole, as ``self_attn*``.
    """
    match = re.search(r"^(.*\.layers\.\d+\.)([^.]+)", name)
    if match is None:
        return None
    top = f"{match.group(1)}{match.group(2)}"
    if any(stem.startswith(f"{top}.") for stem in quantized_stems):
        return f"{name.rsplit('.', 1)[0]}*"
    return f"{top}*"


def _state_key(path: str) -> str:
    """The checkpoint key for a quantizer found at ``path``."""
    return path if path.endswith(".weight") else f"{path}.weight"


def _source_files(model: Any) -> dict[str, str]:
    """Maps a checkpoint tensor name to the safetensors file holding it."""
    src = Path(model.pipeline_config.model.resolved_weight_paths()[0]).parent
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file():
        return {}
    weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    return weight_map if isinstance(weight_map, dict) else {}


def _release_cached_weight(model: Any, checkpoint_name: str) -> None:
    """Drops one safetensors view from the loader cache."""
    cache = getattr(getattr(model, "weights", None), "_st_weight_map", None)
    if isinstance(cache, dict):
        cache.pop(checkpoint_name, None)


def _module_at_paths(quantizers: Mapping[str, Any]) -> Module[Any, Any]:
    """A module whose parameters sit at the same paths as ``quantizers``."""
    root: Module[Any, Any] = Module()
    for path, quantizer in quantizers.items():
        cursor = root
        parts = path.split(".")
        for part in parts[:-1]:
            child = getattr(cursor, part, None)
            if not isinstance(child, Module):
                child = Module()
                setattr(cursor, part, child)
            cursor = child
        setattr(cursor, parts[-1], quantizer)
    return root


@contextlib.contextmanager
def _place_host_quantizers(module: Module[..., Any]) -> Iterator[None]:
    """Moves host-only quantizers onto a GPU for the body of the block.

    The block-scaled quantize kernel runs on a GPU. MiniMax keeps each
    routed expert on the host until ``stacked()`` places it. With no
    accelerator the weights stay where they are. The wrapped tensor is
    moved back to the host when the block ends.
    """
    placed: list[Any] = []
    gpu: Accelerator | None = None
    if accelerator_count() == 0:
        yield
        return
    for cls, _names in _QUANTIZER_LEAVES:
        for quantizer in tree.nodes(module, cls).values():
            tensor = getattr(quantizer, "_tensor", None)
            if not isinstance(tensor, Tensor) or not all(
                isinstance(device, CPU)
                for device in tensor.mapping.mesh.devices
            ):
                continue
            if gpu is None:
                gpu = Accelerator(0)
            placed.append(quantizer)
            quantizer.to(gpu)
    try:
        yield
    finally:
        for quantizer in placed:
            quantizer.to(CPU())


def _materialize_quantizers(
    module: Module[..., Any],
    state_dict: Mapping[str, Any],
) -> dict[str, dict[str, Tensor]]:
    """Runs the quantizers and returns packed weights and scales.

    The generate trace's quant tensors are graph values with no host bytes,
    so this graph produces the buffers that get written.
    """
    with _place_host_quantizers(module):
        declared = tree.map(
            _ExternalWeight.like, module, leaf=Tensor, shared=True
        )
        name_weights(declared)
        registry = shard_checkpoint(declared, state_dict)

        def export_quants() -> dict[str, dict[str, Tensor]]:
            exported: dict[str, dict[str, Tensor]] = {}
            for cls, names in _QUANTIZER_LEAVES:
                for path, quantizer in tree.nodes(declared, cls).items():
                    quantized = quantizer.quant_tensor
                    exported[path] = {
                        name: getattr(quantized, name) for name in names
                    }
            return exported

        # ``compile`` returns a function of input types. This graph takes none.
        # The session is the process-wide one: the weights carry their devices.
        compiled = compilation.compile(export_quants, weights=registry)()
        try:
            return compiled()
        finally:
            del compiled


def export_quantized_checkpoint(
    model: Any, dest: Path, quantization_format: str
) -> None:
    """Writes a new sharded checkpoint from the quantizers on ``model``.

    Quantizers are grouped by the safetensors file that holds their source
    weight. Each file is quantized, written, and dropped from the loader
    cache before the next file is read. Tensors left unquantized are copied
    under their original names. Anything not on the decoder is copied from
    the source shard unchanged.
    """
    module = model.nn_module
    if not isinstance(module, Module):
        raise TypeError(
            f"{type(module).__name__} is not a ModuleV3 module, so its "
            "quantizers cannot be exported"
        )
    state_dict = dict(model.final_state_dict)
    # ``found`` is what ``ready`` will be, before this file is quantized.
    found: list[tuple[str, Any, str]] = []
    for cls, _names in _QUANTIZER_LEAVES:
        for path, quantizer in tree.nodes(module, cls).items():
            state_key = _state_key(path)
            found.append((path, quantizer, state_dict[state_key].name))
    if not found:
        logger.info("No quantizers inserted, aborting export.")
        return

    files = _source_files(model)
    quantized_keys = {_state_key(path) for path, _, _ in found}
    quantized_stems = [name.removesuffix(".weight") for _, _, name in found]

    def source_file(checkpoint_name: str) -> str:
        return str(files.get(checkpoint_name, ""))

    groups: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for path, quantizer, checkpoint_name in found:
        groups[source_file(checkpoint_name)].append((path, quantizer))
    copied: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for key, value in state_dict.items():
        if key not in quantized_keys:
            copied[source_file(value.name)].append((key, value))

    src = Path(model.pipeline_config.model.resolved_weight_paths()[0]).parent
    shards = set(files.values())
    writer = ShardWriter(dest)
    ignore: set[str] = set()
    exported = 0
    for filename in sorted(set(groups) | set(copied) | shards):
        batch = groups.get(filename, [])
        ready: dict[str, dict[str, Tensor]] = {}
        if batch:
            logger.info(
                "Quantizing %s (%d weights).",
                filename or "checkpoint",
                len(batch),
            )
            ready = _materialize_quantizers(
                _module_at_paths(dict(batch)), state_dict
            )
            exported += len(ready)
        saved: dict[str, _SavedTensor] = {}
        for path, leaves in ready.items():
            source = state_dict[_state_key(path)]
            checkpoint_path = source.name.removesuffix(".weight")
            for name, tensor in quantized_host_tensors(checkpoint_path, leaves):
                saved[name] = tensor
            _release_cached_weight(model, source.name)
        if filename in shards:
            for name in writer.copy_except(src / filename, saved):
                _release_cached_weight(model, name)
        else:
            for name, tensor in saved.items():
                writer.add(name, tensor)
        for key, value in copied.get(filename, []):
            if value.name not in files:
                writer.add(value.name, _host_tensor(value))
            pattern = _ignore_pattern(value.name, quantized_stems)
            if pattern is not None:
                ignore.add(pattern)
            _release_cached_weight(model, value.name)
            del state_dict[key]
        del ready
    writer.close()

    write_quantization_config(
        src,
        dest,
        checkpoint_quantization_config(quantization_format, sorted(ignore)),
    )
    copy_auxiliary_files(src, dest)
    logger.info("Exported %d quantized weights to %s", exported, dest)


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


def on_the_fly_config(
    name: str, *, report_error: bool = False, mse_sweep_scales: bool = False
) -> QuantConfig:
    """The config ``--quantization-format`` selects.

    With ``report_error``, the quantizers built from it print their
    quantization error from inside the graph. ``mse_sweep_scales`` is read
    by NVFP4 quantizers: each block's scale is the multiplier of its abs-max
    with the smallest quantization MSE.

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
    return dataclasses.replace(
        config, report_error=report_error, mse_sweep_scales=mse_sweep_scales
    )


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


def _graphs_without_compile(
    model: Any, session: InferenceSession
) -> dict[str, Model]:
    """Builds the graphs' modules and does not compile them.

    ``_load_decoder`` and ``_load_towers`` read ``compiled_modules`` after
    this returns, so the uncompiled modules stand in for the engine models.
    The decoder ``nn_module`` is the one the insert hook stored. The export
    state dict is the decoder. A tower that is not quantized is copied from
    the source shard.
    """
    model._session = session
    state_dict = model._load_state_dict()
    model_config = model._create_model_config(state_dict)
    specs = model.graph_specs(model_config)
    exported: dict[str, Any] = {}
    decoder = specs.get(GraphRole.DECODER)
    if getattr(decoder, "weights", None):
        exported.update(decoder.weights)
    model._export_state_dict = exported
    model.compiled_modules = {role: spec.module for role, spec in specs.items()}
    return {}


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
            # Sparse attention reads this while the decoder is built, after
            # expert-parallel init, so a bf16 checkpoint still dispatches in
            # bf16. The on-the-fly layer tuples are empty, so the projections
            # stay plain tensors for the wrap below. V3 treats a present
            # config as "quantize these layers", so a bf16 V3 checkpoint
            # keeps None.
            if getattr(model_config, "quant_config", None) is None and hasattr(
                model_config, "index_n_heads"
            ):
                model_config.quant_config = quant_config
            nn_model = insert_quantized_linears(
                super()._instantiate_module(model_config),
                targets,
                quant_config,
                verbose=verbose,
            )
            self.nn_module = nn_model
            return nn_model

        @property
        def final_state_dict(self) -> dict[str, Any]:
            prepared = getattr(self, "_export_state_dict", None)
            if prepared is not None:
                return prepared
            state_dict = self._load_state_dict()
            model_config = self._create_model_config(state_dict)
            return self._prepare_state_dict(state_dict, model_config)

        def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
            if self.pipeline_config.export_only:
                return _graphs_without_compile(self, session)
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
            language = insert_quantized_linears(
                language,
                targets,
                quant_config,
                verbose=verbose,
            )
            self.nn_module = language
            if language is not spec.module:
                spec = dataclasses.replace(spec, module=language)
            return spec

        @property
        def final_state_dict(self) -> dict[str, Any]:
            prepared = getattr(self, "_export_state_dict", None)
            if prepared is not None:
                return prepared
            # ``_load_state_dict`` splits the checkpoint and leaves the
            # decoder weights on ``_language_weights_dict``.
            self._load_state_dict()
            return self._language_weights_dict

        def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
            if self.pipeline_config.export_only:
                return _graphs_without_compile(self, session)
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
    mse_sweep_scales: bool = False,
) -> None:
    """Quantizes the targeted layers on load, then generates text.

    With ``--export-path``, the model is built and exported without a
    generate graph. With ``error_report_dir``, every quantized weight's
    error against its native-precision source is measured inside the
    compiled graph and printed as the graph runs.
    """
    quant_config = on_the_fly_config(
        quantization_format,
        report_error=error_report_dir is not None,
        mse_sweep_scales=mse_sweep_scales,
    )
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    if export_path is not None:
        pipeline_config = pipeline_config.model_copy(
            update={"export_only": True}
        )
    retrieved = PIPELINE_REGISTRY.retrieve_factory(
        pipeline_config, plan_memory=export_path is None
    )
    factory = _factory_with_layer_quantizers(
        retrieved.factory,
        targets,
        quant_config,
        verbose=verbose,
        error_report_dir=error_report_dir,
    )
    # Export uses this same pipeline. Its constructor returns once the model
    # exists, before the KV cache and the sampler.
    pipeline = factory()
    if export_path is not None:
        logger.info(
            "Exporting quantized weights without generation to %s",
            export_path,
        )
        # Export compiles shard graphs on the process-wide session and
        # returns from load_graphs before that session's prints are armed.
        # Arm them here, before those graphs compile, so write_report reads
        # this run.
        _enable_error_prints(_session(), error_report_dir)
        export_quantized_checkpoint(
            pipeline._pipeline_model, export_path, quantization_format
        )
        if error_report_dir is not None:
            write_report(error_report_dir)
            logger.info(f"Wrote {error_report_dir / JSON_NAME}")
        return
    tokenizer = retrieved.tokenizer
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
