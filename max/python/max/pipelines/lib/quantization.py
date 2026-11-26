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
"""Layer quantization for pipeline build, export, and checkpoint writing.

Wraps matching linears in on-the-fly quantizers before graphs compile
(``max export``) or before a quantized checkpoint is written
(``max quantize``). Selected from ``PipelineConfig.quant_config`` when
``source`` is :attr:`~max.nn.quant_config.WeightSource.QUANTIZE`.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from max import tree
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.engine.api import PrintStyle
from max.nn import Linear, Module
from max.nn.common_layers.linear import QKVLinear
from max.nn.common_layers.mlp import MLP
from max.nn.common_layers.moe import MoE
from max.nn.common_layers.quant_linear import (
    QuantizedLinear,
    QuantizedMLP,
    QuantizedQKVLinear,
)
from max.nn.common_layers.quant_moe import ExpertParallelMoE, QuantizedMoE
from max.nn.common_layers.quant_ops import quantize_on_load
from max.nn.common_layers.quant_tensor import (
    FP8BlockTensor,
    MXFP4Quantizer,
    MXFP8Quantizer,
    NVFP4TensorQuantizer,
    QTensor,
)
from max.nn.quant_config import QuantConfig, QuantFormat, WeightSource
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lib.interfaces import (
    ModuleV3MultiGraphPipelineModelWithKVCache,
    ModuleV3PipelineModelWithKVCache,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Sharded

logger = logging.getLogger("max.pipelines")

# MoE before MLP. Rebuilding a MoE constructs new experts, so the MLP walk
# must run after that and see those experts rather than the ones discarded.
_FUSED_GROUPS = (
    (MoE, ("gate_proj", "up_proj")),
    (MLP, ("gate_proj", "up_proj")),
    (QKVLinear, ("q_proj", "k_proj", "v_proj")),
)

#: Formats whose on-the-fly quantizer can decode in the graph to measure
#: its own error.
_ERROR_REPORT_FORMATS = frozenset(
    {QuantFormat.NVFP4, QuantFormat.MXFP8, QuantFormat.MXFP4}
)

#: The config each ``quantization_format`` value selects.
#:
#: ``source`` is set here rather than by the caller: the path exists to
#: quantize a native-precision checkpoint, so CHECKPOINT would describe
#: bytes that are not there.
#:
#: The layer tuples are empty because ``QuantConfig`` selects layers by
#: index and this path selects them by name instead:
#: ``insert_quantized_linears`` wraps whatever matches ``targets``.
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
    """The values ``quantization_format`` accepts."""
    return tuple(sorted(_ON_THE_FLY))


def on_the_fly_config(
    name: str,
    *,
    report_error: bool = False,
    targets: Iterable[str] = (),
    error_report_dir: Path | None = None,
    verbose: bool = False,
    mse_sweep_scales: bool = False,
    source_format: str | None = None,
) -> QuantConfig:
    """The :class:`~max.nn.quant_config.QuantConfig` ``name`` selects.

    Sets :attr:`~max.nn.quant_config.WeightSource.QUANTIZE` (already on the
    table row), optional name-based ``targets``, and error-report settings.

    With ``report_error`` or ``error_report_dir``, the quantizers print their
    quantization error from inside the graph. ``mse_sweep_scales`` is read
    by NVFP4 quantizers: each block's scale is the multiplier of its abs-max
    with the smallest quantization MSE. ``source_format`` optionally
    asserts that every targeted weight is stored as ``bf16`` or ``fp8``.

    Raises:
        ValueError: if ``name`` is not a format this path can produce.
    """
    try:
        config = _ON_THE_FLY[name]
    except KeyError:
        raise ValueError(
            f"unknown quantization format {name!r}; this path can "
            f"produce {', '.join(quant_formats())}"
        ) from None
    # A copy, because QuantConfig is mutable and the table is shared.
    return dataclasses.replace(
        config,
        report_error=report_error or error_report_dir is not None,
        targets=tuple(targets),
        error_report_dir=error_report_dir,
        verbose=verbose,
        mse_sweep_scales=mse_sweep_scales,
        source_format=source_format,
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


_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

#: What ``--source-format`` accepts: ``bf16`` is a native-precision weight,
#: ``fp8`` a block-scaled ``float8_e4m3fn`` weight with ``weight_scale_inv``.
SOURCE_FORMATS = ("bf16", "fp8")


def _require_source_format(
    what: str, actual: str, expected: str | None
) -> None:
    """Rejects a targeted weight whose stored format is not the stated one."""
    if expected is not None and actual != expected:
        raise ValueError(
            f"{what} is stored as {actual}, but the source format was "
            f"stated as {expected}. Fix --targets so it matches only "
            f"{expected} weights, or drop --source-format."
        )


def _require_unsharded(path: str, mapping: DeviceMapping) -> None:
    """Rejects a weight that is split across devices.

    The quantize ops have no sharding rule, so a quantizer cannot read a
    sharded source: it would first have to gather the whole weight onto every
    device. A weight on one device, or replicated, is fine. That covers
    routed experts under expert parallelism, which sit whole on one device
    each.

    Raises:
        ValueError: If ``mapping`` shards the weight along any mesh axis.
    """
    for axis_index, placement in enumerate(mapping.placements):
        if isinstance(placement, Sharded):
            shards = mapping.mesh.axis_size(axis_index)
            raise ValueError(
                f"{path} is sharded {shards} ways along axis "
                f"{placement.axis}, and quantizing a sharded weight is not "
                "supported yet. Leave it out of --targets so it stays in FP8, "
                "or use a layout that does not shard it, such as expert "
                "parallelism for routed experts."
            )


def _check_expert_parallel_target(
    path: str,
    moe: ExpertParallelMoE,
    quant_config: QuantConfig,
    patterns: list[re.Pattern[str]],
) -> None:
    """Rejects an expert-parallel MoE the target format cannot serve.

    Expert-parallel dispatch quantizes tokens as FP8, NVFP4 or MXFP4 (the
    FP4 formats travel as packed ``uint8``). MXFP8 has no dispatch yet, so
    NVFP4 and MXFP4 are the targets an FP8 MoE can move to. MXFP4's grouped
    matmul runs on AMD only. When the dispatch fuses the shared expert into
    the routed experts' stack, the shared expert has to change format with
    them.

    Raises:
        ValueError: If the target is not NVFP4 or MXFP4, or the shared expert
            is fused into the routed experts and is not targeted too.
    """
    if not quant_config.is_fp4:
        raise ValueError(
            f"{path} is expert parallel, and expert-parallel dispatch can "
            f"quantize tokens only as FP8, NVFP4 or MXFP4, not "
            f"{quant_config.format.value}. Quantize to nvfp4 or mxfp4."
        )
    fused = moe.ep_config.fused_shared_expert
    if fused and moe.shared_experts is not None:
        shared = _child_path(path, "shared_experts")
        paths = [_child_path(shared, name) for name in _EXPERT_PROJECTIONS]
        hits = _projection_hits(paths, patterns)
        if len(hits) != len(paths):
            raise ValueError(
                f"{path} stacks its shared expert with the routed experts, so "
                "they change format together. Also target "
                f"{[p for p in paths if p not in hits]}."
            )


def _retarget_dispatch(
    moe: ExpertParallelMoE, quant_config: QuantConfig
) -> None:
    """Points the expert-parallel dispatch at the target format.

    The dispatch config is shared by every MoE layer of the model and was
    built from the checkpoint's FP8 config. Experts that now run in NVFP4 or
    MXFP4 need tokens dispatched in that format, as packed ``uint8``, so the
    config follows them. Setting it again from another layer changes
    nothing.
    """
    config = moe.ep_config
    config.dispatch_dtype = DType.uint8
    config.dispatch_quant_config = quant_config
    config.__post_init__()


_QUANTIZER_TYPES = (NVFP4TensorQuantizer, MXFP8Quantizer, MXFP4Quantizer)


def _check_fp8_moe(
    path: str,
    moe: QuantizedMoE,
    patterns: list[re.Pattern[str]],
    quant_config: QuantConfig,
    source_format: str | None,
) -> None:
    """Rejects targets that would leave a block-FP8 MoE half quantized.

    A MoE built from a block-scaled FP8 checkpoint holds FP8 experts and a
    ``quant_config`` that says so. The grouped matmuls read that config, so
    the experts cannot be quantized one projection at a time: every expert's
    gate, up and down projection is targeted, or none is. A MoE whose
    experts are not FP8 is left to the rest of :func:`insert_quantized_linears`.

    Raises:
        ValueError: If only some experts, or only some of an expert's three
            projections, are targeted, a targeted expert weight is not
            block-scaled FP8, ``source_format`` is not ``fp8``, or the MoE is
            expert parallel and cannot serve the target (see
            :func:`_check_expert_parallel_target`).
    """
    layers: list[tuple[str, Any]] = []
    for index, expert in enumerate(moe.experts):
        expert_path = _child_path(_child_path(path, "experts"), str(index))
        for name in _EXPERT_PROJECTIONS:
            layer = getattr(expert, name)
            if isinstance(layer, QuantizedLinear):
                layers.append((_child_path(expert_path, name), layer.weight))
    if not any(isinstance(weight, FP8BlockTensor) for _, weight in layers):
        return
    targeted = 0
    for index in range(len(moe.experts)):
        expert_path = _child_path(_child_path(path, "experts"), str(index))
        paths = [_child_path(expert_path, name) for name in _EXPERT_PROJECTIONS]
        hits = _projection_hits(paths, patterns)
        if not hits:
            continue
        if len(hits) != len(paths):
            raise ValueError(
                "an expert's gate_proj, up_proj and down_proj are "
                "quantized together, so target all three. Matched "
                f"{hits}; left out {[p for p in paths if p not in hits]}."
            )
        targeted += 1
    if targeted == 0:
        return
    if targeted != len(moe.experts):
        raise ValueError(
            "a MoE stacks every expert into one matmul, so quantize all "
            f"experts or none of them. Matched {targeted} of "
            f"{len(moe.experts)} at {path}."
        )
    for layer_path, weight in layers:
        if not isinstance(weight, FP8BlockTensor):
            raise ValueError(
                f"{layer_path} holds a {type(weight).__name__}; only "
                "block-scaled FP8 experts can be quantized again"
            )
        _require_source_format(layer_path, "fp8", source_format)
    if isinstance(moe, ExpertParallelMoE):
        _check_expert_parallel_target(path, moe, quant_config, patterns)


def _check_loaded_containers(
    module: Module[..., Any],
    patterns: list[re.Pattern[str]],
    quant_config: QuantConfig,
    source_format: str | None,
) -> None:
    """Rejects targets that would split a container the model already built.

    A model built from a quantized checkpoint's config holds
    :class:`QuantizedMLP` and :class:`QuantizedMoE` containers, which the
    plain ``MLP`` and ``MoE`` checks in :func:`insert_quantized_linears`
    never see. A gate/up pair is one matmul, so it is targeted whole or not
    at all. Runs before any weight changes.

    Raises:
        ValueError: See :func:`_require_full_group` and
            :func:`_check_fp8_moe`.
    """
    for path in tree.nodes(module, QuantizedMLP):
        names = ("gate_proj", "up_proj")
        paths = [_child_path(path, name) for name in names]
        hits = _projection_hits(paths, patterns)
        if hits:
            _require_full_group(names, paths, hits)
    for path, moe in tree.nodes(module, QuantizedMoE).items():
        _check_fp8_moe(path, moe, patterns, quant_config, source_format)


def _sync_moe_configs(
    module: Module[..., Any], quant_config: QuantConfig
) -> None:
    """Points each block-FP8 MoE whose experts are now quantizers at the target.

    The grouped matmuls dispatch on the MoE's ``quant_config``, so it has to
    follow the experts it holds. Under expert parallelism the dispatch config,
    which is shared by every MoE layer and was built from the checkpoint's FP8
    config, follows as well: tokens are dispatched in the experts' format, as
    packed ``uint8``. The dispatch refuses a config that quantizes on load,
    since it runs before any weight is read. These quantizers are the
    format's own containers, so the MoE and its dispatch take the config of
    a pre-quantized checkpoint of that format.
    """
    for moe in tree.nodes(module, QuantizedMoE).values():
        current = moe.quant_config
        if current is None or current.format is not QuantFormat.BLOCKSCALED_FP8:
            continue
        weights = [
            getattr(expert, name).weight
            for expert in moe.experts
            for name in _EXPERT_PROJECTIONS
        ]
        if not all(isinstance(weight, _QUANTIZER_TYPES) for weight in weights):
            continue
        if isinstance(moe, ExpertParallelMoE):
            container_config = dataclasses.replace(
                quant_config, source=WeightSource.CHECKPOINT
            )
            moe.quant_config = container_config
            _retarget_dispatch(moe, container_config)
        else:
            moe.quant_config = quant_config


def insert_quantized_linears(
    module: Module[..., Any],
    targets: Iterable[str],
    quant_config: QuantConfig,
    *,
    verbose: bool = False,
    source_format: str | None = None,
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

    A model built from a block-scaled FP8 checkpoint already holds quantized
    layers. Targeted ones are wrapped in the target format's quantizer,
    which decodes the FP8 weight to bfloat16 itself; untargeted ones stay
    FP8. The quantizer wraps the weight where it already lives, so an
    expert keeps its device. An FP8 weight sharded across devices is
    rejected, because the quantize ops cannot read one. The routed experts
    of an FP8 :class:`QuantizedMoE` are quantized all together. The MoE's
    own config, and under expert parallelism the dispatch config, then
    follow its experts to the target format (:func:`_sync_moe_configs`).

    ``source_format`` (``bf16`` or ``fp8``) states what the targeted weights
    are stored as. It is optional: when ``None`` each weight's own format
    decides. When given, a targeted weight stored as anything else raises, so
    a mistaken ``targets`` pattern cannot quietly quantize the wrong layers.

    Returns:
        ``module``. A new object when ``module`` itself was the container:
        a constructor cannot change the object the caller already holds.

    Raises:
        ValueError: If ``quant_config.report_error`` is set for a format that
            cannot measure its error, ``source_format`` is not one of
            :data:`SOURCE_FORMATS`, or a targeted weight is stored in a
            different format than ``source_format``.
    """
    if source_format is not None and source_format not in SOURCE_FORMATS:
        raise ValueError(
            f"unknown source format {source_format!r}; expected one of "
            f"{', '.join(SOURCE_FORMATS)}"
        )
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
    _check_loaded_containers(module, patterns, quant_config, source_format)
    # One walk for a Linear, a QuantizedLinear still holding a plain
    # tensor, a projection the model built as its own module, and a block-FP8
    # weight. The container updates above have already landed, so this sees
    # those new children. The walk stops at a quantized weight, which is
    # reported at its layer's path and holds its own plain ``weight`` tensor:
    # it is a layer's weight, not a layer. A weight that is already a
    # quantizer is skipped. The tree cannot be edited while it is walked,
    # so the replacements are applied once afterwards.
    replacements: dict[str, Any] = {}
    replacing: set[int] = set()
    for path, node in tree.nodes(module, Module, leaf=QTensor).items():
        if isinstance(node, QTensor):
            if not any(pattern.search(path) for pattern in patterns):
                continue
            if isinstance(node, _QUANTIZER_TYPES):
                continue
            if not isinstance(node, FP8BlockTensor):
                raise ValueError(
                    f"{path} holds a {type(node).__name__}, which cannot be "
                    "quantized again; only block-scaled FP8 and "
                    "native-precision weights can be"
                )
            _require_source_format(path, "fp8", source_format)
            _require_unsharded(path, node.weight.mapping)
            # The container is already placed, so the quantizer wrapping
            # it is too.
            replacements[path] = quantize_on_load(node, quant_config)
            replacing.add(id(node))
            count += 1
            continue
        weight = getattr(node, "weight", None)
        if type(weight) is not Tensor or weight.rank != 2:
            continue
        if not any(pattern.search(path) for pattern in patterns):
            continue
        _require_source_format(path, "bf16", source_format)
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
    _sync_moe_configs(module, quant_config)
    if verbose:
        print(module)
    logger.info(f"Quantizing {count} weight tensors.")
    return module


# Leaves each on-the-fly quantizer writes, in checkpoint order. The name is
# the safetensors suffix: ``weight`` becomes ``{path}.weight``.
def _enable_error_prints(
    session: InferenceSession, directory: Path | None
) -> None:
    """Has ``session`` dump the graph's prints, which carry the error stats.

    Must run before the graphs compile. Dumps from an earlier run are
    removed, so the report holds only this run's weights. A no-op without a
    directory. Dump names match ``quant_error.*.max`` from the error-report
    helpers.
    """
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("quant_error.*.max"):
        path.unlink()
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
                source_format=quant_config.source_format,
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
                source_format=quant_config.source_format,
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


def quantizing_factory(
    factory: Callable[[], Any],
    config: QuantConfig,
) -> Callable[[], Any]:
    """Returns a factory that compiles graphs with on-the-fly quantizers.

    The compiled graphs bind the checkpoint's native-precision weights, and
    the quantizers run inside the graph.

    ``config`` must use :attr:`~max.nn.quant_config.WeightSource.QUANTIZE`
    and name the linears in :attr:`~max.nn.quant_config.QuantConfig.targets`.
    With ``config.error_report_dir``, each quantizer also measures its error
    against the native-precision weight and prints that measurement from
    inside the compiled graph.

    Args:
        factory: The factory ``retrieve_factory`` built before quantization.
        config: On-the-fly :class:`~max.nn.quant_config.QuantConfig`.

    Returns:
        A factory that builds the same pipeline with those quantizers
        inserted, and compiles its graphs.

    Raises:
        ValueError: If ``config`` is not an on-the-fly config or has no
            targets.
    """
    if not config.quantizes_on_load:
        raise ValueError(
            "on-the-fly quantization needs source=QUANTIZE on QuantConfig"
        )
    if not config.targets:
        raise ValueError(
            "on-the-fly quantization needs targets: the regexes of the "
            "linears to wrap before the graphs compile"
        )
    return _factory_with_layer_quantizers(
        factory,
        config.targets,
        config,
        verbose=config.verbose,
        error_report_dir=config.error_report_dir,
    )
