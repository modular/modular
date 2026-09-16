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
"""Internal utilities for Module.compile / CompiledModel.

Slot descriptors, flatten/unflatten helpers, and signal-buffer detection
extracted from ``module.py`` to keep the public-facing module lean.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from max import driver, graph
from max import tree as _tree
from max.driver import CPU, Accelerator, Buffer, DLPackArray
from max.dtype import DType
from max.engine import Model
from max.experimental.nn._trace_context import ModuleTraceRealizationContext
from max.experimental.realization_context import (
    LazyRealizationContext,
    define_subgraph,
    in_graph_context,
)
from max.experimental.sharding import (
    DeviceMapping,
    TensorLayout,
)
from max.experimental.tensor import (
    GraphValue,
    Tensor,
    current_realization_context,
)
from max.graph import DeviceRef, Value, ops
from max.nn.comm.allreduce import Signals

_logger = logging.getLogger(__name__)

# ─── Type aliases ──────────────────────────────────────────────────────

InputType = graph.Type[Any] | TensorLayout

CastRecord = tuple[DType, DType]

# Dtypes between which the Module loader will silently auto-cast loaded
# weights when the parameter dtype differs. Intentionally narrow; expand as
# new safe pairs are validated.
_SAFE_CAST_DTYPES: frozenset[DType] = frozenset({DType.float32, DType.bfloat16})


# ─── Validation ────────────────────────────────────────────────────────


def _validate_loaded_parameter(
    name: str, existing: Tensor, loaded: Tensor
) -> None:
    """Validates that a loaded tensor matches the existing parameter.

    Args:
        name: Parameter name for error messages.
        existing: The existing parameter tensor (may be distributed).
        loaded: The loaded tensor to validate.

    Raises:
        ValueError: If shape or dtype doesn't match.
    """
    existing_shape = existing.shape
    if loaded.shape != existing_shape or loaded.dtype != existing.dtype:
        raise ValueError(
            f"{name!r}: Loaded tensor (shape={list(loaded.shape)}, "
            f"dtype={loaded.dtype}) not assignable to parameter "
            f"(shape={list(existing_shape)}, dtype={existing.dtype})."
        )


# ─── Slot descriptors ─────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class _InputSlot:
    """Describes how one user-facing input maps to graph-level inputs."""

    start: int
    count: int
    mapping: DeviceMapping | None
    """Device mapping for a distributed input (``None`` for single-device)."""


@dataclasses.dataclass(frozen=True)
class _OutputSlot:
    """Describes how one user-facing output maps to graph-level outputs."""

    start: int
    count: int
    mapping: DeviceMapping | None


# ─── Flatten / unflatten helpers ──────────────────────────────────────


def _flatten_input_types(
    input_types: Sequence[InputType],
) -> tuple[list[graph.Type[Any]], list[_InputSlot]]:
    """Expands distributed input types into per-device local types."""
    graph_types: list[graph.Type[Any]] = []
    slots: list[_InputSlot] = []
    for t in input_types:
        if isinstance(t, TensorLayout):
            local = t.local_types
            slots.append(
                _InputSlot(
                    len(graph_types),
                    len(local),
                    t.mapping,
                )
            )
            graph_types.extend(local)
        else:
            slots.append(_InputSlot(len(graph_types), 1, None))
            graph_types.append(t)
    return graph_types, slots


def _wrap_graph_inputs(
    graph_inputs: Sequence[Value[Any]],
    input_slots: list[_InputSlot],
) -> list[Tensor]:
    """Wraps flat graph inputs back into Tensors."""
    ctx = current_realization_context()
    inputs: list[Tensor] = []
    for slot in input_slots:
        if slot.mapping is not None:
            shards = [
                Tensor.from_graph_value(graph_inputs[slot.start + i])
                for i in range(slot.count)
            ]
            shard_values = tuple(s._graph_value for s in shards)
            inputs.append(
                ctx.create_unrealized(shard_values, mapping=slot.mapping)
            )
        else:
            inputs.append(Tensor.from_graph_value(graph_inputs[slot.start]))
    return inputs


def flatten_input_buffers(
    args: Sequence[Any],
    input_slots: list[_InputSlot],
) -> list[Any]:
    """Flattens sharded call-time arguments into per-shard buffers."""
    flat: list[Any] = []
    if len(args) != len(input_slots):
        error = [
            "Unable to flatten input arguments.",
            f"Expected {len(input_slots)} arguments, got {len(args)}.",
        ]
        error.extend(_args_description(args, input_slots))
        raise ValueError("\n".join(error))
    for arg, slot in zip(args, input_slots, strict=True):
        if (
            slot.mapping is not None
            and isinstance(arg, Tensor)
            and arg.is_distributed
        ):
            for shard in arg.local_shards:
                flat.append(shard.driver_tensor)
        else:
            flat.append(arg)
    return flat


def _flatten_outputs(
    raw_outputs: Tensor | Sequence[Tensor],
) -> tuple[list[GraphValue], list[_OutputSlot], bool]:
    """Normalises forward's return value to a flat graph-value list."""
    if isinstance(raw_outputs, Tensor):
        output_list: list[Tensor] = [raw_outputs]
        unary = True
    else:
        output_list = list(raw_outputs)
        unary = False

    flat: list[GraphValue] = []
    slots: list[_OutputSlot] = []
    for out in output_list:
        if out.is_distributed:
            shards = out.local_shards
            slots.append(_OutputSlot(len(flat), len(shards), out.mapping))
            for shard in shards:
                flat.append(shard._graph_value)
        else:
            slots.append(_OutputSlot(len(flat), 1, None))
            flat.append(out._graph_value)
    return flat, slots, unary


def _reconstruct_outputs(
    raw_results: list[Any],
    output_slots: list[_OutputSlot],
    unary: bool,
) -> Any:
    """Reconstructs sharded Tensors from flat session results."""
    results: list[Tensor] = []
    for slot in output_slots:
        if slot.mapping is not None:
            _mesh = slot.mapping.mesh
            _placements = slot.mapping.to_placements()
            buffers = tuple(
                raw_results[slot.start + i] for i in range(slot.count)
            )
            results.append(
                Tensor._from_shards(
                    buffers,
                    _mesh,
                    _placements,
                )
            )
        else:
            assert isinstance(raw_results[slot.start], driver.Buffer)
            results.append(Tensor(storage=raw_results[slot.start]))
    return results[0] if unary else tuple(results)


def flatten_operands(
    operands: Sequence[Tensor],
) -> tuple[list[GraphValue], list[_InputSlot]]:
    """Flattens operand Tensors to flat graph values plus per-operand slots.

    The Tensor counterpart of :func:`_flatten_input_types`: a distributed
    operand contributes one graph value per shard plus a slot carrying its
    distribution, so :func:`_wrap_graph_inputs` regroups it inside a subgraph
    exactly as for a top-level input.
    """
    values: list[GraphValue] = []
    slots: list[_InputSlot] = []
    for t in operands:
        shards = list(t.graph_values)
        mapping = t.mapping if t.is_distributed else None
        slots.append(_InputSlot(len(values), len(shards), mapping))
        values.extend(shards)
    return values, slots


# ─── subgraph value-tree layer ────────────────────────────────────────


ValueTreeDef = tuple[_tree.TreeDef, list[_InputSlot]]


def flatten_value_tree(tree: Any) -> tuple[list[GraphValue], ValueTreeDef]:
    """Flattens a pytree of Tensors to flat per-shard graph values and a treedef."""
    leaves, treedef = _tree.flatten(tree, leaf=Tensor)
    values, slots = flatten_operands(leaves)
    return values, (treedef, slots)


def unflatten_value_tree(
    values: Sequence[Value[Any]], treedef: ValueTreeDef
) -> Any:
    """Regroups shard values into Tensors and rebuilds the pytree."""
    structure, slots = treedef
    leaves = _wrap_graph_inputs(list(values), slots)
    return _tree.unflatten(structure, leaves, exact=False)


def lower_subgraph(
    resolved: str,
    module: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    key: str | None = None,
) -> Any:
    """Lowers one :func:`~max.experimental.nn.subgraphable` call.

    Inlines in plain eager (and inside a subgraph body, whose cache is ``None``);
    in a graph-compile or lazy capture emits one ``mo.call`` into a shared,
    deduplicated subgraph. ``module`` is the :class:`Module` whose parameters
    thread in as operands. ``key`` is the subgraph dedup key: when given (from
    ``subgraphable(..., name=...)``), calls sharing it share one definition;
    when ``None``, the body's IR hash is used instead.
    """
    if not in_graph_context():
        raise TypeError(
            f"subgraphable({resolved}) is only valid inside a capture "
            "(Module.compile() / Module.trace()); call the module directly "
            "to run eagerly."
        )
    # Subgraphs apply to graph-compile and lazy contexts; plain eager (and
    # nested subgraph bodies, whose cache is None) inline instead.
    ctx = current_realization_context()
    if (
        not isinstance(
            ctx, (ModuleTraceRealizationContext, LazyRealizationContext)
        )
        or ctx.subgraph_cache is None
    ):
        return module.forward(*args, **kwargs)

    # When tracing a Module, weights must be created in the subgraph, and
    # referenced with the module's weight_prefix.
    weight_prefix: str = ""
    create_external_constant: (
        Callable[[str, str, Tensor, bool], Tensor] | None
    ) = None
    if isinstance(ctx, ModuleTraceRealizationContext):
        weight_prefix = ctx.weight_prefixes.get(module, "")
        create_external_constant = ctx.create_external_constant

    arg_values, arg_def = flatten_value_tree((args, kwargs))

    # Compile resolves each weight by name at load time via the call prefix and
    # the weights registry, so weights are not call operands. Lazy has no
    # registry (weights are concrete tensors), so thread them in as operands and
    # rebind them to block args inside the body.
    weight_names: list[str] = []
    weight_values: list[GraphValue] = []
    weight_def: Any = None
    if not isinstance(ctx, ModuleTraceRealizationContext):
        params = list(module.parameters)
        weight_names = [name for name, _ in params]
        weight_values, weight_def = flatten_value_tree([t for _, t in params])

    operand_values = [*arg_values, *weight_values]
    n_args = len(arg_values)
    out_def: list[ValueTreeDef] = []  # Tree structure of the module outputs.

    def build_body(inputs: list[Value[Any]]) -> list[GraphValue]:
        call_args, call_kwargs = unflatten_value_tree(inputs[:n_args], arg_def)
        by_name: dict[str, Tensor] = {}
        if weight_def is not None:
            weights = unflatten_value_tree(inputs[n_args:], weight_def)
            by_name = dict(zip(weight_names, weights, strict=True))

        def rebind(name: str, tensor: Tensor) -> Tensor:
            if name in by_name:  # lazy: bound to a threaded operand
                return by_name[name]
            assert create_external_constant is not None
            return create_external_constant(
                weight_prefix + name, name, tensor, True
            )

        with module._mapped_parameters(rebind):
            out = module.forward(*call_args, **call_kwargs)
        values, treedef = flatten_value_tree(out)
        out_def.append(treedef)
        return values

    subgraph = define_subgraph(
        ctx, resolved, [v.type for v in operand_values], build_body, key=key
    )

    # `out_def` is only defined correctly if `build_body` is called, so
    # cache it in case a cached subgraph is used.
    if out_def:
        out_treedef = out_def[0]
        if key is not None:
            ctx.subgraph_out_defs[key] = out_treedef
    else:
        assert key is not None
        out_treedef = ctx.subgraph_out_defs[key]

    signals = ctx.signal_buffers or []
    results = ops.call(
        subgraph, *operand_values, *signals, prefix=weight_prefix
    )
    return unflatten_value_tree(list(results), out_treedef)


def flatten_distributed_tensors(
    named_tensors: Iterable[tuple[str, Tensor]],
) -> dict[str, Tensor]:
    """Flattens potentially distributed tensors.

    Distributed tensors are flattened into `{name}._shard.{i}` for each shard.
    """
    cpu = CPU()
    result: dict[str, Tensor] = {}
    for name, tensor in named_tensors:
        if tensor.real:
            local_shards = tensor.local_shards
            for i, shard in enumerate(local_shards):
                key = f"{name}._shard.{i}" if len(local_shards) > 1 else name
                result[key] = shard if shard.device == cpu else shard.to(cpu)
        else:
            result[name] = tensor
    return result


class _PreparedWeight(NamedTuple):
    weight: Tensor
    """Tensor holding the weight value."""
    cast_record: CastRecord | None
    """Record of the cast applied, if any."""
    transfer_needed: bool
    """Whether the weight needs to be transferred to the parameter's device(s)."""


def prepare_weight_for_parameter(
    name: str,
    weight: DLPackArray | Tensor,
    param: Tensor,
    *,
    auto_cast: bool,
) -> _PreparedWeight:
    """Validates and prepares a weight for a parameter.

    Handles conversion and validation of the weight. Does not handle device
    transfers; instead any device transfers are performed in the graph.

    1. Converts DLPack array to Tensor if needed
    2. When ``auto_cast`` is true, auto-casts dtype when both loaded and
       parameter dtypes are in the safe-cast whitelist (see
       ``_SAFE_CAST_DTYPES``)
    3. Validates shape and dtype match the parameter
    4. For distributed parameters: validates if mapping matches the parameter's
    mapping. If it doesn't match, then notes that a transfer is needed.

    Args:
        name: Parameter name for error messages.
        weight: User-provided weight (DLPack array or Tensor).
        param: The target parameter tensor.
        auto_cast: Whether to apply safe-cast-set dtype coercion.

    Returns:
        See :class:`_PreparedWeight`.

    Raises:
        ValueError: If shape, dtype, or distribution doesn't match.
    """
    if isinstance(weight, Tensor):
        weight_tensor = weight
    else:
        weight_tensor = Tensor.from_dlpack(weight)

    cast_record: CastRecord | None = None
    if (
        auto_cast
        and weight_tensor.shape == param.shape
        and weight_tensor.dtype != param.dtype
        and weight_tensor.dtype in _SAFE_CAST_DTYPES
        and param.dtype in _SAFE_CAST_DTYPES
    ):
        cast_record = (weight_tensor.dtype, param.dtype)
        weight_tensor = weight_tensor.cast(param.dtype)

    _validate_loaded_parameter(name, param, weight_tensor)

    if not param.is_distributed:
        return _PreparedWeight(weight_tensor, cast_record, False)

    assert param._mapping is not None

    if weight_tensor.is_distributed:
        if weight_tensor._mapping != param._mapping:
            raise ValueError(
                f"Weight '{name}' has incompatible distribution. "
                f"Expected {param._mapping}, got {weight_tensor._mapping}."
            )
        return _PreparedWeight(weight_tensor, cast_record, False)

    return _PreparedWeight(weight_tensor, cast_record, True)


def _emit_cast_summary(cast_counts: Mapping[CastRecord, int]) -> None:
    """Logs a single summary message for a batch of auto-casts.

    Called once per ``load_state_dict`` (or ``compile(weights=...)``) after
    all parameters have been processed, so users see one log line per load
    rather than one per parameter.
    """
    if not cast_counts:
        return
    parts = []
    for (src, dst), count in cast_counts.items():
        # Narrowing casts (e.g. float32 -> bfloat16) truncate precision; flag
        # them so users aren't silently surprised when accuracy regresses.
        qualifier = (
            " (precision loss)" if dst.size_in_bytes < src.size_in_bytes else ""
        )
        parts.append(f"{count} parameter(s) from {src} to {dst}{qualifier}")
    _logger.warning("load_state_dict auto-cast: %s.", "; ".join(parts))


class _PreparedWeights(NamedTuple):
    weights_registry: dict[str, Tensor]
    """Processed weights that are ready to be assigned to the parameters."""

    weights_to_transfer: dict[str, Tensor]
    """The weights that need to be transferred to the parameters.
    The keys are strictly a subset of `weights_registry`."""


def prepare_weights_registry(
    weights: Mapping[str, DLPackArray],
    parameters: Iterable[tuple[str, Tensor]],
    *,
    auto_cast: bool,
) -> _PreparedWeights:
    """Prepares the weight registry given input weights and module parameters.

    Args:
        weights: User-provided weight buffers keyed by parameter name.
        parameters: Module parameters with distribution metadata.
        auto_cast: Whether to permit safe-cast-set dtype coercion when
            shapes match (see :func:`prepare_weight_for_parameter`).

    Returns:
        A :class:`_PreparedWeights` containing the weights registry and
        the weights that need to be transferred.
    """
    weights_registry: dict[str, Tensor] = {}
    weights_to_transfer: dict[str, Tensor] = {}
    cast_counts: dict[CastRecord, int] = {}

    for name, param in parameters:
        if name not in weights:
            raise KeyError(
                f"Weight '{name}' is missing from the provided weights mapping."
            )

        prepared, cast_record, transfer_needed = prepare_weight_for_parameter(
            name, weights[name], param, auto_cast=auto_cast
        )
        if cast_record is not None:
            cast_counts[cast_record] = cast_counts.get(cast_record, 0) + 1
        shards = prepared.local_shards

        if transfer_needed:
            assert len(shards) == 1
            weights_registry[name] = shards[0]
            weights_to_transfer[name] = prepared
        elif not param.is_distributed:
            weights_registry[name] = shards[0]
        else:
            for i, shard in enumerate(shards):
                weights_registry[f"{name}._shard.{i}"] = shard

    _emit_cast_summary(cast_counts)
    return _PreparedWeights(weights_registry, weights_to_transfer)


def _detect_signals(
    input_types: Sequence[InputType],
    parameters: Iterable[tuple[str, Tensor]] | None = None,
) -> Signals | None:
    """Creates :class:`Signals` if inputs or parameters span multiple GPUs.

    Checks both input types (for distributed activations) and module
    parameters (for distributed weights on GPU meshes).

    Returns ``None`` for single-device or CPU-only inputs.
    """
    gpu_refs: list[DeviceRef] = []
    seen: set[int] = set()
    for t in input_types:
        if not isinstance(t, TensorLayout):
            continue
        for dev in t.mesh.devices:
            if isinstance(dev, Accelerator) and dev.id not in seen:
                gpu_refs.append(DeviceRef.GPU(id=dev.id))
                seen.add(dev.id)
    if parameters is not None:
        for _, param in parameters:
            if param.is_distributed and param.mesh is not None:
                for dev in param.mesh.devices:
                    if isinstance(dev, Accelerator) and dev.id not in seen:
                        gpu_refs.append(DeviceRef.GPU(id=dev.id))
                        seen.add(dev.id)
    if len(gpu_refs) < 2:
        return None
    return Signals(devices=gpu_refs)


# ─── Engine-call diagnostics ──────────────────────────────────────────


def _describe_arg(arg: Any) -> str:
    """Format a single user-facing argument for an error message."""
    if isinstance(arg, Tensor):
        if arg.is_distributed:
            return (
                f"distributed Tensor(shape={list(arg.shape)}, "
                f"dtype={arg.dtype}, placements={arg.placements}, "
                f"shards={len(arg.local_shards)})"
            )
        return f"Tensor(shape={list(arg.shape)}, dtype={arg.dtype}, device={arg.device})"
    if isinstance(arg, Buffer):
        return f"Buffer(shape={list(arg.shape)}, dtype={arg.dtype}, device={arg.device})"
    if arg is None:
        return "None"
    return type(arg).__name__


def _describe_slot(slot: _InputSlot) -> str:
    """Format an input slot's expectation for an error message."""
    if slot.mapping is not None:
        return (
            f"distributed Tensor(placements={slot.mapping.placements}, "
            f"expects {slot.count} shards)"
        )
    return "single-device Tensor"


def _args_description(
    user_args: Sequence[Any] | None, input_slots: Sequence[_InputSlot]
) -> list[str]:
    """Generates a description of the arguments for an error message."""
    lines = []
    if user_args is not None:
        lines.append(
            f"  Got {len(user_args)} positional arg(s), expected {len(input_slots)}."
        )
        if user_args:
            lines.append("  Provided arguments:")
            for i, arg in enumerate(user_args):
                lines.append(f"    arg[{i}]: {_describe_arg(arg)}")
    lines.append("  Expected arguments:")
    for i, slot in enumerate(input_slots):
        lines.append(f"    arg[{i}]: {_describe_slot(slot)}")
    return lines


def engine_call_error(
    error: BaseException,
    engine_model: Model,
    user_args: Sequence[Any] | None,
    flat_args: Sequence[Any],
    input_slots: Sequence[_InputSlot],
    signal_buffer_count: int,
) -> TypeError:
    """Wraps an engine call argument-binding error with diagnostics.

    Args:
        error: The original error from the engine call.
        engine_model: The compiled :class:`~max.engine.Model`.
        user_args: User-facing positional args (``None`` for ``execute_raw``,
            which does not have a user-facing layer).
        flat_args: Flattened buffers actually passed to the engine (includes
            appended signal buffers). Empty when flattening itself failed.
        input_slots: Per-user-arg slot descriptors from compile time.
        signal_buffer_count: Number of trailing signal buffers in
            ``flat_args``.

    Returns:
        A new ``TypeError`` chained to *error* with the diagnostic message.
    """
    expected_total = len(engine_model.input_metadata)
    expected_user = expected_total - signal_buffer_count
    got_total = len(flat_args)
    got_user = got_total - signal_buffer_count

    lines = [
        f"Compiled model call failed to bind arguments ({error}).",
        (
            f"  Engine expects {expected_total} flat input(s) "
            f"({expected_user} user args + {signal_buffer_count} signal buffer(s)); "
            f"got {got_total} ({got_user} user args + {signal_buffer_count} signal buffer(s))."
        ),
    ]

    lines.extend(_args_description(user_args, input_slots))

    return TypeError("\n".join(lines))
