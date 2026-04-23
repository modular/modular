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
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from max import driver, graph
from max.driver import Accelerator, Buffer, DLPackArray
from max.engine import Model
from max.experimental.functional import transfer_to
from max.experimental.sharding import (
    DeviceMapping,
    DistributedType,
    PlacementMapping,
    global_shape_from_local,
)
from max.experimental.tensor import GraphValue, Tensor
from max.graph import DeviceRef, Value
from max.nn.comm.allreduce import Signals

# ─── Type aliases ──────────────────────────────────────────────────────

InputType = graph.Type[Any] | DistributedType[Any]


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
    dist: DistributedType[Any] | None


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
        if isinstance(t, DistributedType):
            local = t.local_types
            slots.append(_InputSlot(len(graph_types), len(local), t))
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
    from max.experimental.tensor import current_realization_context

    ctx = current_realization_context()
    inputs: list[Tensor] = []
    for slot in input_slots:
        if slot.dist is not None:
            shards = [
                Tensor.from_graph_value(graph_inputs[slot.start + i])
                for i in range(slot.count)
            ]
            shard_values = tuple(s._graph_value for s in shards)
            inputs.append(
                ctx.create_unrealized(
                    shard_values,
                    mapping=PlacementMapping(
                        slot.dist.mesh, slot.dist.placements
                    ),
                    global_shape=slot.dist.shape,
                )
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
            slot.dist is not None
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
            gshape = global_shape_from_local(
                buffers[0].shape,
                _mesh,
                _placements,
            )
            results.append(
                Tensor._from_shards(
                    buffers,
                    _mesh,
                    _placements,
                    global_shape=gshape,
                )
            )
        else:
            assert isinstance(raw_results[slot.start], driver.Buffer)
            results.append(Tensor(storage=raw_results[slot.start]))
    return results[0] if unary else results


def _flatten_named_buffers(
    named_tensors: Iterable[tuple[str, Tensor]],
) -> dict[str, DLPackArray]:
    """Flattens named parameters to a ``name -> DLPackArray`` mapping."""
    result: dict[str, DLPackArray] = {}
    for name, t in named_tensors:
        if t.real:
            bufs = t.buffers
            for i, buf in enumerate(bufs):
                key = f"{name}._shard.{i}" if len(bufs) > 1 else name
                result[key] = buf
        else:
            result[name] = t
    return result


def _prepare_weight_for_parameter(
    name: str,
    weight: DLPackArray | Tensor,
    param: Tensor,
) -> Tensor:
    """Validates and prepares a weight for a parameter.

    Handles conversion, validation, and sharding:

    1. Converts DLPack array to Tensor if needed
    2. Validates shape and dtype match the parameter
    3. For distributed parameters: validates mapping or shards single-device weights

    Args:
        name: Parameter name for error messages.
        weight: User-provided weight (DLPack array or Tensor).
        param: The target parameter tensor.

    Returns:
        A Tensor ready to be assigned to the parameter.

    Raises:
        ValueError: If shape, dtype, or distribution doesn't match.
    """
    if isinstance(weight, Tensor):
        weight_tensor = weight
    else:
        weight_tensor = Tensor.from_dlpack(weight)

    _validate_loaded_parameter(name, param, weight_tensor)

    if not param.is_distributed:
        return weight_tensor

    assert param._mapping is not None

    if weight_tensor.is_distributed:
        if weight_tensor._mapping != param._mapping:
            raise ValueError(
                f"Weight '{name}' has incompatible distribution. "
                f"Expected {param._mapping}, got {weight_tensor._mapping}."
            )
        return weight_tensor

    return transfer_to(weight_tensor, param._mapping)


def _process_provided_weights(
    weights: Mapping[str, DLPackArray],
    parameters: Iterable[tuple[str, Tensor]],
) -> dict[str, DLPackArray]:
    """Processes user-provided weights based on parameter distribution.

    Handles two cases for each parameter:

    - **Non-distributed parameter**: Weight passes through as-is.
    - **Distributed parameter**: If weight is a single-device buffer, shards
      it using ``transfer_to()``. If weight is already a dtensor, extracts shards.
      Both cases produce ``name._shard.N`` entries.

    Args:
        weights: User-provided weight buffers keyed by parameter name.
        parameters: Module parameters with distribution metadata.

    Returns:
        A weights registry suitable for ``session.load(..., weights_registry=...)``.
    """
    result: dict[str, DLPackArray] = {}

    for name, param in parameters:
        if name not in weights:
            raise KeyError(
                f"Weight '{name}' is missing from the provided weights mapping."
            )

        prepared = _prepare_weight_for_parameter(name, weights[name], param)
        shards = prepared.local_shards

        if not param.is_distributed:
            result[name] = shards[0]
        else:
            for i, shard in enumerate(shards):
                result[f"{name}._shard.{i}"] = shard

    return result


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
        if not isinstance(t, DistributedType):
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
        return f"Tensor(shape={list(arg.shape)}, dtype={arg.dtype})"
    if isinstance(arg, Buffer):
        return f"Buffer(shape={list(arg.shape)}, dtype={arg.dtype})"
    if arg is None:
        return "None"
    return type(arg).__name__


def _describe_slot(slot: _InputSlot) -> str:
    """Format an input slot's expectation for an error message."""
    if slot.dist is not None:
        return (
            f"distributed Tensor(shape={list(slot.dist.shape)}, "
            f"placements={slot.dist.placements}, "
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
