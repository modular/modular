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

"""Functional wrappers for MAX kernel operations used in attention layers."""

import functools
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from max.experimental import functional as F
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMesh,
    Placement,
    PlacementMapping,
    Replicated,
)
from max.experimental.sharding.action import Action, ActionSet, AxisAssignment
from max.experimental.sharding.cost import (
    build_action_set,
    force_replicated_action_set,
)
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.comm.ep.ep_kernels import (
    fused_silu as _fused_silu,
)
from max.nn.kernels import (
    flare_mla_prefill_plan as _flare_mla_prefill_plan,
)
from max.nn.kernels import (
    flash_attention_ragged as _flash_attention_ragged,
)
from max.nn.kernels import (
    grouped_matmul_ragged as _grouped_matmul_ragged,
)
from max.nn.kernels import (
    mla_decode_graph as _mla_decode_graph,
)
from max.nn.kernels import (
    mla_prefill_decode_graph as _mla_prefill_decode_graph,
)
from max.nn.kernels import (
    mla_prefill_graph as _mla_prefill_graph,
)
from max.nn.kernels import (
    moe_create_indices as _moe_create_indices,
)
from max.nn.kernels import (
    moe_router_group_limited as _moe_router_group_limited,
)
from max.nn.kernels import (
    rms_norm_key_cache as _rms_norm_key_cache,
)
from max.nn.kernels import (
    rope_split_store_ragged as _rope_split_store_ragged,
)


def _preserve_orig_mappings(
    orig_mappings: tuple[Any, ...],
) -> Callable[[Action], Action]:
    """Returns a finalize that restores the op's original input mappings."""

    def finalize(action: Action) -> Action:
        return Action(inputs=orig_mappings, outputs=action.outputs)

    return finalize


def grouped_matmul_ragged_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    expert_start_indices: TensorLayout,
    expert_ids: TensorLayout,
    expert_usage_stats: TensorLayout,
) -> ActionSet:
    layouts = (
        hidden_states,
        weight,
        expert_start_indices,
        expert_ids,
        expert_usage_stats,
    )
    n_axes = weight.mapping.mesh.ndim
    rows = []
    for axis in range(n_axes):
        needed = tuple(l.mapping.to_placements()[axis] for l in layouts)
        out = weight.mapping.to_placements()[axis]
        rows.append(AxisAssignment(needed_inputs=needed, output=out))
    return build_action_set(
        rows,
        layouts=layouts,
        finalize=_preserve_orig_mappings(tuple(l.mapping for l in layouts)),
    )


grouped_matmul_ragged = F.functional(
    _grouped_matmul_ragged, rule=grouped_matmul_ragged_rule
)


def _moe_create_indices_rule(lhs: TensorLayout, *args: Any) -> ActionSet:
    return force_replicated_action_set(lhs)


moe_create_indices = F.functional(
    _moe_create_indices, rule=_moe_create_indices_rule
)

inplace_custom = F.functional(ops.inplace_custom)
shard_and_stack = F.functional(ops.shard_and_stack)


# ─── Operations that should be dispatched per-device on distributed inputs ────


def local_map(
    fn: Callable[..., Any],
    distributed_kwargs: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> Any:
    """Applies a single-device function independently to each device's data.

    Args:
        fn: Single-device function invoked once per device with keyword
            arguments.
        distributed_kwargs: Per-device arguments to unroll. A distributed
            :class:`~max.experimental.tensor.Tensor` contributes its
            ``local_shards[i]``; a ``list``/``tuple`` bundle contributes
            element ``i`` (one entry per device); a non-distributed
            ``Tensor`` broadcasts whole.
        kwargs: Broadcast arguments passed unchanged to every ``fn`` call.

    Returns:
        A ``list`` of per-device results for a single-output ``fn``, or a
        ``tuple`` of such lists when ``fn`` returns multiple values.
    """
    # Determine n based strictly on the distributed arguments
    n = _local_map_num_devices(distributed_kwargs.values())

    for k, v in distributed_kwargs.items():
        if isinstance(v, (list, tuple)) and len(v) != n:
            raise ValueError(
                f"Distributed kwarg '{k}' has {len(v)} entries, "
                f"but mesh unrolls to {n} devices."
            )

    # Execute per device
    with ensure_context():
        results = []
        for i in range(n):
            device_kwargs = {}
            for k, v in distributed_kwargs.items():
                if isinstance(v, Tensor):
                    # Distributed tensors contribute their per-device shard.
                    # A non-distributed tensor broadcasts whole so the
                    # single-device path (e.g. the base QuantizedMoE forward,
                    # whose router outputs live on a single-device mesh) can
                    # route its tensors through here too; ``local_shards`` is a
                    # 1-tuple in that case, so ``i`` is always 0.
                    device_kwargs[k] = (
                        v.local_shards[i] if v.is_distributed else v
                    )
                else:
                    device_kwargs[k] = v[i]

            results.append(fn(**device_kwargs, **kwargs))

    # Transpose multi-outputs
    if isinstance(results[0], (list, tuple)):
        return tuple(list(x) for x in zip(*results, strict=True))

    return results


def _local_functional_op(
    op: Callable[..., Any],
    return_input_sharding: str | None = None,
) -> Callable[..., Any]:
    """Wraps a kernel op to dispatch per-device on distributed inputs.

    Args:
        op: The underlying kernel function.
        return_input_sharding: The name of a tensor arg in `op`. If set, the
          sharding of the output tensor is set to the sharding of the input
          tensor at this arg.
    """
    sig = inspect.signature(op)
    if (
        return_input_sharding is not None
        and return_input_sharding not in sig.parameters
    ):
        raise ValueError(
            f"Input tensor arg {return_input_sharding} not found in {op.__name__}"
        )

    def run_graph_op(**kwargs: Any) -> Any:
        return op(
            **{
                k: TensorValue(v) if isinstance(v, Tensor) else v
                for k, v in kwargs.items()
            }
        )

    @functools.wraps(op)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Bind to parameter names so every argument can be routed through
        # local_map by keyword, then split into per-device (distributed)
        # and broadcast groups.
        named = sig.bind(*args, **kwargs).arguments

        num_devices = _local_map_num_devices(named.values())

        distributed_kwargs: dict[str, Any] = {}
        broadcast_kwargs: dict[str, Any] = {}
        for k, v in named.items():
            if isinstance(v, Tensor) and v.is_distributed:
                distributed_kwargs[k] = v
            elif isinstance(v, PagedCacheValues):
                # local_map has no notion of PagedCacheValues; pre-unroll it
                # into a per-device bundle it can index positionally.
                distributed_kwargs[k] = [
                    v.for_device(i) for i in range(num_devices)
                ]
            else:
                broadcast_kwargs[k] = v

        per_device = local_map(
            run_graph_op, distributed_kwargs, broadcast_kwargs
        )

        mapping = _get_mapping(
            op.__name__, sig, return_input_sharding, args, kwargs
        )
        # local_map returns a tuple of per-output shard lists for a
        # multi-output op, or a single per-device shard list otherwise.
        if isinstance(per_device, tuple):
            return [_reassemble(out, mapping) for out in per_device]
        return _reassemble(per_device, mapping)

    return wrapped


def _reassemble(
    shard_values: list[Any], mapping: PlacementMapping
) -> Tensor | None:
    """Reassembles a list of shard values into a distributed tensor."""
    if all(s is None for s in shard_values):
        return None
    return Tensor.from_shard_values(shard_values, mapping)


def _get_mapping(
    op_name: str,
    sig: inspect.Signature,
    return_input_sharding: str | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> PlacementMapping:
    placements: tuple[Placement, ...]
    if return_input_sharding is not None:
        # Get the input specified by return_input_sharding from args and kwargs.
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        input_sharding = bound_args.arguments[return_input_sharding]
        if not isinstance(input_sharding, Tensor):
            raise ValueError(
                f"Input tensor arg {return_input_sharding} passed to {op_name} must be a Tensor"
            )
        mesh = input_sharding.mesh
        placements = input_sharding.placements
    else:
        mesh = _find_mesh(*args, **kwargs)
        placements = tuple(Replicated() for _ in range(mesh.ndim))
    return PlacementMapping(mesh, placements)


def _find_mesh(*args: Any, **kwargs: Any) -> DeviceMesh:
    """Returns the DeviceMesh from the first distributed Tensor arg."""
    for a in (*args, *kwargs.values()):
        if isinstance(a, Tensor) and a.is_distributed:
            return a.mesh
        if isinstance(a, PagedCacheValues) and a.n_devices > 1:
            # Get mesh from one of the PagedCacheValues fields.
            return a.kv_blocks.mesh

    # Backup plan: use the mesh of the first Tensor arg.
    for a in (*args, *kwargs.values()):
        if isinstance(a, Tensor):
            return a.mesh
    raise ValueError("No distributed tensors found in args or kwargs")


def _local_map_num_devices(values: Iterable[Any]) -> int:
    """Infers the per-device unroll count for :func:`local_map`.

    Prefers the shard count of the first distributed :class:`Tensor`, then
    the length of the first per-device ``list``/``tuple`` bundle, else 1.
    """
    for a in values:
        if isinstance(a, Tensor) and a.is_distributed:
            return a.num_shards
        if isinstance(a, (list, tuple)):
            return len(a)
        if isinstance(a, PagedCacheValues) and a.n_devices > 1:
            return a.n_devices
    return 1


flash_attention_ragged = _local_functional_op(_flash_attention_ragged, "input")
rope_split_store_ragged = _local_functional_op(_rope_split_store_ragged, "qkv")
rms_norm_key_cache = _local_functional_op(_rms_norm_key_cache)
flare_mla_prefill_plan = _local_functional_op(_flare_mla_prefill_plan)
mla_prefill_graph = _local_functional_op(_mla_prefill_graph, "q")
mla_decode_graph = _local_functional_op(_mla_decode_graph, "q")
mla_prefill_decode_graph = _local_functional_op(_mla_prefill_decode_graph, "q")

fused_silu = _local_functional_op(_fused_silu, "input")

# Routing decisions must match the placement of the (replicated) router
# scores so every device agrees on expert assignment under TP/EP.
moe_router_group_limited = _local_functional_op(
    _moe_router_group_limited, "expert_scores"
)


__all__ = [
    "flash_attention_ragged",
    "fused_silu",
    "grouped_matmul_ragged",
    "local_map",
    "moe_create_indices",
    "moe_router_group_limited",
    "rms_norm_key_cache",
    "rope_split_store_ragged",
]
