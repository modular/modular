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
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from max.experimental import functional as F
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import (
    DeviceMesh,
    Placement,
    PlacementMapping,
    Replicated,
)
from max.experimental.sharding.rules import RuleSignature
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
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
    rms_norm_key_cache as _rms_norm_key_cache,
)
from max.nn.kernels import (
    rope_split_store_ragged as _rope_split_store_ragged,
)

# Define placement for independent tensors, representing a bundle of tensors
# on different devices, with the same shape and dtype, but will not have the
# same values.
# This placement uses "replicated" operation semantics, but they are not really
# replicated.
Independent = Replicated


def grouped_matmul_ragged_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    expert_start_indices: TensorLayout,
    expert_ids: TensorLayout,
    expert_usage_stats_host: TensorLayout,
) -> RuleSignature:
    return (
        (
            hidden_states,
            weight,
            expert_start_indices,
            expert_ids,
            expert_usage_stats_host,
        ),
        (weight.mapping,),
    )


grouped_matmul_ragged = F.functional(
    _grouped_matmul_ragged, rule=grouped_matmul_ragged_rule
)


def _moe_create_indices_rule(lhs: TensorLayout, *args: object) -> RuleSignature:
    mesh = lhs.mapping.mesh
    return ((lhs, *args), (PlacementMapping(mesh, (Independent(),)),))


moe_create_indices = F.functional(
    _moe_create_indices, rule=_moe_create_indices_rule
)

inplace_custom = F.functional(ops.inplace_custom)
shard_and_stack = F.functional(ops.shard_and_stack)


# ─── KVCache Operations ─────────────────────────────────────


def _wrap_kvcache_op(
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

    @functools.wraps(op)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        results: dict[int, list[Any]] = defaultdict(list)
        for a, kw in _loop_distributed_args(args, kwargs):
            output = op(*a, **kw)
            if isinstance(output, TensorValue):
                results[0].append(output)
            elif isinstance(output, (list, tuple)):
                for i, result in enumerate(output):
                    results[i].append(result)
            elif output is None:
                continue
            else:
                raise TypeError(
                    f"Unexpected result type {type(output)} from {op.__name__}"
                )

        if not results:
            return None

        tensor_results: list[Tensor] = []
        mapping = _get_mapping(
            op.__name__, sig, return_input_sharding, args, kwargs
        )
        for i in range(len(results)):
            result = results[i]
            if isinstance(result, TensorValue):
                tensor_results.append(Tensor.from_shard_values(result, mapping))
            elif isinstance(result, (list, tuple)):
                tensor_results.append(Tensor.from_shard_values(result, mapping))
            else:
                raise TypeError(f"Unexpected result type: {type(results[0])}")
        if len(tensor_results) == 1:
            return tensor_results[0]
        else:
            return tensor_results

    return wrapped


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
        placements = tuple(Independent() for _ in range(mesh.ndim))
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


def _shard(a: Any, i: int) -> Any:
    if isinstance(a, Tensor):
        if a.is_distributed:
            return TensorValue(a.local_shards[i])
        else:
            return TensorValue(a)
    if isinstance(a, PagedCacheValues):
        return a.for_device(i)
    return a


def _loop_distributed_args(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Unwrap distributed args into per-device (args, kwargs) tuples.

    Tensor args are split via ``local_shards``, PagedCacheValues via
    ``for_device``, and everything else is broadcast unchanged.
    """
    num_devices = 1
    for a in (*args, *kwargs.values()):
        if isinstance(a, Tensor) and a.is_distributed:
            num_devices = a.num_shards
            break
        if isinstance(a, PagedCacheValues) and a.n_devices > 1:
            num_devices = a.n_devices
            break

    return [
        (
            tuple(_shard(a, i) for a in args),
            {k: _shard(v, i) for k, v in kwargs.items()},
        )
        for i in range(num_devices)
    ]


flash_attention_ragged = _wrap_kvcache_op(_flash_attention_ragged, "input")
rope_split_store_ragged = _wrap_kvcache_op(_rope_split_store_ragged, "qkv")
rms_norm_key_cache = _wrap_kvcache_op(_rms_norm_key_cache)
flare_mla_prefill_plan = _wrap_kvcache_op(_flare_mla_prefill_plan)
mla_prefill_graph = _wrap_kvcache_op(_mla_prefill_graph, "q")
mla_decode_graph = _wrap_kvcache_op(_mla_decode_graph, "q")
mla_prefill_decode_graph = _wrap_kvcache_op(_mla_prefill_decode_graph, "q")


__all__ = [
    "flash_attention_ragged",
    "grouped_matmul_ragged",
    "moe_create_indices",
    "rms_norm_key_cache",
    "rope_split_store_ragged",
]
