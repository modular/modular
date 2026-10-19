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
"""Applies an expert-parallel load-balancing (EPLB) placement to weights.

A placement assigns a logical expert to every physical expert slot, per MoE
layer. The compiled graph only ever names physical slots, so every MoE layer
has the same weight names modulo the layer prefix and one subgraph serves all
of them. The placement is applied to the weight *data* instead: before
``load_state_dict``, slot ``p`` of layer ``L`` is pointed at the checkpoint
tensors of logical expert ``phy2log[L, p]``.

A placement that only permutes experts (no replicas) can also be folded into
the router: reordering the router's expert rows the same way makes it emit
physical ids, so dispatch needs no id remap at all.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData
from numpy.typing import NDArray

from ..layer import Module

_T = TypeVar("_T")


@runtime_checkable
class EplbExpertContainer(Protocol):
    """A module that owns physical expert slots an EPLB placement can fill."""

    layer_idx: int | None
    """Row of the placement that applies to this module."""

    def eplb_weight_sources(self, phy2log: Sequence[int]) -> dict[str, str]:
        """Maps each physical-slot weight to the checkpoint weight it loads.

        Args:
            phy2log: The logical expert id of every physical slot in this
                layer.

        Returns:
            A map from destination FQN to source FQN, both relative to this
            module.
        """
        ...

    def eplb_router_weights(self) -> list[str] | None:
        """Returns the router weights whose leading axis indexes experts.

        Returns:
            FQNs relative to this module, or ``None`` if the router cannot
            be folded.
        """
        ...


def _containers(
    model: Module, phy2log: NDArray[np.integer]
) -> Iterator[tuple[str, EplbExpertContainer, list[int]]]:
    """Yields every container with its prefix and placement row."""
    for prefix, module in model.named_modules():
        if not isinstance(module, EplbExpertContainer):
            continue
        layer_idx = module.layer_idx
        if layer_idx is None:
            raise ValueError(
                f"EPLB placement needs layer_idx on the MoE at '{prefix}'"
            )
        if not 0 <= layer_idx < phy2log.shape[0]:
            raise ValueError(
                f"EPLB placement has {phy2log.shape[0]} layers but the MoE at "
                f"'{prefix}' has layer_idx={layer_idx}"
            )
        yield prefix, module, [int(log) for log in phy2log[layer_idx]]


def router_fold_supported(model: Module, phy2log: NDArray[np.integer]) -> bool:
    """Returns whether ``phy2log`` can be folded into every MoE's router.

    Requires every placement row to be a permutation of the logical experts
    and every router to expose all of its expert-indexed weights.
    """
    found = False
    for _prefix, module, row in _containers(model, phy2log):
        found = True
        if sorted(row) != list(range(len(row))):
            return False
        if module.eplb_router_weights() is None:
            return False
    return found


def apply_eplb_placement(
    model: Module,
    state_dict: Mapping[str, _T],
    phy2log: NDArray[np.integer],
    *,
    fold_router: bool = False,
) -> dict[str, _T | Buffer | WeightData]:
    """Returns ``state_dict`` rewritten so physical slots hold their experts.

    Walks ``model``'s module tree and asks every
    :class:`EplbExpertContainer` for its slot-to-checkpoint mapping. Expert
    weights are aliased, not copied. With ``fold_router``, each router's
    expert rows are also reordered into physical slot order, which copies
    those small tensors.

    Args:
        model: The unsharded model whose weights ``state_dict`` targets.
        state_dict: Checkpoint weights keyed by logical expert FQNs.
        phy2log: ``[num_layers, num_physical_experts]`` logical expert id of
            each physical slot, indexed by ``layer_idx``.
        fold_router: Whether to reorder router rows too. Requires
            :func:`router_fold_supported`.

    Returns:
        A new state dict keyed by physical expert FQNs, ready for
        ``model.load_state_dict``.

    Raises:
        ValueError: If a container has no ``layer_idx``, ``phy2log`` has no
            row for it, or ``fold_router`` is set for a placement that
            cannot be folded.
        KeyError: If a slot's source weight is not in ``state_dict``.
    """
    if fold_router and not router_fold_supported(model, phy2log):
        raise ValueError("this EPLB placement cannot be folded into the router")

    sources: dict[str, str] = {}
    row_orders: dict[str, list[int]] = {}
    for prefix, module, row in _containers(model, phy2log):
        for dst, src in module.eplb_weight_sources(row).items():
            sources[f"{prefix}{dst}"] = f"{prefix}{src}"
        if fold_router:
            for name in module.eplb_router_weights() or []:
                row_orders[f"{prefix}{name}"] = row

    # Read every source from the input dict and write into a copy: the
    # placement may permute experts in cycles, so updating in place would
    # read slots that were already overwritten.
    remapped: dict[str, _T | Buffer | WeightData] = dict(state_dict)
    for dst, src in sources.items():
        if src not in state_dict:
            raise KeyError(
                f"EPLB placement maps '{dst}' to '{src}', which is not in "
                "the state dict"
            )
        remapped[dst] = state_dict[src]
    for name, row in row_orders.items():
        if name not in state_dict:
            raise KeyError(f"router weight '{name}' is not in the state dict")
        remapped[name] = _gather_rows(state_dict[name], row)
    return remapped


def _gather_rows(value: object, rows: Sequence[int]) -> Buffer | WeightData:
    """Returns ``value[rows]`` along axis 0, for any dtype.

    Rows are moved as raw bytes, so bfloat16 and float8 tensors, which NumPy
    cannot represent, reorder the same way as any other.
    """
    if isinstance(value, WeightData):
        return WeightData(
            data=_gather_rows(value.data, rows),
            name=value.name,
            dtype=value.dtype,
            shape=value.shape,
            quantization_encoding=value.quantization_encoding,
        )
    if isinstance(value, np.ndarray):
        buf = Buffer.from_numpy(np.ascontiguousarray(value))
    else:
        buf = Buffer.from_dlpack(value)
    shape = list(buf.shape)
    row_bytes = math.prod(shape[1:]) * buf.dtype.size_in_bytes
    raw = buf.view(DType.uint8, [shape[0], row_bytes]).to_numpy()
    gathered = Buffer.from_numpy(np.ascontiguousarray(raw[list(rows)]))
    return gathered.view(buf.dtype, shape)
