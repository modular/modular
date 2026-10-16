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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
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


def apply_eplb_placement(
    model: Module,
    state_dict: Mapping[str, _T],
    phy2log: NDArray[np.integer],
) -> dict[str, _T]:
    """Returns ``state_dict`` rewritten so physical slots hold their experts.

    Walks ``model``'s module tree and asks every
    :class:`EplbExpertContainer` for its slot-to-checkpoint mapping. The
    returned dict aliases the input values; no tensor is copied.

    Args:
        model: The unsharded model whose weights ``state_dict`` targets.
        state_dict: Checkpoint weights keyed by logical expert FQNs.
        phy2log: ``[num_layers, num_physical_experts]`` logical expert id of
            each physical slot, indexed by ``layer_idx``.

    Returns:
        A new state dict keyed by physical expert FQNs, ready for
        ``model.load_state_dict``.

    Raises:
        ValueError: If a container has no ``layer_idx`` or ``phy2log`` has
            no row for it.
        KeyError: If a slot's source weight is not in ``state_dict``.
    """
    sources: dict[str, str] = {}
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
        row = [int(log) for log in phy2log[layer_idx]]
        for dst, src in module.eplb_weight_sources(row).items():
            sources[f"{prefix}{dst}"] = f"{prefix}{src}"

    # Read every source from the input dict and write into a copy: the
    # placement may permute experts in cycles, so updating in place would
    # read slots that were already overwritten.
    remapped = dict(state_dict)
    for dst, src in sources.items():
        if src not in state_dict:
            raise KeyError(
                f"EPLB placement maps '{dst}' to '{src}', which is not in "
                "the state dict"
            )
        remapped[dst] = state_dict[src]
    return remapped
