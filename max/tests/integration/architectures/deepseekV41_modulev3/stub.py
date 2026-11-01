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
"""A DeepSeek-V4.1-Flash ModuleV3 graph with the frozen interface and no math.

Every parameter of :func:`~.interface.parameter_specs` becomes a graph
constant, every forward input of the ABI is declared, and the layers share the
six block subgraphs plus ``engram``. Each body only reads one element of each
leaf it holds and folds it, times zero, into a carried scalar, so the graph
compiles and its weight registry names the whole tree while computing nothing.
The model slices replace the bodies; the tree, the inputs and the subgraph
grouping are the contract they build against.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Mapping, Sequence
from typing import Any

from max import functional as F
from max.driver import CPU, Accelerator, Device
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.quant_tensor import (
    NVFP4Tensor,
)
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.nn.sequential import ModuleList
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    ExpertForm,
    ParamSpec,
    V41Config,
    block_group,
    extra_input_names,
    parameter_specs,
)
from max.pipelines.lib import ModuleV3Outputs
from max.tensor import Tensor, default_device
from max.tensor.compilation import as_subgraph
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded

_NVFP4_LEAVES = frozenset(
    {"weight", "weight_scale", "weight_scale_2", "input_scale"}
)
_LAYER_PATH = re.compile(r"^layers\.(\d+)\.(.*)$")


def make_mesh(n_devices: int) -> DeviceMesh:
    """The 1-D tensor-parallel mesh over accelerators ``0..n_devices-1``."""
    return DeviceMesh(
        tuple(Accelerator(i) for i in range(n_devices)), (n_devices,), ("tp",)
    )


def _param(spec: ParamSpec, mesh: DeviceMesh) -> Tensor:
    """A zero parameter placed as ``spec.placement`` says."""
    if spec.placement == "cpu":
        return Tensor.zeros(spec.shape, dtype=spec.dtype, device=CPU())
    if spec.placement.startswith("gpu:"):
        rank = int(spec.placement.removeprefix("gpu:"))
        return Tensor.zeros(
            spec.shape, dtype=spec.dtype, device=mesh.devices[rank]
        )
    weight = Tensor.zeros(spec.shape, dtype=spec.dtype, device=mesh)
    if spec.placement == "rep":
        return weight
    axis = int(spec.placement.removeprefix("s"))
    return weight.to(DeviceMapping(mesh, (Sharded(axis),)))


class _Node(Module[..., Any]):
    """A bag of named parameters and sub-modules, the checkpoint's shape."""


class _Weight(_Node):
    """A module that is one ``weight``."""

    weight: Tensor


class _Ffn(_Node):
    """The routed experts and the rest of a layer's feed-forward."""

    experts: Module[..., Any]


class _Engram(_Node):
    """Engram's projection weights; the subgraph body that reads them."""

    def forward(
        self, carry: list[Tensor], tables: list[Tensor], inputs: list[Tensor]
    ) -> list[Tensor]:
        acc = _Carry(carry)
        for _, leaf in self.parameters:
            acc.touch(leaf)
        for leaf in (*tables, *inputs):
            acc.touch(leaf)
        return acc.values


_MODULES: Mapping[str, type[_Node]] = {
    "embed": _Weight,
    "norm": _Weight,
    "head": _Weight,
    "ffn": _Ffn,
    "engram": _Engram,
}


def _build(node: Any, mesh: DeviceMesh, path: str = "") -> Any:
    """Turns a nested ``{name: ParamSpec}`` mapping into modules and tensors."""
    if isinstance(node, ParamSpec):
        return _param(node, mesh)
    if node.keys() == _NVFP4_LEAVES:
        return NVFP4Tensor(
            **{leaf: _build(spec, mesh) for leaf, spec in node.items()}
        )
    if all(key.isdigit() for key in node):
        return ModuleList(
            [_build(node[str(i)], mesh) for i in range(len(node))]
        )
    module = _MODULES.get(path, _Node)()
    for name, child in node.items():
        setattr(
            module,
            name,
            _build(child, mesh, f"{path}.{name}" if path else name),
        )
    return module


def _nest(specs: Sequence[ParamSpec], strip: str = "") -> dict[str, Any]:
    """Nests spec names, relative to ``strip``, into ``{segment: ...}`` dicts."""
    root: dict[str, Any] = {}
    for spec in specs:
        *parents, leaf = spec.name.removeprefix(strip).split(".")
        node = root
        for segment in parents:
            node = node.setdefault(segment, {})
        node[leaf] = spec
    return root


class _Carry:
    """Per-device scalars the stub folds every parameter into.

    One slot per accelerator plus one for the host, so a weight is read on the
    device that holds it and no body needs a cross-device transfer.
    """

    def __init__(self, values: list[Tensor]) -> None:
        self.values = list(values)

    def touch(self, leaf: Tensor) -> None:
        """Folds one element of ``leaf`` into the slot of its device."""
        # A replicated or sharded leaf is read on its first rank; the graph
        # still declares every rank's shard as a constant.
        shard = leaf.local_shards[0]
        device: Device = shard.mapping.mesh.devices[0]
        slot = device.id if isinstance(device, Accelerator) else -1
        element = shard[(0,) * shard.rank] if shard.rank else shard
        self.values[slot] = (
            self.values[slot] + F.cast(element, DType.float32) * 0.0
        )


class DecoderLayer(Module[..., Any]):
    """One trunk layer: its parameters and a body that only reads them."""

    ffn: _Ffn
    engram: _Engram

    def __init__(
        self,
        node: Mapping[str, Any],
        mesh: DeviceMesh,
        group: str,
        experts: ExpertForm,
    ) -> None:
        for name, child in node.items():
            setattr(self, name, _build(child, mesh, name))
        self.group = group
        self.experts_form = experts
        self.n_devices = mesh.num_devices

    def _touch_experts(self, acc: _Carry) -> None:
        experts = self.ffn.experts
        if self.experts_form == "stacked":
            for _, leaf in experts.parameters:
                acc.touch(leaf)
            return
        assert isinstance(experts, ModuleList)
        per = len(experts) // self.n_devices
        for proj in ("w1", "w3", "w2"):
            for rank in range(self.n_devices):
                local = [
                    getattr(experts[e], proj).weight
                    for e in range(rank * per, (rank + 1) * per)
                ]
                # The per-rank stack of local experts, as the grouped matmul
                # will read them.
                for leaf_name in sorted(_NVFP4_LEAVES):
                    acc.touch(
                        F.stack([getattr(w, leaf_name) for w in local], axis=0)
                    )

    def forward(
        self,
        carry: list[Tensor],
        row_offsets: Tensor,
        layer_ids: list[Tensor],
        routed: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Reads every leaf and operand; returns the carry and what it routes.

        ``routed`` is the top-k (or candidate pool) a reuse or reindex body
        reads; a full or reindex body returns the top-k it computes, and the
        candidate source also its pool.
        """
        acc = _Carry(carry)
        for name, leaf in self.parameters:
            # Engram runs in its own subgraph, ahead of the body.
            if not name.startswith(("engram.", "ffn.experts")):
                acc.touch(leaf)
        self._touch_experts(acc)
        for operand in (row_offsets, *layer_ids, *routed):
            acc.touch(operand)
        produced: list[Tensor] = []
        if self.group.startswith(("full", "reindex")):
            top_k = F.reshape(F.cast(acc.values[0], DType.int32), [1])
            produced.append(top_k)
            if self.group.endswith("_cand"):
                produced.append(top_k + 1)
        return acc.values, produced


class DeepseekV41Stub(Module[..., ModuleV3Outputs]):
    """The frozen V4.1 parameter tree and forward ABI, with no model math."""

    embed: _Weight
    norm: _Weight
    head: _Weight

    def __init__(
        self,
        config: V41Config,
        kv_params: MultiKVCacheParams,
        mesh: DeviceMesh,
        experts: ExpertForm = "per_expert",
    ) -> None:
        self.config = config
        self.kv_params = kv_params
        self.mesh = mesh
        specs = parameter_specs(config, mesh.num_devices, experts)
        top = [s for s in specs if not _LAYER_PATH.match(s.name)]
        by_layer: dict[int, list[ParamSpec]] = {}
        for spec in specs:
            if match := _LAYER_PATH.match(spec.name):
                by_layer.setdefault(int(match[1]), []).append(spec)
        with default_device(mesh):
            for name, child in _nest(top).items():
                setattr(self, name, _build(child, mesh, name))
            self.layers = ModuleList(
                [
                    DecoderLayer(
                        _nest(by_layer[layer], f"layers.{layer}."),
                        mesh,
                        block_group(config, layer),
                        experts,
                    )
                    for layer in range(config.num_hidden_layers)
                ]
            )

    def _layer_ids(self, layer: int) -> list[Tensor]:
        """The leaf-local layer indices a body takes as graph operands.

        A window index is the layer itself; a compressed-leaf index is the
        position, among the sources of its ratio, of the nearest one at or
        before the layer.
        """
        config = self.config
        ids = [layer]
        if ratio := config.ratio(layer):
            owners = [
                l
                for l in config.kv_source_layer_ids
                if config.ratio(l) == ratio
            ]
            ids.append(max(bisect.bisect_right(owners, layer) - 1, 0))
        return [F.constant(i, DType.uint32, device=CPU()) for i in ids]

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *rest: Tensor,
    ) -> ModuleV3Outputs:
        config = self.config
        n_kv = len(self.kv_params.flattened_kv_inputs())
        kv_inputs = self.kv_params.unflatten_kv_inputs(
            iter(x._graph_value for x in rest[:n_kv])
        )
        assert len(kv_inputs) == len(self.kv_params.children)
        extras = dict(zip(extra_input_names(config), rest[n_kv:], strict=True))

        devices = self.mesh.devices
        acc = _Carry(
            [F.constant(0.0, DType.float32, device=d) for d in devices]
            + [F.constant(0.0, DType.float32, device=CPU())]
        )
        for leaf in (
            tokens,
            return_n_logits,
            extras["swa_begin"],
            extras["global_write_floor"],
            self.embed.weight,
            self.norm.weight,
            self.head.weight,
        ):
            acc.touch(leaf)
        carry = acc.values

        tables = [
            extras[f"engram_table_{i}_{part}"]
            for i in range(len(config.engram_layer_ids))
            for part in ("weight", "scale")
        ]
        engram_inputs = [
            extras[name]
            for name in (
                "engram_lookback",
                "engram_lookback_dead",
                "image_token_mask",
                "engram_token_map",
            )
        ]
        top_k: dict[int, Tensor] = {}
        pool: Tensor | None = None
        for index, layer in enumerate(self.layers):
            if index in config.engram_layer_ids:
                ordinal = F.constant(
                    config.engram_layer_ids.index(index),
                    DType.uint32,
                    device=CPU(),
                )
                carry = as_subgraph(layer.engram, name="engram")(
                    carry, tables, [ordinal, *engram_inputs]
                )
            group = layer.group
            reads = {
                "reuse_r2": top_k.get(2),
                "reindex_r1": pool,
                "reuse_r1": top_k.get(1),
            }.get(group)
            carry, produced = as_subgraph(layer, name=group)(
                carry,
                input_row_offsets,
                self._layer_ids(index),
                [] if reads is None else [reads],
            )
            if group == "full_r2":
                (top_k[2],) = produced
            elif group == "full_r1_cand":
                top_k[1], pool = produced
            elif group == "reindex_r1":
                (top_k[1],) = produced

        total = carry[0]
        for value in carry[1:]:
            total = total + value.to(devices[0])
        rows = F.reshape(F.cast(input_row_offsets[1:], DType.float32), [-1, 1])
        logits = F.broadcast_to(rows * 0.0, [rows.shape[0], config.vocab_size])
        return ModuleV3Outputs(next_token_logits=logits + total)
