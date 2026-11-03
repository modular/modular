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
"""Vision transformer for Qwen3.5.

The vision architecture is identical to Qwen3VL-MoE; re-export everything
from that package so we have a single implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from max import tree
from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorValue,
    Type,
    Value,
    Weight,
    ops,
)
from max.graph import Module as GraphModule
from max.nn.legacy.graph_groups import (
    VisionEmbeddings,
    group_fields,
    vision_outputs,
)
from max.nn.legacy.linear import linear
from max.pipelines.architectures.qwen3vl_moe.model_config import VisionConfig
from max.pipelines.architectures.qwen3vl_moe.nn.visual_transformer import (
    BilinearInterpolationPositionEmbedding,
    VisionBlock,
    VisionMLP,
    VisionPatchEmbed,
    VisionPatchMerger,
    VisionRotaryEmbedding,
    VisionTransformer,
)

__all__ = [
    "BilinearInterpolationPositionEmbedding",
    "Qwen3_5VisionTransformer",
    "VisionBlock",
    "VisionMLP",
    "VisionPatchEmbed",
    "VisionPatchMerger",
    "VisionRotaryEmbedding",
    "VisionTransformer",
]


class Qwen3_5VisionPatchEmbed(VisionPatchEmbed):
    """The patch embedding, holding the checkpoint's Conv3D weight.

    The checkpoint stores the projection as ``[embed_dim, in_channels,
    temporal_patch_size, patch_size, patch_size]``. The graph flattens it to
    the linear weight it equals when the stride is the kernel, so the
    checkpoint loads as stored.
    """

    def __init__(
        self,
        dtype: DType,
        devices: Sequence[DeviceRef],
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__(
            dtype,
            devices,
            patch_size,
            temporal_patch_size,
            in_channels,
            embed_dim,
        )
        self.proj.weight = Weight(
            "weight",
            dtype,
            [
                embed_dim,
                in_channels,
                temporal_patch_size,
                patch_size,
                patch_size,
            ],
            device=self.devices[0],
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Projects ``[n_patches, patch_dim]`` patches to the embedding."""
        assert self.proj.bias is not None
        weight = ops.reshape(
            self.proj.weight.to(x.device), [self.embed_dim, self.patch_dim]
        )
        return linear(x.cast(weight.dtype), weight) + self.proj.bias.to(
            x.device
        )


class Qwen3_5VisionTransformer(VisionTransformer):
    """The Qwen3VL vision transformer as the Qwen3.5 vision graph.

    The graph outputs only the merged image embeddings: Qwen3.5 does not use
    deepstack (intermediate visual features injected at multiple LM depths),
    which is a Qwen3VL-MoE feature.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__(config)
        self.patch_embed = Qwen3_5VisionPatchEmbed(
            dtype=config.dtype,
            devices=self.devices,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        self.patch_embed.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        self.patch_embed_shards = self.patch_embed.shard(self.devices)

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the vision graph, whose inputs are :meth:`input_tree`'s leaves.

        Groups the flat graph inputs into :meth:`__call__`'s per-device
        arguments by :meth:`input_tree`, and outputs the image embeddings
        by name.

        Args:
            *input_types: The graph's input types, the leaves of
                :meth:`input_tree`.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """
        treedef = tree.flatten(self.input_tree())[1]

        def forward(
            *inputs: Value[Any],
        ) -> OrderedDict[str, VisionEmbeddings]:
            named = tree.unflatten(treedef, iter(inputs))
            assert isinstance(named, OrderedDict)
            fields = group_fields(named)
            tensors = {
                key: [v.tensor for v in values]
                for key, values in fields.items()
                if key != "signal"
            }
            image_embeddings, _ = self(
                **tensors,
                signal_buffers=[v.buffer for v in fields["signal"]],
            )
            return vision_outputs(image_embeddings)

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
