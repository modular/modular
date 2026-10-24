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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from max import tree
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Type, Value
from max.graph import Module as GraphModule
from max.nn.comm import Signals
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


class Qwen3_5VisionTransformer(VisionTransformer):
    """The Qwen3VL vision transformer as the Qwen3.5 vision graph.

    The graph outputs only the merged image embeddings: Qwen3.5 does not use
    deepstack (intermediate visual features injected at multiple LM depths),
    which is a Qwen3VL-MoE feature.
    """

    def input_tree(self) -> OrderedDict[str, object]:
        """Names the vision graph's inputs; its leaves are its inputs.

        Each input but the signal buffers has one entry per device.

        Returns:
            The input types under :meth:`__call__`'s argument names, in
            graph order.
        """
        devices: list[DeviceRef] = list(self.devices)

        def per_device(
            dtype: DType, shape: list[int | str], *, on_host: bool = False
        ) -> tuple[TensorType, ...]:
            return tuple(
                TensorType(
                    dtype, shape=shape, device=DeviceRef.CPU() if on_host else d
                )
                for d in devices
            )

        return OrderedDict(
            pixel_values=per_device(
                DType.float32, ["vision_seq_len", self.patch_embed.patch_dim]
            ),
            # The bilinear position grid is interpolated once per FRAME while
            # pixels and rotary ids carry one row per temporal PATCH, so these
            # two are sum(h * w) long against pixel_values' sum(t * h * w).
            # Binding them to `vision_seq_len` would be an over-constraint
            # that only holds for still images (t == 1) and rejects every
            # clip.
            weights=per_device(DType.float32, [4, "vision_pos_len", 1]),
            idxs=per_device(DType.int64, [4, "vision_pos_len"]),
            rot_pos_ids=per_device(DType.int32, ["vision_seq_len", 2]),
            max_grid_size=per_device(DType.int32, [], on_host=True),
            grid_thw=per_device(DType.int64, ["n_images", 3]),
            cu_seqlens=per_device(DType.uint32, ["n_seqlens"]),
            max_seqlen=per_device(DType.uint32, [1], on_host=True),
            signal_buffers=tuple(Signals(devices=devices).input_types()),
        )

    def input_types(self) -> tuple[Type[Any], ...]:
        """Returns the vision graph's input types: the leaves of
        :meth:`input_tree`."""
        return tuple(tree.leaves(self.input_tree()))

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the vision graph, whose inputs are :meth:`input_types`.

        Groups the flat graph inputs into :meth:`__call__`'s per-device
        arguments by :meth:`input_tree`, and outputs the image embeddings.

        Args:
            *input_types: The graph's input types, :meth:`input_types`.
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

        def forward(*inputs: Value[Any]) -> list[TensorValue]:
            named = tree.unflatten(treedef, iter(inputs))
            assert isinstance(named, OrderedDict)

            def tensors(key: str) -> list[TensorValue]:
                return [v.tensor for v in named[key]]

            image_embeddings, _ = self(
                pixel_values=tensors("pixel_values"),
                idxs=tensors("idxs"),
                weights=tensors("weights"),
                grid_thw=tensors("grid_thw"),
                rot_pos_ids=tensors("rot_pos_ids"),
                max_grid_size=tensors("max_grid_size"),
                cu_seqlens=tensors("cu_seqlens"),
                max_seqlen=tensors("max_seqlen"),
                signal_buffers=[v.buffer for v in named["signal_buffers"]],
            )
            return image_embeddings

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
