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

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from max import tree
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorValue, Type, Value, ops
from max.graph import Module as GraphModule
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import DecoderOutputs
from max.pipelines.architectures.llama3.llama3 import Llama3
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_batch_inputs,
    vision_merge_inputs,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings


class Idefics3LanguageModel(Llama3):
    """Implements the language model component of Idefics3.

    This model is based on Llama3 and is designed to process
    text tokens and generate language outputs. It supports multimodal inputs
    through embedding merging before passing to the standard Llama3 forward pass.

    The model consists of:
    - Standard Llama3 architecture (inherited)
    - Multimodal embedding merging capability
    - Image token processing
    """

    def __init__(
        self,
        config: Llama3Config,
        image_token_id: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initializes the Idefics3 language model.

        Args:
            config: The Llama3Config containing model parameters for the text component.
            image_token_id: Token ID used to represent image tokens in the text sequence.
            dtype: Data type for the model weights.
            device: Device for the model computation.
        """
        # Initialize the parent Llama3 model with the text config
        super().__init__(config)

        self.config = config
        self.device = device
        self.dtype = dtype

        # Store image token ID for multimodal processing
        self.image_token_id = image_token_id

    def input_tree(
        self,
        kv_params: KVCacheParamInterface,
        needs_hidden_state_input: bool = False,
    ) -> OrderedDict[str, object]:
        """Names the language graph's inputs; its leaves are its inputs.

        Image embeddings are always inputs: a text-only batch feeds zero rows.

        Args:
            kv_params: The KV cache parameters.
            needs_hidden_state_input: Must be ``False``: Idefics3 has no
                draft model that takes hidden states.

        Returns:
            The input types under ``batch``, ``vision`` and ``kv``, in graph
            order.
        """
        if needs_hidden_state_input:
            raise ValueError("Idefics3 takes no hidden-state input")
        return OrderedDict(
            batch=ragged_batch_inputs(
                self.device, ("tokens", "row_offsets", "return_n_logits")
            ),
            vision=vision_merge_inputs(
                [self.device],
                self.dtype,
                self.config.hidden_size,
                rows_dim="num_image_tokens",
            ),
            kv=kv_params.get_symbolic_inputs(),
        )

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the language graph, whose inputs are :meth:`input_tree`'s
        leaves.

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
        # The KV cache comes last, and unflattens through the KV params.
        head = OrderedDict(
            (key, types)
            for key, types in self.input_tree(self.kv_params).items()
            if key != "kv"
        )
        head_treedef = tree.flatten(head)[1]

        def forward(*inputs: Value[Any]) -> DecoderOutputs:
            remaining = iter(inputs)
            named = tree.unflatten(head_treedef, remaining, exact=False)
            assert isinstance(named, OrderedDict)
            batch, vision = named["batch"], named["vision"]
            kv_collections = tree.leaves(
                self.kv_params.unflatten_kv_inputs(remaining),
                leaf=KVCacheInputsPerDevice,
            )
            return self(
                tokens=batch["tokens"].tensor,
                kv_collection=kv_collections[0],
                return_n_logits=batch["return_n_logits"].tensor,
                input_row_offsets=batch["row_offsets"].tensor,
                image_embeddings=vision["image_embeddings"][0].tensor,
                image_token_indices=vision["image_token_indices"][0].tensor,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )

    def __call__(  # type: ignore[override]
        self,
        tokens: TensorValue,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        image_embeddings: TensorValue,
        image_token_indices: TensorValue,
    ) -> DecoderOutputs:
        """Executes the language model forward pass with multimodal embedding merging.

        Args:
            tokens: Input token IDs to process.
            kv_cache_inputs: KV cache inputs for each device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Offsets for flattened input sequences.
            image_embeddings: Image embeddings to merge into text embeddings,
                one per device. Can be empty tensors for text-only inputs.
            image_token_indices: Indices where image tokens should be placed.

        Returns:
            A tuple containing the output logits.
        """
        # Get text embeddings using the inherited embedding layer
        h = self.embed_tokens(tokens)

        h = merge_multimodal_embeddings(
            inputs_embeds=h,
            multimodal_embeddings=image_embeddings,
            image_token_indices=image_token_indices,
        )

        # Run through decoder layers using inherited layers
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=self.rope.freqs_cis,
                input_row_offsets=input_row_offsets,
            )

        return self._postprocess_logits(h, input_row_offsets, return_n_logits)
