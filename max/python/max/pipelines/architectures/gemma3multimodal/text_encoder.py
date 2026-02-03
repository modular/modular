# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Gemma3 text encoder for diffusion-based visual generation pipelines."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import (
    SafetensorWeights,
    Weights,
)
from max.interfaces import TokenBuffer
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
    RaggedKVCacheInputs,
)
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    CompilationTimer,
    SupportedEncoding,
)
from max.pipelines.lib.interfaces.component_model import ComponentModel

from .vision_model.gemma3multimodal import Gemma3LanguageModel
from .model_config import Gemma3ForConditionalGenerationConfig
from .weight_adapters import convert_safetensor_state_dict

logger = logging.getLogger(__name__)


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class DummyKVManager:
    """
    Provides dummy KV-cache inputs required by the compiled graph.

    This class is NOT coupled to KVCacheParams. It derives everything from
    stable Gemma3Config fields (plus a few explicit knobs like page_size).

    Assumed graph signature:
      - blocks       : bfloat16, (total_num_pages, 2, num_layers, page_size, n_kv_heads, head_dim) on GPU
      - cache_lengths: uint32,   (batch_size,) on GPU
      - lookup_table : uint32,   (batch_size, lookup_table_width) on GPU
      - max_lengths  : uint32,   (steps_remaining, 2) on CPU
    """

    def __init__(
        self,
        *,
        total_num_pages: int,
        page_size: int,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        device: Device,
        lookup_table_width: int = 4,
    ) -> None:
        self.total_num_pages = int(total_num_pages)
        self.page_size = int(page_size)
        self.num_layers = int(num_layers)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.device = device
        self.lookup_table_width = int(lookup_table_width)

    @classmethod
    def from_config(
        cls,
        gemma3_config: Gemma3ForConditionalGenerationConfig,
        *,
        device: Device | None = None,
        page_size: int = 128,
        lookup_table_width: int = 4,
    ) -> DummyKVManager:
        """
        Build DummyKVManager from Gemma3Config without using kv_params.

        Derivations:
          - num_layers      = gemma3_config.num_hidden_layers
          - n_kv_heads      = gemma3_config.num_key_value_heads
          - head_dim        = gemma3_config.head_dim
          - total_num_pages = ceil(gemma3_config.max_seq_len / page_size)

        """
        if device is None:
            device = gemma3_config.devices[0]

        max_seq_len = int(gemma3_config.max_seq_len)
        if max_seq_len <= 0:
            raise ValueError(
                f"gemma3_config.max_seq_len must be > 0, got {max_seq_len}"
            )

        total_num_pages = _ceildiv(max_seq_len, int(page_size))

        return cls(
            total_num_pages=total_num_pages,
            page_size=int(page_size),
            num_layers=int(gemma3_config.num_hidden_layers),
            n_kv_heads=int(gemma3_config.num_key_value_heads),
            head_dim=int(gemma3_config.head_dim),
            device=device,
            lookup_table_width=int(lookup_table_width),
        )

    def create_dummy_inputs(
        self, input_row_offsets: Buffer, num_steps: int = 1
    ) -> RaggedKVCacheInputs:
        """
        Create a dummy RaggedKVCacheInputs object for the given ragged batch.

        Args:
            input_row_offsets: uint32 buffer of shape (B+1,) on GPU.
            num_steps: Number of decoding steps to account for (defaults to 1).

        Returns:
            RaggedKVCacheInputs that can be splatted into model.execute(..., *inputs).

        Raises:
            ValueError: If required_pages exceeds the fixed lookup_table width (4).
        """
        # Derive batch size and per-row lengths from input_row_offsets.
        # NOTE: This assumes input_row_offsets.to_numpy() works for a GPU buffer in your environment.
        offs = input_row_offsets.to_numpy()  # shape (B+1,)
        batch_size = int(offs.shape[0] - 1)

        row_lens = [int(offs[i + 1] - offs[i]) for i in range(batch_size)]
        max_seq_len = max(row_lens) + int(num_steps) - 1
        required_pages = _ceildiv(max_seq_len, self.page_size)

        # Your compiled graph has lookup_table width fixed to 4.
        if required_pages > 4:
            raise ValueError(
                f"required_pages={required_pages} > 4, but lookup_table is fixed to (B,4) in this graph. "
                f"max_seq_len={max_seq_len}, page_size={self.page_size}"
            )

        # Allocate device-side buffers.
        blocks = Buffer(
            DType.bfloat16,
            (
                self.total_num_pages,
                2,
                self.num_layers,
                self.page_size,
                self.n_kv_heads,
                self.head_dim,
            ),
            self.device,
        )
        cache_lengths = Buffer(DType.uint32, (batch_size,), self.device)
        lookup_table = Buffer(DType.uint32, (batch_size, 4), self.device)

        # max_lengths is expected on CPU for your graph (Buffer(...) defaults to CPU in your setup).
        max_lengths = Buffer(DType.uint32, (1, 2))

        # Allocate CPU staging buffers for initializing/copying metadata.
        cache_lengths_host = Buffer(DType.uint32, (batch_size,))
        lookup_table_host = Buffer(DType.uint32, (batch_size, 4))

        # Initialize cache_lengths = 0 (no cached tokens).
        cache_np = cache_lengths_host.to_numpy()
        cache_np.fill(0)

        # Initialize lookup_table with sentinel, then fill required pages with [0..required_pages-1].
        lut_np = lookup_table_host.to_numpy()
        lut_np.fill(
            np.uint32(self.total_num_pages)
        )  # sentinel = invalid page index
        for b in range(batch_size):
            for p in range(required_pages):
                lut_np[b, p] = np.uint32(p)

        # Initialize max_lengths conservatively to the maximum prompt length in the batch.
        # (These values are often used for bounds checks / slice limits.)
        ml_np = max_lengths.to_numpy()
        ml_np[0, 0] = np.uint32(max(row_lens))  # max_prompt_len
        ml_np[0, 1] = np.uint32(max(row_lens))  # max_cached_len (prefill-style)

        # Copy CPU metadata -> GPU buffers.
        cache_lengths.inplace_copy_from(cache_lengths_host)
        lookup_table.inplace_copy_from(lookup_table_host)

        return RaggedKVCacheInputs(
            blocks=blocks,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_lengths=max_lengths,
            kv_scales=None,
        )


class Gemma3TextEncoderModel(ComponentModel):
    """Gemma3 text encoder wrapper implementing ComponentModel interface.

    This class builds and compiles a Gemma3 model graph directly for text encoding,
    without depending on PipelineModel or Gemma3Tokenizer. It exposes a simple
    interface that returns hidden states from all layers.

    Note: Although text encoding is a single forward pass operation and doesn't
    actually use KV cache for multi-step generation, the compiled model graph
    requires KV cache inputs as part of its interface. We allocate minimal KV
    cache to satisfy the graph requirements.
    """

    def __init__(
        self,
        config: dict,
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
    ) -> None:
        """Initialize Gemma3TextEncoderModel.

        Args:
            config: Configuration dictionary from model config file.
            encoding: Supported encoding for the model.
            devices: List of devices to use.
            weights: Model weights.
        """
        super().__init__(config, encoding, devices, weights)

        # Lazy initialization attributes (set in load_model)
        self._model: Model | None = None
        self._session: InferenceSession | None = None
        self._kv_manager: DummyKVManager | None = None
        self._config = config

        # Load model during initialization
        self.load_model()

    def load_model(self) -> Callable[..., Any]:
        """Load pretrained model weights and compile the model graph.

        This method builds and compiles the Gemma3 graph directly without
        depending on PipelineModel. It initializes the KV cache manager
        directly for use during inference.

        Returns:
            Compiled model callable (Model instance).
        """
        self._session = InferenceSession(devices=self.devices)

        text_config = self._config["text_config"]
        dtype = getattr(DType, self._config["dtype"])

        gemma3_config = Gemma3ForConditionalGenerationConfig(
            hidden_size=int(text_config["hidden_size"]),
            num_attention_heads=int(text_config["num_attention_heads"]),
            num_key_value_heads=int(text_config["num_key_value_heads"]),
            num_hidden_layers=int(text_config["num_hidden_layers"]),
            head_dim=int(text_config["head_dim"]),
            vocab_size=int(text_config["vocab_size"]),
            rope_theta=float(text_config.get("rope_theta", 10000.0)),
            rms_norm_eps=float(text_config.get("rms_norm_eps", 1e-5)),
            feed_forward_length=int(text_config.get("intermediate_size", 0)),
            dtype=dtype,
            max_seq_len=int(text_config.get("max_position_embeddings", 0)),
            kv_params=KVCacheParams(
                dtype=dtype,
                n_kv_heads=int(text_config["num_key_value_heads"]),
                head_dim=int(text_config["head_dim"]),
                num_layers=int(text_config["num_hidden_layers"]),
                devices=[DeviceRef.from_device(d) for d in self.devices],
            ),
            attention_multiplier=math.sqrt(1 / text_config["head_dim"]),
            devices=list(self.devices),
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.ALL_LAYERS,
        )

        self._kv_manager = DummyKVManager.from_config(gemma3_config)

        self._model = self._build_and_load_model(gemma3_config)

        return self._model

    def _build_and_load_model(
        self,
        gemma3_config: Gemma3ForConditionalGenerationConfig,
    ) -> Model:
        """Build and load the Gemma3 model graph.

        Args:
            gemma3_config: HuggingFace text model configuration.
            adapter: Optional weight adapter.

        Returns:
            Compiled Model instance.
        """
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are currently supported."
            )

        state_dict = convert_safetensor_state_dict(dict(self.weights.items()))

        # Build graph inputs
        device_ref = DeviceRef.from_device(self.devices[0])
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        # Get KV cache input types
        kv_inputs = gemma3_config.kv_params.get_symbolic_inputs()

        graph_inputs = (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *kv_inputs[0],
        )

        # Build the neural network model
        nn_model = Gemma3LanguageModel(gemma3_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=False,
        )

        # Build the graph
        timer = CompilationTimer("text_encoder_model")
        with Graph("gemma3_text_encoder", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collection = PagedCacheValues(
                kv_blocks=kv_cache_inputs[0].buffer,
                cache_lengths=kv_cache_inputs[1].tensor,
                lookup_table=kv_cache_inputs[2].tensor,
                max_lengths=kv_cache_inputs[3].tensor,
            )
            outputs = nn_model(
                tokens.tensor,
                kv_collection,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)

        timer.mark_build_complete()
        model = self._session.load(
            graph, weights_registry=nn_model.state_dict()
        )
        timer.done()

        return model

    def __call__(
        self,
        tokens: TokenBuffer,
        attention_mask: np.ndarray | None = None,
        position_ids: np.ndarray | None = None,
    ) -> tuple[Buffer, ...]:
        """Apply Gemma3 text encoder forward pass.

        Args:
            input_ids: Input token IDs as numpy array.
            attention_mask: Attention mask (not used, kept for compatibility).
            position_ids: Position IDs (not used, kept for compatibility).

        Returns:
            Tuple of hidden states from all layers as MAX Buffers.

        Raises:
            RuntimeError: If model is not loaded.
        """
        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [tokens.active_length],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        next_tokens_batch = Buffer.from_numpy(tokens.active).to(self.devices[0])

        return_n_logits = Buffer.from_numpy(np.array([1], dtype=np.int64))

        dummy_inputs = self._kv_manager.create_dummy_inputs(input_row_offsets)

        # Execute the model
        model_outputs = self._model.execute(
            next_tokens_batch,
            input_row_offsets,
            return_n_logits,
            *dummy_inputs,
        )

        hidden_states = tuple(model_outputs[1:])
        return hidden_states

    @property
    def session(self) -> InferenceSession:
        """Return the InferenceSession instance.

        Returns:
            InferenceSession: The compiled inference session.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._session
