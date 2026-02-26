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

import gc

import torch
from max._core.engine import PrintStyle
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.nn import Signals
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3TextScaledWordEmbedding,
)


def generate_torch_outputs(
    text_config: Gemma3TextConfig,
    input_indices: torch.Tensor,
    embedding_weights: torch.Tensor,
    embed_scale: float = 1.0,
) -> torch.Tensor:
    layer = (
        Gemma3TextScaledWordEmbedding(
            num_embeddings=text_config.vocab_size,
            embedding_dim=text_config.hidden_size,
            padding_idx=text_config.pad_token_id,
            embed_scale=embed_scale,
        )
        .to(torch.bfloat16)
        .to("cuda")
    )

    layer.weight.data = embedding_weights.to(torch.bfloat16).to("cuda")
    return layer(input_indices.to("cuda")).to(torch.bfloat16)


def generate_max_outputs(
    text_config: Gemma3TextConfig,
    input_indices: torch.Tensor,
    embedding_weights: torch.Tensor,
    embed_scale: float = 1.0,
) -> torch.Tensor:
    devices = [DeviceRef.GPU()]
    layer = ScaledWordEmbedding(
        vocab_size=text_config.vocab_size,
        hidden_dim=text_config.hidden_size,
        dtype=DType.bfloat16,
        devices=devices,
        quantization_encoding=None,
        name="embeddings",
        embed_scale=embed_scale,
    )

    # Weights need to be passed as CPU tensors
    state_dict = {"embeddings": embedding_weights.cpu()}
    layer.load_state_dict(state_dict)

    session = InferenceSession(devices=[Accelerator()])
    session.set_debug_print_options(style=PrintStyle.COMPACT)

    # Create signal types for distributed communication
    signals = Signals(devices=devices)

    with Graph(
        "ScaledWordEmbedding",
        input_types=[
            TensorType(
                DType.int64,
                Shape(input_indices.shape),
                device=DeviceRef.GPU(),
            ),
            *signals.input_types(),
        ],
    ) as graph:
        indices, *signal_inputs = graph.inputs
        signal_buffers = [s.buffer for s in signal_inputs]

        # Call the layer with signal buffers
        result = layer(indices.tensor, signal_buffers)
        graph.output(result[0])  # Take the first device output

    compiled = session.load(graph, weights_registry=state_dict)

    # Create signal buffers for execution
    signal_tensor = Buffer.zeros(
        shape=(signals.NUM_BYTES,), dtype=DType.uint8, device=Accelerator()
    )

    max_output = compiled.execute(input_indices.to("cuda"), signal_tensor)
    return from_dlpack(max_output[0]).to(torch.bfloat16)


# Flagged as flaky with "no room left in memory manager" in MODELS-958.
# Optimistically re-enabled with larger gpu-memory requirement and explicit
# garbage collection on 20250128.
def test_scaled_word_embedding(
    text_config: Gemma3TextConfig,
    input_indices: torch.Tensor,
    embedding_weights: torch.Tensor,
) -> None:
    """Test ScaledWordEmbedding with default scale (1.0)."""
    # Clear CUDA cache before test
    torch.cuda.empty_cache()

    # Garbage collection to migitate MODELS-958
    gc.collect()

    torch_output = generate_torch_outputs(
        text_config, input_indices, embedding_weights
    )
    max_output = generate_max_outputs(
        text_config, input_indices, embedding_weights
    )

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=1e-3,
        atol=1e-3,
    )

    # Clear CUDA cache after test
    torch.cuda.empty_cache()

    # Garbage collection to migitate MODELS-958
    gc.collect()


# Flagged as flaky with "no room left in memory manager" in MODELS-958.
# Optimistically re-enabled with larger gpu-memory requirement and explicit
# garbage collection on 20250128.
def test_scaled_word_embedding_with_scale(
    text_config: Gemma3TextConfig,
    input_indices: torch.Tensor,
    embedding_weights: torch.Tensor,
) -> None:
    """Test `ScaledWordEmbedding` with a non-default scale factor."""
    # Clear CUDA cache before test
    torch.cuda.empty_cache()

    # Garbage collection to migitate MODELS-958
    gc.collect()

    embed_scale = 0.5
    torch_output = generate_torch_outputs(
        text_config, input_indices, embedding_weights, embed_scale=embed_scale
    )
    max_output = generate_max_outputs(
        text_config, input_indices, embedding_weights, embed_scale=embed_scale
    )

    torch.testing.assert_close(
        torch_output,
        max_output,
        rtol=1e-3,
        atol=1e-3,
    )

    # Clear CUDA cache after test
    torch.cuda.empty_cache()

    # Garbage collection to migitate MODELS-958
    gc.collect()
