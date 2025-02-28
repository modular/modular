# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Token sampling algorithms."""

from max.dtype import DType
from max.graph import Dim, Graph, Shape, TensorType, TensorValue, ops
from max.pipelines import SamplingConfig


def _bitmask_sampler(sampling_config: SamplingConfig) -> Graph:
    logits_in_type = TensorType(
        sampling_config.in_dtype, ["batch", "vocab_size"]
    )
    prev_tokens_type = TensorType(DType.int64, ["batch", "num_prev_steps"])
    bitmask_type = TensorType(DType.bool, ["batch", "vocab_size"])

    with Graph(
        "bitmask_sampler",
        input_types=[logits_in_type, prev_tokens_type, bitmask_type],
    ) as graph:
        # Deconstruct inputs and cast.
        logits, prev_tokens, bitmask = (val.tensor for val in graph.inputs)
        logits = ops.cast(logits, sampling_config.out_dtype)

        # Mask the logits out.
        logits = ops.select(
            bitmask, logits, ops.constant(-10000, dtype=DType.float32)
        )

        # Apply top_k or argmax sampling.
        shape = Shape(logits.shape)
        shape[-1] = Dim(1)
        tokens = ops.custom(
            "topk_fused_sampling",
            [
                ops.constant(sampling_config.top_k, dtype=DType.int64),
                logits,
            ],
            [TensorType(DType.int64, shape)],
        )[0]
        assert isinstance(tokens, TensorValue)

        all_tokens = ops.concat([prev_tokens, tokens], -1)
        tokens = ops.squeeze(tokens, -1)
        graph.output(tokens, all_tokens)

        return graph


def _vanilla_sampler(sampling_config: SamplingConfig) -> Graph:
    logits_in_type = TensorType(
        sampling_config.in_dtype, ["batch", "vocab_size"]
    )
    prev_tokens_type = TensorType(DType.int64, ["batch", "num_prev_steps"])
    with Graph(
        "token_sampler", input_types=[logits_in_type, prev_tokens_type]
    ) as graph:
        logits, prev_tokens = (val.tensor for val in graph.inputs)
        logits = ops.cast(logits, sampling_config.out_dtype)

        shape = Shape(logits.shape)
        shape[-1] = Dim(1)
        tokens = ops.custom(
            "topk_fused_sampling",
            [
                ops.constant(sampling_config.top_k, dtype=DType.int64),
                logits,
            ],
            [TensorType(DType.int64, shape)],
        )[0]
        assert isinstance(tokens, TensorValue)

        all_tokens = ops.concat([prev_tokens, tokens], -1)
        tokens = ops.squeeze(tokens, -1)
        graph.output(tokens, all_tokens)

        return graph


def token_sampler(sampling_config: SamplingConfig) -> Graph:
    if sampling_config.enable_structured_output:
        return _bitmask_sampler(sampling_config)
    else:
        return _vanilla_sampler(sampling_config)
