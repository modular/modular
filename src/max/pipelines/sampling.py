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
from max.pipelines.max_config import SamplingConfig


def _sampling_input_types(sampling_config: SamplingConfig) -> list[TensorType]:
    input_types = []

    # Logits are always provided
    if sampling_config.enable_variable_logits:
        logits_in_type = TensorType(
            sampling_config.in_dtype, ["total_output_len", "vocab_size"]
        )
        input_types.append(logits_in_type)
    else:
        logits_in_type = TensorType(
            sampling_config.in_dtype, ["batch", "vocab_size"]
        )
        input_types.append(logits_in_type)

    # We are currently, always passing tokens through
    prev_tokens_type = TensorType(DType.int64, ["batch", "num_prev_steps"])
    input_types.append(prev_tokens_type)

    # If we have variable token logits enabled
    if sampling_config.enable_variable_logits:
        logit_offset_type = TensorType(DType.uint32, ["logit_offsets_len"])
        input_types.append(logit_offset_type)

    # If we have structured_outputs enabled
    if sampling_config.enable_structured_output:
        bitmask_type = TensorType(DType.bool, ["batch", "vocab_size"])
        input_types.append(bitmask_type)

    return input_types


def token_sampler(sampling_config: SamplingConfig) -> Graph:
    _input_types = _sampling_input_types(sampling_config)
    with Graph("top_k_sampler", input_types=_input_types) as graph:
        # Deconstruct inputs
        # TODO: Explore better ways of indexing into these input values
        # tightly coupling the input order with element indices feels
        # quite brittle.
        logits = graph.inputs[0].tensor
        logits = ops.cast(logits, sampling_config.out_dtype)
        prev_tokens = graph.inputs[1].tensor

        if sampling_config.enable_variable_logits:
            logit_offsets = graph.inputs[2].tensor
            logits = ops.gather(logits, logit_offsets[1:] - 1, axis=0)
            logits = ops.rebind(logits, shape=("batch", "vocab_size"))

        if sampling_config.enable_structured_output:
            bitmask = graph.inputs[-1].tensor
            logits = ops.select(
                bitmask, logits, ops.constant(-10000, dtype=DType.float32)
            )

        # Apply top_k sampling
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
