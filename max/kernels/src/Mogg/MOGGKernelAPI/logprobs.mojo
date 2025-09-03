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

from algorithm.functional import parallelize
from compiler_internal import register
from gpu import global_idx
from gpu.host.info import is_cpu, is_gpu
from math import ceildiv, exp, inf, log
from nn._ragged_utils import get_batch_from_row_offsets
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from tensor_internal.transitional import managed_tensor_slice_to_ndbuffer
from utils.index import IndexList


struct FixedHeightMinHeap[k_dtype: DType, v_dtype: DType, levels: Int]:
    alias num_elements = 2**levels - 1
    var k_array: InlineArray[Scalar[k_dtype], Self.num_elements]
    var v_array: InlineArray[Scalar[v_dtype], Self.num_elements]

    fn __init__(out self, *, fill_k: Scalar[k_dtype], fill_v: Scalar[v_dtype]):
        self.k_array = InlineArray[size = Self.num_elements](fill=fill_k)
        self.v_array = InlineArray[size = Self.num_elements](fill=fill_v)

    @always_inline
    fn swap(mut self, a: Int, b: Int) -> None:
        self.k_array[a], self.k_array[b] = self.k_array[b], self.k_array[a]
        self.v_array[a], self.v_array[b] = self.v_array[b], self.v_array[a]

    fn heap_down(mut self) -> None:
        var current_index = 0

        @parameter
        for level in range(levels - 1):
            # Must ensure:
            # arr[cur] < arr[left] && arr[cur] < arr[right]
            var left_index = current_index * 2 + 1
            var right_index = current_index * 2 + 2
            var smaller_index = left_index
            if self.v_array[right_index] < self.v_array[left_index]:
                smaller_index = right_index
            if self.v_array[current_index] < self.v_array[smaller_index]:
                # Full heap property is satisfied.  We could stop here,
                # but this is an unrolled loop, so just continue on.
                # (Useless but harmless work.)
                pass
            else:
                self.swap(current_index, smaller_index)
                current_index = smaller_index


alias logit_dtype = DType.float32
alias token_dtype = DType.uint32
alias offset_dtype = DType.uint32


fn compute_log_probabilities_1tok[
    target: StaticString, levels: Int
](
    output_token_index: Int,
    lp_logits: OutputTensor[dtype=logit_dtype, rank=2],
    lp_tokens: OutputTensor[dtype=token_dtype, rank=2],
    logits: InputTensor[dtype=logit_dtype, rank=2],
    tokens: InputTensor[dtype=token_dtype, rank=1],
    sampled_tokens: InputTensor[dtype=token_dtype, rank=1],
    logit_row_offsets: InputTensor[dtype=offset_dtype, rank=1],
    token_row_offsets: InputTensor[dtype=offset_dtype, rank=1],
    lp_output_offsets: InputTensor[dtype=offset_dtype, rank=1],
) -> None:
    var vocab_size = logits.shape()[1]
    var batch_index = get_batch_from_row_offsets(
        managed_tensor_slice_to_ndbuffer(lp_output_offsets), output_token_index
    )
    var reverse_index_in_seq = (
        lp_output_offsets[batch_index + 1] - output_token_index - 1
    )
    var token_end_index = token_row_offsets[batch_index + 1]
    var sampled_token: Scalar[token_dtype]
    if reverse_index_in_seq == 0:
        sampled_token = sampled_tokens[batch_index]
    else:
        sampled_token = tokens[Int(token_end_index - reverse_index_in_seq)]

    var logit_end_index = logit_row_offsets[batch_index + 1]
    var logit_index = Int(logit_end_index - reverse_index_in_seq - 1)
    var x_max = logits[logit_index, 0]
    for token_value in range(1, vocab_size):
        x_max = max(x_max, logits[logit_index, token_value])
    var sum_exp = Scalar[logit_dtype](0.0)
    for token_value in range(vocab_size):
        sum_exp += exp(logits[logit_index, token_value] - x_max)
    var log_sum_exp = log(sum_exp)
    var normalizer = -(x_max + log_sum_exp)

    var post_heap_idx: Int

    @parameter
    if levels <= 0:
        post_heap_idx = 0
    else:
        var heap = FixedHeightMinHeap[token_dtype, logit_dtype, levels](
            fill_k=vocab_size, fill_v=-inf[logit_dtype]()
        )
        for token_value in range(vocab_size):
            var logit_value = logits[logit_index, token_value]
            if logit_value > heap.v_array[0]:
                heap.k_array[0] = token_value
                heap.v_array[0] = logit_value
                heap.heap_down()
        for i in range(heap.num_elements):
            lp_tokens[output_token_index, i] = heap.k_array[i]
            lp_logits[output_token_index, i] = heap.v_array[i] + normalizer
        post_heap_idx = heap.num_elements
    lp_tokens[output_token_index, post_heap_idx] = sampled_token
    lp_logits[output_token_index, post_heap_idx] = (
        logits[logit_index, Int(sampled_token)] + normalizer
    )


@register("compute_log_probabilities_ragged")
struct LogProbabilitiesRagged:
    @staticmethod
    fn execute[
        target: StaticString, levels: Int
    ](
        lp_logits: OutputTensor[dtype=logit_dtype, rank=2],
        lp_tokens: OutputTensor[dtype=token_dtype, rank=2],
        logits: InputTensor[dtype=logit_dtype, rank=2],
        tokens: InputTensor[dtype=token_dtype, rank=1],
        sampled_tokens: InputTensor[dtype=token_dtype, rank=1],
        logit_row_offsets: InputTensor[dtype=offset_dtype, rank=1],
        token_row_offsets: InputTensor[dtype=offset_dtype, rank=1],
        lp_output_offsets: InputTensor[dtype=offset_dtype, rank=1],
        lp_output_offsets_host: InputTensor[dtype=offset_dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises -> None:
        var num_output_tokens = lp_logits.shape()[0]
        if lp_tokens.shape()[0] != num_output_tokens:
            raise Error("Mismatch in axis 0 of lp_logits and lp_tokens")
        if lp_logits.shape()[1] != 2**levels:
            raise Error("Axis 1 of lp_logits inconsistent with level setting")
        if lp_tokens.shape()[1] != 2**levels:
            raise Error("Axis 1 of lp_tokens inconsistent with level setting")

        @parameter
        if is_cpu[target]():

            @parameter
            fn lp_idx_kernel(output_token_index: Int) -> None:
                compute_log_probabilities_1tok[target, levels](
                    output_token_index=output_token_index,
                    lp_logits=lp_logits,
                    lp_tokens=lp_tokens,
                    logits=logits,
                    tokens=tokens,
                    sampled_tokens=sampled_tokens,
                    logit_row_offsets=logit_row_offsets,
                    token_row_offsets=token_row_offsets,
                    lp_output_offsets=lp_output_offsets,
                )

            parallelize[lp_idx_kernel](num_output_tokens)
        elif is_gpu[target]():

            @parameter
            @__copy_capture(num_output_tokens)
            fn raw_lp_kernel():
                var output_token_index = global_idx.x
                if output_token_index < UInt(num_output_tokens):
                    compute_log_probabilities_1tok[target, levels](
                        output_token_index=output_token_index,
                        lp_logits=lp_logits,
                        lp_tokens=lp_tokens,
                        logits=logits,
                        tokens=tokens,
                        sampled_tokens=sampled_tokens,
                        logit_row_offsets=logit_row_offsets,
                        token_row_offsets=token_row_offsets,
                        lp_output_offsets=lp_output_offsets,
                    )

            alias block_size = 64
            ctx.get_device_context().enqueue_function_checked[
                raw_lp_kernel, raw_lp_kernel
            ](
                grid_dim=ceildiv(num_output_tokens, block_size),
                block_dim=block_size,
            )
        else:
            constrained[False, "unsupported target"]()

    @staticmethod
    fn shape[
        levels: Int
    ](
        logits: InputTensor[dtype=logit_dtype, rank=2],
        tokens: InputTensor[dtype=token_dtype, rank=1],
        sampled_tokens: InputTensor[dtype=token_dtype, rank=1],
        logit_row_offsets: InputTensor[dtype=offset_dtype, rank=1],
        token_row_offsets: InputTensor[dtype=offset_dtype, rank=1],
        lp_output_offsets: InputTensor[dtype=offset_dtype, rank=1],
        lp_output_offsets_host: InputTensor[dtype=offset_dtype, rank=1],
    ) -> IndexList[2]:
        return IndexList[2](
            Int(lp_output_offsets_host[lp_output_offsets_host.shape()[0] - 1]),
            2**levels,
        )
