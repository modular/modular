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

from std.testing import assert_false, assert_true

from nn.attention.gpu.mla_graph import mla_sparse_prefill_on_decode_kernel


def main() raises:
    # A packed NVFP4 prefill runs the sparse decode kernel at every head count
    # that kernel serves; `nvfp4_prefill_converts_once` picks the converted
    # FP8 copy or the packed rows inside the graph.
    comptime for heads in [8, 16, 24, 32, 40, 48, 56, 64, 128]:
        assert_true(
            mla_sparse_prefill_on_decode_kernel[
                DType.bfloat16, DType.uint8, heads, cache_is_nvfp4=True
            ]()
        )
        assert_true(
            mla_sparse_prefill_on_decode_kernel[
                DType.bfloat16, DType.float8_e4m3fn, heads
            ]()
        )
        assert_false(
            mla_sparse_prefill_on_decode_kernel[
                DType.bfloat16,
                DType.uint8,
                heads,
                cache_is_nvfp4=True,
                has_extra_k=True,
            ]()
        )
        assert_false(
            mla_sparse_prefill_on_decode_kernel[
                DType.bfloat16,
                DType.uint8,
                heads,
                cache_is_nvfp4=True,
                fold_shared_index=True,
            ]()
        )
    comptime for heads in [1, 4, 33, 65, 96, 129]:
        assert_false(
            mla_sparse_prefill_on_decode_kernel[
                DType.bfloat16, DType.uint8, heads, cache_is_nvfp4=True
            ]()
        )
