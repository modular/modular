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

from max.gpu.host import DeviceContext
from _mla_test_shell import MLATests
from _mla_test_vendors import AMDVendor


comptime T = MLATests[AMDVendor]


def main() raises:
    with DeviceContext() as ctx:
        T.test_decoding[1, 1, False](ctx, False)
        T.test_decoding[27, 1, False](ctx, False)
        T.test_decoding[128, 1, False](ctx, False)
        T.test_decoding[0, 1, False](ctx, False)

        T.test_decoding[1, 4, False](ctx, False)
        T.test_decoding[27, 2, False](ctx, False)
        # Default (None) — exercise the AMD heuristic.
        T.test_decoding[1, None, False](ctx, False)
        T.test_decoding[
            1,
            4,
            False,
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
        ](ctx, False)

        T.test_decoding[
            1,
            1,
            False,
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
        ](ctx, False)
        T.test_decoding[
            27,
            1,
            False,
            qkv_type=DType.float8_e4m3fn,
            output_type=DType.bfloat16,
        ](ctx, False)

        # 12 heads needs BM rounded up to 16; 24 leaves a partial last
        # head tile at BM=16.
        T.test_decoding_k3_head_counts[2, 1, False](ctx, 1, 512)
        T.test_decoding_k3_head_counts[1, 1, False](ctx, 1, 4096)
        # AMD reaches its combine kernel through partitions, not warp
        # split-K.
        T.test_decoding_k3_head_counts[1, 4, False](ctx, 1, 4096)

        # An explicit zero-offset tensor must match the no-offset reference.
        T.test_prefill[
            DType.bfloat16,
            DType.bfloat16,
            depth=192,
            num_heads=128,
            kv_depth=128,
            cache_depth=576,
            cache_num_heads=1,
            batch_size=2,
            use_cache_offsets=True,
        ](32, 64, ctx)

        # test mla prefill
        T.test_mla_prefill[2, DType.bfloat16, DType.bfloat16](ctx)
        T.test_mla_prefill[4, DType.bfloat16, DType.bfloat16](ctx)
        T.test_mla_prefill[0, DType.bfloat16, DType.bfloat16](ctx)
