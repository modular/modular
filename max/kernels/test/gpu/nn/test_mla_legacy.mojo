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
from _mla_test_vendors import LegacyNVVendor


comptime T = MLATests[LegacyNVVendor]


def main() raises:
    with DeviceContext() as ctx:
        T.test_decoding[1, 1, False](ctx, False)
        T.test_decoding[27, 1, False](ctx, False)
        T.test_decoding[128, 1, False](ctx, False)
        T.test_decoding[0, 1, False](ctx, False)

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
