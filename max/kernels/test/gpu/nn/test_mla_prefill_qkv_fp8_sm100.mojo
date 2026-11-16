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
from nn.attention.gpu.mla import SM100MLA
from _mla_prefill_qkv_fp8_shell import MLAPrefillQKVFP8Tests

comptime T = MLAPrefillQKVFP8Tests[SM100MLA]


def main() raises:
    with DeviceContext() as ctx:
        T.test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            0,
        ](ctx)
        T.test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            1,
        ](ctx)
        T.test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            2,
        ](ctx)
        T.test_mla_prefill_qkv_fp8[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            4,
        ](ctx)
