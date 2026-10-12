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

"""Smoke test for MLA_SM100_Decode_Sparse_KV_BF16.

Type-level test that monomorphizes the kernel struct against its containers
(TMA descriptors, OffsetPosition, pipelines).  Full correctness coverage
lives in `test_mla_decode_sparse_kv_bf16.mojo`.
"""

from std.sys import default_accelerator
from max.gpu.host import DeviceContext

from nn.attention.gpu.nvidia.sm100.mla_decode_sparse_kv_bf16 import (
    MLA_SM100_Decode_Sparse_KV_BF16,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
)


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        return

    _ = DeviceContext()
    print("MLA_SM100_Decode_Sparse_KV_BF16 struct compiled.")
