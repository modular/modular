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

from gpu import thread_idx
from gpu.host._compile import _compile_code_asm


fn outer[y: Int]():
    @parameter
    fn param[x: Int](y: SIMD[DType.float32, y], /):
        pass

    print(_compile_code_asm[param[y]]())


fn main():
    # CHECK: .debug_
    outer[2]()
