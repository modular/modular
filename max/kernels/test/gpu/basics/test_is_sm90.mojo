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

from sys.info import _is_sm_9x

from gpu.host._compile import _compile_code_asm, get_gpu_target
from testing import *


fn check_sm() -> Bool:
    alias v = _is_sm_9x()
    return v


def test_is_sm_9x():
    assert_true(
        "ret i1 true"
        in _compile_code_asm[
            check_sm,
            emission_kind="llvm",
            target = get_gpu_target["sm_90"](),
        ]()
    )
    assert_true(
        "ret i1 true"
        in _compile_code_asm[
            check_sm,
            emission_kind="llvm",
            target = get_gpu_target["sm_90a"](),
        ]()
    )


def main():
    test_is_sm_9x()
