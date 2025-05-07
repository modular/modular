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
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from testing import *


fn register_intrinsics():
    warpgroup_reg_alloc[64]()
    warpgroup_reg_dealloc[64]()


def test_register_intrinsics_sm80():
    var asm = _compile_code_asm[
        register_intrinsics, target = _get_gpu_target["sm_80"]()
    ]()
    assert_false("setmaxnreg.inc.sync.aligned.u32" in asm)
    assert_false("setmaxnreg.dec.sync.aligned.u32" in asm)


def test_register_intrinsics_sm90():
    var asm = _compile_code_asm[
        register_intrinsics, target = _get_gpu_target["sm_90a"]()
    ]()
    assert_true("setmaxnreg.inc.sync.aligned.u32" in asm)
    assert_true("setmaxnreg.dec.sync.aligned.u32" in asm)


def main():
    test_register_intrinsics_sm80()
    test_register_intrinsics_sm90()
