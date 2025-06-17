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

from time import sleep

from gpu.host.compile import _compile_code_asm, get_gpu_target
from testing import *


fn sleep_intrinsics():
    sleep(0.0000001)


@always_inline
fn _verify_sleep_intrinsics_nvidia(asm: StringSlice) raises -> None:
    assert_true("nanosleep.u32" in asm)


@always_inline
fn _verify_sleep_intrinsics_mi300x(asm: StringSlice) raises -> None:
    assert_true("s_sleep" in asm)


def test_sleep_intrinsics_sm80():
    var asm = _compile_code_asm[
        sleep_intrinsics, target = get_gpu_target["sm_80"]()
    ]()
    _verify_sleep_intrinsics_nvidia(asm)


def test_sleep_intrinsics_sm90():
    var asm = _compile_code_asm[
        sleep_intrinsics, target = get_gpu_target["sm_90"]()
    ]()
    _verify_sleep_intrinsics_nvidia(asm)


def test_sleep_intrinsics_mi300x():
    var asm = _compile_code_asm[
        sleep_intrinsics, target = get_gpu_target["mi300x"]()
    ]()
    _verify_sleep_intrinsics_mi300x(asm)


def main():
    test_sleep_intrinsics_sm80()
    test_sleep_intrinsics_sm90()
    test_sleep_intrinsics_mi300x()
