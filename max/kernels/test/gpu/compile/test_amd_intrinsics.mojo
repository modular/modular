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

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.intrinsics import implicitarg_ptr, sendmsg
from std.testing import assert_true

comptime MI355X_TARGET = get_gpu_target["mi355x"]()


def sendmsg_kernel():
    sendmsg(Int32(1), Int32(0))


def implicitarg_ptr_kernel(y: MutPointer[UInt8, MutAnyOrigin]):
    y[0] = implicitarg_ptr()[0]


def test_sendmsg() raises:
    var ir = _compile_code[
        sendmsg_kernel, target=MI355X_TARGET, emission_kind="llvm"
    ]().asm
    assert_true("llvm.amdgcn.s.sendmsg" in ir, "expected s.sendmsg intrinsic")


def test_implicitarg_ptr() raises:
    var ir = _compile_code[
        implicitarg_ptr_kernel, target=MI355X_TARGET, emission_kind="llvm"
    ]().asm
    assert_true(
        "llvm.amdgcn.implicitarg.ptr" in ir,
        "expected implicitarg.ptr intrinsic",
    )


def main() raises:
    test_sendmsg()
    test_implicitarg_ptr()
