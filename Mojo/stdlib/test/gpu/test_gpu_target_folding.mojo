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

# Verifies that an accelerator `CompilationTarget` used as a type parameter
# stays compact. The accelerator-table lookup can't fold while types are
# parsed, and if it is inlined into the type, the whole table is serialized
# into the symbol name of every function whose signature mentions it. That
# costs hundreds of KB per symbol and gigabytes of compiler memory in
# kernel-heavy programs.
#
# Compile-only: every check is a `comptime assert` on a linkage name.
# RUN: %mojo-build --emit object --target-accelerator sm_90  %s -o %t
# RUN: %mojo-build --emit object --target-accelerator gfx950 %s -o %t

from std.sys.info import CompilationTarget
from std._gpu.host.info import get_gpu_target
from std.reflection import get_linkage_name


struct _Parameterized[target: CompilationTarget]:
    pass


def _takes_default_accelerator(
    x: _Parameterized[CompilationTarget.default_accelerator()],
):
    pass


def _takes_gpu_target_literal(x: _Parameterized[get_gpu_target["sm_90"]()]):
    pass


# Far above the length of a symbol carrying one `#kgen.target`, far below one
# carrying the inlined lookup table.
comptime _MAX_SYMBOL_BYTES = 2048


def _check_compact[name: StaticString]():
    comptime prefix = name[byte = : min(name.byte_length(), 400)]

    comptime assert "TargetAccelerator" not in name, String(
        "accelerator target table was inlined into the type; symbol is ",
        name.byte_length(),
        " bytes: ",
        prefix,
    )
    comptime assert name.byte_length() <= _MAX_SYMBOL_BYTES, String(
        "symbol is ", name.byte_length(), " bytes: ", prefix
    )


def main():
    _check_compact[get_linkage_name[_takes_default_accelerator]()]()
    _check_compact[get_linkage_name[_takes_gpu_target_literal]()]()
