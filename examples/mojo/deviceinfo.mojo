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

from collections.string import StaticString
from sys import (
    CompilationTarget,
    has_intel_amx,
    is_apple_m1,
    is_apple_m2,
    is_apple_m3,
    num_logical_cores,
    num_physical_cores,
    os_is_linux,
    os_is_macos,
)

# This sample prints the current host system information using APIs from the
# sys module.
from sys.info import _current_arch, _triple_attr


def main():
    var os: StaticString
    if os_is_linux():
        os = "linux"
    elif os_is_macos():
        os = "macOS"
    else:
        os = "windows"
    var cpu = _current_arch()
    var arch = StaticString(_triple_attr())
    var cpu_features = String()
    if CompilationTarget.has_sse4():
        cpu_features += " sse4"
    if CompilationTarget.has_avx():
        cpu_features += " avx"
    if CompilationTarget.has_avx2():
        cpu_features += " avx2"
    if CompilationTarget.has_avx512f():
        cpu_features += " avx512f"
    if CompilationTarget.has_vnni():
        if CompilationTarget.has_avx512f():
            cpu_features += " avx512_vnni"
        else:
            cpu_features += " avx_vnni"
    if has_intel_amx():
        cpu_features += " intel_amx"
    if CompilationTarget.has_neon():
        cpu_features += " neon"
    if is_apple_m1():
        cpu_features += " Apple M1"
    if is_apple_m2():
        cpu_features += " Apple M2"
    if is_apple_m3():
        cpu_features += " Apple M3"

    print("System information: ")
    print("    OS             : ", os)
    print("    CPU            : ", cpu)
    print("    Arch           : ", arch)
    print("    Physical Cores : ", num_physical_cores())
    print("    Logical Cores  : ", num_logical_cores())
    print("    CPU Features   :", cpu_features)
