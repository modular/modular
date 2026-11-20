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
#
# Guards the deterministic bundling order in Elaborator::bundleOffloadModules:
# offload ops are bundled sorted by the auto-mangled kernel symbol name, not
# in the order elaboration discovered them (which is thread-timing dependent).
# Each kernel is requested from its own exported entrypoint, so the offload
# ops are discovered by independent elaboration roots rather than in one
# body's statement order. Device-side diagnostics are emitted in bundling
# order, so the constraint failures below must appear in mangled-name order
# (alpha_kern, bb, middle_kern, yy, zebra_kern) even though the kernels are
# defined and requested in a different order.
#
# The bb/yy kernels have @__name linkage names in the opposite order ("zz"
# vs "aa"); bb still sorts before yy because linkage names do not affect the
# bundling order.
#
# ===----------------------------------------------------------------------=== #
# RUN: not %mojo %s 2>&1 | FileCheck %s

# CHECK: constraint failed: alpha_kern is not supported on NVIDIA GPU
# CHECK: constraint failed: bb is not supported on NVIDIA GPU
# CHECK: constraint failed: middle_kern is not supported on NVIDIA GPU
# CHECK: constraint failed: yy is not supported on NVIDIA GPU
# CHECK: constraint failed: zebra_kern is not supported on NVIDIA GPU

from std.compile import compile_info
from std.sys.info import is_nvidia_gpu, CompilationTarget

from max.gpu.host.info import A100


def zebra_kern(x: Float64) raises -> Float64:
    comptime assert (
        not is_nvidia_gpu()
    ), "zebra_kern is not supported on NVIDIA GPU"
    return x


def middle_kern(x: Float64) raises -> Float64:
    comptime assert (
        not is_nvidia_gpu()
    ), "middle_kern is not supported on NVIDIA GPU"
    return x


def alpha_kern(x: Float64) raises -> Float64:
    comptime assert (
        not is_nvidia_gpu()
    ), "alpha_kern is not supported on NVIDIA GPU"
    return x


@__name("zz")
def bb(x: Float64) raises -> Float64:
    comptime assert not is_nvidia_gpu(), "bb is not supported on NVIDIA GPU"
    return x


@__name("aa")
def yy(x: Float64) raises -> Float64:
    comptime assert not is_nvidia_gpu(), "yy is not supported on NVIDIA GPU"
    return x


@no_inline
@export
def zebra_entry() abi("Mojo") raises:
    print(compile_info[zebra_kern, emission_kind="llvm", target=A100.target]())


@no_inline
@export
def middle_entry() abi("Mojo") raises:
    print(compile_info[middle_kern, emission_kind="llvm", target=A100.target]())


@no_inline
@export
def alpha_entry() abi("Mojo") raises:
    print(compile_info[alpha_kern, emission_kind="llvm", target=A100.target]())


@no_inline
@export
def bb_entry() abi("Mojo") raises:
    print(compile_info[bb, emission_kind="llvm", target=A100.target]())


@no_inline
@export
def yy_entry() abi("Mojo") raises:
    print(compile_info[yy, emission_kind="llvm", target=A100.target]())


def main() raises:
    zebra_entry()
    middle_entry()
    bb_entry()
    yy_entry()
    alpha_entry()
