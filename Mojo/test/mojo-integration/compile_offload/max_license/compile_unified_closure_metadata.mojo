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
# Regression test: `@__llvm_metadata` decorators on a closure must flow
# through to the GPU entry point. A closure's `__call__` method forwards to
# the user's impl; without intervention `compile_offload` would target the
# forwarder and the metadata (attached to the impl) would be lost.
#
# The fix marks `__call__` as a transparent thunk so `processCompileOffload`
# redirects the offload to the wrapped impl. See `IREvaluatorContext.h`.
#
# This test covers both capturing and non-capturing closures.
#
# ===----------------------------------------------------------------------=== #

# RUN: env MOJO_ENABLE_PARAMETRIC_CLOSURE_TRAIT=1 %mojo %s | FileCheck %s
# RUN: %mojo %s | FileCheck %s

from std.compile import compile_info
from max.gpu.host.info import A100
from std.sys.info import CompilationTarget
from std.utils.static_tuple import StaticTuple


comptime MAX_THREADS_PER_BLOCK_METADATA = "nvvm.maxntid".value


def main():
    var a: UInt32 = 8

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
    )
    @__name("custom_capturing_closure")
    def metadata_closure() {var a}:
        _ = a + 1

    # CHECK: .entry custom_capturing_closure
    # CHECK: .maxntid 256
    enqueue_function(metadata_closure)

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(64))
    )
    @__name("custom_thin_closure")
    def metadata_thin_closure():
        pass

    # CHECK: .entry custom_thin_closure
    # CHECK: .maxntid 64
    enqueue_function(metadata_thin_closure)


def enqueue_function[
    FuncType: TrivialRegisterPassable & def() -> None, //
](func: FuncType):
    print(
        compile_info[
            FuncType.__call__,
            target=A100.target,
        ]()
    )
