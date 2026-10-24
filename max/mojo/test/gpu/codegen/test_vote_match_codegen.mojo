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
"""Checks that each GPU vendor's MAX plugin lowers the warp vote and match
primitives."""

from std.compile import compile_info
from std.sys.info import CompilationTarget
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from max.gpu.host import get_gpu_target
from max.gpu.primitives.warp import match_all, match_any, vote


def _compile_vote_match_kernel[target: CompilationTarget]() raises -> String:
    # Nested so `__functions_in_module()` doesn't compile it for the host,
    # where these are rejected.
    def kernel(data: Pointer[UInt32, MutAnyOrigin]):
        var key = data[]
        data[unsafe_offset=1] = vote[.uint32](key == 0)
        data[unsafe_offset=2] = match_any(key).cast[.uint32]()
        data[unsafe_offset=3] = match_all(key).cast[.uint32]()
        var wide_key = data.unsafe_bitcast[UInt64]()[]
        data[unsafe_offset=4] = match_any(wide_key).cast[.uint32]()
        data[unsafe_offset=5] = match_all(wide_key).cast[.uint32]()

    return String(compile_info[kernel, emission_kind="llvm", target=target]())


def _assert_nvidia_vote_match(llvm: String) raises:
    assert_true("call i32 @llvm.nvvm.vote.ballot.sync(i32 -1, i1" in llvm)
    assert_true("match.any.sync.b32" in llvm)
    assert_true("match.all.sync.b32" in llvm)
    assert_true("match.any.sync.b64" in llvm)
    assert_true("match.all.sync.b64" in llvm)


def test_nvidia() raises:
    _assert_nvidia_vote_match(
        _compile_vote_match_kernel[get_gpu_target["sm_80"]()]()
    )


def test_nvidia_bare_triple() raises:
    # Named only by its triple, as `--target-triple` builds it: no
    # `stdlib_plugin` field and no plugin payload.
    comptime target = CompilationTarget[
        _mlir_value=__mlir_attr[
            `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
            `arch = "sm_80", `,
            `features = "+ptx81", `,
            `data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
            `simd_bit_width = 128,`,
            `index_bit_width = 64`,
            `> : !kgen.target`,
        ]
    ]()
    _assert_nvidia_vote_match(_compile_vote_match_kernel[target]())


def test_amd() raises:
    var llvm = _compile_vote_match_kernel[get_gpu_target["mi300x"]()]()
    assert_true("call i32 @llvm.amdgcn.ballot.i32(i1" in llvm)
    # The 64-lane warp's match mask, from the `readfirstlane` loop rather
    # than the shuffle-based default.
    assert_true("call i64 @llvm.amdgcn.ballot.i64(i1" in llvm)
    assert_true("@llvm.amdgcn.readfirstlane" in llvm)
    assert_false("@llvm.amdgcn.ds.bpermute" in llvm)


def test_apple() raises:
    var llvm = _compile_vote_match_kernel[get_gpu_target["apple-m4"]()]()
    assert_true("call i32 @air.simd_ballot.i32.i1(i1" in llvm)
    # No match instruction, so both matches use the default, which
    # broadcasts lanes' bits.
    assert_true("call i32 @air.simd_shuffle.i32.i16(i32" in llvm)


def test_apple_match_any() raises:
    def kernel(data: Pointer[UInt32, MutAnyOrigin]):
        data[unsafe_offset=1] = match_any(data[]).cast[.uint32]()

    var llvm = String(
        compile_info[
            kernel, emission_kind="llvm", target=get_gpu_target["apple-m4"]()
        ]()
    )
    # The default loops once per distinct value rather than unrolling a
    # shuffle per lane: one shuffle and one ballot in the loop, and one
    # ballot for the active lanes.
    assert_equal(llvm.count("call i32 @air.simd_shuffle.i32.i16(i32"), 1)
    assert_equal(llvm.count("call i32 @air.simd_ballot.i32.i1(i1"), 2)


def test_apple_match_all() raises:
    def kernel(data: Pointer[UInt32, MutAnyOrigin]):
        data[unsafe_offset=1] = match_all(data[]).cast[.uint32]()

    var llvm = String(
        compile_info[
            kernel, emission_kind="llvm", target=get_gpu_target["apple-m4"]()
        ]()
    )
    # The default broadcasts lane 0's bits once and ballots the rest.
    assert_equal(llvm.count("call i32 @air.simd_shuffle.i32.i16(i32"), 1)
    assert_equal(llvm.count("call i32 @air.simd_ballot.i32.i1(i1"), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
