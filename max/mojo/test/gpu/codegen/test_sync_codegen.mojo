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
"""Checks that each GPU vendor's MAX plugin lowers the block, warp, and
async-copy-group synchronization primitives."""

from std.compile import compile_info
from std.sys.info import CompilationTarget
from std.testing import TestSuite, assert_false, assert_true

from max.gpu.host import get_gpu_target
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
)
from max.gpu.sync import barrier, syncwarp


def _compile_sync_kernel[target: CompilationTarget]() raises -> String:
    # Nested so `__functions_in_module()` doesn't compile it for the host,
    # where `barrier()` is rejected.
    def kernel():
        barrier()
        syncwarp()
        async_copy_commit_group()
        async_copy_wait_group(1)
        async_copy_wait_all()

    return String(compile_info[kernel, emission_kind="llvm", target=target]())


def test_nvidia() raises:
    var llvm = _compile_sync_kernel[get_gpu_target["sm_80"]()]()
    assert_true(
        "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)" in llvm
    )
    assert_true("call void @llvm.nvvm.bar.warp.sync(i32 -1)" in llvm)
    assert_true("call void @llvm.nvvm.cp.async.commit.group()" in llvm)
    assert_true("call void @llvm.nvvm.cp.async.wait.group(i32 1)" in llvm)
    assert_true("call void @llvm.nvvm.cp.async.wait.all()" in llvm)


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
    var llvm = _compile_sync_kernel[target]()
    assert_true(
        "call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)" in llvm
    )
    assert_true("call void @llvm.nvvm.cp.async.commit.group()" in llvm)


def test_amd() raises:
    var llvm = _compile_sync_kernel[get_gpu_target["mi300x"]()]()
    assert_true('fence syncscope("workgroup") release' in llvm)
    assert_true("call void @llvm.amdgcn.s.barrier()" in llvm)
    assert_true('fence syncscope("workgroup") acquire' in llvm)
    assert_true("call void @llvm.amdgcn.wave.barrier()" in llvm)
    assert_false("cp.async" in llvm)


def test_apple() raises:
    var llvm = _compile_sync_kernel[get_gpu_target["apple-m4"]()]()
    assert_true("call void @air.wg.barrier(i32 2, i32 1)" in llvm)
    assert_true("call void @air.simdgroup.barrier(i32 0, i32 4)" in llvm)
    assert_false("cp.async" in llvm)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
