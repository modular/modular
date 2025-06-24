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

from gpu.host.compile import _compile_code_asm
from gpu.host import get_gpu_target
from gpu.sync import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    named_barrier,
)


# CHECK-LABEL: test_cp_async_bulk_wait_group
fn test_cp_async_bulk_wait_group():
    print("== test_cp_async_bulk_wait_group")

    fn cp_async_bulk_wait_group_kernel[n: Int32]():
        # CHECK: cp.async.bulk.wait_group.read 0;
        cp_async_bulk_wait_group[0]()
        # CHECK: cp.async.bulk.wait_group 2;
        cp_async_bulk_wait_group[n, False]()

    print(
        _compile_code_asm[
            cp_async_bulk_wait_group_kernel[2],
            target = get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_cp_async_bulk_commit_group
fn test_cp_async_bulk_commit_group():
    print("== test_cp_async_bulk_commit_group")

    fn cp_async_bulk_commit_group_kernel():
        # CHECK: cp.async.bulk.commit_group;
        cp_async_bulk_commit_group()

    print(
        _compile_code_asm[
            cp_async_bulk_commit_group_kernel,
            target = get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_named_barrier
fn test_named_barrier():
    print("== test_named_barrier")

    fn test_test_named_barrier_kernel():
        # CHECK: bar.sync 10 256
        named_barrier[256, 10]()

    print(
        _compile_code_asm[
            test_test_named_barrier_kernel,
            target = get_gpu_target["sm_90"](),
        ]()
    )


fn main():
    test_cp_async_bulk_wait_group()
    test_cp_async_bulk_commit_group()
    test_named_barrier()
