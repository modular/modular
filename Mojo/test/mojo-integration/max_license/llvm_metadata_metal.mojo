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

# REQUIRES: system-darwin
# RUN: mojo %s | FileCheck %s

from max.gpu import thread_idx
from max.gpu.host.compile import _compile_code
from std.utils import IndexList
from std.utils.static_tuple import StaticTuple
from max.gpu.host import get_gpu_target


@__llvm_metadata(`pop.air.max_work_group_size`=SIMDLength(128))
def my_kernel():
    pass


@__llvm_metadata(`pop.air.max_work_group_size`=StaticTuple[Int32, 1](64))
def my_kernel_array():
    pass


# CHECK-DAG: define dso_local void @{{.*}}my_kernel{{.*}}() #[[MY_KERNEL_ATTRS:.+]] {
# CHECK-DAG: attributes #[[MY_KERNEL_ATTRS]] = { {{.*}} "air.max_work_group_size"="128" "metal.kernel"="true" {{.*}} }

# CHECK-DAG: define dso_local void @{{.*}}my_kernel_a{{.*}}() #[[MY_KERNEL_ARRAY_ATTRS:.+]] {
# CHECK-DAG: attributes #[[MY_KERNEL_ARRAY_ATTRS]] = { {{.*}} "air.max_work_group_size"="64" "metal.kernel"="true" {{.*}} }


def main():
    print(
        _compile_code[
            my_kernel,
            emission_kind="llvm",
            target=get_gpu_target["metal:3"](),
        ]()
    )

    print(
        _compile_code[
            my_kernel_array,
            emission_kind="llvm",
            target=get_gpu_target["metal:3"](),
        ]()
    )
