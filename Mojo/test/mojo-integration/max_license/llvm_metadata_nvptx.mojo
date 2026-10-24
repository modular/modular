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
# REQUIRES: NVIDIA-GPU
# RUN: mojo %s | FileCheck %s

from max.gpu import thread_idx
from max.gpu.host.compile import _compile_code
from std.utils import IndexList
from std.utils.static_tuple import StaticTuple
from max.gpu.host import get_gpu_target


@__llvm_metadata(`nvvm.maxntid`=StaticTuple[Int32, 1](128))
def my_kernel():
    pass


@__llvm_metadata(`nvvm.maxntid`=StaticTuple[Int32, 1](Int32(x)))
def my_kernel_with_param[x: Int]():
    pass


@__llvm_metadata(`nvvm.maxntid`=StaticTuple[Int32, 1](x))
def my_kernel_apply[x: Int32]():
    pass


def nested_function_metadata():
    @__llvm_metadata(`nvvm.maxntid`=StaticTuple[Int32, 1](x))
    def my_nested_kernel[x: Int32]():
        pass

    print(_compile_code[my_nested_kernel[64]]())


@__llvm_metadata(`nvvm.cluster_dim`=cluster_dims)
def test_cluster_dims_attribute_kernel_with_param[
    cluster_dims: StaticTuple[Int32, 3]
]():
    pass


# CHECK: .maxntid 128
# CHECK: .maxntid 256
# CHECK: .maxntid 512
# CHECK: .maxntid 64
# CHECK: .explicitcluster
# CHECK: .reqnctapercluster 2, 4, 1


struct MemStruct:
    var val: Int


# CHECK-LABEL: llvm_metadata_nvptx_my_grid
# CHECK-SAME:  byval({ i64 }) align 64
@__llvm_arg_metadata(x, `nvvm.grid_constant`)
def my_grid_constant_kernel[
    offset: Int
](g: Pointer[Int, MutAnyOrigin], x: MemStruct):
    g[unsafe_offset=thread_idx.x] = x.val + offset


# CHECK-DAG: "nvvm.grid_constant" %1


def main():
    print(_compile_code[my_kernel]())
    print(_compile_code[my_kernel_with_param[256]]())
    print(_compile_code[my_kernel_apply[512]]())
    nested_function_metadata()
    print(
        _compile_code[
            test_cluster_dims_attribute_kernel_with_param[
                StaticTuple[Int32, 3](2, 4, 1)
            ],
            target=get_gpu_target["sm_90"](),
        ]()
    )
    print(
        _compile_code[
            my_grid_constant_kernel[3],
            emission_kind="llvm",
            target=get_gpu_target["sm_90"](),
        ]()
    )
