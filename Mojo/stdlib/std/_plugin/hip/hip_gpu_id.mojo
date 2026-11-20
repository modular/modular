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
"""AMDGCN reads of the GPU indices and sizes `HIPPlugin` provides."""

from std._gpu.primitives.id import _GPUIdPluginHookFnType, _is_cluster_id
from std.collections import OptionalReg
from std.collections.string.string_span import get_static_string
from std.sys import llvm_intrinsic


def _amdgcn_gpu_id_fn[
    name: StaticString, dim: StaticString
]() -> OptionalReg[_GPUIdPluginHookFnType]:
    """Returns the AMDGCN read of `name` along `dim`, or `None` for a thread
    block cluster id, which AMD GPUs do not have."""
    comptime if _is_cluster_id(name):
        return None
    else:
        return comptime (
            OptionalReg[_GPUIdPluginHookFnType](_amdgcn_gpu_id[name, dim])
        )


@inline(.always)
def _implicitarg_load[offset: Int, dtype: DType]() -> Int:
    var ptr = llvm_intrinsic[
        "llvm.amdgcn.implicitarg.ptr",
        Pointer[Scalar[dtype], MutUntrackedOrigin, address_space=.CONSTANT],
        has_side_effect=False,
    ]()
    return Int(ptr.unsafe_load(offset))


def _dim_index[dim: StaticString]() -> Int:
    comptime if dim == "x":
        return 0
    elif dim == "y":
        return 1
    else:
        comptime assert dim == "z", "the dimension must be x, y, or z"
        return 2


@inline(.always)
def _amdgcn_gpu_id[name: StaticString, dim: StaticString]() -> Int:
    """Reads the GPU index or size `name` along `dim`.

    Block and grid sizes come from the kernel's implicit arguments: the grid
    size in blocks as three `u32` values at the start, and the block size as
    three `u16` values after them.
    """
    comptime if name == "lane_id":
        comptime none = Int32(-1)
        comptime zero = Int32(0)
        var t = llvm_intrinsic[
            "llvm.amdgcn.mbcnt.lo", Int32, has_side_effect=False
        ](none, zero)
        var i = llvm_intrinsic[
            "llvm.amdgcn.mbcnt.hi", Int32, has_side_effect=False
        ](none, t)
        return Int(i)
    elif name == "thread_idx":
        return Int(
            llvm_intrinsic[
                get_static_string["llvm.amdgcn.workitem.id.", dim](),
                UInt32,
                has_side_effect=False,
            ]()
        )
    elif name == "block_idx":
        return Int(
            llvm_intrinsic[
                get_static_string["llvm.amdgcn.workgroup.id.", dim](),
                UInt32,
                has_side_effect=False,
            ]()
        )
    elif name == "block_dim":
        return _implicitarg_load[6 + _dim_index[dim](), DType.uint16]()
    else:
        comptime assert name == "grid_dim", "unknown GPU id"
        return _implicitarg_load[_dim_index[dim](), DType.uint32]()
