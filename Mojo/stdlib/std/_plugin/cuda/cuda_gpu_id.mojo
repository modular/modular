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

from std._gpu.primitives.id import _GPUIdPluginHookFnType, _is_cluster_id
from std.collections import OptionalReg
from std.collections.string.string_span import get_static_string
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget


def _nvptx_gpu_id_fn[
    name: StaticString, dim: StaticString, target: CompilationTarget
]() -> OptionalReg[_GPUIdPluginHookFnType]:
    """Returns the PTX read of `name` along `dim` on `target`, or `None` for
    a thread block cluster id on a GPU older than SM90, which has no clusters.
    """
    comptime if _is_cluster_id(name) and not target._is_sm_9x_or_newer():
        return None
    else:
        return comptime (
            OptionalReg[_GPUIdPluginHookFnType](_nvptx_gpu_id[name, dim])
        )


@inline(.always)
def _sreg_read[register: StaticString, dtype: DType]() -> Int:
    return Int(
        llvm_intrinsic[
            get_static_string["llvm.nvvm.read.ptx.sreg.", register](),
            Scalar[dtype],
            has_side_effect=False,
        ]()
    )


@inline(.always)
def _nvptx_gpu_id[name: StaticString, dim: StaticString]() -> Int:
    """Reads the GPU index or size `name` along `dim` from its PTX special
    register.
    """
    comptime if name == "lane_id":
        return _sreg_read["laneid", DType.int32]()
    elif name == "thread_idx":
        return _sreg_read[get_static_string["tid.", dim](), DType.uint32]()
    elif name == "block_idx":
        return _sreg_read[get_static_string["ctaid.", dim](), DType.uint32]()
    elif name == "block_dim":
        return _sreg_read[get_static_string["ntid.", dim](), DType.int32]()
    elif name == "cluster_dim":
        return _sreg_read[
            get_static_string["cluster.nctaid.", dim](), DType.int32
        ]()
    elif name == "cluster_idx":
        return _sreg_read[
            get_static_string["clusterid.", dim](), DType.uint32
        ]()
    elif name == "block_id_in_cluster":
        return _sreg_read[
            get_static_string["cluster.ctaid.", dim](), DType.uint32
        ]()
    else:
        comptime assert name == "grid_dim", "unknown GPU id"
        return _sreg_read[get_static_string["nctaid.", dim](), DType.int32]()
