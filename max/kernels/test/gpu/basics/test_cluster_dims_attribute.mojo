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

from gpu.cluster import block_rank_in_cluster
from gpu.host import DeviceContext, Dim, FuncAttribute
from gpu.host.compile import get_gpu_target
from gpu.id import block_idx, cluster_dim, cluster_idx

from utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.cluster_dim`=cluster_dims)
fn test_cluster_dims_attribute_kernel_with_param[
    cluster_dims: StaticTuple[Int32, 3]
]():
    print(
        "CLUSTER DIMS(",
        cluster_dim.x,
        cluster_dim.y,
        cluster_dim.z,
        ")",
        "BLOCK(",
        block_idx.x,
        block_idx.y,
        block_idx.z,
        ")",
        "CLUSTER(",
        cluster_idx.x,
        cluster_idx.y,
        cluster_idx.z,
        ")",
    )


@__llvm_metadata(`nvvm.cluster_dim`=StaticTuple[Int32, 3](2, 1, 1))
fn test_cluster_dims_attribute_kernel():
    print(
        "CLUSTER DIMS(",
        cluster_dim.x,
        cluster_dim.y,
        cluster_dim.z,
        ")",
        "BLOCK(",
        block_idx.x,
        block_idx.y,
        block_idx.z,
        ")",
        "CLUSTER(",
        cluster_idx.x,
        cluster_idx.y,
        cluster_idx.z,
        ")",
    )


# CHECK-LABEL: test_cluster_dims_attribute
# CHECK-DAG: CLUSTER DIMS( 2 1 1 ) BLOCK( 0 0 0 ) CLUSTER( 0 0 0 )
# CHECK-DAG: CLUSTER DIMS( 2 1 1 ) BLOCK( 1 0 0 ) CLUSTER( 0 0 0 )
# CHECK-DAG: CLUSTER DIMS( 2 1 1 ) BLOCK( 1 1 0 ) CLUSTER( 0 1 0 )
# CHECK-DAG: CLUSTER DIMS( 2 1 1 ) BLOCK( 0 1 0 ) CLUSTER( 0 1 0 )
fn test_cluster_dims_attribute(ctx: DeviceContext) raises:
    print("== test_cluster_dims_attribute")
    ctx.enqueue_function[test_cluster_dims_attribute_kernel](
        grid_dim=(2, 2, 1), block_dim=(1)
    )
    ctx.synchronize()


# CHECK-LABEL: test_cluster_dims_attribute_with_param
# CHECK-DAG: CLUSTER DIMS( 1 2 1 ) BLOCK( 0 1 0 ) CLUSTER( 0 0 0 )
# CHECK-DAG: CLUSTER DIMS( 1 2 1 ) BLOCK( 0 0 0 ) CLUSTER( 0 0 0 )
# CHECK-DAG: CLUSTER DIMS( 1 2 1 ) BLOCK( 1 0 0 ) CLUSTER( 1 0 0 )
# CHECK-DAG: CLUSTER DIMS( 1 2 1 ) BLOCK( 1 1 0 ) CLUSTER( 1 0 0 )
fn test_cluster_dims_attribute_with_param(ctx: DeviceContext) raises:
    print("== test_cluster_dims_attribute_with_param")
    alias x = StaticTuple[Int32, 3](1, 2, 1)
    ctx.enqueue_function[test_cluster_dims_attribute_kernel_with_param[x]](
        grid_dim=(2, 2, 1), block_dim=(1)
    )
    ctx.synchronize()


def main():
    with DeviceContext() as ctx:
        test_cluster_dims_attribute(ctx)
        test_cluster_dims_attribute_with_param(ctx)
