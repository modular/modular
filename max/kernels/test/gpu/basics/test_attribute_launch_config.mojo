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
"""Tests launching with the NVIDIA configurations that carry launch attributes.

Needs thread block clusters and programmatic dependent launch (PDL), so
NVIDIA sm_90 and newer only.
"""

from std.sys import size_of
from std.sys.info import align_of
from std.testing import assert_equal, assert_raises

from max.gpu import block_idx, thread_idx
from max.gpu.host import (
    DeviceContext,
    Dim,
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)
from max.gpu.host.nvidia import (
    ClusterAttributeLaunchConfig,
    GridBlockAttributeLaunchConfig,
)
from max.gpu.memory import external_memory
from max.gpu.primitives.cluster import block_rank_in_cluster
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import barrier

comptime NUM_BLOCKS = 4
comptime BLOCK_SIZE = 128
comptime N = NUM_BLOCKS * BLOCK_SIZE

# Above CUDA's 48 KB default cap on dynamic shared memory.
comptime LARGE_SMEM_BYTES = 64 * 1024


def produce(dst: MutPointer[Int32, MutUntrackedOrigin], value: Int32):
    # The previous iteration's `consume` reads `dst`, so wait for it.
    wait_on_dependent_grids()
    var i = Int(block_idx.x * BLOCK_SIZE + thread_idx.x)
    dst[i] = value + Int32(i)
    launch_dependent_grids()


def consume(
    dst: MutPointer[Int32, MutUntrackedOrigin],
    src: MutPointer[Int32, MutUntrackedOrigin],
):
    wait_on_dependent_grids()
    var i = Int(block_idx.x * BLOCK_SIZE + thread_idx.x)
    dst[i] = 2 * src[i]
    launch_dependent_grids()


def noop():
    pass


def cluster_large_smem(
    dst: MutPointer[Int32, MutUntrackedOrigin],
    ranks: MutPointer[Int32, MutUntrackedOrigin],
):
    comptime last = LARGE_SMEM_BYTES // size_of[Int32]() - BLOCK_SIZE
    wait_on_dependent_grids()
    var smem = external_memory[
        Int32, address_space=.SHARED, alignment=align_of[Int32]()
    ]()
    var t = Int(thread_idx.x)
    smem[t] = Int32(t)
    smem[last + t] = Int32(t)
    barrier()
    dst[Int(block_idx.x) * BLOCK_SIZE + t] = smem[t] + smem[last + t]
    if t == 0:
        ranks[Int(block_idx.x)] = Int32(block_rank_in_cluster())


def apply_closure[
    F: ImplicitlyCopyable & RegisterPassable & def(Int32) -> Int32
](dst: MutPointer[Int32, MutUntrackedOrigin], f: F):
    wait_on_dependent_grids()
    var i = Int(block_idx.x * BLOCK_SIZE + thread_idx.x)
    dst[i] = f(Int32(i))


def test_execution_config_type() raises:
    comptime assert (
        DeviceContext[].ExecutionConfigType == ClusterAttributeLaunchConfig
    ), "an sm_90+ target must resolve to ClusterAttributeLaunchConfig"


def test_pdl_chain(ctx: DeviceContext) raises:
    comptime iterations = 8
    var a = ctx.enqueue_create_buffer[.int32](N)
    var b = ctx.enqueue_create_buffer[.int32](N)
    a.enqueue_fill(0)
    b.enqueue_fill(0)

    for k in range(iterations):
        ctx.enqueue_function[produce](
            GridBlockAttributeLaunchConfig(
                grid_dim=NUM_BLOCKS,
                block_dim=BLOCK_SIZE,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            ),
            a,
            Int32(k),
        )
        ctx.enqueue_function[consume](
            GridBlockAttributeLaunchConfig(
                grid_dim=NUM_BLOCKS,
                block_dim=BLOCK_SIZE,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            ),
            b,
            a,
        )

    with b.map_to_host() as h:
        for i in range(N):
            assert_equal(h[i], Int32(2 * (iterations - 1 + i)))


def test_host_args(ctx: DeviceContext) raises:
    var offset: Int32 = 7

    @inline(.always)
    def add_offset(x: Int32) {var offset} -> Int32:
        return x + offset

    var dst = ctx.enqueue_create_buffer[.int32](N)
    dst.enqueue_fill(0)
    ctx.enqueue_function[apply_closure[type_of(add_offset)]](
        GridBlockAttributeLaunchConfig(
            grid_dim=NUM_BLOCKS,
            block_dim=BLOCK_SIZE,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        ),
        dst,
        host_args=(add_offset,),
    )
    with dst.map_to_host() as h:
        for i in range(N):
            assert_equal(h[i], Int32(i) + offset)


def test_attributes_reach_launch(ctx: DeviceContext) raises:
    # A cooperative launch fails when its blocks can't all be resident at once,
    # which only happens if the config's attributes reach the driver.
    with assert_raises():
        ctx.enqueue_function[noop](
            GridBlockAttributeLaunchConfig(
                grid_dim=1 << 16,
                block_dim=1024,
                attributes=[
                    LaunchAttribute(
                        LaunchAttributeID.COOPERATIVE,
                        LaunchAttributeValue(True),
                    )
                ],
            )
        )


def test_cluster_dim_attribute_rejected(ctx: DeviceContext) raises:
    with assert_raises(contains="cluster_dim"):
        ctx.enqueue_function[noop](
            GridBlockAttributeLaunchConfig(
                grid_dim=2,
                block_dim=1,
                attributes=[LaunchAttribute.from_cluster_dim(Dim(2, 1, 1))],
            )
        )
    with assert_raises(contains="cluster_dim"):
        ctx.enqueue_function[noop](
            ClusterAttributeLaunchConfig(
                grid_dim=2,
                block_dim=1,
                cluster_dim=Dim(2, 1, 1),
                attributes=[LaunchAttribute.from_cluster_dim(Dim(2, 1, 1))],
            )
        )


def test_cluster_large_smem_pdl(ctx: DeviceContext) raises:
    comptime cluster_size = 2
    var dst = ctx.enqueue_create_buffer[.int32](N)
    var ranks = ctx.enqueue_create_buffer[.int32](NUM_BLOCKS)
    dst.enqueue_fill(-1)
    ranks.enqueue_fill(-1)

    ctx.enqueue_function[cluster_large_smem](
        ClusterAttributeLaunchConfig(
            grid_dim=NUM_BLOCKS,
            block_dim=BLOCK_SIZE,
            cluster_dim=Dim(cluster_size, 1, 1),
            shared_mem_bytes=LARGE_SMEM_BYTES,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        ),
        dst,
        ranks,
    )

    with dst.map_to_host() as h:
        for i in range(N):
            assert_equal(h[i], Int32(2 * (i % BLOCK_SIZE)))
    with ranks.map_to_host() as h:
        for i in range(NUM_BLOCKS):
            assert_equal(h[i], Int32(i % cluster_size))


def main() raises:
    test_execution_config_type()
    with DeviceContext() as ctx:
        test_pdl_chain(ctx)
        test_host_args(ctx)
        test_cluster_large_smem_pdl(ctx)
        test_cluster_dim_attribute_rejected(ctx)
        test_attributes_reach_launch(ctx)
