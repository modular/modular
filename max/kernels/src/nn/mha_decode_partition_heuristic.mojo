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

from std.bit import next_power_of_two
from std.gpu.host import DeviceAttribute, DeviceContext
from std.math import ceildiv


def cuda_mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    sm_count: Int,
) -> Int:
    if num_keys > 512:
        return min(
            next_power_of_two(
                min(
                    sm_count // (batch_size * heads_per_group),
                    num_keys // 512,
                )
            ),
            32,
        )
    return 1


def hip_mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    sm_count: Int,
) -> Int:
    # AMD split-k strategy: scale partitioning based on occupancy.
    # 256: min context length where split-k overhead is worthwhile.
    if num_keys <= 256:
        return 1

    # Compute total work items (occupancy).
    work_items = batch_size * heads_per_group

    # High occupancy when work_items >= sm_count (>=1 work item per CU).
    if work_items >= sm_count:
        # High occupancy: scale partition size to avoid over-partitioning.
        # 256: base partition size matching the kernel block width.
        # 64: scaling factor that reduces partitions as occupancy increases.
        occupancy_scale = work_items // 64
        return min(ceildiv(num_keys, 256 * occupancy_scale), 64)

    # Low occupancy: aggressive partitioning for more parallelism.
    # 256: keys per partition matching the kernel block width.
    # 64: max partitions, matching the AMD wavefront reduction limit.
    return min(ceildiv(num_keys, 256), 64)


def mha_decoding_num_partitions(
    batch_size: Int,
    num_keys: Int,
    heads_per_group: Int,
    ctx: DeviceContext,
) raises -> Int:
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    if ctx.api() == "hip":
        return hip_mha_decoding_num_partitions(
            batch_size,
            num_keys,
            heads_per_group,
            sm_count,
        )
    if ctx.api() == "cuda":
        return cuda_mha_decoding_num_partitions(
            batch_size,
            num_keys,
            heads_per_group,
            sm_count,
        )
    raise Error("Expected a CUDA or HIP device context.")
