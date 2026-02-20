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
"""Provides device query utilities for communication primitives. """

from sys.info import _accelerator_arch
from internal_utils import TuningConfig, Table
from gpu.host.info import GPUInfo


@fieldwise_init
struct TuningConfigAllreduce(TrivialRegisterPassable, TuningConfig):
    """
    Parameters:
        ngpus: Number of GPUs for running allreduce.
        num_bytes: Total number of input bytes supported by the config.
        sm_version: SM version (as string).
        num_blocks: Number of thread blocks for running allreduce.
    """

    var ngpus: Int
    var num_bytes: Int
    var sm_version: StaticString
    var num_blocks: Int

    fn __str__(self) -> String:
        return String(
            self.ngpus, self.num_bytes, self.sm_version, self.num_blocks
        )


comptime allreduce_table = Table(
    [
        # default for sm90 (encoded with ngpus=-1, num_bytes=-1)
        TuningConfigAllreduce(
            ngpus=-1, num_bytes=-1, sm_version="sm_90a", num_blocks=216
        ),
        TuningConfigAllreduce(
            ngpus=4, num_bytes=(1 << 27), sm_version="sm_90a", num_blocks=232
        ),
        # default for sm100 (encoded with ngpus=-1, num_bytes=-1)
        TuningConfigAllreduce(
            ngpus=-1, num_bytes=-1, sm_version="sm_100a", num_blocks=512
        ),
        # Tuning results for sm100 (2xB200, 4xB200)
        TuningConfigAllreduce(
            ngpus=2, num_bytes=(1 << 23), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=2, num_bytes=(1 << 24), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=2, num_bytes=(1 << 25), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=2, num_bytes=(1 << 26), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=2, num_bytes=(1 << 27), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=4, num_bytes=(1 << 23), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=4, num_bytes=(1 << 24), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=4, num_bytes=(1 << 25), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=4, num_bytes=(1 << 26), sm_version="sm_100a", num_blocks=512
        ),
        TuningConfigAllreduce(
            ngpus=4, num_bytes=(1 << 27), sm_version="sm_100a", num_blocks=512
        ),
    ],
    "allreduce_table",
)


@always_inline
fn _dispatch_max_num_blocks[
    ngpus: Int, sm_version: StaticString
](num_bytes: Int) -> Int:
    """
    This function searches for tuning configs with matching sm_version
    and ngpus. If such configs are found, then the search continues for
    finding the config x where num_bytes <= x.num_bytes.

    If no matching config is found then falls back to default configs
    (encoded with ngpus=-1 and num_bytes=-1)
    """

    # get default entry
    # TODO: first search for default for that sm
    # if not found look for a generic config
    @parameter
    fn rule_eq_arch_default(x: TuningConfigAllreduce) -> Bool:
        return x.ngpus == -1 and x.num_bytes == -1

    comptime default_idx = allreduce_table.query_index[rule_eq_arch_default]()
    comptime assert len(default_idx) > 0
    comptime default_entry = allreduce_table.configs[default_idx[0]]
    var default_num_blocks = default_entry.num_blocks

    # Override defaults for specific AMD CDNA3 parts regardless of sm_version aliasing
    comptime arch = _accelerator_arch()

    comptime if "gfx950" in arch:  # MI355 family
        default_num_blocks = 64
    elif "gfx942" in arch:  # MI300 family
        default_num_blocks = 32

    # narrowing the search space to matching sm_version and ngpus
    @parameter
    fn rule_eq_arch_ngpus(x: TuningConfigAllreduce) -> Bool:
        return x.sm_version == sm_version and x.ngpus == ngpus

    comptime search_domain = allreduce_table.query_index[rule_eq_arch_ngpus]()

    comptime if not search_domain:
        return default_num_blocks

    # get all static num_bytes values in table within the search space
    @parameter
    fn rule_get_num_bytes(x: TuningConfigAllreduce) -> Int:
        return x.num_bytes

    comptime all_num_bytes_values = allreduce_table.query_values[
        Int, rule_get_num_bytes, search_domain
    ]()

    comptime for nb in all_num_bytes_values:

        @parameter
        fn rule_eq_nb(x: TuningConfigAllreduce) -> Bool:
            return x.num_bytes == nb

        # Find the fist config x with input 'num_bytes <= x.num_bytes'
        if num_bytes <= nb:
            comptime idx_list = allreduce_table.query_index[
                rule_eq_nb, domain=search_domain
            ]()

            comptime if idx_list:
                comptime entry = allreduce_table.configs[idx_list[0]]
                return entry.num_blocks
            else:
                break

    return default_num_blocks


fn get_sm_version() -> StaticString:
    comptime default_device_info = GPUInfo.from_name[_accelerator_arch()]()
    return default_device_info.version
