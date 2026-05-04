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
"""Provides device query utilities for communication primitives."""

from std.sys.info import _accelerator_arch
from internal_utils import TuningConfig, Table
from std.gpu.host.info import GPUInfo


trait CommTuningConfig(TuningConfig):
    def get_num_blocks(self) -> Int:
        ...

    def get_num_bytes(self) -> Int:
        ...

    def get_sm_version(self) -> StaticString:
        ...

    def get_ngpus(self) -> Int:
        ...


@fieldwise_init
struct DefaultCommTuningConfig(CommTuningConfig, TrivialRegisterPassable):
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

    def get_num_blocks(self) -> Int:
        return self.num_blocks

    def get_num_bytes(self) -> Int:
        return self.num_bytes

    def get_sm_version(self) -> StaticString:
        return self.sm_version

    def get_ngpus(self) -> Int:
        return self.ngpus

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tuning config as a string.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            self.ngpus, self.num_bytes, self.sm_version, self.num_blocks
        )


@always_inline
def dispatch_select_comm_config[
    TuningTableType: CommTuningConfig,
    //,
    ngpus: Int,
    sm_version: StaticString,
    tuning_table: Table[TuningTableType],
](num_bytes: Int) -> TuningTableType:
    """
    This function searches for tuning configs with matching sm_version
    and ngpus. If such configs are found, then the search continues for
    finding the config x where num_bytes <= x.num_bytes.

    Falls back to the arch-specific default (ngpus=-1, num_bytes=-1,
    matching sm_version), or if none exists, to the global default
    (ngpus=-1, num_bytes=-1, sm_version="default").
    """

    # Validate that every entry has num_blocks <= 512 (MAX_NUM_BLOCKS_UPPER_BOUND
    # from sync.mojo). _multi_gpu_barrier indexes Signal.self_counter and
    # Signal.peer_counter with block_idx.x; those arrays are statically sized
    # to MAX_NUM_BLOCKS_UPPER_BOUND, so an entry exceeding 512 would silently
    # corrupt barrier state.
    @parameter
    def _entry_exceeds_block_bound(x: tuning_table.type) -> Bool:
        return x.get_num_blocks() > 512

    comptime _over_limit = tuning_table.query_index[
        _entry_exceeds_block_bound
    ]()
    comptime assert (
        len(_over_limit) == 0
    ), "tuning_table entry has num_blocks > MAX_NUM_BLOCKS_UPPER_BOUND (512)"

    # get default entry: prefer arch-specific, fall back to global default
    @parameter
    def rule_eq_arch_default(x: tuning_table.type) -> Bool:
        return (
            x.get_ngpus() == -1
            and x.get_num_bytes() == -1
            and x.get_sm_version() == sm_version
        )

    @parameter
    def rule_eq_global_default(x: tuning_table.type) -> Bool:
        return (
            x.get_ngpus() == -1
            and x.get_num_bytes() == -1
            and x.get_sm_version() == "default"
        )

    comptime arch_default_idx = tuning_table.query_index[rule_eq_arch_default]()
    comptime global_default_idx = tuning_table.query_index[
        rule_eq_global_default
    ]()
    comptime default_idx = arch_default_idx if len(
        arch_default_idx
    ) > 0 else global_default_idx
    comptime assert len(default_idx) > 0, (
        "tuning_table must have a default entry for sm_version: "
        + sm_version
        + " or a global default entry (sm_version='default')"
    )
    comptime default_entry = tuning_table.configs[default_idx[0]]

    # narrowing the search space to matching sm_version and ngpus
    @parameter
    def rule_eq_arch_ngpus(x: tuning_table.type) -> Bool:
        return x.get_sm_version() == sm_version and x.get_ngpus() == ngpus

    comptime search_domain = tuning_table.query_index[rule_eq_arch_ngpus]()

    comptime if not search_domain:
        return default_entry

    # get all static num_bytes values in table within the search space
    @parameter
    def rule_get_num_bytes(x: tuning_table.type) -> Int:
        return x.get_num_bytes()

    comptime all_num_bytes_values = tuning_table.query_values[
        Int, rule_get_num_bytes, search_domain
    ]()

    comptime for nb in all_num_bytes_values:

        @parameter
        def rule_eq_nb(x: tuning_table.type) -> Bool:
            return x.get_num_bytes() == nb

        # Find the fist config x with input 'num_bytes <= x.num_bytes'
        if num_bytes <= nb:
            comptime idx_list = tuning_table.query_index[
                rule_eq_nb, domain=search_domain
            ]()

            comptime if idx_list:
                comptime entry = tuning_table.configs[idx_list[0]]
                return entry
            else:
                break

    return default_entry


def get_sm_version() -> StaticString:
    comptime default_device_info = GPUInfo.from_name[_accelerator_arch()]()
    return default_device_info.version
