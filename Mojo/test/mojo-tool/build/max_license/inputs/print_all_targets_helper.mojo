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

# Helper file for testing --print-target-accelerators synchronization.
# This prints the _all_target_accelerators() values accepted by max.gpu.host.info so
# they can be compared against the C++ implementation's output.

from max.gpu.host.info import _all_target_accelerators


def main():
    # Print each normalized value on its own line for set comparison.

    comptime values = _all_target_accelerators()

    comptime for i in range(len(values)):
        comptime value = values[i]

        # TODO(MSTDL-3241): Remove empty target sentinel value.
        comptime if i == 0 and value == "":
            continue

        print(value)
