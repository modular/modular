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

from gpu.host.info import *
from testing import *


def a100_occupancy_test():
    assert_equal(
        A100.occupancy(threads_per_block=256, registers_per_thread=32), 1
    )
    assert_equal(
        A100.occupancy(threads_per_block=128, registers_per_thread=33), 0.75
    )
    assert_equal(
        A100.occupancy(threads_per_block=256, registers_per_thread=41), 0.625
    )


def main():
    a100_occupancy_test()
