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

from internal_utils import Table, TuningConfig


# Setting up HW-specific tuning parameters
@fieldwise_init
struct TuningConfigAMD(TrivialRegisterPassable, TuningConfig):
    # keys
    var m: Int
    var n: Int
    var k: Int

    # values
    var bm: Int
    var bn: Int

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tuning config as a string.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "m:",
            self.m,
            "/n:",
            self.n,
            "/k:",
            self.k,
            "/bm:",
            self.bm,
            "/bn:",
            self.bn,
        )


# Put the tuning results in this file.
comptime configs_amd: List[TuningConfigAMD] = [
    TuningConfigAMD(m=1, n=1, k=1, bm=11, bn=11),
    TuningConfigAMD(m=1, n=2, k=1, bm=11, bn=11),
    TuningConfigAMD(m=2, n=1, k=1, bm=22, bn=22),
    TuningConfigAMD(m=3, n=1, k=3, bm=33, bn=33),
    TuningConfigAMD(m=16, n=1, k=1, bm=33, bn=33),
]

# Make sure to register the above configs in the ConfigTable.
comptime TuningTableAMD = Table(configs_amd, "TuningTableAMD")
