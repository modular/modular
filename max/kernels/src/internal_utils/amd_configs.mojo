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

from internal_utils import Table, TuningConfig


# Setting up HW-specific tuning parameters
@fieldwise_init
@register_passable("trivial")
struct TuningConfigAMD(TuningConfig):
    # keys
    var m: Int
    var n: Int
    var k: Int

    # values
    var bm: Int
    var bn: Int

    fn __str__(self) -> String:
        var s = List[String]()
        s += ["m:" + String(self.m)]
        s += ["n:" + String(self.n)]
        s += ["k:" + String(self.k)]
        s += ["bm:" + String(self.bm)]
        s += ["bn:" + String(self.bn)]
        return "/".join(s)


# Put the tuning results in this file.
alias configs_amd = List[TuningConfigAMD](
    TuningConfigAMD(m=1, n=1, k=1, bm=11, bn=11),
    TuningConfigAMD(m=1, n=2, k=1, bm=11, bn=11),
    TuningConfigAMD(m=2, n=1, k=1, bm=22, bn=22),
    TuningConfigAMD(m=3, n=1, k=3, bm=33, bn=33),
    TuningConfigAMD(m=16, n=1, k=1, bm=33, bn=33),
)

# Make sure to register the above configs in the ConfigTable.
alias TuningTableAMD = Table(configs_amd, "TuningTableAMD")
