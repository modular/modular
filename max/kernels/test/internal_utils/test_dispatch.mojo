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

from bit import next_power_of_two

# from internal_utils import Table, TuningConfig
from internal_utils import (
    TuningConfigAMD,
    TuningConfigNvidia,
    TuningTableAMD,
    TuningTableNvidia,
    arg_parse,
)
from testing import assert_equal

# Highly recommended to use "vendor_arch_dtype" format for table names.
# For example:
# nvidia_sm90_fp8 = Table[...]
# import nvidia_sm90_fp8_configs


# Kernel developer can modify the dispatch accordingly.
# You can design your own dispatch queries based on available data and parameters.
fn dispatch_matmul_amd[static_n: Int, static_k: Int](m: Int) raises:
    print("Dispatch for m=", m, "/N=", static_n, "/K=", static_k, sep="")

    # First, check on exact value of M
    print("Checking exact m configs ...")

    @parameter
    @always_inline
    fn rule_eq_nk(x: TuningConfigAMD) -> Bool:
        return x.k == static_k and x.n == static_n

    # First, filter by static params N and K
    alias nk_idx_list = TuningTableAMD.query_index[rule_eq_nk]()

    # equivalently:
    # - select n==static_n
    # alias n_idx_list = TuningTable.query_index[rule(n==static_n)]()
    # - select k==static_k where n==static_n
    # alias nk_idx_list = TuningTable.query_index[rule(k==static_k), n_idx_list]()

    # Get unique the values of M in the config for the subset of NK indices.
    # Note: this is faster if numerically close values of M are placed close together in the list.
    @parameter
    @always_inline
    fn get_m(x: TuningConfigAMD) -> Int:
        return x.m

    alias m_values = TuningTableAMD.query_values[Int, get_m, nk_idx_list]()
    alias expected_m_values = List[Int](1, 2, 16)
    constrained[len(m_values) == len(expected_m_values)]()

    @parameter
    for i in range(len(m_values)):
        constrained[m_values[i] == expected_m_values[i]]()

    @parameter
    for i in range(1, len(m_values)):

        @parameter
        @always_inline
        fn rule_m(x: TuningConfigAMD) -> Bool:
            return x.m == materialize[m_values[i]]()

        if materialize[m_values[i - 1]]() < m <= materialize[m_values[i]]():
            print(
                "Searching for m: prev_m=",
                materialize[m_values[i - 1]](),
                "/m=",
                m,
                "/next_m=",
                materialize[m_values[i]](),
                sep="",
            )
            alias idx_list = TuningTableAMD.query_index[
                rule_m, domain=nk_idx_list
            ]()

            @parameter
            if idx_list:
                print("Found dispatch for next value of m", m)
                print(
                    String(materialize[TuningTableAMD.configs[idx_list[0]]]())
                )
                # call dispatch with this config
                # return


fn dispatch_matmul_nvidia[static_n: Int, static_k: Int](m: Int) raises:
    print("Dispatch for m=", m, "/N=", static_n, "/K=", static_k, sep="")

    # First, check on exact value of M
    print("Checking exact m configs ...")

    @parameter
    @always_inline
    fn rule_eq_nk(x: TuningConfigNvidia) -> Bool:
        return x.N == static_k and x.N == static_n

    # First, filter by static params N and K
    alias nk_idx_list = TuningTableNvidia.query_index[rule_eq_nk]()

    """
    equivalently:
    - select n==static_n
    alias n_idx_list = TuningTable.query_index[rule(n==static_n)]()
    - select k==static_k where n==static_n
    alias nk_idx_list = TuningTable.query_index[rule(k==static_k), n_idx_list]()
    """

    # Get unique the values of M in the config for the subset of NK indices.
    # Note: this is faster if numerically close values of M are placed close together in the list.
    @parameter
    @always_inline
    fn get_m(x: TuningConfigNvidia) -> Int:
        return x.M

    alias m_values = TuningTableNvidia.query_values[Int, get_m, nk_idx_list]()

    alias expected_m_values = List[Int](
        1, 8, 16, 32, 64, 128, 256, 65536, 128000
    )
    constrained[len(m_values) == len(expected_m_values)]()

    @parameter
    for i in range(len(m_values)):
        constrained[m_values[i] == expected_m_values[i]]()

    @parameter
    for i in range(1, len(m_values)):

        @parameter
        @always_inline
        fn rule_m(x: TuningConfigNvidia) -> Bool:
            return x.M == materialize[m_values[i]]()

        if materialize[m_values[i - 1]]() < m <= materialize[m_values[i]]():
            print(
                "Searching for m: prev_m=",
                materialize[m_values[i - 1]](),
                "/m=",
                m,
                "/next_m=",
                materialize[m_values[i]](),
                sep="",
            )
            alias idx_list = TuningTableNvidia.query_index[
                rule_m, domain=nk_idx_list
            ]()

            @parameter
            if idx_list:
                print("Found dispatch for next value of m", m)
                print(
                    String(
                        materialize[TuningTableNvidia.configs[idx_list[0]]]()
                    )
                )
                # call dispatch with this config
                # return


def main():
    var m = arg_parse("m", 0)
    print(String(materialize[TuningTableAMD]()))
    dispatch_matmul_amd[static_n=1, static_k=1](m)

    print("-----------------------------------------------------------")
    print(String(materialize[TuningTableNvidia]()))
    dispatch_matmul_nvidia[static_n=8192, static_k=8192](m)
