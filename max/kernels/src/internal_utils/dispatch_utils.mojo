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

from builtin.sort import _quicksort
from os import abort


# DO NOT CHANGE
@register_passable("trivial")
trait TuningConfig(ImplicitlyCopyable, Movable, Stringable):
    ...


# DO NOT CHANGE
struct Table[type: TuningConfig](Stringable):
    var configs: List[type]
    var name: String
    var num_configs: UInt

    fn __init__(out self, configs: List[type], name: String):
        self.configs = configs.copy()
        self.name = name
        self.num_configs = UInt(len(configs))

        if not self.check():
            abort(String("Failed to Compile Table: [", self.name, "]"))

    # Method to check there are no redundancies in table (based on __str__).
    fn check(self) -> Bool:
        var keys = List[String]()
        var is_valid = True

        for i in range(len(self.configs)):
            var cfg = self.configs[i]
            var res = String(cfg)
            if res in keys:
                print(
                    "ERROR: Redundant Entry [",
                    self.name,
                    "][",
                    String(i),
                    "] ",
                    String(cfg),
                    sep="",
                )
                is_valid = False
                continue
            keys.append(res)
        return is_valid

    fn __str__(self) -> String:
        var s = List[String](self.name)
        for i in range(len(self.configs)):
            var cfg = self.configs[i]
            s += [String("[", i, "] ", String(cfg))]
        return "\n".join(s)

    # Method `query_index` queries a unique list of values for each parameter.
    # Find the indices of all matching values in the list.
    # Notes:
    #   - `domain` is a list of indices to narrow down the search.
    #     These indices are marked valid in the flag and may not represent the entire domain.
    #   - Returns a list of matching indices, not the entire domain.
    fn query_index[
        rule: fn (type) capturing -> Bool, domain: List[Int] = List[Int]()
    ](self) -> List[Int]:
        var flag: List[Bool]

        @parameter
        if len(domain):
            flag = List[Bool](length=self.num_configs, fill=False)
            for idx in materialize[domain]():
                flag[idx] = True
        else:
            flag = List[Bool](length=self.num_configs, fill=True)

        for i in range(self.num_configs):
            flag[i] &= rule(self.configs[i])
        var result_idx_list = List[Int]()

        for i in range(self.num_configs):
            if flag[i]:
                result_idx_list.append(i)
        return result_idx_list^

    # Apply rule on all configs in the table and return list of all the unique results.
    fn query_values[
        ret_type: Comparable & ImplicitlyCopyable & Movable,
        rule: fn (type) capturing -> ret_type,
        idx_list: List[Int] = List[Int](),
    ](self) -> List[ret_type]:
        var result = List[ret_type]()

        @always_inline
        @parameter
        fn _get_search_idx_list() -> List[Int]:
            @parameter
            if idx_list:
                return materialize[idx_list]()
            else:
                return [idx for idx in range(self.num_configs)]

        var search_idx_list = _get_search_idx_list()

        for idx in search_idx_list:
            value = rule(self.configs[idx])
            if value not in result:
                result.append(value)

        @parameter
        fn _cmp(lsh: ret_type, rhs: ret_type) -> Bool:
            return lsh < rhs

        _quicksort[_cmp](result)
        return result^
