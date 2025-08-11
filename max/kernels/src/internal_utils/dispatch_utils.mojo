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

from builtin.sort import _quicksort, _SortWrapper


# DO NOT CHANGE
@register_passable("trivial")
trait TuningConfig(Copyable, Movable, Stringable):
    ...


# DO NOT CHANGE
# @fieldwise_init
@register_passable("trivial")
struct Table[
    type: TuningConfig, //, configs: List[type], name: String = "TuningTable"
](Stringable):
    alias num_configs = len(configs)

    @always_inline("nodebug")
    fn __init__(out self):
        Self.check()

    # Method to check there are no redundancies in table (based on __str__).
    @staticmethod
    @always_inline
    fn check():
        @parameter
        @always_inline
        fn _check() -> Bool:
            var keys = List[String]()
            var is_valid = True

            @parameter
            for i in range(len(configs)):
                var cfg = configs[i]
                var res = String(cfg)
                if res in keys:
                    print(
                        "ERROR: Redundant Entry [",
                        Self.name,
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

        constrained[_check()]()

    fn __str__(self) -> String:
        var s = List[String](name)

        @parameter
        for i in range(len(configs)):
            var cfg = configs[i]
            s += [String("[", i, "] ", String(cfg))]
        return "\n".join(s)

    # Method `query_index` queries a unique list of values for each parameter.
    # Find the indices of all matching values in the list.
    # Notes:
    #   - `domain` is a list of indices to narrow down the search.
    #     These indices are marked valid in the flag and may not represent the entire domain.
    #   - Returns a list of matching indices, not the entire domain.
    @staticmethod
    fn query_index[
        rule: fn (type) capturing -> Bool, domain: List[Int] = List[Int]()
    ]() -> List[Int]:
        var flag: InlineArray[Bool, Self.num_configs]

        @parameter
        if len(domain):
            flag = InlineArray[Bool, Self.num_configs](fill=False)

            @parameter
            for idx in domain:
                flag[idx] = True
        else:
            flag = InlineArray[Bool, Self.num_configs](fill=True)

        @parameter
        for i in range(Self.num_configs):
            flag[i] &= rule(configs[i])

        var result_idx_list = List[Int]()

        @parameter
        for i in range(Self.num_configs):
            if flag[i]:
                result_idx_list.append(i)
        return result_idx_list

    # Apply rule on all configs in the table and return list of all the unique results.
    @staticmethod
    fn query_values[
        ret_type: Comparable & Copyable & Movable,
        rule: fn (type) capturing -> ret_type,
        idx_list: List[Int] = List[Int](),
    ]() -> List[ret_type]:
        var result = List[ret_type]()

        @parameter
        if idx_list:

            @parameter
            for idx in idx_list:
                alias cfg = configs[idx]
                alias value = rule(cfg)
                if value not in result:
                    result.append(value)
        else:

            @parameter
            for idx in range(Self.num_configs):
                alias cfg = configs[idx]
                alias value = rule(cfg)
                if value not in result:
                    result.append(value)

        @parameter
        fn _cmp(
            lsh: _SortWrapper[ret_type], rhs: _SortWrapper[ret_type]
        ) -> Bool:
            return lsh.data < rhs.data

        _quicksort[_cmp](result)

        return result
