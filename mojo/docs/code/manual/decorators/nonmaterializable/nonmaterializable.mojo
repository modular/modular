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


@fieldwise_init
@register_passable("trivial")
struct RuntimeStruct:
    var value: Int

    @implicit
    fn __init__(out self, nms: ParamStruct):
        self.value = nms.param_value


@nonmaterializable(RuntimeStruct)
@register_passable("trivial")
struct ParamStruct[param_value: Int]:
    var thing: UnsafePointer[Int]

    fn __init__(out self):
        self.thing = UnsafePointer[Int].alloc(1)
        self.thing[0] = self.param_value

    fn __add__(
        self, rhs: ParamStruct
    ) -> ParamStruct[self.param_value + rhs.param_value]:
        return ParamStruct[self.param_value + rhs.param_value]()


def main():
    alias still_param_struct = ParamStruct[1]() + ParamStruct[2]()
    print(still_param_struct.param_value)
    # When materializing to a run-time variable, it is automatically converted,
    # even without a type annotation.
    var runtime_struct = still_param_struct
    print(runtime_struct.value)
