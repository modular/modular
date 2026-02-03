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


struct ExampleStruct:
    fn example[
        T: Writable & Copyable,  # type parameter with bounds
        count: Int,  # parameter
    ](
        self,
        data: String,  # function argument
        init_value: T,  # parametric function argument
    ) -> String:
        var out = data.copy()
        for _ in range(count):
            out = out + "{}".format(init_value)
        return out

    fn __init__(out self):
        pass


fn main():
    s = ExampleStruct()
    result = s.example[String, 3]("Value: ", "42")
    print(result)  # Expected output: "Value: 424242"
