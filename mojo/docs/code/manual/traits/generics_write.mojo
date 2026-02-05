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


fn process[T: AnyType](value: T):
    @parameter
    if conforms_to(T, Writable & ImplicitlyCopyable):
        var w = trait_downcast[Writable & ImplicitlyCopyable](value)
        print(w)
    else:
        print("<not writable>")


fn main():
    process(42)  # Int is Writable
    process("Hello, Mojo!")  # String is Writable
    process(3.14)  # Float is Writable
    process([1, 2, 3])  # List[Int] is not Writable
    process({"key": "value"})  # Dict[String, String] is not Writable
