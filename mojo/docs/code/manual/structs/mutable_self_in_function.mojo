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

# DOC: mojo/manual/struct.mdx


struct MyStruct:
    var value: Int

    fn increment(mut self):
        self.value += 1  # Works. Mutable `self` allows assignment
        # But without `mut`:
        # ERROR: expression must be mutable in assignment

    fn __init__(out self, value: Int):
        self.value = value


fn main():
    my_struct = MyStruct(1)
    my_struct.increment()
    print(my_struct.value)
