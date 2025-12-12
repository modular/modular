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


@deprecated(use=OtherStruct)
@fieldwise_init
struct MyStruct:
    var x: Int


@deprecated(use=Honkable)
trait Quackable:
    fn quack(self):
        ...


trait Honkable:
    fn honk(self):
        ...


struct OtherStruct:
    var x: Int
    var y: Int

    @deprecated("Don't touch this")
    fn mchammer(self):
        pass

    fn __init__(out self, x: Int):
        self.x = x
        self.y = 0


@deprecated("Use tau instead")
comptime pi = 3.141592


fn main():
    _ = MyStruct(x=5)

    OtherStruct(2).mchammer()

    print(pi)

    # Demonstrate that only warnings are issued
    print("This is a functioning app")
