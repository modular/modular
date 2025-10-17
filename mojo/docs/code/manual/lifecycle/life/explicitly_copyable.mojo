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


struct BigExpensiveStruct(Copyable, Movable):
    var value: Int

    @implicit
    fn __init__(out self, value: Int = 0):
        self.value = value

    fn copy(self, out copy: Self):
        copy = Self(self.value)


struct ExplicitCopyOnly[ElementType: Copyable & Movable](Copyable, Movable):
    var ptr: UnsafePointer[ElementType]

    fn __init__(out self, var elt: ElementType):
        """Constructs a new container, storing the given value."""
        self.ptr = UnsafePointer[ElementType].alloc(1)
        self.ptr.init_pointee_move(elt^)

    fn copy(self) -> Self:
        """Performs a deep copy of this container."""
        elt_copy = self.ptr[].copy()
        copy = Self(elt_copy^)
        return copy^

    fn __getitem__(ref self) -> ref [self] ElementType:
        """Returns a reference to the stored value."""
        return self.ptr[]


def main():
    big = BigExpensiveStruct()
    original = ExplicitCopyOnly(big^)
    # implicit_copy = original  # error: 'ExplicitCopyOnly[BigExpensiveStruct]'
    #                         is not copyable because it has no '__copyinit__'
    copy = original.copy()
    copy[].value = 300
    print(copy[].value)
    print(original[].value)
