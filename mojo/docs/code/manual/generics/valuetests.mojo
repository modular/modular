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


struct SizedListWrapper[capacity: Int, T: Writable & Copyable](
    Sized, Writable where conforms_to(T, Writable) and capacity > 0
):
    var data: List[Self.T]

    def __init__(out self, value: Self.T):
        self.data = List[Self.T](capacity=Self.capacity)
        for _ in range(Self.capacity):
            self.data.append(value.copy())

    def __len__(self) -> Int:
        return len(self.data)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(repr(self.data))

    def first(self) -> Self.T where Self.capacity > 0:
        return self.data[0].copy()


def main() raises:
    from std.testing import assert_equal, assert_true

    var s = SizedListWrapper[5, Int](42)
    assert_equal(s.first(), 42)
    assert_true(conforms_to(type_of(s), Writable))

    comptime CollectionElement = ImplicitlyCopyable & ImplicitlyDestructible

    def make_filled[T: CollectionElement, size: Int](splat_value: T) -> List[T]:
        var result = List[T](capacity=size)
        for _ in range(size):
            result.append(splat_value)
        return result^

    var three_zeros = make_filled[Int, 3](0)
    var five_hellos = make_filled[String, 5]("hello")
    assert_equal(three_zeros, [0, 0, 0])
    assert_equal(five_hellos, ["hello", "hello", "hello", "hello", "hello"])
