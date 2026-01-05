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


comptime CollectionElement = Copyable & ImplicitlyDestructible


struct _SlidingWindowIterator[T: CollectionElement]:
    """Iterates over a snapshot of values."""

    var _values: List[Self.T]
    var _index: Int

    fn __init__(out self, var values: List[Self.T]):
        self._values = values^
        self._index = 0

    fn __has_next__(self) -> Bool:
        return self._index < len(self._values)

    fn __next__(mut self) -> Self.T:
        value = self._values[self._index].copy()
        self._index += 1
        return value^

    fn __len__(self) -> Int:
        return len(self._values) - self._index


struct SlidingWindow[T: CollectionElement, Capacity: Int](Sized, Writable):
    var slots: List[Optional[Self.T]]
    var count: Int

    fn __init__(out self):
        self.slots = List[Optional[Self.T]]()
        for _ in range(Self.Capacity):
            self.slots.append(None)
        self.count = 0

    fn push(mut self, var value: Self.T) -> Optional[Self.T]:
        evicted = self.slots.pop(0)
        self.slots.append(Optional(value^))

        if not evicted:
            self.count += 1

        return evicted^

    fn __len__(self) -> Int:
        return self.count

    fn pull(self, request_count: Int) -> List[Self.T]:
        if request_count < 1:
            return List[Self.T]()

        actual = request_count if request_count <= self.count else self.count

        result = List[Self.T]()
        for i in range(actual):
            index = Self.Capacity - (i + 1)
            result.append(self.slots[index].value().copy())

        return result^

    fn push_all(mut self, var *values: Self.T) -> List[Self.T]:
        """Push multiple values, returning any evicted elements."""
        evicted = List[Self.T]()
        for value in values:
            dropped = self.push(value.copy())
            if dropped:
                evicted.append(dropped.take())
        return evicted^

    fn __iter__(ref self) -> _SlidingWindowIterator[Self.T]:
        """Iterate over current values, most recent first."""
        values = self.pull(Self.Capacity)
        return _SlidingWindowIterator(values^)

    fn write_to[W: Writer](self, mut writer: W):
        """Display the window's contents."""
        writer.write("[")
        for i in range(Self.Capacity - 1, -1, -1):
            if self.slots[i]:
                value = self.slots[i].value().copy()
                ref element = trait_downcast[Representable](value)
                writer.write(repr(element))
            else:
                writer.write("[-]")
            if i > 0:
                writer.write(", ")
        writer.write("]")


fn main():
    window = SlidingWindow[Int, 5]()
    print("Empty:", window)

    for value in range(1, 8):
        evicted = window.push(value)
        if evicted:
            print("Pushed {}, evicted {}".format(value, evicted.value()))
        else:
            print("Pushed {}, nothing evicted".format(value))
        print("Window:", window)

    print("\nPulling values:")
    for count in range(6):
        values = window.pull(count)
        print("Pull {}: {}".format(count, values))

    print("\nIterating:")
    for value in window:
        print("  ", value)
