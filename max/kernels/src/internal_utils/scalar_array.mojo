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

from std.collections import check_bounds
from std.memory import alloc, free, Layout


struct ScalarArray[dtype: DType](Movable, Writable):
    comptime ScalarType = Scalar[Self.dtype]

    var _layout: Layout[Self.ScalarType]
    var _data: UnsafePointer[Self.ScalarType, MutExternalOrigin]

    @always_inline
    def __init__(out self, *, count: Int):
        self._layout = Layout[Self.ScalarType](count=count)
        self._data = alloc(self._layout)

    @always_inline
    def __init__(out self, *, count: Int, alignment: Int):
        self._layout = Layout[Self.ScalarType](count=count, alignment=alignment)
        self._data = alloc(self._layout)

    @always_inline
    def __del__(deinit self):
        free(self._data, self._layout)

    @always_inline("nodebug")
    def as_span(ref self) -> Span[Self.ScalarType, origin_of(self)]:
        return Span(ptr=self.unsafe_ptr(), length=self._layout.count())

    @always_inline("nodebug")
    def unsafe_ptr(ref self) -> UnsafePointer[Self.ScalarType, origin_of(self)]:
        comptime origin = origin_of(self)
        return self._data.unsafe_mut_cast[origin.mut]().unsafe_origin_cast[
            origin
        ]()

    @always_inline("nodebug")
    def __getitem__(
        ref self, index: Some[Indexer]
    ) -> ref[self._data] Self.ScalarType:
        check_bounds(index, self._layout.count())
        return self._data[index]
