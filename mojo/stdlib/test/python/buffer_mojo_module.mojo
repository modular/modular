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

# ===----------------------------------------------------------------------=== #
# Test for BufferProtocolBuilder.
#
# Exposes a FloatBuffer type to Python that supports the buffer protocol,
# allowing zero-copy access via memoryview and numpy.frombuffer.
# ===----------------------------------------------------------------------=== #

from std.memory import UnsafePointer
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from std.python.buffer import BufferInfo, BufferProtocolBuilder


struct FloatBuffer(Defaultable, Movable, Writable):
    """A 1-D array of Float64 values that exposes itself via the buffer protocol.
    """

    var data: List[Float64]

    def __init__(out self):
        self.data = []

    @staticmethod
    def from_count(n: PythonObject) raises -> PythonObject:
        """Create a FloatBuffer with `n` elements: [0.0, 1.0, ..., n-1.0]."""
        var result = FloatBuffer()
        var count = Int(py=n)
        for i in range(count):
            result.data.append(Float64(i))
        return PythonObject(alloc=result^)

    @staticmethod
    def get_buffer(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], flags: Int32
    ) raises -> BufferInfo:
        """Return a BufferInfo describing the internal Float64 array."""
        var data_ptr = self_ptr[].data.unsafe_ptr()
        return BufferInfo(
            buf=rebind[UnsafePointer[UInt8, MutAnyOrigin]](data_ptr),
            nitems=len(self_ptr[].data),
            itemsize=8,  # sizeof(Float64)
            format="d",  # Python struct code for C double
            readonly=True,
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write("FloatBuffer(len=", len(self.data), ")")


@export
def PyInit_buffer_mojo_module() -> PythonObject:
    try:
        var b = PythonModuleBuilder("buffer_mojo_module")
        ref tb = (
            b.add_type[FloatBuffer]("FloatBuffer")
            .def_init_defaultable[FloatBuffer]()
            .def_staticmethod[FloatBuffer.from_count]("from_count")
        )
        var bpb = BufferProtocolBuilder[FloatBuffer](tb)
        _ = bpb.def_getbuffer[FloatBuffer.get_buffer]().def_releasebuffer()
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
