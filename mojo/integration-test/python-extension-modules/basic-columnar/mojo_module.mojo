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

from os import abort

from buffer import NDBuffer
from memory import UnsafePointer, alloc
from utils import IndexList
from python import Python, PythonObject
from python._cpython import PyObjectPtr, Py_LT, Py_GT
from python.bindings import PythonModuleBuilder, TypeObjectSlot


fn _extent(
    pos: NDBuffer[DType.float64, 1, MutAnyOrigin]
) -> Tuple[Float64, Float64]:
    """Return the min and max value in the buffer."""
    v_min = Float64.MAX
    v_max = Float64.MIN
    for i in range(pos.size()):
        var v = pos[i]
        v_min = min(v_min, v)
        v_max = max(v_max, v)
    return (v_min, v_max)


fn _compute_bounding_box_area(
    pos_x: NDBuffer[DType.float64, 1, MutAnyOrigin],
    pos_y: NDBuffer[DType.float64, 1, MutAnyOrigin],
) -> Float64:
    if pos_x.size() == 0:
        return 0.0
    ext_x = _extent(pos_x)
    ext_y = _extent(pos_y)
    return (ext_x[1] - ext_x[0]) * (ext_y[1] - ext_y[0])


struct DataFrame(Defaultable, Movable, Representable):
    """A simple columnar data structure.

    This struct contains points with the x,y coordinates stored in columns. Some algorithms
    are a lot more efficient in this representation.
    """

    var pos_x: NDBuffer[DType.float64, 1, MutAnyOrigin]
    var pos_y: NDBuffer[DType.float64, 1, MutAnyOrigin]

    # The bounding box area that contains all points.
    var _bounding_box_area: Float64

    fn __init__(out self):
        """Default initializer."""
        self.pos_x = NDBuffer[DType.float64, 1, MutAnyOrigin]()
        self.pos_y = NDBuffer[DType.float64, 1, MutAnyOrigin]()
        self._bounding_box_area = 0

    fn __init__(
        out self,
        x: NDBuffer[DType.float64, 1, MutAnyOrigin],
        y: NDBuffer[DType.float64, 1, MutAnyOrigin],
    ):
        self._bounding_box_area = _compute_bounding_box_area(x, y)
        self.pos_x = x
        self.pos_y = y

    fn __del__(deinit self):
        """Free the allocated memory for the buffers."""
        if self.pos_x.data:
            self.pos_x.data.free()
        if self.pos_y.data:
            self.pos_y.data.free()

    @staticmethod
    fn _get_self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            return abort[UnsafePointer[Self, MutAnyOrigin]](
                String(
                    (
                        "Python method receiver object did not have the"
                        " expected type:"
                    ),
                    e,
                )
            )

    @staticmethod
    fn with_columns(
        pos_x: PythonObject, pos_y: PythonObject
    ) raises -> PythonObject:
        var len_x = Int(pos_x.__len__())
        var len_y = Int(pos_y.__len__())
        if len_x != len_y:
            raise Error("The length of the two columns does not match.")

        # Allocate memory for the buffers
        var ptr_x = alloc[Float64](len_x)
        var ptr_y = alloc[Float64](len_x)

        # Copy values from Python objects
        var i = 0
        for value in pos_x:
            ptr_x[i] = Float64(value)
            i += 1
        i = 0
        for value in pos_y:
            ptr_y[i] = Float64(value)
            i += 1

        # Create NDBuffers with dynamic shape
        var m_x = NDBuffer[DType.float64, 1, MutAnyOrigin](
            ptr_x, IndexList[1](len_x)
        )
        var m_y = NDBuffer[DType.float64, 1, MutAnyOrigin](
            ptr_y, IndexList[1](len_x)
        )

        return PythonObject(alloc=DataFrame(m_x, m_y))

    @staticmethod
    fn py__getitem__(
        py_self: PythonObject, index: PythonObject
    ) raises -> PythonObject:
        """For __getitem__ we will return a row of the DataFrame."""
        var self_ptr = Self._get_self_ptr(py_self)
        var index_mojo = Int(index)
        return Python().tuple(
            self_ptr[].pos_x[index_mojo], self_ptr[].pos_y[index_mojo]
        )

    fn __repr__(self) -> String:
        return String("DataFrame( length=", self.pos_x.size(), ")")

    @staticmethod
    fn rich_compare(
        self_ptr: PythonObject, other: PythonObject, op: Int
    ) raises -> Bool:
        """Implement the rich compare functionality.

        We use the bounding box area to order the DataFrame objects.
        """
        var self_df = Self._get_self_ptr(self_ptr)
        var other_df = Self._get_self_ptr(other)
        if op == Py_LT:
            return self_df[]._bounding_box_area < other_df[]._bounding_box_area
        elif op == Py_GT:
            return self_df[]._bounding_box_area > other_df[]._bounding_box_area
        # For other comparisons, return False
        return False


@export
fn PyInit_mojo_module() -> PythonObject:
    """Create a Python module with a type binding for `DataFrame`."""

    try:
        var b = PythonModuleBuilder("mojo_module")
        _ = (
            b.add_type[DataFrame]("DataFrame")
            .def_init_defaultable[DataFrame]()
            .def_staticmethod[DataFrame.with_columns]("with_columns")
            # Mapping protocol.
            .def_special_method[
                DataFrame.py__getitem__, TypeObjectSlot.MappingGetItem
            ]("__getitem__")
            .def_rich_compare[DataFrame.rich_compare]()
        )
        return b.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )
