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

from memory import UnsafePointer, alloc
from utils import IndexList
from python import Python, PythonObject
from python._cpython import PyObjectPtr, Py_LT, Py_GT
from python.bindings import (
    PythonModuleBuilder,
    PyTypeObjectSlot,
)

comptime Coord1DColumn = List[Float64]


fn _extent(pos: Coord1DColumn) -> Tuple[Float64, Float64]:
    """Return the min and max value in the buffer."""
    v_min = Float64.MAX
    v_max = Float64.MIN
    for v in pos:
        v_min = min(v_min, v)
        v_max = max(v_max, v)
    return (v_min, v_max)


fn _compute_bounding_box_area(
    pos_x: Coord1DColumn,
    pos_y: Coord1DColumn,
) -> Float64:
    if len(pos_x) == 0:
        return 0.0
    ext_x = _extent(pos_x)
    ext_y = _extent(pos_y)
    return (ext_x[1] - ext_x[0]) * (ext_y[1] - ext_y[0])


struct DataFrame(Defaultable, Movable, Representable):
    """A simple columnar data structure.

    This struct contains points with the x,y coordinates stored in columns. Some algorithms
    are a lot more efficient in this representation.
    """

    var pos_x: Coord1DColumn
    var pos_y: Coord1DColumn

    # The bounding box area that contains all points.
    var _bounding_box_area: Float64

    fn __init__(out self):
        """Default initializer."""
        self.pos_x = []
        self.pos_y = []
        self._bounding_box_area = 0

    fn __init__(
        out self,
        var x: Coord1DColumn,
        var y: Coord1DColumn,
    ):
        self._bounding_box_area = _compute_bounding_box_area(x, y)
        self.pos_x = x^
        self.pos_y = y^

    @staticmethod
    fn _get_self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(
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
        var ptr_x = Coord1DColumn(capacity=len_x)
        var ptr_y = Coord1DColumn(capacity=len_y)

        # Copy values from Python objects
        for value in pos_x:
            ptr_x.append(Float64(py=value))
        for value in pos_y:
            ptr_y.append(Float64(py=value))

        return PythonObject(alloc=DataFrame(ptr_x^, ptr_y^))

    @staticmethod
    fn py__len__(py_self: PythonObject) raises -> Int:
        """For __len__ we return the number of rows in the DataFrame."""
        var self_ptr = Self._get_self_ptr(py_self)
        return len(self_ptr[].pos_x)

    @staticmethod
    fn py__getitem__(
        py_self: PythonObject, index: PythonObject
    ) raises -> PythonObject:
        """For __getitem__ we will return a row of the DataFrame."""
        var self_ptr = Self._get_self_ptr(py_self)
        var index_mojo = Int(py=index)
        return Python().tuple(
            self_ptr[].pos_x[index_mojo], self_ptr[].pos_y[index_mojo]
        )

    @staticmethod
    fn py__setitem__(
        py_self: PythonObject, index: PythonObject, value: PythonObject
    ) raises -> None:
        """For __setitem__ we set the x and y values at the given index."""
        var self_ptr = Self._get_self_ptr(py_self)
        var index_mojo = Int(py=index)
        # Expect value to be a tuple of (x, y)
        self_ptr[].pos_x[index_mojo] = Float64(py=value[0])
        self_ptr[].pos_y[index_mojo] = Float64(py=value[1])

    fn __repr__(self) -> String:
        return String("DataFrame( length=", len(self.pos_x), ")")

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
            .def_method[DataFrame.py__len__, PyTypeObjectSlot.mp_length]()
            .def_method[DataFrame.py__getitem__, PyTypeObjectSlot.mp_getitem]()
            .def_method[DataFrame.py__setitem__, PyTypeObjectSlot.mp_setitem]()
            .def_method[
                DataFrame.rich_compare, PyTypeObjectSlot.tp_richcompare
            ]()
        )
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
