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
from python import Python, PythonObject
from python._cpython import PyObjectPtr, Py_LT, Py_GT
from python.bindings import PythonModuleBuilder, TypeObjectSlot


fn _extent(pos: List[Float64]) -> Tuple[Float64, Float64]:
    """Return the min and max value in the list."""
    v_min = Float64.MAX
    v_max = Float64.MIN
    for v in pos:
        v_min = min(v_min, v)
        v_max = max(v_min, v)
    return (v_min, v_max)


fn _compute_bounding_box_area(
    pos_x: List[Float64], pos_y: List[Float64]
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

    var pos_x: List[Float64]
    var pos_y: List[Float64]

    # The bounding box area that contains all points.
    var _bounding_box_area: Float64

    fn __init__(out self):
        """Default initializer."""
        self.pos_x = {}
        self.pos_y = {}
        self._bounding_box_area = 0

    fn __init__(out self, var x: List[Float64], var y: List[Float64]):
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

        m_x = List[Float64](capacity=len_x)
        for value in pos_x:
            var f = Float64(value)
            m_x.append(f)
        m_y = List[Float64](capacity=len_x)
        for value in pos_y:
            var f = Float64(value)
            m_y.append(f)

        return PythonObject(alloc=DataFrame(m_x^, m_y^))

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
        return String("DataFrame( length=", len(self.pos_x), ")")

    @staticmethod
    fn rich_compare(
        self_ptr: PyObjectPtr, other: PyObjectPtr, op: Int
    ) raises -> Bool:
        """Implement the rich compare functionality.

        We use the bounding box area to order the DataFrame objects.
        """
        var self_df = Self._get_self_ptr(PythonObject(from_borrowed=self_ptr))
        var other_df = Self._get_self_ptr(PythonObject(from_borrowed=other))
        if op == Py_LT:
            return (
                self_df[]._bounding_box_area < other_df[]._bounding_box_area
            )
        elif op == Py_GT:
            return (
                self_df[]._bounding_box_area > other_df[]._bounding_box_area
            )
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
