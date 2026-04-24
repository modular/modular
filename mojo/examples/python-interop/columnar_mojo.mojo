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
# Example from https://github.com/modular/modular/pull/5562, adapted to use
# the pontoneer library instead of the stdlib additions directly.
#
# Exposes a simple columnar DataFrame to Python that supports:
#   - len(df)          via mp_length
#   - df[i]            via mp_subscript
#   - df[i] = (x, y)  via mp_ass_subscript
#   - del df[i]        via mp_ass_subscript (null value)
#   - df < other       via tp_richcompare
#   - -df              via nb_negative
#   - abs(df)          via nb_absolute
#   - bool(df)         via nb_bool
#   - df + other       via nb_add
#   - df * scalar      via nb_multiply
#   - df ** exp        via nb_power
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.memory import UnsafePointer
from std.utils import Variant
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.ffi import _Global

from std.python.utils import NotImplementedError, RichCompareOps
from std.python.builders import (
    TypeProtocolBuilder,
    MappingProtocolBuilder,
    NumberProtocolBuilder,
)

comptime Coord1DColumn = List[Float64]


def _extent(pos: Coord1DColumn) -> Tuple[Float64, Float64]:
    """Return the (min, max) of a column."""
    v_min = Float64.MAX
    v_max = Float64.MIN
    for v in pos:
        v_min = min(v_min, v)
        v_max = max(v_max, v)
    return (v_min, v_max)


def _compute_bounding_box_area(
    pos_x: Coord1DColumn,
    pos_y: Coord1DColumn,
) -> Float64:
    if len(pos_x) == 0:
        return 0.0
    ext_x = _extent(pos_x)
    ext_y = _extent(pos_y)
    return (ext_x[1] - ext_x[0]) * (ext_y[1] - ext_y[0])


def _init_call_count() -> Dict[String, Int]:
    return {}


# Track call counts for testing reasons.
comptime _global_call_count = _Global["call_count", _init_call_count]


def _get_global_call_count(
    out result: UnsafePointer[Dict[String, Int], MutExternalOrigin]
):
    """Gets the global calls count.

    Returns:
        A pointer to the global calls count dictionary.
    """
    try:
        result = _global_call_count.get_or_create_ptr()
    except:
        abort("Failed to initialize global calls count")


struct DataFrame(Defaultable, Movable, Writable):
    """A simple columnar data structure storing 2-D points.

    x and y coordinates are stored in separate columns for cache-friendly
    access patterns.  Used here to demonstrate the mapping protocol and
    rich comparison protocol via pontoneer.
    """

    var pos_x: Coord1DColumn
    var pos_y: Coord1DColumn
    var _bounding_box_area: Float64

    def __init__(out self):
        self.pos_x = []
        self.pos_y = []
        self._bounding_box_area = 0

    def __init__(
        out self,
        var x: Coord1DColumn,
        var y: Coord1DColumn,
    ):
        self._bounding_box_area = _compute_bounding_box_area(x, y)
        self.pos_x = x^
        self.pos_y = y^

    # ------------------------------------------------------------------
    # Regular methods
    # ------------------------------------------------------------------

    @staticmethod
    def get_call_count(
        py_self: PythonObject, name: PythonObject
    ) raises -> PythonObject:
        """Return the number of times a named method was called (for testing).
        """
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var key = "{}{}".format(Int(self_ptr), String(py=name))
        return _get_global_call_count()[].get(key, 0)

    @staticmethod
    def with_columns(
        pos_x: PythonObject, pos_y: PythonObject
    ) raises -> PythonObject:
        var len_x = Int(pos_x.__len__())
        var len_y = Int(pos_y.__len__())
        if len_x != len_y:
            raise Error("The length of the two columns does not match.")
        var ptr_x = Coord1DColumn(capacity=len_x)
        var ptr_y = Coord1DColumn(capacity=len_y)
        for value in pos_x:
            ptr_x.append(Float64(py=value))
        for value in pos_y:
            ptr_y.append(Float64(py=value))
        return PythonObject(alloc=DataFrame(ptr_x^, ptr_y^))

    # ------------------------------------------------------------------
    # Mapping protocol
    # ------------------------------------------------------------------

    def py__len__(self) raises -> Int:
        return len(self.pos_x)

    def py__getitem__(self, index: PythonObject) raises -> PythonObject:
        var i = Int(py=index)
        var length = len(self.pos_x)
        if i < 0 or i >= length:
            raise Error("index out of range")
        return Python().tuple(self.pos_x[i], self.pos_y[i])

    def py__setitem__(
        mut self,
        index: PythonObject,
        value: Variant[PythonObject, Int],
    ) raises -> None:
        var i = Int(py=index)
        var length = len(self.pos_x)
        if i < 0 or i >= length:
            raise Error("index out of range")
        if value.isa[PythonObject]():
            # Assignment: value is a (x, y) tuple.
            self.pos_x[i] = Float64(py=value[PythonObject][0])
            self.pos_y[i] = Float64(py=value[PythonObject][1])
        else:
            # Deletion (value is null / Int(0)).
            _ = self.pos_x.pop(i)
            _ = self.pos_y.pop(i)

    # ------------------------------------------------------------------
    # Rich comparison protocol
    # ------------------------------------------------------------------

    def rich_compare(
        self,
        other: PythonObject,
        op: Int,
    ) raises -> Bool:
        """Compare DataFrames by bounding-box area.

        Only LT and EQ are implemented; all other operations raise
        NotImplementedError so Python falls back to the reflected call.
        """
        var invocation = "{}rich_compare[{}]".format(
            Int(UnsafePointer(to=self)), op
        )
        var call_count = _get_global_call_count()
        call_count[][invocation] = call_count[].get(invocation, 0) + 1
        var other_df = other.downcast_value_ptr[Self]()
        if op == RichCompareOps.Py_LT:
            return self._bounding_box_area < other_df[]._bounding_box_area
        if op == RichCompareOps.Py_EQ:
            return self._bounding_box_area == other_df[]._bounding_box_area
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # Number protocol — unary
    # ------------------------------------------------------------------

    @staticmethod
    def py__neg__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var result_x = Coord1DColumn(capacity=len(self_ptr[].pos_x))
        var result_y = Coord1DColumn(capacity=len(self_ptr[].pos_y))
        for v in self_ptr[].pos_x:
            result_x.append(-v)
        for v in self_ptr[].pos_y:
            result_y.append(-v)
        return PythonObject(alloc=DataFrame(result_x^, result_y^))

    @staticmethod
    def py__abs__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin]
    ) raises -> PythonObject:
        var result_x = Coord1DColumn(capacity=len(self_ptr[].pos_x))
        var result_y = Coord1DColumn(capacity=len(self_ptr[].pos_y))
        for v in self_ptr[].pos_x:
            result_x.append(abs(v))
        for v in self_ptr[].pos_y:
            result_y.append(abs(v))
        return PythonObject(alloc=DataFrame(result_x^, result_y^))

    # ------------------------------------------------------------------
    # Number protocol — bool
    # ------------------------------------------------------------------

    @staticmethod
    def py__bool__(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> Bool:
        return len(self_ptr[].pos_x) > 0

    # ------------------------------------------------------------------
    # Number protocol — binary
    # ------------------------------------------------------------------

    @staticmethod
    def py__add__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], other: PythonObject
    ) raises -> PythonObject:
        """Concatenate two DataFrames row-wise. Returns NotImplemented for non-DataFrames.
        """
        try:
            var other_ptr = other.downcast_value_ptr[Self]()
            var n = len(self_ptr[].pos_x) + len(other_ptr[].pos_x)
            var result_x = Coord1DColumn(capacity=n)
            var result_y = Coord1DColumn(capacity=n)
            for v in self_ptr[].pos_x:
                result_x.append(v)
            for v in self_ptr[].pos_y:
                result_y.append(v)
            for v in other_ptr[].pos_x:
                result_x.append(v)
            for v in other_ptr[].pos_y:
                result_y.append(v)
            return PythonObject(alloc=DataFrame(result_x^, result_y^))
        except:
            raise NotImplementedError()

    @staticmethod
    def py__mul__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], other: PythonObject
    ) raises -> PythonObject:
        """Scale all coordinates by a numeric scalar. Returns NotImplemented otherwise.
        """
        try:
            var scale = Float64(py=other)
            var result_x = Coord1DColumn(capacity=len(self_ptr[].pos_x))
            var result_y = Coord1DColumn(capacity=len(self_ptr[].pos_y))
            for v in self_ptr[].pos_x:
                result_x.append(v * scale)
            for v in self_ptr[].pos_y:
                result_y.append(v * scale)
            return PythonObject(alloc=DataFrame(result_x^, result_y^))
        except:
            raise NotImplementedError()

    # ------------------------------------------------------------------
    # Number protocol — ternary
    # ------------------------------------------------------------------

    @staticmethod
    def py__pow__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        exp: PythonObject,
        mod: PythonObject,
    ) raises -> PythonObject:
        """Raise all coordinates to a power. The `mod` argument is ignored."""
        var e = Float64(py=exp)
        var result_x = Coord1DColumn(capacity=len(self_ptr[].pos_x))
        var result_y = Coord1DColumn(capacity=len(self_ptr[].pos_y))
        for v in self_ptr[].pos_x:
            result_x.append(v**e)
        for v in self_ptr[].pos_y:
            result_y.append(v**e)
        return PythonObject(alloc=DataFrame(result_x^, result_y^))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DataFrame( length=", len(self.pos_x), ")")


@export
def PyInit_columnar_mojo() -> PythonObject:
    """Entry point: create the Python extension module."""
    try:
        var b = PythonModuleBuilder("columnar_mojo")

        ref tb = (
            b.add_type[DataFrame]("DataFrame")
            .def_init_defaultable[DataFrame]()
            .def_staticmethod[DataFrame.with_columns]("with_columns")
            .def_method[DataFrame.get_call_count]("get_call_count")
        )
        var tpb = TypeProtocolBuilder[DataFrame](tb)
        _ = tpb.def_richcompare[DataFrame.rich_compare]()
        var mpb = MappingProtocolBuilder[DataFrame](tb)
        _ = (
            mpb.def_len[DataFrame.py__len__]()
            .def_getitem[DataFrame.py__getitem__]()
            .def_setitem[DataFrame.py__setitem__]()
        )
        var npb = NumberProtocolBuilder[DataFrame](tb)
        _ = (
            npb.def_neg[DataFrame.py__neg__]()
            .def_abs[DataFrame.py__abs__]()
            .def_bool[DataFrame.py__bool__]()
            .def_add[DataFrame.py__add__]()
            .def_mul[DataFrame.py__mul__]()
            .def_pow[DataFrame.py__pow__]()
        )

        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
