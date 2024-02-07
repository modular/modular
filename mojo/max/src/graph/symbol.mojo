# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional
from memory.unsafe import _LITRef
from tensor import Tensor, TensorShape, TensorSpec

import mlir

from .attr import AttrMap
from .graph import Graph
from .type import *
from .ops import *

# TODO: The overloads are incomplete, and make unverified assumptions about
# dtype, etc.


# TODO: Cull the sea of operator overloads. Should be simplifiable with traits.
#
# Some development notes - tried implementing a Reifiable trait, with
# Symbol implementing it, along with a few adapters, e.g. ReifiableInt, so that
# we can then implement operators like e.g.
#
#     fn __add__[T: Reifiable](self, rhs: T) raises -> Symbol:
#         var g = self.graph()
#         return add(self, rhs.to_symbol(g, self.tensor_type()))
#
# Alas that falls on its face for some reason. Bug?


@value
struct Symbol(CollectionElement, Stringable):
    var s: mlir.Value

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn graph(self) -> Graph:
        let parent = self.s.parent()
        let block: mlir.Block
        if parent.isa[mlir.Block]():
            block = parent.get[mlir.Block]()
        else:
            let op = parent.get[mlir.Operation]()
            block = op.block()

        let graph_op = block.parent()
        return Graph(graph_op)

    fn type(self) raises -> AnyMOType:
        return AnyMOType.from_mlir(self.s.type())

    fn tensor_type(self) raises -> MOTensor:
        return self.type().tensor()

    fn __str__(self) -> String:
        return str(self.s)

    # ===------------------------------------------------------------------=== #
    # Overloaded operators
    # ===------------------------------------------------------------------=== #

    fn __getitem__(self, i: Symbol, axis: Int = 0) raises -> Symbol:
        return ops.slice(self, i, axis=axis)

    fn __getitem__(self, i: Int, axis: Int = 0) raises -> Symbol:
        let g = self.graph()
        return ops.slice(self, g.scalar(Int64(i)), axis=axis)

    fn __getitem__(self, *s: SymbolicSlice) raises -> Symbol:
        var slices = DynamicVector[SymbolicSlice]()
        for sval in s:
            slices.append(sval[])
        return ops.slice(self, slices)

    fn __getitem__(self, s: Slice) raises -> Symbol:
        return ops.slice(self, s)

    # ===------------------------------------------------------------------=== #
    # ... to tidy up ...
    # ===------------------------------------------------------------------=== #

    fn list_get(self, i: Int) raises -> Symbol:
        return self.list_get(self.graph().constant[DType.int64](i))

    fn list_get(self, i: Symbol) raises -> Symbol:
        let g = self.graph()
        let result_type = self.type().list().eltype
        return g.op("mo.list.get", (self, i), result_type)

    fn list_insert(self, i: Int, v: Symbol) raises -> Symbol:
        return self.list_insert(self.graph().constant[DType.int64](i), v)

    fn list_insert(self, i: Symbol, v: Symbol) raises -> Symbol:
        let g = self.graph()
        let result_type = self.type().list().eltype
        return g.op("mo.list.insert", (self, v, i), result_type)

    fn print(self, label: StringRef = "debug_tensor") raises:
        var g = self.graph()
        let attrs = AttrMap(g.module().string_attr("label", label))
        _ = g.nvop("mo.debug.tensor.print", self, TypeTuple(), attrs)

    fn transpose(self) raises -> Symbol:
        return transpose(self, -1, -2)

    fn transpose(self, dim1: Int, dim2: Int) raises -> Symbol:
        return transpose(self, dim1, dim2)

    fn __neg__(self) raises -> Symbol:
        var g = self.graph()
        return g.op("mo.negative", self, self.tensor_type())

    fn __matmul__(self, rhs: Symbol) raises -> Symbol:
        return matmul(self, rhs)

    fn __add__(self, rhs: Symbol) raises -> Symbol:
        return add(self, rhs)

    fn __add__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self + g.constant(rhs)

    fn __add__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return self + g.scalar(rhs)

    fn __add__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self + g.scalar(Int64(rhs))

    fn __add__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self + g.scalar(Float32(rhs))

    fn __sub__(self, rhs: Symbol) raises -> Symbol:
        return sub(self, rhs)

    fn __sub__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self - g.constant(rhs)

    fn __sub__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return self - g.scalar(rhs)

    fn __sub__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self - g.scalar(Int64(rhs))

    fn __sub__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self - g.scalar(Float32(rhs))

    fn __mul__(self, rhs: Symbol) raises -> Symbol:
        return mul(self, rhs)

    fn __mul__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self * g.constant(rhs)

    fn __mul__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return self * g.scalar(rhs)

    fn __mul__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self * g.scalar(Int64(rhs))

    fn __mul__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self * g.scalar(Float32(rhs))

    fn __truediv__(self, rhs: Symbol) raises -> Symbol:
        return div(self, rhs)

    fn __truediv__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self / g.constant(rhs)

    fn __truediv__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return self / g.scalar(rhs)

    fn __truediv__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self / g.scalar(Int64(rhs))

    fn __truediv__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self / g.scalar(Float32(rhs))

    fn __pow__(self, rhs: Symbol) raises -> Symbol:
        return pow(self, rhs)

    fn __pow__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return self ** g.constant(rhs)

    fn __pow__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return self ** g.scalar(rhs)

    fn __pow__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return self ** g.scalar(Int64(rhs))

    fn __pow__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return self ** g.scalar(Float32(rhs))

    fn __radd__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) + self

    fn __radd__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.scalar(rhs) + self

    fn __radd__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Int64(rhs)) + self

    fn __radd__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Float32(rhs)) + self

    fn __rsub__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) - self

    fn __rsub__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.scalar(rhs) - self

    fn __rsub__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Int64(rhs)) - self

    fn __rsub__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Float32(rhs)) - self

    fn __rmul__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) * self

    fn __rmul__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.scalar(rhs) * self

    fn __rmul__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Int64(rhs)) * self

    fn __rmul__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Float32(rhs)) * self

    fn __rtruediv__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) / self

    fn __rtruediv__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.scalar(rhs) / self

    fn __rtruediv__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Int64(rhs)) / self

    fn __rtruediv__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Float32(rhs)) / self

    fn __rpow__[dtype: DType](self, rhs: Tensor[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.constant(rhs) ** self

    fn __rpow__[dtype: DType](self, rhs: Scalar[dtype]) raises -> Symbol:
        var g = self.graph()
        return g.scalar(rhs) ** self

    fn __rpow__(self, rhs: Int) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Int64(rhs)) ** self

    fn __rpow__(self, rhs: FloatLiteral) raises -> Symbol:
        var g = self.graph()
        return g.scalar(Float32(rhs)) ** self

    # ===------------------------------------------------------------------=== #
    # Graph manipulation
    # ===------------------------------------------------------------------=== #

    fn replace_all_uses_with(self, other: Symbol):
        self.s.replace_all_uses_with(other.s)

    fn replace_all_uses_with(
        self, transform: fn (Symbol) raises -> Symbol
    ) raises:
        let dummy = self.graph().constant[DType.float32](0)
        self.replace_all_uses_with(dummy)
        dummy.replace_all_uses_with(transform(self))


@value
struct SymbolicSlice(CollectionElement):
    var start: Optional[Symbol]
    var stop: Optional[Symbol]
    var step: Optional[Symbol]

    def __init__(inout self, g: Graph, s: Slice):
        self.start = Optional[Symbol]()
        self.stop = Optional[Symbol]()
        self.step = Optional[Symbol]()
        if s.start:
            self.start = g.scalar(Int64(s.start))
        if s.end:
            self.stop = g.scalar(Int64(s.end))
        if s.step:
            self.step = g.scalar(Int64(s.step))


@value
struct SymbolTuple(Sized):
    var symbols: DynamicVector[Symbol]

    # ===------------------------------------------------------------------=== #
    # Basic constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *symbols: Symbol):
        self.symbols = DynamicVector[Symbol]()
        for symbol in symbols:
            self.symbols.append(symbol[])

    # ===------------------------------------------------------------------=== #
    # Convenience tuple adapters
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, owned symbols: ()):
        self.__init__()

    fn __init__(inout self, owned symbols: (Symbol, Symbol)):
        let ptr = Pointer.address_of(symbols).bitcast[Int8]()
        self.__init__(
            __get_address_as_lvalue(ptr.bitcast[Symbol]().address),
            __get_address_as_lvalue(
                ptr.offset(symbols._offset[1]()).bitcast[Symbol]().address
            ),
        )

    fn __init__(inout self, owned symbols: (Symbol, Symbol, Symbol)):
        let ptr = Pointer.address_of(symbols).bitcast[Int8]()
        self.__init__(
            __get_address_as_lvalue(ptr.bitcast[Symbol]().address),
            __get_address_as_lvalue(
                ptr.offset(symbols._offset[1]()).bitcast[Symbol]().address
            ),
            __get_address_as_lvalue(
                ptr.offset(symbols._offset[2]()).bitcast[Symbol]().address
            ),
        )

    fn __init__(inout self, owned symbols: (Symbol, Symbol, Symbol, Symbol)):
        let ptr = Pointer.address_of(symbols).bitcast[Int8]()
        self.__init__(
            __get_address_as_lvalue(ptr.bitcast[Symbol]().address),
            __get_address_as_lvalue(
                ptr.offset(symbols._offset[1]()).bitcast[Symbol]().address
            ),
            __get_address_as_lvalue(
                ptr.offset(symbols._offset[2]()).bitcast[Symbol]().address
            ),
            __get_address_as_lvalue(
                ptr.offset(symbols._offset[3]()).bitcast[Symbol]().address
            ),
        )

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        return len(self.symbols)

    fn __getitem__(self, idx: Int) -> Symbol:
        return self.symbols[idx]

    fn as_values(self) -> DynamicVector[mlir.Value]:
        var values = DynamicVector[mlir.Value]()
        for i in range(len(self.symbols)):
            values.append(self.symbols[i].s)
        return values

    # ===------------------------------------------------------------------=== #
    # Mutators
    # ===------------------------------------------------------------------=== #

    fn append(inout self, symbol: Symbol):
        self.symbols.append(symbol)
