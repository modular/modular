# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

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

    # ===------------------------------------------------------------------=== #
    # Type accessors
    # ===------------------------------------------------------------------=== #

    fn tensor_type(self) raises -> MOTensor:
        # TODO: Assert that this is an actual Tensor type, raise otherwise.
        return MOTensor.from_mlir(self.s.type())

    # ===------------------------------------------------------------------=== #
    # Stringable trait
    # ===------------------------------------------------------------------=== #

    fn __str__(self) -> String:
        return str(self.s)

    # ===------------------------------------------------------------------=== #
    # Overloaded operators
    # ===------------------------------------------------------------------=== #

    fn __getitem__(self, i: Int, axis: Int = 0) raises -> Symbol:
        let g = self.graph()
        return self[g.scalar(Int64(i)), axis=axis]

    fn __getitem__(self, i: Symbol, axis: Int = 0) raises -> Symbol:
        return ops.slice(self, i, axis=axis)

    # ===------------------------------------------------------------------=== #
    # ... to tidy up ...
    # ===------------------------------------------------------------------=== #

    fn __getitem__(self, span: slice) raises -> Symbol:
        if not span._has_end():
            raise "slice expects stop to be specified"

        let spec = self.tensor_type()
        if len(spec.dims) != 1:
            raise "__getitem__ with slice object on `Symbol` only supports 1D tensors"

        # TODO: Use ops.get, once it supports passing a static type.

        var g = self.graph()
        return g.op(
            "mo.slice",
            (
                self,
                g.scalar(Int64(span.start), rank=1),
                g.scalar(Int64(span.end), rank=1),
                g.scalar(Int64(span.step), rank=1),
            ),
            MOTensor(DType.int64, len(span)),
        )

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

    # TODO: issue
    # fn __init__(inout self, symbols: VariadicListMem[Symbol]):
    #     self.symbols = DynamicVector[Symbol]()
    #     for symbol in symbols:
    #         self.symbols.append(symbol[])

    # ===------------------------------------------------------------------=== #
    # Convenience tuple adapters
    # ===------------------------------------------------------------------=== #

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

    fn get[idx: Int, T: AnyType](self) -> Symbol:
        return self.symbols[idx]

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
