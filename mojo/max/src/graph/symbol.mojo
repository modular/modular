# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .attr import AttrMap
from .graph import Graph
from .capi import SymbolPtr, TuplePtr
from .type import *
from .ops import *

from tensor import Tensor, TensorShape, TensorSpec

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
@register_passable  # TODO: Use with Tuple shouldn't require reg-only?
struct Symbol(CollectionElement, Stringable):
    var s: SymbolPtr

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn graph(self) -> Graph:
        return Graph(capi.symbol_get_graph(self.s))

    # ===------------------------------------------------------------------=== #
    # Type accessors
    # ===------------------------------------------------------------------=== #

    fn tensor_type(self) -> MOTensor:
        # TODO: Assert that this is an actual Tensor type, raise otherwise.
        let ranked = capi.tensor_type_is_ranked(self.s)
        let dims = capi.tensor_type_get_shape(self.s)
        let dtype = capi.tensor_type_get_dtype(self.s)
        return MOTensor(ElementType(dtype), dims, ranked)

    # ===------------------------------------------------------------------=== #
    # Stringable trait
    # ===------------------------------------------------------------------=== #

    fn __str__(self) -> String:
        return capi.symbol_to_string(self.s)

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
        let attrs = AttrMap(g.m.string_attr("label", label))
        _ = g.nvop("mo.debug.tensor.print", self, Arity(), attrs)

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
    var t: TuplePtr

    # ===------------------------------------------------------------------=== #
    # Basic constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *symbols: Symbol):
        self.__init__(symbols)

    fn __init__(inout self, symbols: VariadicList[Symbol]):
        self.t = capi.tuple_new()
        for symbol in symbols:
            capi.tuple_append_symbol(self.t, symbol.s)

    fn __init__(inout self, symbols: DynamicVector[Symbol]):
        self.t = capi.tuple_new()
        for i in range(len(symbols)):
            capi.tuple_append_symbol(self.t, symbols[i].s)

    # ===------------------------------------------------------------------=== #
    # Convenience tuple adapters
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, symbols: (Symbol, Symbol)):
        self.__init__(symbols.get[0, Symbol]().s, symbols.get[1, Symbol]().s)

    fn __init__(inout self, symbols: (Symbol, Symbol, Symbol)):
        self.__init__(
            symbols.get[0, Symbol]().s,
            symbols.get[1, Symbol]().s,
            symbols.get[2, Symbol]().s,
        )

    fn __init__(inout self, symbols: (Symbol, Symbol, Symbol, Symbol)):
        self.__init__(
            symbols.get[0, Symbol]().s,
            symbols.get[1, Symbol]().s,
            symbols.get[2, Symbol]().s,
            symbols.get[3, Symbol]().s,
        )

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        return capi.tuple_size(self.t)

    fn __getitem__(self, pos: UInt32) -> Symbol:
        return Symbol(capi.tuple_get_symbol(self.t, pos))

    # ===------------------------------------------------------------------=== #
    # Mutators
    # ===------------------------------------------------------------------=== #

    fn append(self, s: Symbol):
        capi.tuple_append_symbol(self.t, s.s)
