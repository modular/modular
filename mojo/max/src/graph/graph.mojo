# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""

from collections import Optional
from memory.unsafe import Pointer
from tensor import TensorShape, TensorSpec

from .attr import AttrMap
from .module import Module
from .symbol import Symbol, SymbolTuple
from .type import dyn, MOType, MOTensor, TypeTuple


@value
struct Graph:
    """The core unit of computation in MAX Engine.

    `Graph`s are callable routines in MAX Engine, similar to functions in
    Mojo. Like functions, graphs have a name and signature. Unlike functions,
    which follow an imperative programming model, `Graph`s follow a dataflow
    programming model, using lazily-executed, parallel operations instead of
    sequential instructions. `Graph`s aren't called directly from Mojo, but are
    instead compiled and executed by MAX Engine, for example using the MAX
    Engine API.
    """

    # TODO: Add an exmple, after we cleaned up the TypeTuple thing.
    # TODO: Refer to the concepts doc for the meaning of Symbol, etc.
    # TODO: Link to max engine page, wikipedia, etc.

    var g: mlir.Operation
    """A handle to the `Graph`'s internal implementation."""

    fn __init__(inout self, g: mlir.Operation):
        self.g = g

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn module(self) -> Module:
        try:
            return Module(mlir.Module.from_op(self.g.parent()))
        except:
            trap("Invalid state: Graph has no parent module")
            return __get_address_as_lvalue(Pointer[Module]().address)

    fn __getitem__(self, n: Int) -> Symbol:
        """Returns the `n`th argument of this `Graph`.

        This allows ops inside the `Graph` to take this argument as input.

        Args:
            n: The argument's position.

        Returns:
            A `Symbol` representing the argumen't symbolic value, as seen from
            within the `Graph`'s body.
        """
        # TODO: Add an exmple, after we cleaned up the Arity thing.
        return Symbol(self._body().argument(n))

    fn _body(self) -> mlir.Block:
        try:
            return self.g.region(0).first_block()
        except:
            trap("Invalid state: Graph has no associated regions")
            return Optional[mlir.Block](None).value()

    # ===------------------------------------------------------------------=== #
    # nvop - the most generic op builder
    # ===------------------------------------------------------------------=== #

    fn nvop(self, name: StringRef, inputs: SymbolTuple) -> SymbolTuple:
        return self.nvop(name, inputs, TypeTuple(), AttrMap())

    fn nvop(
        self, name: StringRef, inputs: SymbolTuple, out_types: TypeTuple
    ) -> SymbolTuple:
        return self.nvop(name, inputs, out_types, AttrMap())

    fn nvop(
        self, name: StringRef, out_types: TypeTuple, attrs: AttrMap
    ) -> SymbolTuple:
        return self.nvop(name, SymbolTuple(), out_types, attrs)

    fn nvop(
        self,
        name: StringRef,
        inputs: SymbolTuple,
        out_types: TypeTuple,
        attrs: AttrMap,
    ) -> SymbolTuple:
        let ctx = self.g.context()

        let op = mlir.Operation(
            name=name,
            location=mlir.Location.unknown(ctx),
            operands=inputs.as_values(),
            results=out_types.to_mlir(self.module()),
            attributes=attrs.attrs,
        )
        self._body().append(op)
        var tup = SymbolTuple()
        for i in range(op.num_results()):
            tup.append(op.result(i))
        return tup

    # ===------------------------------------------------------------------=== #
    # op - shorthands for single-result ops
    # ===------------------------------------------------------------------=== #

    fn op(
        self, name: StringRef, inputs: SymbolTuple, out_type: AnyMOType
    ) -> Symbol:
        return self.nvop(name, inputs, out_type, AttrMap())[0]

    fn op(self, name: StringRef, out_type: AnyMOType, attrs: AttrMap) -> Symbol:
        return self.nvop(name, out_type, attrs)[0]

    fn op(
        self,
        name: StringRef,
        inputs: SymbolTuple,
        out_type: AnyMOType,
        attrs: AttrMap,
    ) -> Symbol:
        return self.nvop(name, inputs, out_type, attrs)[0]

    # ===------------------------------------------------------------------=== #
    # Factories for various nullary ops
    # ===------------------------------------------------------------------=== #

    fn constant[
        dtype: DType
    ](self, owned value: Tensor[dtype]) raises -> Symbol:
        # Note: Unlike TensorSpec, Tensor *is* the canonical "tensor value" for
        # MO attributes. So we don't need extra auxiliaty structures for it.
        return self.op(
            "mo.constant",
            MOTensor(value.spec()),
            AttrMap(self.module().tensor_attr("value", value)),
        )

    fn vector[
        dtype: DType
    ](self, values: DynamicVector[Scalar[dtype]]) raises -> Symbol:
        return self.op(
            "mo.constant",
            MOTensor(dtype, len(values)),
            AttrMap(self.module().vector_attr[dtype]("value", values)),
        )

    fn scalar[
        dtype: DType
    ](self, value: Scalar[dtype], rank: Int = 0) raises -> Symbol:
        # Note: while this could generalize to something like splat, the
        # canonical way of achieving that is using a broadcast instead.
        var shape = DynamicVector[Int](rank)
        for i in range(rank):
            shape.append(1)
        return self.constant[dtype](Tensor(shape, value))

    fn range[
        dtype: DType
    ](
        self, start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]
    ) raises -> Symbol:
        return self.op(
            "mo.range",
            (
                self.scalar[dtype](start),
                self.scalar[dtype](stop),
                self.scalar[dtype](step),
            ),
            MOTensor(dtype, len(range(start, stop, step))),
        )

    fn output(self) raises:
        _ = self.nvop("mo.output", SymbolTuple())

    fn output(self, outs: SymbolTuple) raises:
        _ = self.nvop("mo.output", outs)
