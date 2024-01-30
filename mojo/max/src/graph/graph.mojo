# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Core graph primitives."""

from .attr import AttrMap
from .capi import GraphPtr, SymbolPtr, TuplePtr
from .module import Module
from .symbol import Symbol, SymbolTuple
from .type import dyn, MOType, MOTensor, TypeTuple

from memory.unsafe import Pointer
from tensor import TensorShape, TensorSpec


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

    var g: GraphPtr
    """A handle to the `Graph`'s internal implementation."""

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(
        inout self,
        m: Module,
        name: StringRef,
        in_types: TypeTuple,
        out_types: TypeTuple,
    ):
        let unknown = m.unknown_loc()
        self.g = capi.graph_new(
            m.m, unknown, name, in_types.to_mlir(m), out_types.to_mlir(m)
        )

    fn __init__(
        inout self, name: StringRef, in_types: TypeTuple, out_types: TypeTuple
    ):
        self.__init__(Module(), name, in_types, out_types)

    fn module(self) -> Module:
        """Returns the `Module` that owns this `Graph`."""
        return Module(capi.graph_get_module(self.g))

    fn __getitem__(self, n: UInt32) -> Symbol:
        """Returns the `n`th argument of this `Graph`.

        This allows ops inside the `Graph` to take this argument as input.

        Args:
            n: The argument's position.

        Returns:
            A `Symbol` representing the argumen't symbolic value, as seen from
            within the `Graph`'s body.
        """
        # TODO: Add an exmple, after we cleaned up the Arity thing.
        return Symbol(capi.graph_get_arg(self.g, n))

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
        let outputs = capi.graph_new_op(
            self.g,
            self.module().unknown_loc(),
            name,
            inputs.t,
            out_types.to_mlir(self.module()),
            attrs.m,
        )
        return SymbolTuple(outputs)

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
