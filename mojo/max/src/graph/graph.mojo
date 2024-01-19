# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .attr import AttrMap
from .capi import AttrPtr, GraphPtr, SymbolPtr, TuplePtr, TypePtr
from .module import Module
from .symbol import Symbol, Tup
from .type import dyn, MOType, MOTensor, Arity

from memory.unsafe import Pointer
from tensor import TensorShape, TensorSpec


# TODO: Drop inout self - it's not reflective of any Mojo-side mutation.


@value
struct Graph:
    var g: GraphPtr
    var m: Module

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, g: GraphPtr):
        self.g = g
        self.m = Module(capi.graph_get_module(g))

    fn __getitem__(self, pos: UInt32) -> Symbol:
        return Symbol(capi.graph_get_arg(self.g, pos))

    # ===------------------------------------------------------------------=== #
    # nvop - the most generic op builder
    # ===------------------------------------------------------------------=== #

    fn nvop(inout self, name: StringRef, inputs: Tup) -> Tup:
        return self.nvop(name, inputs, Arity(), AttrMap())

    fn nvop(inout self, name: StringRef, inputs: Tup, out_types: Arity) -> Tup:
        return self.nvop(name, inputs, out_types, AttrMap())

    fn nvop(
        inout self, name: StringRef, out_types: Arity, attrs: AttrMap
    ) -> Tup:
        return self.nvop(name, Tup(), out_types, attrs)

    fn nvop(
        inout self,
        name: StringRef,
        inputs: Tup,
        out_types: Arity,
        attrs: AttrMap,
    ) -> Tup:
        let loc = self.m.unknown_loc()
        let outputs = capi.graph_new_op(
            self.g, loc, name, inputs.t, out_types.to_mlir(self.m), attrs.m
        )
        return Tup(outputs)

    # ===------------------------------------------------------------------=== #
    # op - shorthands for single-result ops
    # ===------------------------------------------------------------------=== #

    # TODO: s/T/Arity/ when MOTensor can be autoboxed inside an Arity.

    fn op[
        T: MOType
    ](inout self, name: StringRef, inputs: Tup, out_type: T) -> Symbol:
        return self.nvop(
            name, inputs, Arity(out_type.to_mlir(self.m)), AttrMap()
        )[0]

    fn op[
        T: MOType
    ](inout self, name: StringRef, out_type: T, attrs: AttrMap) -> Symbol:
        return self.nvop(name, Arity(out_type.to_mlir(self.m)), attrs)[0]

    fn op[
        T: MOType
    ](
        inout self,
        name: StringRef,
        inputs: Tup,
        out_type: T,
        attrs: AttrMap,
    ) -> Symbol:
        return self.nvop(name, inputs, Arity(out_type.to_mlir(self.m)), attrs)[
            0
        ]

    # ===------------------------------------------------------------------=== #
    # Factories for various nullary ops
    # ===------------------------------------------------------------------=== #

    fn constant[
        dtype: DType
    ](inout self, owned value: Tensor[dtype]) raises -> Symbol:
        # Note: Unlike TensorSpec, Tensor *is* the canonical "tensor value" for
        # MO attributes. So we don't need extra auxiliaty structures for it.
        return self.op(
            "mo.constant",
            MOTensor(value.spec()),
            AttrMap(self.m.tensor_attr("value", value)),
        )

    fn vector[
        dtype: DType
    ](inout self, values: DynamicVector[Scalar[dtype]]) raises -> Symbol:
        return self.op(
            "mo.constant",
            MOTensor(dtype, len(values)),
            AttrMap(self.m.vector_attr[dtype]("value", values)),
        )

    fn scalar[
        dtype: DType
    ](inout self, value: Scalar[dtype], rank: Int = 0) raises -> Symbol:
        # Note: while this could generalize to something like splat, the
        # canonical way of achieving that is using a broadcast instead.
        var shape = DynamicVector[Int](rank)
        for i in range(rank):
            shape.append(1)
        return self.constant[dtype](Tensor(shape, value))

    fn range[
        dtype: DType
    ](
        inout self,
        start: Scalar[dtype],
        stop: Scalar[dtype],
        step: Scalar[dtype],
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

    fn output(inout self) raises:
        _ = self.nvop("mo.output", Tup())

    fn output(inout self, outs: Tup) raises:
        _ = self.nvop("mo.output", outs)
