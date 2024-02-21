# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import Tensor
from pathlib import Path

import _mlir

from .type import MOTensor, TypeTuple
import ._c


@value
struct Module(Stringable):
    var _module: _mlir.Module

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self):
        var ctx = _mlir.Context()
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        self._module = _mlir.Module(_mlir.Location.unknown(ctx))

    fn __str__(self) -> String:
        return str(self._module)

    # ===------------------------------------------------------------------=== #
    # High level utilities
    # ===------------------------------------------------------------------=== #

    fn verify(self) raises:
        if not self._module.as_op().verify():
            raise "Module did not verify"

    fn save_to_file(self, path: Path) raises:
        with open(path, "w") as file:
            self._module.as_op().write(file)

    # ===------------------------------------------------------------------=== #
    # Location factories
    # ===------------------------------------------------------------------=== #

    fn unknown_loc(self) -> _mlir.Location:
        return _mlir.Location.unknown(self._module.context())

    # ===------------------------------------------------------------------=== #
    # Attribute factories
    # ===------------------------------------------------------------------=== #

    fn tensor_attr[
        dtype: DType
    ](self, name: String, owned value: Tensor[dtype]) -> _mlir.NamedAttribute:
        let t = MOTensor(value.spec()).to_mlir(self)
        return _c.attr_new_tensor(
            self._module,
            name,
            value._steal_ptr().bitcast[DType.invalid](),
            t,
            is_owned=True,
        )

    fn tensor_resource_attr(
        self, name: String, file_name: String, type: MOTensor
    ) -> _mlir.NamedAttribute:
        return _c.attr_new_tensor_from_file(
            self._module, name, file_name, type.to_mlir(self)
        )

    fn vector_attr[
        dtype: DType
    ](
        self, name: String, values: DynamicVector[Scalar[dtype]]
    ) -> _mlir.NamedAttribute:
        return _c.attr_new_tensor(
            self._module,
            name,
            values,
            MOTensor(dtype, len(values)).to_mlir(self),
            is_owned=False,
        )

    fn scalar_attr[
        dtype: DType
    ](
        self, name: String, value: Scalar[dtype], rank: Int = 0
    ) raises -> _mlir.NamedAttribute:
        # Note: while this could generalize to something like splat, MO doesn't
        # really make use of those.
        var shape = DynamicVector[Int](capacity=rank)
        for i in range(rank):
            shape.append(1)
        return self.tensor_attr[dtype](name, Tensor(shape, value))

    fn string_attr(self, name: String, value: String) -> _mlir.NamedAttribute:
        let ctx = self._module.context()
        return _mlir.NamedAttribute(
            name=_mlir.Identifier(ctx, name),
            attr=_mlir.builtin_attributes.StringAttr(ctx, value),
        )

    # ===------------------------------------------------------------------=== #
    # Graph factories
    # ===------------------------------------------------------------------=== #

    fn graph(
        self, name: String, in_types: TypeTuple, out_types: TypeTuple
    ) -> Graph:
        let ctx = self._module.context()
        let loc = _mlir.Location.unknown(ctx)

        let function_type = _mlir.builtin_types.FunctionType(
            ctx, in_types.to_mlir(self), out_types.to_mlir(self)
        )
        let op = _c.graph_new(
            self._module, loc, name._strref_dangerous(), function_type
        )
        name._strref_keepalive()

        return Graph(op)

    fn graph(self, in_types: TypeTuple, out_types: TypeTuple) -> Graph:
        return self.graph("graph", in_types, out_types)
