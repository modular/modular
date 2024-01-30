# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import Tensor

import mlir

from .type import MOTensor, TypeTuple


@value
struct Module:
    var m: mlir.Module

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self):
        var ctx = mlir.Context()
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        self.m = mlir.Module(mlir.Location.unknown(ctx))

    fn __str__(self) -> String:
        return str(self.m)

    # ===------------------------------------------------------------------=== #
    # High level utilities
    # ===------------------------------------------------------------------=== #

    fn verify(self) raises:
        if not self.m.as_op().verify():
            raise "Module did not verify"

    fn save_to_file(self, path: Path) raises:
        with open(path, "w") as file:
            self.m.as_op().write(file)

    # ===------------------------------------------------------------------=== #
    # Location factories
    # ===------------------------------------------------------------------=== #

    fn unknown_loc(self) -> mlir.Location:
        return mlir.Location.unknown(self.m.context())

    # ===------------------------------------------------------------------=== #
    # Attribute factories
    # ===------------------------------------------------------------------=== #

    fn tensor_attr[
        dtype: DType
    ](self, name: StringRef, owned value: Tensor[dtype]) -> mlir.NamedAttribute:
        let t = MOTensor(value.spec()).to_mlir(self)
        return capi.attr_new_tensor(
            self.m,
            name,
            value._steal_ptr().bitcast[DType.invalid](),
            t,
            is_owned=True,
        )

    fn tensor_resource_attr(
        self, name: StringRef, file_name: StringRef, type: MOTensor
    ) -> mlir.NamedAttribute:
        return capi.attr_new_tensor_from_file(
            self.m, name, file_name, type.to_mlir(self)
        )

    fn vector_attr[
        dtype: DType
    ](
        self, name: StringRef, values: DynamicVector[Scalar[dtype]]
    ) -> mlir.NamedAttribute:
        return capi.attr_new_tensor(
            self.m,
            name,
            values,
            MOTensor(dtype, len(values)).to_mlir(self),
            is_owned=False,
        )

    fn scalar_attr[
        dtype: DType
    ](
        self, name: StringRef, value: Scalar[dtype], rank: Int = 0
    ) raises -> mlir.NamedAttribute:
        # Note: while this could generalize to something like splat, MO doesn't
        # really make use of those.
        var shape = DynamicVector[Int](rank)
        for i in range(rank):
            shape.append(1)
        return self.tensor_attr[dtype](name, Tensor(shape, value))

    fn string_attr(
        self, name: StringRef, value: StringRef
    ) -> mlir.NamedAttribute:
        return capi.attr_new_string(self.m, name, value)

    # ===------------------------------------------------------------------=== #
    # Graph factories
    # ===------------------------------------------------------------------=== #

    fn graph(
        self, name: StringRef, in_types: TypeTuple, out_types: TypeTuple
    ) -> Graph:
        return Graph(self, name, in_types, out_types)
