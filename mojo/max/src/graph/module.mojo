# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines the Module Graph container."""

from tensor import Tensor
from pathlib import Path

import _mlir

from .type import MOTensor, TypeTuple
import ._c


@value
struct Module(Stringable):
    var _module: _mlir.Module
    """A Module is a container that holds a Graph."""

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self):
        """Constructs an empty Module instance."""
        var ctx = _mlir.Context()
        ctx.load_modular_dialects()
        ctx.load_all_available_dialects()
        self._module = _mlir.Module(_mlir.Location.unknown(ctx))

    fn __str__(self) -> String:
        """Constructs a human-readable string representation of the module.

        The string is in MLIR text format, and will include representations
        of any graphs and ops inside it.

        Returns:
            A human-readable string representation of the module.
        """
        return str(self._module)

    # ===------------------------------------------------------------------=== #
    # High level utilities
    # ===------------------------------------------------------------------=== #

    fn verify(self) raises:
        """Verifies the module, and the validity of the graph structure.

        The module is valid if every op inside of it is valid.

        A graph is valid if
        1. it has an `output` op whose types match the `out_types` of the graph
        2. every op in the graph is valid
            a. it has a valid op name
            b. it has the right number and type of operands and attributes
                for that operation (see the [`mo`](/engine/reference/mlir/mo)
                reference for op documentation).
        3. there are no cycles in the graph
        4. every symbol in the graph is defined before its first use

        If a graph is constructed forwards with only op construction and no
        op mutations, 3) and 4) will always be true.

        Raises:
            If the module is invalid. In this case it will also print a diagonstic
            to help debug why the graph is invalid.
        """
        if not self._module.as_op().verify():
            raise "Module did not verify"

    fn save_to_file(self, path: Path) raises:
        """Save the module to a file.

        > ⚠️⚠️⚠️ **THIS IS NOT A DURABLE REPRESENTATION!!!** Saved modules may
        > not correctly load or execute in future versions of the API or engine.

        Args:
            path: The path to save the module to.

        Raises:
            If writing to the file fails.
        """
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
        var t = MOTensor(value.spec()).to_mlir(self)
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
        var ctx = self._module.context()
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
        """Constructs a new Graph object in the Module.

        The constructed Graph will not be valid unless it has no outputs;
        a graph with outputs will need a `graph.output` call to tell it
        what to return. The graph's validity can be checked by calling
        `graph.verify()`.

        Args
            name: A name for the graph.
            in_types: The input types of the graph's computation.
            out_types: The output types of the graph's computation.

        Returns:
            A new `Graph` instance inside the module.
        """
        var ctx = self._module.context()
        var loc = _mlir.Location.unknown(ctx)

        var function_type = _mlir.builtin_types.FunctionType(
            ctx, in_types.to_mlir(self), out_types.to_mlir(self)
        )
        var op = _c.graph_new(
            self._module, loc, name._strref_dangerous(), function_type
        )
        name._strref_keepalive()

        return Graph(op)

    fn graph(self, in_types: TypeTuple, out_types: TypeTuple) -> Graph:
        return self.graph("graph", in_types, out_types)
