# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Module primitives.

A `Module` is a very basic container that holds `Graph`s for the purpose of
compiling or exporting them as a whole.

Note: `Module`s are not to be confused with layers or modules found in other
high-level APIs (like `torch.nn.Module`, or `tf.Module`). The MAX Graph API
is a low level library. Rather, a `Module` is closer to ONNX Models.
"""

from tensor import Tensor
from pathlib import Path

import _mlir

from .type import MOTensor, TypeTuple
import ._c


@value
struct Module(Stringable):
    """A Module is a container that holds `Graph`s."""

    var _module: _mlir.Module

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
        """Verifies the `Module` and its contents.

        Examples of cases when a `Graph` may not be valid (the list is not
        exhaustive):
        1. it has an `output` op whose types don't match its `out_types`
        2. it has an op with an invalid name, number, type of operands,
            output types, etc.
        3. it contains cycles

        Raises:
            If the `Module` did not pass verification. In this case it will also
            print a diagnostic message indicating the error.
        """
        if not self._module.as_op().verify():
            raise "module did not verify"

    fn save_to_file(self, path: Path) raises:
        """Saves this `Module` to a file.

        Important: The file contents uses internal MLIR Bytecode format that is
        not guaranteed to be cross-version compatible and may change without
        notice.

        Args:
            path: The path to save the `Module` to.
        """
        with open(path, "w") as file:
            self._module.as_op().write(file)

    # ===------------------------------------------------------------------=== #
    # Attribute factories
    # ===------------------------------------------------------------------=== #

    fn tensor_attr[
        dtype: DType
    ](self, name: String, owned value: Tensor[dtype]) -> _mlir.NamedAttribute:
        """Creates a new `Tensor`-valued `Attribute`.

        The value of this attribute will have the type `MOTensor` with the same
        shape and dtype as `value`.
        This method takes ownership of `value` and is suitable for use with
        very large `Tensor` values (such as model weights).

        Parameters:
            dtype: The attribute tensor's element type.

        Args:
            name: The `Attribute` name.
            value: The `Attribute` value.

        Returns:
            An internal representation of an `Attribute`.
        """
        var t = MOTensor(value.spec()).to_mlir(self._module.context())
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
        """Creates a new `Tensor` `Attribute` from an external file.

        The value of this constant will have the type `type`.
        The file must contain the `Tensor`s raw data, as returned by
        `Tensor.data`. No endianness transformation is performed.

        Args:
            name: The `Attribute` name.
            file_name: The file name to load from.
            type: The `Tensor` type (element type, shape).

        Returns:
            An internal representation of an `Attribute`.
        """
        return _c.attr_new_tensor_from_file(
            self._module, name, file_name, type.to_mlir(self._module.context())
        )

    fn vector_attr[
        dtype: DType
    ](self, name: String, values: List[Scalar[dtype]]) -> _mlir.NamedAttribute:
        """Creates a new `Tensor`-valued `Attribute`.

        The value of this attribute will have the type `MOTensor` with 1D shape,
        consistent with the size of `values`.

        Parameters:
            dtype: The attribute tensor's element type.

        Args:
            name: The `Attribute` name.
            values: A vector representing the attribute's value.

        Returns:
            An internal representation of an `Attribute`.
        """
        return _c.attr_new_tensor(
            self._module,
            name,
            values,
            MOTensor(dtype, len(values)).to_mlir(self._module.context()),
            is_owned=False,
        )

    fn scalar_attr[
        dtype: DType
    ](
        self, name: String, value: Scalar[dtype], rank: Int = 0
    ) raises -> _mlir.NamedAttribute:
        """Creates a new `Tensor`-valued `Attribute`.

        The `Tensor` is considered to contain a single element, and its shape
        be of the specified rank (for example, `rank=0` denotes a scalar).

        Parameters:
            dtype: The attribute tensor's element type.

        Args:
            name: The `Attribute` name.
            value: The `Attribute` value.
            rank: The attribute tensor's rank.

        Returns:
            An internal representation of an `Attribute`.
        """
        # Note: while this could generalize to something like splat, MO doesn't
        # really make use of those.
        var shape = List[Int](capacity=rank)
        for i in range(rank):
            shape.append(1)
        return self.tensor_attr[dtype](name, Tensor(shape, value))

    fn string_attr(self, name: String, value: String) -> _mlir.NamedAttribute:
        """Creates a new `String`-valued `Attribute`.

        Args:
            name: The `Attribute` name.
            value: The `Attribute` value.

        Returns:
            An internal representation of an `Attribute`.
        """
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
        """Adds an empty `Graph` to this `Module`.

        Args:
            name: The `Graph`'s name.
            in_types: The `Graphs`'s input types.
            out_types: The `Graphs`'s return types.

        Returns:
            An empty `Graph` ready to be filled with ops.
        """
        var ctx = self._module.context()
        var loc = _mlir.Location.unknown(ctx)

        var function_type = _mlir.builtin_types.FunctionType(
            ctx, in_types.to_mlir(ctx), out_types.to_mlir(ctx)
        )
        var op = _c.graph_new(
            self._module,
            loc,
            name,
            _mlir.builtin_types.FunctionType(
                ctx, in_types.to_mlir(ctx), out_types.to_mlir(ctx)
            ),
        )

        return Graph(op)

    fn graph(self, in_types: TypeTuple, out_types: TypeTuple) -> Graph:
        """Adds an empty `Graph` with a default name to this `Module`.

        Args:
            in_types: The `Graphs`'s input types.
            out_types: The `Graphs`'s return types.

        Returns:
            An empty `Graph` ready to be filled with ops.
        """
        return self.graph("graph", in_types, out_types)
