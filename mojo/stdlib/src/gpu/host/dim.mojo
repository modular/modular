# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the dim type."""

from utils.index import Index, IndexList


@value
@register_passable("trivial")
struct Dim(Stringable, Writable):
    """Represents a dimension with up to three components (x, y, z).

    This struct is commonly used to represent grid and block dimensions
    for kernel launches.
    """

    var _value: IndexList[3]
    """Internal storage for the three dimension components (x, y, z).

    This field stores the values for all three dimensions using an IndexList
    with a fixed size of 3. The dimensions are accessed in order: x, y, z.
    """

    @implicit
    fn __init__(out self, value: IndexList[3]):
        """Initializes Dim with an IndexList[3].

        Args:
            value: The IndexList[3] representing the dimension.
        """
        self._value = value

    @implicit
    fn __init__[I: Indexer](out self, x: I):
        """Initializes Dim with a single indexable value for x.

        y and z dimensions are set to 1.

        Parameters:
            I: The type of the indexable value.

        Args:
            x: The value for the x dimension.
        """
        self._value = IndexList[3](index(x), 1, 1)

    fn __init__[I0: Indexer, I1: Indexer](out self, x: I0, y: I1):
        """Initializes Dim with indexable values for x and y.

        z dimension is set to 1.

        Parameters:
            I0: The type of the first indexable value.
            I1: The type of the second indexable value.

        Args:
            x: The value for the x dimension.
            y: The value for the y dimension.
        """
        self._value = IndexList[3](index(x), index(y), 1)

    fn __init__[
        I0: Indexer, I1: Indexer, I2: Indexer
    ](out self, x: I0, y: I1, z: I2):
        """Initializes Dim with indexable values for x, y, and z.

        Parameters:
            I0: The type of the first indexable value.
            I1: The type of the second indexable value.
            I2: The type of the third indexable value.

        Args:
            x: The value for the x dimension.
            y: The value for the y dimension.
            z: The value for the z dimension.
        """
        self._value = IndexList[3](index(x), index(y), index(z))

    @implicit
    fn __init__[I: Indexer](out self, dims: (I,)):
        """Initializes Dim with a tuple containing a single indexable value.

        y and z dimensions are set to 1.

        Parameters:
            I: The type of the indexable value in the tuple.

        Args:
            dims: A tuple with one element for x dimension.
        """
        self._value = IndexList[3](index(dims[0]), 1, 1)

    @implicit
    fn __init__[I0: Indexer, I1: Indexer](out self, dims: (I0, I1)):
        """Initializes Dim with a tuple of two indexable values.

        The z dimension is set to 1.

        Parameters:
            I0: The type of the first indexable value in the tuple.
            I1: The type of the second indexable value in the tuple.

        Args:
            dims: A tuple with two elements: x and y dimensions.
        """
        self._value = IndexList[3](index(dims[0]), index(dims[1]), 1)

    @implicit
    fn __init__[
        I0: Indexer, I1: Indexer, I2: Indexer
    ](out self, dims: (I0, I1, I2)):
        """Initializes Dim with a tuple of three indexable values.

        Parameters:
            I0: The type of the first indexable value in the tuple.
            I1: The type of the second indexable value in the tuple.
            I2: The type of the third indexable value in the tuple.

        Args:
            dims: Tuple with three elements: x, y, and z dimensions.
        """
        self._value = IndexList[3](
            index(dims[0]), index(dims[1]), index(dims[2])
        )

    fn __getitem__(self, idx: Int) -> Int:
        """Gets the dimension value at the specified index.

        Args:
            idx: The index (0 for x, 1 for y, 2 for z).

        Returns:
            The value of the dimension at the given index.
        """
        return self._value[idx]

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of the Dim.

        Returns:
            String representation of this Dim object.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """Returns a string representation of the Dim.

        Returns:
            String representation of this Dim object.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """Writes a formatted string representation of the Dim.

        Parameters:
            W: The type of writer to use for output. Must implement the Writer trait.

        Args:
            writer: The Writer to write to.
        """
        writer.write("(x=", self.x(), ", ")
        if self.y() != 1 or self.z() != 1:
            writer.write("y=", self.y())
            if self.z() != 1:
                writer.write(", z=", self.z())
        writer.write(")")

    fn z(self) -> Int:
        """Returns the z dimension.

        Returns:
            The value of the z dimension.
        """
        return self[2]

    fn y(self) -> Int:
        """Returns the y dimension.

        Returns:
            The value of the y dimension.
        """
        return self[1]

    fn x(self) -> Int:
        """Returns the x dimension.

        Returns:
            The value of the x dimension.
        """
        return self[0]
