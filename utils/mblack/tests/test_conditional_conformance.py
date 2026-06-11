# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for formatting conditional trait conformances in struct definitions.

Conditional conformance allows a struct to conform to a trait only when certain
conditions are met. The syntax is:
    TraitName where condition

For example:
    struct Wrapper[T: Movable](
        Copyable where conforms_to(T, Copyable),
        Movable,
    ):
"""

import pytest

from tests.util import assert_mojo_format

# ============================================ #
# Basic conditional conformance in struct defs
# ============================================ #


def test_simple_conditional_conformance():
    """Test a simple conditional conformance with conforms_to."""
    source = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_conditional_conformances():
    """Test struct with multiple conditional conformances."""
    source = (
        "struct Multi[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Intable where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Multi[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Intable where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_conditional_conformance_with_and():
    """Test conditional conformance with 'and' operator."""
    source = (
        "struct Wrapper[T: Movable](\n"
        "    TraitA where conforms_to(T, Copyable) and conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Wrapper[T: Movable](\n"
        "    Movable,\n"
        "    TraitA where conforms_to(T, Copyable) and conforms_to(T, Intable),\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_conditional_conformance_with_or():
    """Test conditional conformance with 'or' operator."""
    source = (
        "struct Wrapper[T: Movable](\n"
        "    Base where conforms_to(T, Copyable) or conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Wrapper[T: Movable](\n"
        "    Base where conforms_to(T, Copyable) or conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_conditional_conformance_single_line():
    """Test compact conditional conformance that fits on one line."""
    source = "struct W[T: M](A where cond(T, B), M): pass"
    expected = (
        "struct W[T: M](A where cond(T, B), M):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ================================================ #
# Long conditional conformances that need wrapping
# ================================================ #


def test_long_conditional_conformance():
    """Test conditional conformance that needs line wrapping."""
    source = (
        "struct VeryLongWrapper[T: ImplicitlyDeletable & Movable]("
        "Copyable where conforms_to(T, Copyable) and conforms_to(T, Intable) and conforms_to(T, Writable), "
        "Movable): pass"
    )
    expected = (
        "struct VeryLongWrapper[T: ImplicitlyDeletable & Movable](\n"
        "    Copyable where (\n"
        "        conforms_to(T, Copyable)\n"
        "        and conforms_to(T, Intable)\n"
        "        and conforms_to(T, Writable)\n"
        "    ),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_long_conditional_conformances():
    """Test multiple conditional conformances that need line wrapping."""
    source = (
        "struct Diamond[T: Movable](\n"
        "Base where conforms_to(T, Copyable) or conforms_to(T, Intable),\n"
        "DerivedA where conforms_to(T, Copyable),\n"
        "DerivedB where conforms_to(T, Intable),\n"
        "Movable,\n"
        "): pass"
    )
    expected = (
        "struct Diamond[T: Movable](\n"
        "    Base where conforms_to(T, Copyable) or conforms_to(T, Intable),\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    DerivedB where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ================================================ #
# Mixed conditional and unconditional conformances
# ================================================ #


def test_mixed_conditional_unconditional():
    """Test struct with both conditional and unconditional conformances."""
    source = (
        "struct Mixed[T: Movable](\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    DerivedB,\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Mixed[T: Movable](\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    DerivedB,\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_explicit_ancestor_with_conditional():
    """Test explicit ancestor conformance with conditional."""
    source = (
        "struct Explicit[T: Movable](\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    Base,\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    # Formatter sorts all conformances alphabetically, including conditional ones
    expected = (
        "struct Explicit[T: Movable](\n"
        "    Base,\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ================================== #
# Conditional conformance with Self.T
# ================================== #


def test_conditional_conformance_with_self_type():
    """Test conditional conformance using Self.T syntax."""
    source = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(Self.T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(Self.T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ======================================== #
# Complex struct definitions with generics
# ======================================== #


def test_complex_generic_with_conditional():
    """Test complex generic struct with conditional conformance."""
    source = (
        "struct Container[T: ImplicitlyDeletable & Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    ImplicitlyDeletable,\n"
        "    Movable,\n"
        "):\n"
        "    var value: Self.T\n"
    )
    expected = (
        "struct Container[T: ImplicitlyDeletable & Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    ImplicitlyDeletable,\n"
        "    Movable,\n"
        "):\n"
        "    var value: Self.T\n"
    )
    assert_mojo_format(source, expected)


# ====================================== #
# Whitespace normalization in conditions
# ====================================== #


def test_whitespace_normalization_in_condition():
    """Test that extra whitespace in conditions is normalized."""
    source = (
        "struct W[T: M](\n"
        "    A where   conforms_to(T,   B),\n"
        "    M,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct W[T: M](\n"
        "    A where conforms_to(T, B),\n"
        "    M,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_condition_with_complex_expression():
    """Test conditional conformance with complex boolean expression."""
    source = (
        "struct W[T: M](\n"
        "    A where (conforms_to(T, B) and conforms_to(T, C)) or conforms_to(T, D),\n"
        "    M,\n"
        "):\n"
        "    pass\n"
    )
    # Expression fits on one line, so no wrapping is applied
    expected = (
        "struct W[T: M](\n"
        "    A where (conforms_to(T, B) and conforms_to(T, C)) or conforms_to(T, D),\n"
        "    M,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ============================================================== #
# Sorting correctness: where clauses must stay with their traits
# ============================================================== #


def test_sorting_keeps_where_clause_with_trait():
    """Sorting must move the where clause together with its trait name.

    Previously, sorting only operated on bare NAME leaves, so conditional
    conformances were either skipped or could get detached from their
    where clause.
    """
    source = (
        "struct Foo[T: Movable](\n"
        "    Zebra where conforms_to(T, Copyable),\n"
        "    Alpha,\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    # Alpha < Movable < Zebra — and the where clause stays on Zebra.
    expected = (
        "struct Foo[T: Movable](\n"
        "    Alpha,\n"
        "    Movable,\n"
        "    Zebra where conforms_to(T, Copyable),\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_sorting_multiple_conditional_conformances():
    """Multiple conditional conformances are sorted with their where clauses."""
    source = (
        "struct Foo[T: Movable](\n"
        "    Zebra where conforms_to(T, Copyable),\n"
        "    Alpha where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Foo[T: Movable](\n"
        "    Alpha where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "    Zebra where conforms_to(T, Copyable),\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
