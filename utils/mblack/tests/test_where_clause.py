# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import mblack
from tests.util import assert_mojo_format

# ====================== #
# Fns with where clauses
# ====================== #

def test_simple_fn_where_clause():
    source = "def where_simple[x: Bool]() where x: pass"
    expected = (
        "def where_simple[x: Bool]() where x:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_composite_fn_where_clause_with_return():
    source = "def where_composite[x: Bool]() -> Int where x and y + z: pass"
    expected = (
        "def where_composite[x: Bool]() -> Int where x and y + z:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_long_fn_where_clause():
    source = (
        "def where_long[x: Bool, y: Int, z: Int](\n"
        "    a: Int, b: Int, c: Int\n"
        ") where x and   y + z and a == b and c   == d and (e == f or  g):\n"
        "    pass\n"
    )
    expected = (
        "def where_long[\n"
        "    x: Bool, y: Int, z: Int\n"
        "](a: Int, b: Int, c: Int) where (\n"
        "    x and y + z and a == b and c == d and (e == f or g)\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_very_long_fn_where_clause_with_return():
    source = (
        "def where_very_long() -> (Int, Int, String) where (\n"
        "    x and y + z and a == b and c == d and (e == f or g) and xx == yy and zz == aa or (ds or dd)\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "def where_very_long() -> (\n"
        "    Int,\n"
        "    Int,\n"
        "    String,\n"
        ") where (\n"
        "    x\n"
        "    and y + z\n"
        "    and a == b\n"
        "    and c == d\n"
        "    and (e == f or g)\n"
        "    and xx == yy\n"
        "    and zz == aa\n"
        "    or (ds or dd)\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_fn_where_clauses():
    source = (
        "def where_multiple[x: Bool, y: Int]() "
        "where x and y + z and a == b "
        "where c == d and (e == f or g) and xx == yy "
        "where zz == aa or (ds or dd):\n"
        "    pass\n"
    )
    expected = (
        "def where_multiple[\n"
        "    x: Bool, y: Int\n"
        "]() where x and y + z and a == b where (\n"
        "    c == d and (e == f or g) and xx == yy\n"
        ") where zz == aa or (ds or dd):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ============================ #
# Structs with where clauses
# ============================ #


def test_simple_struct_where_clause():
    source = "struct WhereSimple[T: Bool] where T: pass"
    expected = (
        "struct WhereSimple[T: Bool] where T:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_struct_where_clause_no_params():
    """A bare `where` clause with no parameter list and no parent traits."""
    source = "struct NoParams where some_predicate(): pass"
    expected = (
        "struct NoParams where some_predicate():\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_long_struct_where_clause():
    source = (
        "struct WhereLong[T: Bool, U: Int, V: Int]() "
        "where T and U + V == W and a == b and c == d and (e == f or g): pass"
    )
    expected = (
        "struct WhereLong[T: Bool, U: Int, V: Int]() where (\n"
        "    T and U + V == W and a == b and c == d and (e == f or g)\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_very_long_struct_where_clause():
    source = (
        "struct WhereVeryLong[T: Bool] where (\n"
        "    x and y + z and a == b and c == d and (e == f or g)"
        " and xx == yy and zz == aa or (ds or dd)\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct WhereVeryLong[T: Bool] where (\n"
        "    x\n"
        "    and y + z\n"
        "    and a == b\n"
        "    and c == d\n"
        "    and (e == f or g)\n"
        "    and xx == yy\n"
        "    and zz == aa\n"
        "    or (ds or dd)\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_struct_where_clauses_no_parent():
    source = (
        "struct WhereMultiple[T: Movable] "
        "where conforms_to(T, Copyable) where conforms_to(T, Intable): pass"
    )
    expected = (
        "struct WhereMultiple[T: Movable] where conforms_to(\n"
        "    T, Copyable\n"
        ") where conforms_to(T, Intable):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_trailing_where_clause_on_struct_decl():
    source = (
        "struct Wrapper[\n"
        "T: Movable, U: Movable\n"
        "](\n"
        "Movable\n"
        ") where conforms_to(T, Copyable) and conforms_to(U, Intable): pass"
    )
    expected = (
        "struct Wrapper[T: Movable, U: Movable](Movable) where conforms_to(\n"
        "    T, Copyable\n"
        ") and conforms_to(U, Intable):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ============================== #
# Param decls with where clauses
# ============================== #

def test_where_clause_in_params():
    source = (
        "def where_in_params["
        "x: Bool where x is False, "
        "y: Int where y > 0 and x is True]():\n"
        "    pass\n"
    )
    expected = (
        "def where_in_params[\n"
        "    x: Bool where x is False, y: Int where y > 0 and x is True\n"
        "]():\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_long_where_clause_in_params():
    source = (
        "def where_in_params["
        "x: Bool where x and y + z and a == b and c == d and (e == f or g) and xx == yy and zz == aa or (ds or dd), "
        "y: Int where y > 0 and x is True]():\n"
        "    pass\n"
    )
    expected = (
        "def where_in_params[\n"
        "    x: Bool where (\n"
        "        x\n"
        "        and y + z\n"
        "        and a == b\n"
        "        and c == d\n"
        "        and (e == f or g)\n"
        "        and xx == yy\n"
        "        and zz == aa\n"
        "        or (ds or dd)\n"
        "    ),\n"
        "    y: Int where y > 0 and x is True,\n"
        "]():\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_where_clauses_in_params():
    source = (
        "def where_in_params["
        "x: Bool where x and y + z and a == b where c == d and (e == f or g) and xx == yy where zz == aa or (ds or dd), "
        "y: Int where y > 0 where x is True]():\n"
        "    pass\n"
    )
    expected = (
        "def where_in_params[\n"
        "    x: Bool where x and y + z and a == b where (\n"
        "        c == d and (e == f or g) and xx == yy\n"
        "    ) where zz == aa or (ds or dd),\n"
        "    y: Int where y > 0 where x is True,\n"
        "]():\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_composite_param_where_clause_with_default():
    source = "def where_composite[x: Bool where x and y + z = True]() -> Int: pass"
    expected = (
        "def where_composite[x: Bool where x and y + z = True]() -> Int:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
