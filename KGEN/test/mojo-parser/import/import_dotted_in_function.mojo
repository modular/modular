# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

# Test that imports inside a function body work.


def test_package_import_in_function():
    import test_package

    test_package.method_defined_in_init()  # ok


# // -----


def test_dotted_module_import_in_function():
    import test_package.module

    # FIXME: They don't work: this shouldn't be an error - MOCO-3509.
    test_package.module.function()  # expected-error {{use of unknown declaration 'test_package'}}


# // -----


def test_dotted_package_import_in_function():
    import test_package.test_nested_package

    # FIXME: They don't work: this shouldn't be an error - MOCO-3509.
    test_package.test_nested_package.nested_function()  # expected-error {{use of unknown declaration 'test_package'}}


# // -----


def test_from_package_import_module_in_function():
    from test_package import module

    module.function()  # ok


# // -----


def test_from_package_import_package_in_function():
    from test_package import test_nested_package

    test_nested_package.nested_function()  # ok


# // -----


def test_from_nested_package_import_symbol_in_function():
    from test_package.test_nested_package.module import nested_function

    nested_function()  # ok


# // -----


def test_from_nested_package_import_symbol_in_function():
    from test_package.test_nested_package.deep_package.leaf import deep_function

    deep_function()  # ok
