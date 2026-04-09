# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file -I=%S -verify-diagnostics %s

# Test that importing a package allows access to arbitrarily deep sub-packages.

import test_package


def main():
    # 2 levels deep
    _ = test_package.test_nested_package

    # 3 levels deep
    test_package.test_nested_package.module.nested_function()

    # 4 levels deep
    test_package.test_nested_package.deep_package.leaf.deep_function()


# // -----

# Same but importing one extra level

import test_package.test_nested_package


def main():
    # 1 level deep
    _ = test_package

    # 2 levels deep
    _ = test_package.test_nested_package

    # 3 levels deep
    test_package.test_nested_package.module.nested_function()

    # 4 levels deep
    test_package.test_nested_package.deep_package.leaf.deep_function()


# // -----

# Same but importing two extra levels

import test_package.test_nested_package.module


def main():
    # expected-error @+1 {{use of unknown declaration 'deep_package'}}
    test_package.test_nested_package.deep_package.leaf.deep_function()


# // -----

# Same but importing one extra level as an alias

import test_package.test_nested_package as test_nested


def main():
    # 2 levels deep
    _ = test_nested

    # 3 levels deep
    test_nested.module.nested_function()

    # 4 levels deep
    test_nested.deep_package.leaf.deep_function()


# // -----

# Aliased import should NOT leak the parent package.

import test_package.test_nested_package as test_nested


def main():
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    _ = test_package.test_nested_package

    # 3 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    test_package.test_nested_package.module.nested_function()

    # 4 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    test_package.test_nested_package.deep_package.leaf.deep_function()


# // -----

# 'from' import of a subpackage: the imported name works, but the parent
# package is not accessible.

from test_package import test_nested_package


def main():
    # The imported name is in scope and works.
    _ = test_nested_package

    # 3 levels deep through the imported name
    test_nested_package.module.nested_function()

    # 4 levels deep through the imported name
    test_nested_package.deep_package.leaf.deep_function()


# // -----

# 'from' import should NOT leak the parent package.

from test_package import test_nested_package


def main():
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    _ = test_package

    # 2 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    _ = test_package.test_nested_package

    # 3 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    test_package.test_nested_package.module.nested_function()

    # 4 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    test_package.test_nested_package.deep_package.leaf.deep_function()
