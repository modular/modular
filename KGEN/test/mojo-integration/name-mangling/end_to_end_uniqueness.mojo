# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Regression test: a host program with a parameterized function and a custom
# linkage name. The linkage name is not a product of the function's parameters
# so will clash on repeated instantiations. Test that with 'mangle=True', we
# are able to handle this with uniqueness suffixes.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build --emit asm %s -o %t.s
# RUN: FileCheck %s --input-file=%t.s


@no_inline
@__name("0this_name_is_long_and_will_clash_so_mangle_it", mangle=True)
def make_me_unique[n: Int]():
    print("hello", n)


# The two instantiations of the function 'make_me_unique' are separate concrete
# functions whose linkage names would clash if it weren't for mangle=True.
# Check that the output assembly contains both functions with unique suffixes.
# Check also that the names are not sanitized or shortened, as this is not a
# GPU target.

# CHECK-DAG: {{^_?}}0this_name_is_long_and_will_clash_so_mangle_it_{{[0-9a-f]+}}:
# CHECK-DAG: {{^_?}}0this_name_is_long_and_will_clash_so_mangle_it_{{[0-9a-f]+}}:


def main() raises:
    make_me_unique[0]()
    make_me_unique[1]()
