# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-error @+1 {{@__name must have at least 1 argument}}
@__name
def name_no_args():
    ...


# expected-error @+1 {{@__name must have at most 1 name argument}}
@__name("name1", "name2")
def name_two_args():
    ...


# expected-error @+1 {{function has conflicting linkage name from a previous @__name or @export decorator}}
@__name("first_name")
@export("different_name")
def name_export_conflict() abi("C"):
    ...


# expected-error @+1 {{function has conflicting linkage name from a previous @__name or @export decorator}}
@export("first_name")
@__name("different_name")
def export_name_conflict() abi("C"):
    ...


# expected-warning @+2 {{ABI="C" is deprecated; use abi("C") instead}}
# expected-error @+1 {{function has conflicting linkage name from a previous @__name or @export decorator}}
@export("one_name", ABI="C")
@__name("different_name")
def c_export_name_conflict():
    ...
