# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# expected-error @+1 {{@__name must have 1 argument}}
@__name
def name_no_args():
    ...

# expected-error @+1 {{@__name must have 1 argument}}
@__name("name1", "name2")
def name_two_args():
    ...

# expected-error @+1 {{function has conflicting linkage name from a previous @__name or @export decorator}}
@__name("first_name")
@export("different_name")
def name_export_conflict():
    ...

# expected-error @+1 {{function has conflicting linkage name from a previous @__name or @export decorator}}
@export("first_name")
@__name("different_name")
def export_name_conflict():
    ...


# expected-error @+1 {{function has conflicting linkage name from a previous @__name or @export decorator}}
@export("one_name", ABI="C")
@__name("different_name")
def c_export_name_conflict():
    ...

