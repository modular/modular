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


# ---------------------------------------------------------------------------
# mangle= argument errors
# ---------------------------------------------------------------------------


# expected-error @+1 {{'mangle' argument to @__name must be True or False}}
@__name("foo", mangle=42)
def mangle_int_value():
    ...


# expected-error @+1 {{'mangle' argument to @__name must be True or False}}
@__name("foo", mangle="yes")
def mangle_string_value():
    ...


# expected-error @+1 {{@__name requires a name argument when 'mangle' is specified}}
@__name(mangle=True)
def mangle_true_no_name():
    ...


# expected-error @+1 {{@__name requires a name argument when 'mangle' is specified}}
@__name(mangle=False)
def mangle_false_no_name():
    ...


# expected-error @+1 {{@__name requires at most 2 arguments}}
@__name("foo", "extra", mangle=True)
def mangle_too_many_args():
    ...


# @export does not accept a 'mangle' keyword — it falls through to the
# catch-all error that requires a plain string for the symbol name.
# expected-error @+1 {{@export requires a string specifying the name of the exported symbol}}
@export("foo", mangle=True)
def export_with_mangle():
    ...
