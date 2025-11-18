# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait SomeTrait:
    pass


struct StructWithField:
    var field: __mlir_type.index


# Issue #6879: Qualified lookup is looking up names wrong
fn unqualified_name_lookup(a: StructWithField):
    # expected-error @+1 {{StructWithField' value has no attribute 'badPropertyError'}}
    a.badPropertyError

    # expected-error @+1 {{StructWithField' value has no attribute 'badPropertyError'}}
    StructWithField.badPropertyError

    # expected-error @+1 {{'SomeTrait' value has no attribute 'value'}}
    SomeTrait.value

    # expected-error @+1 {{cannot access instance field 'field' without an instance of 'StructWithField'}}
    StructWithField.field


struct DirectInstanceReference:
    comptime my_alias: Int = 8
    var value: Int

    fn fxn(self):
        # expected-error @+1 {{cannot access instance field 'value' directly; did you mean 'self.'?}}
        var xx = value
        # expected-error @+1 {{cannot access comptime 'my_alias' directly; did you mean 'Self.'?}}
        _ = my_alias

    @staticmethod
    fn stat():
        # expected-error @+1 {{cannot access method 'fxn' directly; did you mean 'Self.'?}}
        _ = fxn

    fn direct_ref(self):
        # expected-error @+1 {{cannot access method 'fxn' directly; did you mean 'self.'?}}
        fxn(self)
        # expected-error @+1 {{cannot access method 'stat' directly; did you mean 'Self.'?}}
        stat()


fn field_indexes(a: DirectInstanceReference):
    # expected-error @+1 {{'DirectInstanceReference' value has no attribute 'badField'}}
    a.badField = 42


trait DirectTraitMemberReference:
    comptime my_alias: Int

    fn fxn(self):
        # expected-error @+1 {{cannot access comptime 'my_alias' directly; did you mean 'Self.'?}}
        _ = my_alias

    @staticmethod
    fn stat():
        # expected-error @+1 {{cannot access method 'fxn' directly; did you mean 'Self.'?}}
        _ = fxn

    fn direct_ref(self):
        # expected-error @+1 {{cannot access method 'fxn' directly; did you mean 'Self.'?}}
        fxn(self)
        # expected-error @+1 {{cannot access method 'stat' directly; did you mean 'Self.'?}}
        stat()
