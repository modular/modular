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


struct StructWithParam[a: Int]:
    pass


@fieldwise_init
struct UnqualifiedStructParameterAccess[
    my_param: Int,  # expected-note {{parameter 'my_param' declared here}}
    other_param: Int,
    struct_param: StructWithParam[my_param],  # this should be okay
]:
    # expected-warning @+1 {{unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead}}
    comptime my_alias = my_param

    fn bar(self) -> Int:
        fn nested_fn():
            # expected-warning @+1 {{unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead}}
            comptime my_different_alias = my_param

        fn shadowing_nested_fn[my_param: Int]():
            # There should be no warning here because the comptime is shadowed.
            comptime my_different_alias = my_param

        nested_fn()
        shadowing_nested_fn[4]()

        # expected-warning @+1 {{unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead}}
        comptime my_other_alias = my_param
        # expected-warning @+1 {{unqualified access to struct parameter 'my_param'; use 'Self.my_param' instead}}
        return my_param
