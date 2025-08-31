# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
@register_passable("trivial")
struct RP:
    pass


@fieldwise_init
struct NonRP:
    pass


trait Foo:
    fn rp(self) -> RP:
        return RP()

    fn non_rp(self) -> NonRP:
        return NonRP()

    @no_inline
    fn no_inline(self, x: RP) -> RP:
        return RP()

    @always_inline
    @staticmethod
    fn always_inline(x: RP) -> RP:
        return RP()


# CHECK-LABEL: lit.struct.decl @Bar
struct Bar(Foo):
    # CHECK: lit.fn @"rp(default_trait_methods::Bar)Foo"{{.*}}([[SELF:%[^:]+]]: {{.*}}) -> !RP
    # CHECK: lit.call @default_trait_methods::@Foo::@"rp($0)"{{.*}}([[SELF]])

    # CHECK: lit.fn @"non_rp(default_trait_methods::Bar)Foo"{{.*}}([[SELF:%[^:]+]]: {{.*}}, {{.*}}, [[RESULT:%[^:]+]]: {{.*}}) -> !kgen.none
    # CHECK: lit.call @default_trait_methods::@Foo::@"non_rp($0)"{{.*}}([[SELF]], [[RESULT]])

    # Make sure we preserve inline annotations on the wrapper methods
    # CHECK: lit.fn @"no_inline
    # CHECK-SAME: no_inline
    # CHECK: lit.fn @"always_inline
    # CHECK-SAME: always_inline

    # CHECK: kgen.conformance{{.*}}:Foo
    # CHECK-DAG: kgen.witness "rp"
    # CHECK-DAG: kgen.witness "non_rp"
    # CHECK-DAG: kgen.witness "no_inline"
    # CHECK-DAG: kgen.witness "always_inline"
    pass


@fieldwise_init
@register_passable("trivial")
struct Zork(Copyable, Movable):
    pass


trait AA1:
    alias X: Copyable

    fn zork(self, x: Self.X) -> Self.X:
        return x


# Check that we handle traits with associated aliases properly
# CHECK-LABEL: lit.struct.decl @TAA
struct TAA(AA1):
    alias X = Zork

    # CHECK: lit.fn @"zork(default_trait_methods::TAA,default_trait_methods::Zork)AA1"{{.*}}([[SELF:%[^:]+]]: {{.*}}, [[X:%[^:]+]]: {{.*}}, {{.*}}, [[RESULT:%[^:]+]]: {{.*}}) -> !kgen.none
    # CHECK: lit.call @default_trait_methods::@AA1::@"zork($0,$0.X)"{{.*}}([[SELF]], [[X]], [[RESULT]])

    # CHECK: kgen.conformance{{.*}}:AA1
    # CHECK-DAG: kgen.witness "zork"


# Test parameterized types in default trait methods
@fieldwise_init
@register_passable("trivial")
struct ParamRPType[x: Int, y: Int]:
    var value: Int


trait Barable:
    fn bar(self):
        ...


trait ParamInputTrait:
    @staticmethod
    fn process_parameterized[T: Barable](item: T) -> Int:
        item.bar()
        return 100

    fn return_parameterized[x: Int, y: Int](self) -> ParamRPType[x, y]:
        return ParamRPType[x, y](x * y)


# CHECK-LABEL: lit.struct.decl @ParamTestStruct
struct ParamTestStruct(ParamInputTrait):
    # Check that we generate proper wrapper for parameterized input method
    # CHECK: lit.fn @"process_parameterized
    # CHECK: lit.call @default_trait_methods::@ParamInputTrait::@"process_parameterized

    # Check that we generate proper wrapper for parameterized return type
    # CHECK: lit.fn @"return_parameterized
    # CHECK: lit.call @default_trait_methods::@ParamInputTrait::@"return_parameterized

    # CHECK: kgen.conformance{{.*}}:ParamInputTrait
    # CHECK-DAG: kgen.witness "process_parameterized"
    # CHECK-DAG: kgen.witness "return_parameterized"
    pass


@fieldwise_init
struct BarableStruct(Barable):
    fn bar(self):
        pass
