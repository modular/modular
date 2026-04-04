# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Argument Packs.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


trait SomeTrait:
    pass


struct SomeMem(ImplicitlyCopyable, SomeTrait):
    pass


struct SomeReg(RegisterPassable, SomeTrait):
    def __init__(out self):
        pass


# ===----------------------------------------------------------------------=== #
# Trait packs
# ===----------------------------------------------------------------------=== #


# This function takes a pack of owned values by Trait.
def takeOwnedAnyTypePack[*Ts: AnyType](var *rest: *Ts):
    pass


# Test mangling:
# CHECK-LABEL: lit.fn @"takeOwnedAnyTypePack[*::AnyType

# Test implicit lifetimes / param list.
# CHECK-SAME: <Ts: param_list<!AnyType> pos_vararg

# Check the argument pack.
# CHECK-SAME: (%rest: !lit.ref<{{.*}}#VariadicPack <:!Bool {:i1 1}, :origin<1> *"rest.origin._mlir_origin``"
# CHECK-SAME: :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> Ts>>, mut *"rest{{.*}}"> owned_in_mem|pack_vararg)


# Check the argument pack.
# CHECK-LABEL: lit.fn @"takeOwnedSomeTraitPack
# CHECK-SAME: (%rest: !lit.ref<{{.*}}#VariadicPack <:!Bool {:i1 1}, {{.*}}origin<1> *"rest
# CHECK-SAME: :!lit.anytrait<!AnyType> !SomeTrait, :param_list<!SomeTrait> Ts>>, mut *"rest`2"> owned_in_mem|pack_vararg)
def takeOwnedSomeTraitPack[*Ts: SomeTrait](var *rest: *Ts):
    pass


# CHECK-LABEL: lit.fn @"test_owned_trait
def test_owned_trait():
    # CHECK-NEXT: %value1 = lit.var.decl
    var value1: SomeMem
    # CHECK-NEXT: %value2 = lit.var.decl
    var value2: SomeMem

    # Argument expressions emitted first
    # CHECK-NEXT: lit.ownership.use %value1
    # CHECK-NEXT: [[ANONSLOT:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.memcpy %value2, [[ANONSLOT]]

    # Coerce to common origin
    # CHECK-NEXT: [[V1C:%.*]] = kgen.rebind %value1 : !lit.ref<!SomeMem, mut *"value1`"> to !lit.ref<!SomeMem, mut {*"anonymous*`2", *"value1`"}>
    # CHECK-NEXT: [[V2C:%.*]] = kgen.rebind [[ANONSLOT]] : !lit.ref<!SomeMem, mut *"{{.*}}> to !lit.ref<!SomeMem, mut {*"anonymous*`2", *"value1`"}>

    # Form pack and call
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V1C]], [[V2C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKVAL:%.*]] = lit.call tail @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACK]])
    # CHECK-NEXT: [[PACKTMP:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[PACKVAL]], [[PACKTMP]]

    # CHECK-NEXT: lit.call {{.*}}takeOwnedAnyTypePack{{.*}}([[PACKTMP]])
    takeOwnedAnyTypePack(value1^, value2)

    # Test register types.
    # CHECK-NEXT: %value3 = lit.var.decl
    var value3: SomeReg

    # Argument expressions emitted first
    # CHECK-NEXT: lit.ownership.use %value3
    # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}SomeReg::@"__init__{{.*}}()
    # CHECK-NEXT: [[ANONSLOT:%.*]] = lit.var.decl "anonymous
    # CHECK-NEXT: lit.ref.store [[RES]], [[ANONSLOT]]

    # Coerce to common origin
    # CHECK-NEXT: [[V3C:%.*]] = kgen.rebind %value3
    # CHECK-NEXT: [[V4C:%.*]] = kgen.rebind [[ANONSLOT]]
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V3C]], [[V4C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKVAL:%.*]] = lit.call tail @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACK]])
    # CHECK-NEXT: [[PACKTMP:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[PACKVAL]], [[PACKTMP]]
    # CHECK-NEXT: lit.call {{.*}}takeOwnedAnyTypePack{{.*}}([[PACKTMP]])
    takeOwnedAnyTypePack(value3^, SomeReg())


# Check the argument pack.
# CHECK-LABEL: lit.fn @"takeInoutSomeTraitPack
# CHECK-SAME: (%rest: !lit.ref<{{.*}}#VariadicPack <:!Bool {:i1 1}, {{.*}}origin<1> *"rest
# CHECK-SAME: :!lit.anytrait<!AnyType> !SomeTrait, :param_list<!SomeTrait> Ts>>, imm *"rest`2"> mut|pack_vararg)
def takeInoutSomeTraitPack[*Ts: SomeTrait](mut*rest: *Ts):
    pass


# CHECK-LABEL: lit.fn @"test_inout
def test_inout():
    # CHECK-NEXT: %value1 = lit.var.decl
    var value1: SomeMem
    # CHECK-NEXT: %value2 = lit.var.decl
    var value2: SomeMem

    # Coerce to common origin
    # CHECK-NEXT: [[V1C:%.*]] = kgen.rebind %value1 : !lit.ref<!SomeMem, mut *"value1`"> to !lit.ref<!SomeMem, mut {*"value1`", *"value2`1"}>
    # CHECK-NEXT: [[V2C:%.*]] = kgen.rebind %value2 : !lit.ref<!SomeMem, mut *"value2`1"> to !lit.ref<!SomeMem, mut {*"value1`", *"value2`1"}>

    # Form pack and call
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create([[V1C]], [[V2C]])

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKVAL:%.*]] = lit.call tail @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACK]])
    # CHECK-NEXT: [[VARIADICPACK:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[PACKVAL]], [[VARIADICPACK]]

    # CHECK-NEXT: [[PACKIMM:%.*]] = lit.ref.immut [[VARIADICPACK]]
    # CHECK-NEXT: lit.call {{.*}}takeInoutSomeTraitPack{{.*}}([[PACKIMM]])
    takeInoutSomeTraitPack(value1, value2)

    # Test register types.
    # CHECK-NEXT: %value3 = lit.var.decl
    var value3: SomeReg

    # Coerce to common origin
    # CHECK-NEXT: [[PACK:%.*]] = lit.ref.pack.create(%value3)

    # Create the VariadicPack
    # CHECK-NEXT: [[PACKVAL:%.*]] = lit.call tail @{{.*}}@VariadicPack::@"__init__{{.*}}([[PACK]])
    # CHECK-NEXT: [[VARIADICPACK:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[PACKVAL]], [[VARIADICPACK]]
    # CHECK-NEXT: [[PACKIMM:%.*]] = lit.ref.immut [[VARIADICPACK]]

    # CHECK-NEXT: lit.call {{.*}}takeInoutSomeTraitPack{{.*}}([[PACKIMM]])
    takeInoutSomeTraitPack(value3)


struct not_nested_struct[*Ts: AnyType]:
    @implicit
    def __init__(out self, mut*args: * Self.Ts):
        pass


# CHECK-LABEL: lit.fn @"test_empty_pack
def test_empty_pack():
    # Make sure we pass an immortal origin for the pack.
    # CHECK: lit.call {{.*}}VariadicPack::@"__init__{{.*}}origin<1> {}
    var s1 = not_nested_struct()


# ===----------------------------------------------------------------------=== #
# Other tests
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @MyTuple
# CHECK-SAME: <Ts: param_list<!AnyType> pos_vararg>
struct MyTuple[*Ts: AnyType]:
    @implicit
    def __init__(out self, *args: * Self.Ts):
        pass


# CHECK-LABEL: lit.fn @"pack
# CHECK-SAME: Ts: param_list<!AnyType> pos_vararg
# CHECK-SAME: (%args: !lit.ref<{{.*}}#VariadicPack <:!Bool {:i1 0}, :origin<0> *"args.origin{{.*}}, :!Bool {:i1 0}, :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> Ts>>, imm *"args`2"> read_mem|pack_vararg)
def pack[*Ts: AnyType](*args: *Ts):
    pass


# CHECK-LABEL: lit.fn @"packBorrowed[
# CHECK-SAME: Ts: param_list<!AnyType> pos_vararg
# CHECK-SAME: (%args: !lit.ref<{{.*}}#VariadicPack <:!Bool {:i1 0}, :origin<0> *"args.origin{{.*}}, :!Bool {:i1 0}, :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> Ts>>, imm *"args`2"> read_mem|pack_vararg)
def packBorrowed[*Ts: AnyType](*args: *Ts):
    pass


# Ensure that parameters can be bound correctly.
def variadicParameter[*Ts: TrivialRegisterPassable](x: Int):
    pass


# CHECK-LABEL: lit.fn @"usePacks
# CHECK-SAME: [[ARGX:%.*]]: !FloatDyn
# CHECK-SAME: [[ARGY:%.*]]: !Int
def usePacks(x: FloatDyn, y: Int):
    # CHECK: lit.var.decl {{.*}} : !lit.ref<{{.*}}#MyTuple <:param_list<!AnyType> [!Int]>
    var a: MyTuple[Int]
    # CHECK: lit.var.decl {{.*}} : !lit.ref<{{.*}}#MyTuple <:param_list<!AnyType> [!Int, !FloatDyn, !Int]>
    var b: MyTuple[Int, FloatDyn, Int]
    # CHECK: lit.var.decl {{.*}} : !lit.ref<{{.*}}#MyTuple <:param_list<!AnyType> [!Int]>
    var c = MyTuple[Int](1)
    # CHECK: lit.var.decl {{.*}} : !lit.ref<{{.*}}#MyTuple <:param_list<!AnyType> [!FloatDyn, [{{.*}}@__MLIRType<:non_struct_type index>, index]]>
    var d = MyTuple(3.14, Int(6)._mlir_value)
    # CHECK: lit.var.decl {{.*}} : !lit.ref<{{.*}}#MyTuple <:param_list<!AnyType> []>
    var e = MyTuple()

    pack(Int(1)._mlir_value)
    pack(Int(1)._mlir_value, 3.14)
    pack()

    pack(Int(1)._mlir_value, x, y)
    pack[Int, FloatDyn, Int](Int(1), x, y)

    packBorrowed(Int(1)._mlir_value, x, y)

    # CHECK: lit.call {{.*}}variadicParameter{{.*}}<:param_list<!TrivialRegisterPassable> [!Int, !FloatDyn]>
    variadicParameter[Int, FloatDyn](1)
    # CHECK: lit.call {{.*}}variadicParameter{{.*}}<:param_list<!TrivialRegisterPassable> []>
    variadicParameter(Int(2))


# CHECK-LABEL: test_comptime_call
def test_comptime_call[a: Int]():
    # CHECK: lit.alias.decl *"foo`": none =
    # CHECK-SAME: <apply(:!lit.generator<[1](
    # CHECK-SAME: "args": !lit.ref<{{.*}}#VariadicPack <:!Bool {:i1 0}, :origin<0> #lit.comptime.origin, {{.*}}, :!Bool {:i1 0}, :!lit.anytrait<!AnyType> !AnyType, :param_list<!AnyType> [!Int]>>, imm #lit.comptime.origin> read_mem|pack_vararg)
    # CHECK-SAME: <store_to_mem(a)>))
    comptime foo = pack(a)


@fieldwise_init
struct MyMemoryStruct(ImplicitlyCopyable):
    var x: Int


# CHECK-LABEL: lit.fn @"create_pack_direct
def create_pack_direct(x: MyMemoryStruct, y: String):
    # Create Tuple
    var ptr_tuple = Tuple(UnsafePointer(to=x), UnsafePointer(to=y))
    comptime PackType = VariadicPack[
        origin=origin_of(x, y), False, AnyType, MyMemoryStruct, String
    ]
    # CHECK: %[[PACK:.*]] = lit.ref.pack.from_pointer_pack
    # CHECK: lit.call {{.*}}@VariadicPack::@"__init__(!lit.ref.pack{{.*}}(%[[PACK]])
    var built_pack = PackType(
        __mlir_op.`lit.ref.pack.from_pointer_pack`[
            _type=PackType._mlir_pack_type
        ](ptr_tuple._mlir_value)
    )
    pack(*built_pack)


# CHECK-LABEL: lit.fn @"create_pack_indirect
def create_pack_indirect(x: MyMemoryStruct, y: String):
    # Create Tuple
    var ptr_tuple = Tuple(UnsafePointer(to=x), UnsafePointer(to=y))
    comptime PackType = VariadicPack[
        origin=origin_of(x, y), False, AnyType, MyMemoryStruct, String
    ]
    # CHECK: %[[PACK:.*]] = lit.ref.pack.from_pointer_pack
    # CHECK: %[[VAR_RAW_PACK:.*]] = lit.var.decl "raw_pack" var
    # CHECK: lit.ref.store %[[PACK]], %[[VAR_RAW_PACK]]
    # CHECK: %[[VAR_LOAD_PACK:.*]] = lit.ref.load %[[VAR_RAW_PACK]]
    # CHECK: lit.call {{.*}}@VariadicPack::@"__init__(!lit.ref.pack{{.*}}(%[[VAR_LOAD_PACK]])
    var raw_pack = __mlir_op.`lit.ref.pack.from_pointer_pack`[
        _type=PackType._mlir_pack_type
    ](ptr_tuple._mlir_value)
    var built_pack = PackType(raw_pack)
    pack(*built_pack)


# ===----------------------------------------------------------------------=== #
# Forwarding
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: forward_pack
def forward_pack[*Ts: AnyType](*args: *Ts):
    # CHECK: lit.call {{.*}}@"pack{{.*}}"
    pack(*args)

# CHECK-LABEL: variadic_pack_intable_sink_borrowed
# CHECK-SAME: read_mem|pack_vararg
def variadic_pack_intable_sink_borrowed[*Ts: Intable](*elts: *Ts):
    pass

# CHECK-LABEL: forward_variadic_pack_borrowed_intable
# CHECK-SAME: [imm *[[IMP_ORIGIN_0:.*]]]
# CHECK-SAME: read_mem|pack_vararg
def forward_variadic_pack_borrowed_intable[*Ts: Intable](*pack: *Ts):
    # CHECK-NEXT: lit.call tail {{.*}}variadic_pack_intable_sink_borrowed
    # CHECK-SAME:[imm *[[IMP_ORIGIN_0]]]
    # CHECK-SAME: :origin<0> *"pack.origin
    variadic_pack_intable_sink_borrowed(*pack)

# CHECK-LABEL: variadic_pack_intable_sink_mutable
# CHECK-SAME: mut|pack_vararg
def variadic_pack_intable_sink_mutable[*Ts: Intable](mut *elts: *Ts):
    pass

# CHECK-LABEL: forward_variadic_pack_mut_intable
# CHECK-SAME: [imm *[[IMP_ORIGIN_0:.*]]]
# CHECK-SAME: mut|pack_vararg
def forward_variadic_pack_mut_intable[*Ts: Intable](mut *pack: *Ts):
    # CHECK-NEXT: lit.call tail {{.*}}variadic_pack_intable_sink_mutable
    # CHECK-SAME: [imm *[[IMP_ORIGIN_0]]]
    # CHECK-SAME: :origin<1> *"pack.origin
    variadic_pack_intable_sink_mutable(*pack)


# CHECK-LABEL: forward_variadic_pack_mut_intable_as_borrowed
# CHECK-SAME: [imm *[[IMP_ORIGIN_1:.*]]]
# CHECK-SAME: mut|pack_vararg
def forward_variadic_pack_mut_intable_as_borrowed[*Ts: Intable](mut *pack: *Ts):
    # CHECK-NEXT: kgen.rebind %pack
    # CHECK-NEXT: lit.call tail {{.*}}variadic_pack_intable_sink_borrowed
    # CHECK-SAME: [imm *[[IMP_ORIGIN_1]]]
    # CHECK-SAME: :origin<0> (mutcast mut *"pack.origin
    variadic_pack_intable_sink_borrowed(*pack)
