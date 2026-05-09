# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Basic IR checks for type refinement.
#
# Keep this test parser-only and focused on refinement rebinds.
#
# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Base:
    pass


trait Extra:
    pass


trait SomeTrait:
    pass


def use_base[T: Base](read x: T):
    pass


def use_extra[T: Extra](read x: T):
    pass


struct MiniPack[element_trait: type_of(AnyType), //, *element_types: element_trait]:
    def get_element[
        index: Int
    ](self) -> ref[origin_of(self)] Self.element_types[index]:
        while True:
            pass

    # CHECK-LABEL: lit.fn @"refine_param_list_get_after_assert
    # CHECK: kgen.param.assert <{{.*}}#kgen.type_conforms_to_trait<#kgen.param_list.get<{{.*}}element_types.values{{.*}}, #lit.struct.extract<:!Int i, "_mlir_value">> : !kgen.param<{{.*}}element_trait>, [{{.*}}@type_refinement_ir::@Extra]{{.*}}>
    # CHECK: [[ITEM:%.*]] = lit.call tail @type_refinement_ir::@MiniPack::@"get_element
    # CHECK: [[REBIND:%.*]] = kgen.rebind [[ITEM]] : {{.*}}#kgen.param_list.get<{{.*}}element_types.values{{.*}}, #lit.struct.extract<:!Int i, "_mlir_value">>{{.*}} to {{.*}}downcast(:!kgen.param<{{.*}}> #kgen.param_list.get<{{.*}}element_types.values{{.*}}, #lit.struct.extract<:!Int i, "_mlir_value">>){{.*}}
    # CHECK: lit.call{{.*}}@"use_extra{{.*}}([[REBIND]])
    def refine_param_list_get_after_assert[i: Int](self):
        comptime element_type = Self.element_types[i]
        comptime assert conforms_to(element_type, Extra)
        use_extra(self.get_element[i]())


struct BaseMiniPack[element_trait: type_of(Base), //, *element_types: element_trait]:
    def get_element[
        index: Int
    ](self) -> ref[origin_of(self)] Self.element_types[index]:
        while True:
            pass

    # CHECK-LABEL: lit.fn @"refine_param_list_get_preserves_parametric_bound
    # CHECK: kgen.param.assert <{{.*}}#kgen.type_conforms_to_trait<#kgen.param_list.get<{{.*}}element_types.values{{.*}}, #lit.struct.extract<:!Int i, "_mlir_value">> : !kgen.param<{{.*}}element_trait>, [{{.*}}@type_refinement_ir::@Extra]{{.*}}>
    # CHECK: [[EXTRA_ITEM:%.*]] = lit.call tail @type_refinement_ir::@BaseMiniPack::@"get_element
    # CHECK: [[EXTRA_REBIND:%.*]] = kgen.rebind [[EXTRA_ITEM]] : {{.*}}#kgen.param_list.get<{{.*}}element_types.values{{.*}}, #lit.struct.extract<:!Int i, "_mlir_value">>{{.*}} to {{.*}}Base_Extra downcast(:!kgen.param<{{.*}}> #kgen.param_list.get<{{.*}}element_types.values{{.*}}, #lit.struct.extract<:!Int i, "_mlir_value">>){{.*}}
    # CHECK: lit.call{{.*}}@"use_extra{{.*}}([[EXTRA_REBIND]])
    # CHECK: [[BASE_ITEM:%.*]] = lit.call tail @type_refinement_ir::@BaseMiniPack::@"get_element
    # CHECK: [[BASE_REBIND:%.*]] = kgen.rebind [[BASE_ITEM]] : {{.*}}#kgen.param_list.get<{{.*}}element_types.values{{.*}}, #lit.struct.extract<:!Int i, "_mlir_value">>{{.*}} to {{.*}}Base_Extra downcast(:!kgen.param<{{.*}}> #kgen.param_list.get<{{.*}}element_types.values{{.*}}, #lit.struct.extract<:!Int i, "_mlir_value">>){{.*}}
    # CHECK: lit.call{{.*}}@"use_base{{.*}}([[BASE_REBIND]])
    def refine_param_list_get_preserves_parametric_bound[i: Int](self):
        comptime element_type = Self.element_types[i]
        comptime assert conforms_to(element_type, Extra)
        use_extra(self.get_element[i]())
        use_base(self.get_element[i]())


# CHECK-LABEL: lit.fn @"refine_from_where
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!Base T, imm *"x`"> read_mem)
# CHECK: [[WHERE_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
def refine_from_where[T: Base](read x: T) where conforms_to(T, Extra):
    use_base(x)
    use_extra(x)


# CHECK-LABEL: lit.fn @"refine_in_comptime_if
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!Base T, imm *"x`"> read_mem)
# CHECK: kgen.param.if <{{.*}}conforms_to(:!Base T, [{{.*}}@type_refinement_ir::@Extra]){{.*}}> {
# CHECK: [[IF_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
def refine_in_comptime_if[T: Base](read x: T):
    comptime if conforms_to(T, Extra):
        use_base(x)
        use_extra(x)


# CHECK-LABEL: lit.fn @"refine_after_comptime_assert
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!Base T, imm *"x`"> read_mem)
# CHECK: kgen.param.assert <{{.*}}conforms_to(:!Base T, [{{.*}}@type_refinement_ir::@Extra]){{.*}}>
# CHECK: [[ASSERT_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
def refine_after_comptime_assert[T: Base](read x: T):
    comptime assert conforms_to(T, Extra)
    use_base(x)
    use_extra(x)


# CHECK-LABEL: lit.fn @"refinement_does_not_leak_after_comptime_if
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!Base T, imm *"x`"> read_mem)
# CHECK: kgen.param.if <{{.*}}conforms_to(:!Base T, [{{.*}}@type_refinement_ir::@Extra]){{.*}}> {
# CHECK: [[IF_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
# CHECK-NOT: kgen.rebind [[ARG]]
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
def refinement_does_not_leak_after_comptime_if[T: Base](read x: T):
    comptime if conforms_to(T, Extra):
        use_extra(x)
    use_base(x)


# CHECK-LABEL: lit.fn @"no_refinement_before_comptime_assert
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!Base T, imm *"x`"> read_mem)
# CHECK-NOT: kgen.rebind [[ARG]]
# CHECK: lit.call{{.*}}@{{.*}}@"use_base
# CHECK: kgen.param.assert <{{.*}}conforms_to(:!Base T, [{{.*}}@type_refinement_ir::@Extra]){{.*}}>
# CHECK: [[ASSERT_REF:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}Base_Extra downcast(:!Base T){{.*}}
# CHECK: lit.call{{.*}}@{{.*}}@"use_extra
def no_refinement_before_comptime_assert[T: Base](read x: T):
    use_base(x)
    comptime assert conforms_to(T, Extra)
    use_extra(x)


# CHECK-LABEL: lit.fn @"refine_in_comptime_if_no_call
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!SomeTrait T, mut *"x`"> owned_in_mem)
# CHECK: kgen.param.if <{{.*}}conforms_to(:!SomeTrait T, [{{.*}}@ImplicitlyDestructible]){{.*}}> {
# CHECK: [[IF_REBIND:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}ImplicitlyDestructible{{.*}}downcast(:!SomeTrait T){{.*}}
# CHECK: lit.ownership.use [[IF_REBIND]]
def refine_in_comptime_if_no_call[T: SomeTrait](var x: T):
    comptime if conforms_to(T, ImplicitlyDestructible):
        _ = x


# CHECK-LABEL: lit.fn @"refine_after_comptime_assert_no_call
# CHECK-SAME: ([[ARG:%.*]]: !lit.ref<:!SomeTrait T, mut *"x`"> owned_in_mem)
# CHECK: kgen.param.assert <{{.*}}conforms_to(:!SomeTrait T, [{{.*}}@ImplicitlyDestructible]){{.*}}>
# CHECK: [[ASSERT_REBIND:%.*]] = kgen.rebind [[ARG]] : {{.*}} to {{.*}}ImplicitlyDestructible{{.*}}downcast(:!SomeTrait T){{.*}}
# CHECK: lit.ownership.use [[ASSERT_REBIND]]
def refine_after_comptime_assert_no_call[T: SomeTrait](var x: T):
    comptime assert conforms_to(T, ImplicitlyDestructible)
    _ = x
