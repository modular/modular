# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s| FileCheck %s

# COM: Check that a closure that captures a few things generates the right
# COM: implementation struct.


@value
struct MemType:
    fn __del__(owned self):
        pass


fn use(y: MemType, z: Int, u: int):
    pass


# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK-SAME: isSynthetic
# CHECK:      lit.struct.field field0 : !MemType
# CHECK-NEXT: lit.struct.field field1 : !Int
# CHECK-NEXT: lit.struct.field field2 : index

# CHECK:      lit.func @"__del__
# CHECK-NEXT:    [[VAR0:%.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:    lit.ownership.mark_destroyed %self
# CHECK-NEXT:    lit.return [[VAR0]] : !kgen.none
# CHECK-NEXT:    lit.end_func
# CHECK-NEXT: }

# CHECK-LABEL: lit.func @"__copyinit__(
# CHECK-SAME:   %self: !lit.ref<{{.*}}> init_self,
# CHECK-SAME:   %other: !lit.ref<{{.*}}> borrow_in_mem
# CHECK-SAME: ) -> !kgen.none {{.*}}specialFnKind = 3 : i8
# CHECK-NEXT:   [[V0:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:   [[V1:%.*]] = lit.ref.struct.ger %other[field0]
# CHECK-NEXT:   [[V2:%.*]] = lit.call @{{.*}}__copyinit__{{.*}}([[V0]], [[V1]])
# CHECK-NEXT:   [[V3:%.*]] = lit.ref.struct.ger %self[field1]
# CHECK-NEXT:   [[V4:%.*]] = lit.ref.struct.ger %other[field1]
# CHECK-NEXT:   [[V5:%.*]] = lit.ref.load [[V4]]
# CHECK-NEXT:   lit.ref.store [[V5]], [[V3]]
# CHECK-NEXT:   [[V6:%.*]] = lit.ref.struct.ger %self[field2]
# CHECK-NEXT:   [[V7:%.*]] = lit.ref.struct.ger %other[field2]
# CHECK-NEXT:   [[V8:%.*]] = lit.ref.load [[V7]]
# CHECK-NEXT:   lit.ref.store [[V8]], [[V6]]
# CHECK-NEXT:   [[V9:%.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:   lit.return [[V9]] : !kgen.none
# CHECK-NEXT:   lit.end_func
# CHECK-NEXT: }

# CHECK-LABEL: lit.func @"__moveinit__(
# CHECK-SAME:   %self: !lit.ref<{{.*}}> init_self,
# CHECK-SAME:   %other: !lit.ref<{{.*}}> owned_in_mem
# CHECK-SAME: ) -> !kgen.none {{.*}}specialFnKind = 4 : i8
# CHECK-NEXT:   [[W0:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:   [[W1:%.*]] = lit.ref.struct.ger %other[field0]
# CHECK-NEXT:   [[W2:%.*]] = lit.call @{{.*}}__moveinit__{{.*}}([[W0]], [[W1]])
# CHECK-NEXT:   [[W3:%.*]] = lit.ref.struct.ger %self[field1]
# CHECK-NEXT:   [[W4:%.*]] = lit.ref.struct.ger %other[field1]
# CHECK-NEXT:   [[W5:%.*]] = lit.load.consume [[W4]]
# CHECK-NEXT:   lit.ref.store [[W5]], [[W3]]
# CHECK-NEXT:   [[W6:%.*]] = lit.ref.struct.ger %self[field2]
# CHECK-NEXT:   [[W7:%.*]] = lit.ref.struct.ger %other[field2]
# CHECK-NEXT:   [[W8:%.*]] = lit.load.consume [[W7]]
# CHECK-NEXT:   lit.ref.store [[W8]], [[W6]]
# CHECK-NEXT:   [[W9:%.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:   lit.ownership.mark_destroyed %other
# CHECK-NEXT:   lit.return %none : !kgen.none
# CHECK-NEXT:   lit.end_func
# CHECK-NEXT: }

# CHECK-LABEL: lit.func @"__init__
# CHECK-NEXT:   [[Q0:%.*]] = lit.ref.struct.ger %self[field0]
# CHECK-NEXT:   [[Q1:%.*]] = lit.call @{{.*}}::@"__copyinit__{{.*}}([[Q0]], %fld0)
# CHECK-NEXT:   [[Q2:%.*]] = lit.ref.struct.ger %self[field1]
# CHECK-NEXT:   lit.ref.store %fld1, [[Q2]]
# CHECK-NEXT:   [[Q3:%.*]] = lit.ref.struct.ger %self[field2]
# CHECK-NEXT:   lit.ref.store %fld2, [[Q3]]
# CHECK-NEXT:   [[Q4:%.*]] = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT:   lit.return [[Q4]] : !kgen.none
# CHECK-NEXT:   lit.end_func


fn makes_escaping_closure(m: MemType, z: MemType, y: Bool):
    var register_passable_var: Int = 3
    var mlir_type_var: __mlir_type.index = register_passable_var.value

    fn dummy(n: MemType):
        use(m, register_passable_var, mlir_type_var)
