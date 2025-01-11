# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.struct.decl{{.*}}<{{.*}}: !lit.generator<() capturing -> !Int>
# CHECK:      lit.struct.field param_capture : !lit.struct<#ParameterClosureCaptureList{{.*}}__call__
# CHECK-NEXT: lit.struct.field field0 : !Int

# CHECK: lit.fn @"__call__
# CHECK: [[GEP:%.*]] = lit.ref.struct.ger {{.*}}[param_capture]
# CHECK: lit.call {{.*}}expand{{.*}}([[GEP]])

# CHECK: lit.fn @"__init__
# CHECK: [[GEP:%.*]] = lit.ref.struct.ger %self[param_capture]
# CHECK: [[RES:%.*]] = lit.call {{.*}}@__ParameterClosureCaptureList::@"__init__{{.*}}()
# CHECK-NEXT: lit.ref.store [[RES]], [[GEP]]


fn func[pf: fn () capturing -> Int](x: Int):
    fn escaping(y: Int) -> Int:
        return y + x + pf()  # this use of 'pf' is a parameter capture
        # ParamDeclRefAttr


fn pass_it(x: Int):
    @parameter
    fn closure() -> Int:
        return x

    func[closure](17)
