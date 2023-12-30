# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo %s | FileCheck %s

# CHECK: lit.struct.decl{{.*}}<[[PARAMNAME:.*]]: !lit.signature<() capturing -> !Int>
# CHECK-NEXT: lit.struct.field field0 : !Int
# CHECK-NEXT: lit.struct.field field1 : !kgen.capture_list<!lit.signature<() capturing -> !Int> : [[PARAMNAME]]>

# CHECK: lit.func @"__del__
# CHECK: [[GEP:%.*]] = lit.ref.struct.ger %self[field1]
# CHECK: [[CLIST:%.*]] = lit.ref.load [[GEP]]
# CHECK: kgen.capture_list.destroy [[CLIST]]

# CHECK: lit.func @"__init__
# CHECK: [[CLIST:%.*]] = kgen.capture_list.create : <!lit.signature<() capturing -> !Int> : [[PARAMNAME]]>
# CHECK: [[GEP:%.*]] = lit.ref.struct.ger %self[field1]
# CHECK: lit.ref.store [[CLIST]], [[GEP]]

# CHECK: lit.func @"__call__
# CHECK: [[GEP:%.*]] = lit.struct.gep {{.*}}[field1]
# CHECK: [[CLIST:%.*]] = pop.load [[GEP]]
# CHECK: kgen.capture_list.expand [[CLIST]]

fn take_escaping(ef: fn(y: Int) escaping -> Int):
    print(ef(23))

fn func[pf: fn() capturing -> Int](x: Int):
    fn escaping(y: Int) escaping -> Int:
        return y+ x + pf() # this use of 'pf' is a parameter capture
                           # ParamDeclRefAttr
    take_escaping(escaping)

fn pass_it(x: Int):
    @parameter
    fn closure() -> Int: return x
    func[closure](17)
