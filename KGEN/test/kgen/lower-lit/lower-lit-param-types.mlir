// RUN: kgen-opt %s -lower-lit -verify-parameters --kgen-print-inline-vtables | FileCheck %s

lit.struct.decl @Thing<T: trait<@Foo>> {
}

lit.func @x() {
  kgen.return
}

// CHECK: !kgen.declref<@Thing<:trait<@Foo> [!kgen.paramref<:trait<@Bar> T>, {"f" : () -> () = @x}]>>
lit.func @g<T: trait<@Bar>>() -> !kgen.declref<@Thing<:trait<@Foo> [!kgen.paramref<:trait<@Bar> T>, {"f": !lit.signature<() -> ()> = @x}]>> {
  kgen.unreachable
}
