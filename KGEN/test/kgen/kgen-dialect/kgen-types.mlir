// RUN: kgen-opt %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s

// CHECK-LABEL: @genericSugar<scalar: type, T: type>
// CHECK-SAME: %arg0: !kgen.pointer<*"scalar">, %arg1: !kgen.pointer<T>
kgen.generator @genericSugar<scalar: type, T: type>(
  %arg0: !kgen.pointer<*"scalar">, %arg1: !kgen.pointer<T>
) {
  kgen.return
}

// CHECK-LABEL: @int_literal
// CHECK-SAME: %arg0: !kgen.int_literal
kgen.func @int_literal(%arg0: !kgen.int_literal) {
  kgen.return
}

// CHECK-LABEL: @float_literal
// CHECK-SAME: %arg0: !kgen.float_literal
kgen.func @float_literal(%arg0: !kgen.float_literal) {
  kgen.return
}

// CHECK-LABEL: @memory_only_struct
// CHECK-SAME: %arg0: !kgen.struct<()>,
// CHECK-SAME: %arg1: !kgen.struct<() memoryOnly>,
// CHECK-SAME: %arg2: !kgen.struct<(index, index)>,
// CHECK-SAME: %arg3: !kgen.struct<(index, index) memoryOnly>,
// CHECK-SAME: %arg4: !kgen.struct<((index, index) -> index)>,
// CHECK-SAME: %arg5: !kgen.struct<((index, index) -> index) memoryOnly>
kgen.func @memory_only_struct(
  %arg0: !kgen.struct<()>,
  %arg1: !kgen.struct<() memoryOnly>,
  %arg2: !kgen.struct<(index, index)>,
  %arg3: !kgen.struct<(index, index) memoryOnly>,
  %arg4: !kgen.struct<((index, index) -> index)>,
  %arg5: !kgen.struct<((index, index) -> index) memoryOnly>
) {
  kgen.return
}

// CHECK-LABEL: @type_printing
kgen.generator @type_printing() {
  // CHECK: type = <struct<()>>
  kgen.param.declare atype: type = <[struct<()>, {}]>
  // CHECK: type = <struct<()>>
  kgen.param.declare btype: type = <[struct<()>, {}]>
  // CHECK: type = <[struct<()>, {"method" : () -> () = @method}]>
  kgen.param.declare btype: type = <[struct<()>, {"method" : () -> () = @method}]>
  // CHECK: type = <[source_struct<"Foo">, {"method" : () -> () = @method}]>
  kgen.param.declare btype: type = <[source_struct<"Foo">, {"method" : () -> () = @method}]>
  // CHECK: type = <[source_struct<"Bar"[elemT: dtype, size]<:dtype f32, 16>(data: struct<()>) memoryOnly>, {"method" : () -> () = @method}]>
  kgen.param.declare btype: type = <[source_struct<"Bar"[elemT: dtype, size]<:dtype f32, 16>(data: struct<()>) memoryOnly>, {"method" : () -> () = @method}]>
  kgen.return
}

// CHECK-LABEL: kgen.generator @variadic_variant
// CHECK-SAME: !kgen.variant<[values]>
// CHECK-SAME-LITERAL: !kgen.variant<[[]]>
kgen.generator @variadic_variant<values: variadic<type>>(%arg0: !kgen.variant<[values]>, %arg1: !kgen.variant<[[]]>) {
  kgen.return
}

// CHECK-LABEL: kgen.func @type_value
kgen.func @type_value() {
  // CHECK: type = <struct<()>>
  kgen.param.declare atype: type = <[typevalue<[struct<()>]>]>
  // CHECK: type = <[struct<(typevalue<[struct<()>, {"method" : () -> () = @method}]>)>, struct<(struct<()>)>, {"method2" : () -> () = @method2}]>
  kgen.param.declare atype: type = <[struct<(typevalue<[struct<()>, {"method" : () -> () = @method}]>)>, struct<(struct<()>)>, {"method2" : () -> () = @method2}]>
  kgen.return
}

// CHECK-LABEL: @restrict_pointer
kgen.generator @restrict_pointer<b: i1>(
    // CHECK-SAME: !kgen.pointer<index, 7 exclusive(1)>
    %arg0: !kgen.pointer<index, 7 exclusive(1)>,
    // CHECK-SAME: !kgen.pointer<i32 exclusive(b)>
    %arg1: !kgen.pointer<i32 exclusive(b)>) {
  kgen.return
}
