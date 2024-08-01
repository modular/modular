// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect --kgen-print-inline-type-values | FileCheck %s

// CHECK: *"mangled_fn{{.*}}int
"some.op"() {decl = #kgen<param.decl *"mangled_fn(Pointer[!lit.struct<_\22int\22::_Int>])" : index>} : () -> ()

kgen.generator @return_one() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK: a = #kgen.type
// CHECK-SAME: b = #kgen.type
"some.op"() {
  a = #kgen.type<array<1, i1>> : !kgen.type,
  b = #kgen.type<array<apply(:() -> index @return_one), i1>> : !kgen.type
} : () -> ()

// CHECK: #kgen.param.index.ref<0, false, 0> : index
"some.op"() {ref = #kgen.param.index.ref<0, false, 0> : index} : () -> ()

// CHECK: #kgen.int_literal<5> : !kgen.int_literal
"some.op"() {data = #kgen.int_literal<5> : !kgen.int_literal} : () -> ()

// CHECK: #kgen.float_literal<5|3> : !kgen.float_literal
"some.op"() {data = #kgen.float_literal<5|3> : !kgen.float_literal} : () -> ()
// CHECK: #kgen.float_literal<neg_zero> : !kgen.float_literal
"some.op"() {data = #kgen.float_literal<neg_zero> : !kgen.float_literal} : () -> ()
// CHECK: #kgen.float_literal<inf> : !kgen.float_literal
"some.op"() {data = #kgen.float_literal<inf> : !kgen.float_literal} : () -> ()

// CHECK: #kgen.env<{bar = 1 : index, foo}>
"some.op"() {env = #kgen.env<{bar = 1 : index, foo}>} : () -> ()

// CHECK: #kgen<decorators[1 : i64]>
"some.op"() {decorators = #kgen<decorators[1 : i64]>} : () -> ()

// CHECK: #kgen.int_literal<1234>
// CHECK-SAME: #kgen.int_literal<12345678901234567899012345678901234567890>
"some.op"() {a = #kgen.int_literal<1234> : !kgen.int_literal,
             b = #kgen.int_literal<12345678901234567899012345678901234567890> : !kgen.int_literal} : () -> ()

// CHECK-LABEL: @struct_constants
kgen.generator @struct_constants<T: type, A: !kgen.paramref<T>, value: !pop.scalar<f32>>() {
  // CHECK: struct<(index, f32)> = <{ 1, 2.5{{0+}}e+00 }>
  kgen.param.constant: struct<(index, f32)> = <{ 1, 2.5 }>
  // CHECK: struct<(scalar<f32>)> = <{ value }>
  kgen.param.constant: struct<(scalar<f32>)> = <{ value }>
  // CHECK: struct<(T)> = <{ A }>
  kgen.param.constant: struct<(T)> = <{ A }>
  kgen.return
}

// CHECK-LABEL: @pack_constants
kgen.generator @pack_constants<Ts: variadic<i32>>() {
  // CHECK: !kgen.pack<[i8, ui4, i32]> = <<3, 1, 4>>
  %0 = kgen.param.constant: !kgen.pack<[i8, ui4, i32]> = <<3, 1, 4>>
  // CHECK: !kgen.pack<[]> = <<>>
  %1 = kgen.param.constant: !kgen.pack<[]> = <<>>
  kgen.return
}


// CHECK-LABEL: @variant_constants
kgen.generator @variant_constants<T: type, U: type, value: !kgen.paramref<T>>() {
  // CHECK: variant<f32, f64> = <{:f32 2.5{{0+}}e+00, 0}>
  %0 = kgen.param.constant: variant<f32, f64> = <{:f32 2.5, 0}>
  // CHECK: variant<T, U> = <{:!kgen.paramref<T> value, 0}>
  %1 = kgen.param.constant: variant<T, U> = <{:!kgen.paramref<T> value, 0}>
  kgen.return
}

kgen.generator @entry1() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}
kgen.generator @entry2() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK: vtableEntry = #kgen<vtable.entry"entry1" : () -> index = @entry1>
"some.op"() {vtableEntry = #kgen<vtable.entry "entry1" : () -> index = @entry1>} : () -> ()
// CHECK: vtable = #kgen<vtable"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2
"some.op"() {vtable = #kgen<vtable "entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2>} : () -> ()

// CHECK: #kgen.type<index, {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
"some.op"() {type = #kgen.type<index, {"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type} : () -> ()
// CHECK: a = #kgen.type<array<1, i1>, {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
// CHECK: b = #kgen.type<array<apply(:() -> index @return_one), i1>, {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
// CHECK: c = #kgen.type<array<1, i1>, array<2, i1>, {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
// CHECK: d = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type
// CHECK: e = #kgen.type<array<1, i1>> : !kgen.type
"some.op"() {
  a = #kgen.type<array<1, i1>, {"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type,
  b = #kgen.type<array<apply(:() -> index @return_one), i1>, {"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type,
  c = #kgen.type<array<1, i1>, array<2, i1>, {"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type,
  d = #kgen.type<array<1, i1>, array<2, i1>> : !kgen.type,
  e = #kgen.type<array<1, i1>, array<1, i1>> : !kgen.type
} : () -> ()

// CHECK: a = #kgen.struct_def<"Foo"> : !kgen.struct_def<>
// CHECK: b = #kgen.struct_def<"Bar"[elemT: dtype, size](data: struct<()>) memoryOnly> : !kgen.struct_def<[elemT: dtype, size]>
// CHECK: c = #kgen.struct_def<"Baz"[elemT: dtype, size](data: simd<*("size"), *("elemT")>)> : !kgen.struct_def<[elemT: dtype, size]>
// CHECK: d = #kgen.struct_def<"Node"[size](data: pointer<array<*("size"), applied_struct<[size] #kgen.struct_def.self<0>, <*("size")>>>>)> : !kgen.struct_def<[size]>
// CHECK: e = #kgen.struct_def<"A"[size](data: pointer<applied_struct<[size] #kgen.struct_def<"B"[size](data: pointer<applied_struct<[size] #kgen.struct_def.self<1>, <*("size")>>>)>, <*("size")>>>)> : !kgen.struct_def<[size]>
"some.op"() {
  a = #kgen.struct_def<"Foo"> : !kgen.struct_def<>,
  b = #kgen.struct_def<"Bar"[elemT: dtype, size](data: struct<()>) memoryOnly> : !kgen.struct_def<[elemT: dtype, size]>,
  c = #kgen.struct_def<"Baz"[elemT: dtype, size](data: simd<*("size"), *("elemT")>)> : !kgen.struct_def<[elemT: dtype, size]>,
  d = #kgen.struct_def<"Node"[size](data: pointer<array<*("size"), applied_struct<[size] #kgen.struct_def.self<0>, <*("size")>>>>)> : !kgen.struct_def<[size]>,
  e = #kgen.struct_def<"A"[size](data: pointer<applied_struct<[size] #kgen.struct_def<"B"[size](data: pointer<applied_struct<[size] #kgen.struct_def.self<1>, <*("size")>>>)>, <*("size")>>>)> : !kgen.struct_def<[size]>
} : () -> ()

// CHECK: #kgen.tailkind<none>
// CHECK: #kgen.tailkind<musttail>
// CHECK: #kgen.tailkind<notail>
"some.op"() {
  a = #kgen.tailkind<none>,
  c = #kgen.tailkind<musttail>,
  d = #kgen.tailkind<notail>
} : () -> ()
