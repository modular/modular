// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: *"mangled_fn{{.*}}$int
"some.op"() {decl = #kgen<param.decl *"mangled_fn(Pointer[!kgen.declref<_\22$int\22::_Int>])" : index>} : () -> ()

kgen.generator @return_one() -> index {
  %0 = index.constant 1
  kgen.return %0 : index
}

// CHECK: a = #kgen.concretetype.constant
// CHECK-SAME: b = #kgen.parameterizedtype.constant
"some.op"() {
  a = #kgen.parameterizedtype.constant<!pop.array<1, i1>> : !kgen.type,
  b = #kgen.parameterizedtype.constant<!pop.array<apply(:() -> index @return_one), i1>> : !kgen.type
} : () -> ()

// CHECK: #kgen.param.index.ref<0, false, 0> : index
"some.op"() {ref = #kgen.param.index.ref<0, false, 0> : index} : () -> ()

// CHECK: #kgen.int_literal<5> : !kgen.int_literal
"some.op"() {data = #kgen.int_literal<5> : !kgen.int_literal} : () -> ()

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
  // CHECK: variant<f32, f64> = <#kgen.variant<:f32 2.5{{0+}}e+00, 0>>
  %0 = kgen.param.constant: variant<f32, f64> = <#kgen.variant<:f32 2.5, 0>>
  // CHECK: variant<T, U> = <#kgen.variant<:!kgen.paramref<T> value, 0>>
  %1 = kgen.param.constant: variant<T, U> = <#kgen.variant<:!kgen.paramref<T> value, 0>>
  kgen.return
}

// CHECK: #kgen.package.archive<target = {{.*}}, elaboratedModule = {{.*}}, archive = {{.*}}>
"some.op"() {a = #kgen.package.archive<
  target = #M.target<triple = "arm64-apple-darwin21.6.0", arch="apple-m1">,
  elaboratedModule = dense_resource<foo> : tensor<42xui8>,
  archive = dense_resource<bar> : tensor<13xui8>,
  dependencies = [
    <dense_resource<baz> : tensor<29xui8> as "libBaz">
  ]
>} : () -> ()

// CHECK: #kgen<package.archives[<{{.*}}>, <{{.*}}>]>
"some.op"() {a = #kgen<package.archives[
  <target = #M.target<triple = "", arch="">,
   elaboratedModule = dense_resource<a> : tensor<1xui8>,
   archive = dense_resource<b> : tensor<2xui8>>,
  <target = #M.target<triple = "", arch="">,
   elaboratedModule = dense_resource<c> : tensor<3xui8>,
   archive = dense_resource<d> : tensor<4xui8>>
]>} : () -> ()

// CHECK: #kgen.link.dependency<dense_resource<a862fa0> : tensor<422xui8> as "ffmpeg">
"some.op"() {
  a = #kgen.link.dependency<dense_resource<a862fa0> : tensor<422xui8> as "ffmpeg">
} : () -> ()

// CHECK: #kgen<link.dependencies[
// CHECK-SAME: <dense_resource<c95292> : tensor<659xui8> as "libavcodec">,
// CHECK-SAME: <dense_resource<e062d0> : tensor<861xui8> as "libavutil">]>
"some.op"() {a = #kgen<link.dependencies[
  <dense_resource<c95292> : tensor<659xui8> as "libavcodec">,
  <dense_resource<e062d0> : tensor<861xui8> as "libavutil">
]>} : () -> ()

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

// CHECK: #kgen.concretetype.constant<index, vtable = {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
"some.op"() {type = #kgen.concretetype.constant<index, vtable={"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type} : () -> ()
// CHECK: a = #kgen.concretetype.constant<!pop.array<1, i1>, vtable = {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
// CHECK: b = #kgen.parameterizedtype.constant<!pop.array<apply(:() -> index @return_one), i1>, vtable = {"entry1" : () -> index = @entry1,  "entry2" : () -> index = @entry2}> : !kgen.type
"some.op"() {
  a = #kgen.parameterizedtype.constant<!pop.array<1, i1>, vtable={"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type,
  b = #kgen.parameterizedtype.constant<!pop.array<apply(:() -> index @return_one), i1>, vtable={"entry1" : () -> index = @entry1, "entry2" : () -> index = @entry2}> : !kgen.type
} : () -> ()
