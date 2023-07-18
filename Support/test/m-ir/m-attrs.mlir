// RUN: support-dialect-opt -allow-unregistered-dialect %s | support-dialect-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: ui7: 126, 0, 2, 20
"M"() {a = #M.primitives_array<ui7: -2, 0, 2, 20>} : () -> ()

// CHECK: i1: true, false
"M"() {a = #M.primitives_array<i1: true, false>} : () -> ()

// CHECK: bf16: 3.1{{[0-9]+}}e+00, 1.7{{[0-9]+}}e+00
"M"() {a = #M.primitives_array<bf16: 3.14, 1.73>} : () -> ()

// CHECK: primitives_array<i64>
"M"() {a = #M.primitives_array<i64>} : () -> ()

// CHECK: primitives_array<index: -3, 1, 3>
"M"() {a = #M.primitives_array<index: -3, 1, 3>} : () -> ()

// CHECK: dense_array<1, 2, 3, 4> : tensor<2x2xi32>
"M"() {a = #M.dense_array<1, 2, 3, 4> : tensor<2x2xi32>} : () -> ()

// CHECK: dense_array<0.{{0+}}e+00> : vector<f32>
"M"() {a = #M.dense_array<0.> : vector<f32>} : () -> ()

// CHECK: dense_array<65534, 1, 4> : !M.array<3xui16>
"M"() {a = #M.dense_array<-2, 1, 4> : !M.array<3xui16>} : () -> ()

// CHECK: dense_array<-3, 1, 3> : !M.array<3xindex>
"M"() {a = #M.dense_array<-3, 1, 3> : !M.array<3xindex>} : () -> ()

// CHECK: aligned_bytes<"0x01020304", align 64>
"M"() {a = #M.aligned_bytes<"0x01020304", align 64>} : () -> ()

// CHECK: #M.target<triple = "a", cpu = "b", features = "", data_layout = "p:64:64-i64:64:64", simd_bit_width = 128>
"M"() {a = #M.target<triple = "a", cpu = "b", features = "", data_layout = "p:64:64-i64:64:64", simd_bit_width = 128>} : () -> ()

// CHECK: #M<multiline["a", "b", "c"]>
"M"() {a = #M<multiline["a", "b", "c"]>} : () -> ()
