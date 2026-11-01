// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t
// RUN: mef-translate -import-mef -allow-unregistered-dialect %t | FileCheck %s

// RUN: mef-translate -export-mef -allow-unregistered-dialect -enable-mef-metadata %s -o %t2
// RUN: mef-translate -import-mef -allow-unregistered-dialect %t2 | FileCheck %s --check-prefix=MD

// Check to make sure the magic number is right, and the first section is there.

// CHECK: module {

// CHECK-LABEL: "mef.callable" ["trivial"]
// CHECK-NEXT:    %{{.*}}:2 = mef.opaque "simple.primitive" []() : () -> (!mef.unknown, !mef.unknown)
// CHECK-NEXT:    mef.opaque "grt.invoke" [@trivial]() : () -> ()
// CHECK-NEXT:    mef.output
// CHECK-NEXT:  }
// MD-LABEL: "mef.callable" ["trivial"]
// MD-NEXT:    %{{.*}}:2 = "simple.primitive"() : () -> (i8, i16)
// MD-NEXT:    "grt.invoke"() {target = @trivial} : () -> ()
// MD-NEXT:    mef.output
// MD-NEXT:  }
mef.callable @trivial() {
  %a, %b = "simple.primitive"() : () -> (i8, i16)
  grt.invoke @trivial() : () -> ()
  mef.output
}

// CHECK-LABEL:  "mef.callable" ["args_results"]
// CHECK-NEXT:   ^bb0(%[[ARG0:.*]]: !mef.unknown, %[[ARG1:.*]]: !mef.unknown):
// CHECK-NEXT:     mef.output %[[ARG1]], %[[ARG0]] : !mef.unknown, !mef.unknown
// CHECK-NEXT:   }
// MD-LABEL:  "mef.callable" ["args_results"]
// MD-NEXT:   ^bb0(%[[ARG0:.*]]: si32, %[[ARG1:.*]]: si8):
// MD-NEXT:     mef.output %[[ARG1]], %[[ARG0]] : si8, si32
// MD-NEXT:   }
mef.callable @args_results(%a: si32, %b: si8) -> (si8, si32) {
  mef.output %b, %a : si8, si32
}

// CHECK-LABEL:  "mef.callable" ["attributes"]
// MD-LABEL:  "mef.callable" ["attributes"]
mef.callable @attributes() {
  // CHECK-NEXT: "a.b" [true]
  // MD-NEXT: "a.b"() {a = true}
  "a.b"() { a = 1 : i1 }: () -> ()

  // CHECK-NEXT: "a.b" [42 : si8, 17 : ui8]
  // MD-NEXT: "a.b"() {b = 42 : si8, x = 17 : ui8}
  "a.b"() { b = 42 : si8, x = 17 : ui8 }: () -> ()

  // CHECK-NEXT: "a.b" [258 : si16]
  // MD-NEXT: "a.b"() {c = 258 : si16}
  "a.b"() { c = 0x0102 : si16 }: () -> ()

  // CHECK-NEXT: "a.b" [32751241 : si32]
  "a.b"() { d = 32751241 : si32 }: () -> ()

   // CHECK-NEXT: "a.b" [3275124123121 : si64]
  "a.b"() { a = 3275124123121 : si64 }: () -> ()

  // CHECK-NEXT: "a.b" ["foo\00bar"]
  "a.b"() { a = "foo\00bar" : si64 }: () -> ()

  // CHECK-NEXT: "a.b" [2.000000e+00 : f32]
  "a.b"() { a = 2.0 : f32 }: () -> ()

  // CHECK-NEXT: "a.b" [3.000000e+00]
  "a.b"() { a = 3.0 : f64 }: () -> ()

  // Dictionary attribute.
  // CHECK-NEXT: mef.opaque "a.b" [{b = 1 : si8, c = [true, false, false]}]
  "a.b"() { a = { b = 1 : si8, c = [true, false, false] } }: () -> ()

  // List Attribute.
  // CHECK-NEXT: mef.opaque "a.b"  [1 : si8, [1 : ui32, 2 : si8]]
  // MD: "a.b"() {a = 1 : si8, b = [1 : ui32, 2 : si8]} : () -> ()
  "a.b"() {a = 1: si8, b = [1: ui32, 2: si8] } : () -> ()

  // Type attribute.
  // CHECK-NEXT: mef.opaque "type.i32" [si32]()
  // MD: "type.i32"() {t = si32}
  "type.i32"() {t = si32} : () -> ()

  // CHECK-NEXT: mef.opaque "type.f32" [f32]()
  // MD:  "type.f32"() {t = f32}
  "type.f32"() {t = f32} : () -> ()

  // CHECK-NEXT: mef.opaque "type.tensor" [tensor<4x1xf32>]()
  // MD: "type.tensor"() {t = tensor<4x1xf32>}
  "type.tensor"() {t = tensor<4x1xf32>} : () -> ()

  // CHECK-NEXT: mef.opaque "type.tensor" [tensor<0x1x0xcomplex<si32>>]()
  // MD: "type.tensor"() {t = tensor<0x1x0xcomplex<si32>>
  "type.tensor"() {t = tensor<0x1x0xcomplex<si32>>} : () -> ()

  // CHECK-NEXT: mef.opaque "type.quanttensor" [tensor<1x2x!quant.uniform<si16:f32:1, {0.20000000298023224:128,0.10000000149011612:-128}>>]()
  // MD: "type.quanttensor"() {t = tensor<1x2x!quant.uniform<si16:f32:1, {0.20000000298023224:128,0.10000000149011612:-128}>>}
  "type.quanttensor"() {t = tensor<1x2x!quant.uniform<i16:f32:1, {0.2:128, 0.1:-128}>>} : () -> ()

  // CHECK-NEXT: mef.opaque "type.quant" [!quant.uniform<ui8:f32, 0.0078119998797774315:128>]()
  // MD: "type.quant"() {t = !quant.uniform<ui8:f32, 0.0078119998797774315:128>
  "type.quant"() {t = !quant.uniform<u8:f32, 0.0078119998797774315:128>} : () -> ()

  // CHECK-NEXT: mef.opaque "type.quant" [!quant.uniform<si8:f32, 0.0078119998797774315:-128>]()
  // MD: "type.quant"() {t = !quant.uniform<si8:f32, 0.0078119998797774315:-128>
  "type.quant"() {t = !quant.uniform<i8:f32, 0.0078119998797774315:-128>} : () -> ()

  // CHECK-NEXT: mef.opaque "type.fn" [(si32, !quant.uniform<si8:f32, 0.0078119998797774315:-128>) -> tensor<4x100xf32>]()
  // MD: "type.fn"() {t = (si32, !quant.uniform<si8:f32, 0.0078119998797774315:-128>) -> tensor<4x100xf32>
  "type.fn"() {t = (si32, !quant.uniform<i8:f32, 0.0078119998797774315:-128>) -> tensor<4x100xf32>} : () -> ()

  // CHECK-NEXT: mef.opaque "type.fn" [() -> (si8, i1)]()
  // MD: "type.fn"() {t = () -> (si8, i1)
  "type.fn"() {t = () -> (si8, i1)} : () -> ()

  // We're dropping the quantization information here.
  // CHECK-NEXT: mef.opaque "type.calibrated" [tensor<1x28x28xf32>]()
  // MD: "type.calibrated"() {t = tensor<1x28x28xf32>
  "type.calibrated"() {t = tensor<1x28x28x!quant.calibrated<f32<0.000000e+00:2.550000e+02>>>} : () -> ()

  // Check that we correctly serialize and deserialize the chain type.
  // CHECK-NEXT: mef.opaque "type.chain" [!mef.chain]()
  // MD: "type.chain"() {t = !mef.chain
  "type.chain"() {t = !mef.chain} : () -> ()

  // CHECK-NEXT: mef.opaque "type.aligned_bytes" [#M.dense_array<222, 173, 190, 239> : !M.array<4xui8>]() : () -> ()
  // MD: "type.aligned_bytes"() {t = #M.dense_array<222, 173, 190, 239> : !M.array<4xui8>
  "type.aligned_bytes"() {t = #M.aligned_bytes<"0xDEADBEEF", align 64>} : () -> ()

  mef.output
   // CHECK-NEXT:     mef.output
}  // CHECK-NEXT:   }

// CHECK-LABEL: "mef.callable" ["opaque_round_trip"]
// CHECK-NEXT:    %[[V0:.*]]:2 = mef.opaque "simple.primitive" []() : () -> (!mef.unknown, !mef.unknown)
// CHECK-NEXT:    mef.opaque "other.primitive" [42 : si8](%[[V0]]#1, %[[V0]]#0) : (!mef.unknown, !mef.unknown) -> ()
// CHECK-NEXT:    mef.output
// CHECK-NEXT:  }
mef.callable @opaque_round_trip() {
  %0:2 = "mef.opaque"() {attrs = [], opName = "simple.primitive"} : () -> (!mef.unknown, !mef.unknown)
  "mef.opaque"(%0#1, %0#0) {attrs = [42 : si8], opName = "other.primitive"} : (!mef.unknown, !mef.unknown) -> ()
  mef.output
}

// CHECK-LABEL: mef.opaque "some.container" ["multi_region"]() : () -> () {
// CHECK-NEXT:   mef.opaque "a.firstregion" []() : () -> ()
// CHECK-NEXT:   mef.output
// CHECK-NEXT: }, {
// CHECK-NEXT:   mef.opaque "a.secondregion" []() : () -> ()
// CHECK-NEXT:   mef.output
// CHECK-NEXT: }
"some.container"() ({
  "a.firstregion"() : () -> ()
}, {
  "a.secondregion"() : () -> ()
}) {sym_name = "multi_region"} : () -> ()

// Test mef.model export/import round-trip.
// CHECK-LABEL: mef.opaque "mef.model" [
// CHECK-SAME:    "mof"
// CHECK-SAME:    (si32) -> si32
// CHECK-SAME:    (si32, tensor<?xf32>) -> tensor<?xf32>
// CHECK-SAME:    (si32, tensor<?xf32>) -> tensor<?xf32>
// CHECK-SAME:    "test_model"
// CHECK-SAME:  ]() : () -> () {
// CHECK-NEXT: ^bb0(%[[INIT_ARG:.*]]: !mef.unknown):
// CHECK-NEXT:   mef.output %[[INIT_ARG]] : !mef.unknown
// CHECK-NEXT: }, {
// CHECK-NEXT: ^bb0(%[[EXEC_ARG0:.*]]: !mef.unknown, %[[EXEC_ARG1:.*]]: !mef.unknown):
// CHECK-NEXT:   mef.output %[[EXEC_ARG1]] : !mef.unknown
// CHECK-NEXT: }
// MD-LABEL: mef.opaque "mef.model" [
// MD-SAME:    "mof"
// MD-SAME:    (si32) -> si32
// MD-SAME:    (si32, tensor<?xf32>) -> tensor<?xf32>
// MD-SAME:    (si32, tensor<?xf32>) -> tensor<?xf32>
// MD-SAME:    "test_model"
// MD-SAME:  ]() : () -> () {
// MD-NEXT: ^bb0(%[[INIT_ARG:.*]]: si32):
// MD-NEXT:   mef.output %[[INIT_ARG]] : si32
// MD-NEXT: }, {
// MD-NEXT: ^bb0(%[[EXEC_ARG0:.*]]: si32, %[[EXEC_ARG1:.*]]: tensor<?xf32>):
// MD-NEXT:   mef.output %[[EXEC_ARG1]] : tensor<?xf32>
// MD-NEXT: }
mef.model @test_model : mof (si32, tensor<?xf32>) -> tensor<?xf32> devices [#M.device_ref<"cpu", 0>] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%arg0: si32) -> (si32) {
    mef.output %arg0 : si32
  } execute (%arg0: si32, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {
    mef.output %arg1 : tensor<?xf32>
  }

// CHECK-LABEL: "mef.callable" ["controlflow_if"]
// CHECK-NEXT: ^bb0(%[[ARG0:.*]]: !mef.unknown, %[[ARG1:.*]]: !mef.unknown, %[[ARG2:.*]]: !mef.unknown):
// CHECK-NEXT:   %[[V0:.*]] = mef.opaque "grt.if" nonstrict [](%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (!mef.unknown, !mef.unknown, !mef.unknown) -> !mef.unknown {
// CHECK-NEXT:   ^bb0(%[[ARG3:.*]]: !mef.unknown, %[[ARG4:.*]]: !mef.unknown):
// CHECK-NEXT:     mef.output %[[ARG3]] : !mef.unknown
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb0(%[[ARG3:.*]]: !mef.unknown, %[[ARG4:.*]]: !mef.unknown):
// CHECK-NEXT:     mef.output %[[ARG4]] : !mef.unknown
// CHECK-NEXT:   }
// CHECK-NEXT:   mef.output %[[V0]] : !mef.unknown
// CHECK-NEXT: }
mef.callable @controlflow_if(%cond: i1, %v1: si32, %v2: si32) -> si32 {
  %res = grt.if %cond (%v1: si32, %v2: si32) -> (si32) {
    mef.output %v1 : si32
  } else {
    mef.output %v2 : si32
  }

  mef.output %res : si32
}

// CHECK-NEXT: }
