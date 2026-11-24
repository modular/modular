// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t
// RUN: od -t x1 %t | FileCheck %s

// Check to make sure the magic number is right, and the first section is there.

// CHECK: 0000000 ee f0 03

mef.callable @trivial() {
  "simple.primitive"() { x = 42 : si8 }: () -> ()
  mef.output
}

mef.callable @args_results(%a: si32, %b: si8) -> (si8, si32) {
  mef.output %b, %a : si8, si32
}

mef.callable @unknown_args(%a: !mef.unknown, %b: !mef.unknown) -> !mef.unknown {
  mef.output %b : !mef.unknown
}

// Test that mef.model can be exported to MEF.
mef.model @test_model : mof (si32, tensor<?xf32>) -> tensor<?xf32> devices [] inputDeviceIndices [] resultDeviceIndices [] argumentNames [] resultNames [] init (%arg0: si32) -> (si32) {
    mef.output %arg0 : si32
  } execute (%arg0: si32, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {
    mef.output %arg1 : tensor<?xf32>
  }
