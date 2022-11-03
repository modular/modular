// RUN: kgen -execute -func="test_print:()" %s | FileCheck %s

kgen.generator @impl<lb, ub, step>(%buf: !pop.pointer<scalar<si64>>) {
  %zero = index.constant 0
  %lb = kgen.param.constant = <lb>
  %ub = kgen.param.constant = <ub>
  %step = kgen.param.constant = <step>
  zap.print "values:\n"()
  scf.for %i = %lb to %ub step %step {
    %ptr = pop.offset %buf[%i] : !pop.pointer<scalar<si64>>
    %v0 = pop.load %ptr : !pop.pointer<scalar<si64>>
    %v = pop.simd.extractelement %v0[%zero] : !pop.simd<1, si64>
    // Cast the index to i64
    %is = index.casts %i : index to si64
    zap.print "  buf[%lli] = %lli\n"(%is, %v) : si64, !pop.simd<1, si64>
  }
  kgen.return
}

kgen.generator @test_print() {
  %0 = pop.global_constant(#M.dense_array<0, 11, 22, 33> : !M.array<4xsi64>) : !pop.array<4, scalar<si64>>
  %1 = pop.pointer.bitcast %0 : !pop.pointer<array<4, scalar<si64>>> to !pop.pointer<scalar<si64>>
  kgen.call @impl<lb = 0, ub = 4, step = 2>(%1) : (!pop.pointer<scalar<si64>>) -> ()
  kgen.call @impl<lb = 1, ub = 4, step = 1>(%1) : (!pop.pointer<scalar<si64>>) -> ()
  kgen.return
}

kgen.export [@test_print]

// CHECK: values:
// CHECK-NEXT: buf[0] = 0
// CHECK-NEXT: buf[2] = 22
// CHECK-NEXT: values:
// CHECK-NEXT: buf[1] = 11
// CHECK-NEXT: buf[2] = 22
// CHECK-NEXT: buf[3] = 33
// CHECK-NEXT: 'test_print' finished
