// RUN: kgen -execute -func="test_print:()" %s | FileCheck %s

kgen.generator @impl<lb, ub, step>(%buf: !zap.buffer<?, si64>) {
  %zero = index.constant 0
  %lb = kgen.param.constant = <lb>
  %ub = kgen.param.constant = <ub>
  %step = kgen.param.constant = <step>
  zap.print "values:\n"()
  scf.for %i = %lb to %ub step %step {
    %v0 = zap.buffer.load %buf[%i] : !zap.buffer<?, si64>, !pop.simd<1, si64>
    %v = pop.simd.extractelement %v0[%zero] : !pop.simd<1, si64>
    // Cast the index to i64
    %is = index.casts %i : index to si64
    zap.print "  buf[%lli] = %lli\n"(%is, %v) : si64, !pop.simd<1, si64>
  }
  kgen.return
}

kgen.generator @test_print() {
  %0 = zap.buffer.constant(#M.dense_array<0, 11, 22, 33> : !M.array<4xsi64>) : si64
  %1 = zap.buffer.bitcast %0 : !zap.buffer<4, si64> to !zap.buffer<?, si64>
  kgen.call @impl<lb = 0, ub = 4, step = 2>(%1) : (!zap.buffer<?, si64>) -> ()
  kgen.call @impl<lb = 1, ub = 4, step = 1>(%1) : (!zap.buffer<?, si64>) -> ()
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
