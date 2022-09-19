// RUN: kgen -execute -func="test_print:():%t.o" %s | FileCheck %s

kgen.generator @impl<lb, ub, step>(%buf: !meta.buffer<?, si64>) {
  %lb = kgen.param.constant = <lb>
  %ub = kgen.param.constant = <ub>
  %step = kgen.param.constant = <step>
  zap.print "values:\n"()
  scf.for %i = %lb to %ub step %step {
    %v = zap.buffer.load %buf[%i] : !meta.buffer<?, si64>
    // Cast the index to i64
    %is = index.casts %i : index to si64
    zap.print "  buf[%lli] = %lli\n"(%is, %v) : si64, !meta.scalar<si64>
  }
  kgen.return
}

kgen.generator public @test_print() {
  %0 = zap.buffer.constant(dense<[0, 11, 22, 33]> : tensor<4xsi64>) : si64
  %1 = meta.buffer.convert %0 : !meta.buffer<4, si64> to !meta.buffer<?, si64>
  kgen.call @impl<lb = 0, ub = 4, step = 2>(%1) : (!meta.buffer<?, si64>) -> ()
  kgen.call @impl<lb = 1, ub = 4, step = 1>(%1) : (!meta.buffer<?, si64>) -> ()
  kgen.return
}

// CHECK: values:
// CHECK-NEXT: buf[0] = 0
// CHECK-NEXT: buf[2] = 22
// CHECK-NEXT: values:
// CHECK-NEXT: buf[1] = 11
// CHECK-NEXT: buf[2] = 22
// CHECK-NEXT: buf[3] = 33
// CHECK-NEXT: 'test_print' finished
