// RUN: not mo-opt %s 2>&1 | FileCheck %s

// CHECK: 'mo.overlay_test' op overlay verifier ran
kgen.func @overlay_test(%chain: !mo.chain) -> !mo.chain {
  %0 = mo.overlay_test %chain
  hlcf.return %0 : !mo.chain
}
