// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp | FileCheck %s

// A buffer typed by a parameter is allocated after the op declaring the
// parameter, even when that op is deeper in the dataflow than the buffer.

// CHECK-LABEL: mgp.model @declare_param_before_use
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK-NOT:   mgp.buffer.alloc %[[CH]],
// CHECK:       mogg.routine.async
// CHECK:       mgp.buffer.alloc %[[CH]],
mo.graph @declare_param_before_use<N>(%n: !mo.tensor<[N], si64>) -> !mo.tensor<[P], ui8> {
  %q = mo.rebind<() -> Q>(%n) : !mo.tensor<[N], si64> -> !mo.tensor<[Q], si64>
  %a = mo.add(%q, %q) : !mo.tensor<[Q], si64>
  %p = mo.rebind<() -> P>(%a) : !mo.tensor<[Q], si64> -> !mo.tensor<[P], si64>
  %buf = mo.buffer.create : !mo.buffer<[P], ui8>
  %ch0 = mo.chain.create(%buf) : !mo.buffer<[P], ui8>
  %v, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[P], ui8>, !mo.chain) -> (!mo.tensor<[P], ui8>, !mo.chain)
  mo.output %v : !mo.tensor<[P], ui8>
}

// The same through a kgen.param.declare: the buffer is typed by S, which a
// declare derives from P, so the alloc still waits for the rebind declaring P.

// CHECK-LABEL: mgp.model @declare_derived_param_before_use
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK-NOT:   mgp.buffer.alloc %[[CH]],
// CHECK:       mogg.routine.async
// CHECK-NOT:   mgp.buffer.alloc %[[CH]],
// CHECK:       mogg.routine.async
// CHECK:       mgp.buffer.alloc %[[CH]],
mo.graph @declare_derived_param_before_use<N>(%n: !mo.tensor<[N], si64>) -> !mo.tensor<[S], ui8> {
  %q = mo.rebind<() -> Q>(%n) : !mo.tensor<[N], si64> -> !mo.tensor<[Q], si64>
  kgen.param.declare S: scalar<si64> = <add(P, 1)>
  %buf = mo.buffer.create : !mo.buffer<[S], ui8>
  %a0 = mo.add(%q, %q) : !mo.tensor<[Q], si64>
  %a = mo.mul(%a0, %a0) : !mo.tensor<[Q], si64>
  %p = mo.rebind<() -> P>(%a) : !mo.tensor<[Q], si64> -> !mo.tensor<[P], si64>
  %ch0 = mo.chain.create(%buf) : !mo.buffer<[S], ui8>
  %v, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[S], ui8>, !mo.chain) -> (!mo.tensor<[S], ui8>, !mo.chain)
  mo.output %v : !mo.tensor<[S], ui8>
}

// And through two declares: T is derived from S, which is derived from P.

// CHECK-LABEL: mgp.model @declare_chain_param_before_use
// CHECK:       } execute (%[[CH:[^:]*]]: !mef.chain
// CHECK-NOT:   mgp.buffer.alloc %[[CH]],
// CHECK:       mogg.routine.async
// CHECK-NOT:   mgp.buffer.alloc %[[CH]],
// CHECK:       mogg.routine.async
// CHECK:       mgp.buffer.alloc %[[CH]],
mo.graph @declare_chain_param_before_use<N>(%n: !mo.tensor<[N], si64>) -> !mo.tensor<[T], ui8> {
  %q = mo.rebind<() -> Q>(%n) : !mo.tensor<[N], si64> -> !mo.tensor<[Q], si64>
  kgen.param.declare T: scalar<si64> = <add(S, 1)>
  kgen.param.declare S: scalar<si64> = <add(P, 1)>
  %buf = mo.buffer.create : !mo.buffer<[T], ui8>
  %a0 = mo.add(%q, %q) : !mo.tensor<[Q], si64>
  %a = mo.mul(%a0, %a0) : !mo.tensor<[Q], si64>
  %p = mo.rebind<() -> P>(%a) : !mo.tensor<[Q], si64> -> !mo.tensor<[P], si64>
  %ch0 = mo.chain.create(%buf) : !mo.buffer<[T], ui8>
  %v, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[T], ui8>, !mo.chain) -> (!mo.tensor<[T], ui8>, !mo.chain)
  mo.output %v : !mo.tensor<[T], ui8>
}
