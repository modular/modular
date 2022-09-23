// RUN: kgen-opt %s -verify-diagnostics -split-input-file -o /dev/null

// expected-error @+1 {{expected attribute value}}
kgen.func @unknown_size_simd(%arg0: !pop.simd<?, f32>) -> !pop.simd<?, f32> {
  kgen.return %arg0 : !pop.simd<?, f32>
}

// -----

// expected-error @+1 {{expected attribute value}}
kgen.func @unknown_type_simd(%arg0: !pop.simd<4, ?>) -> !pop.simd<4, ?> {
  kgen.return %arg0 : !pop.simd<4, ?>
}
