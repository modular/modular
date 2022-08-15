// RUN: kgen-opt -lower-hlkgen %s -verify-diagnostics -split-input-file -o /dev/null 

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf(%arg0: si32)

// expected-error @+1 {{'kgen.generator' op generator has 2 arguments but interface expects 1}}
hlkgen.generator @impl(%arg0 : f64, %arg1 : f64) implements @itf {
  kgen.return
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf(%arg0: i32)

// expected-error @+1 {{argument #0 has type 'i12' but interface expected type 'i32'}}
hlkgen.generator @impl(%arg0 : i12) implements @itf {
  kgen.return
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf(%arg0: !meta.buffer<?, f32>)

// expected-error @+1 {{argument #0: dynamic `?` value cannot have static constraint: '4 : index'}}
hlkgen.generator @impl(%arg0 : !meta.buffer<4, f32>) implements @itf {
  kgen.return
}
