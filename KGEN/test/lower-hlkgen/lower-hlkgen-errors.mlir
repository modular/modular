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


// -----

kgen.generator.interface @bufitf<size, ty: dtype -> index>(!meta.buffer<size, ty>) -> index

// This implementation infers that the ty argument must be f32.

// expected-error @+2 {{constraint contradiction detected: "argument #0 specifies 'ty' = f32"}}
// expected-note @+1 {{generator declared here}}
hlkgen.generator @impl<size, ty: dtype -> index>(%arg0: !meta.buffer<size, f32>) -> index
  // This has an explicit constraint saying it must be f64.
  // expected-note @below {{previously constrained "someone told us 'ty' should be f64 dontcha know"}}
  constraints <[eq(:dtype ty, f64), "someone told us 'ty' should be f64 dontcha know"]>
  implements @bufitf {
  %0 = meta.buffer.size %arg0 : !meta.buffer<size, f32>
  kgen.return<add(size, 2)> %0 : index
}


// -----

// expected-note @below {{generator declared here}}
hlkgen.generator @impl<size>()
  // This has an explicit constraint saying it must be f64.
  // expected-note @below {{previously constrained "someone told us size smells like 3"}}
  constraints <[eq(size, 3), "someone told us size smells like 3"],
  // expected-error @below {{constraint contradiction detected: "I sez that size should be 12!"}}
               [eq(size, 12), "I sez that size should be 12!"]> {
  kgen.return
}

// -----

// expected-note @below {{generator declared here}}
hlkgen.generator @impl<size>()
  constraints <
    // expected-note @below {{previously constrained "three'or'fore"}}
    [in(size, [3, 4]), "three'or'fore"],
    // expected-error @below {{constraint contradiction detected: "someone told us size smells like 3 but can we believe them?"}}
    [eq(size, 5), "someone told us size smells like 3 but can we believe them?"]
   > {
  kgen.return
}

// -----

// expected-note @below {{generator declared here}}
hlkgen.generator @impl<size>()
  constraints <
    // expected-note @below {{previously constrained "someone told us size smells like 3 but can we believe them?"}}
    [eq(size, 5), "someone told us size smells like 3 but can we believe them?"],
    // expected-error @below {{constraint contradiction detected: "three'or'fore"}}
    [in(size, [3, 4]), "three'or'fore"]
   > {
  kgen.return
}

// -----

// expected-note @+1 {{generator declared here}}
hlkgen.generator @impl<size>()
  constraints <
    // expected-note @below {{previously constrained "seven ate 9"}}
    [in(size, [7, 8]), "seven ate 9"],
    // expected-error @below {{constraint contradiction detected: "three'or'fore"}}
    [in(size, [3, 4]), "three'or'fore"]
   > {
  kgen.return
}

// -----

// expected-note @+1 {{generator declared here}}
hlkgen.generator @equality<a, b>()
  constraints <
    // expected-error @+1 {{constraint contradiction detected: "a is one, and a and b are same"}}
    [eq(a, 1), "a is one"],
    // expected-note @+1 {{previously constrained "b is two"}}
    [eq(b, 2), "b is two"],
    [eq(a, b), "a and b are same"]
   > {
  kgen.return
}


// -----

kgen.generator.interface @itf<ty: dtype>(!meta.scalar<f64>) -> !meta.scalar<ty>

// This implementation infers that the ty argument must be f32.

// expected-note @+1 {{generator declared here}}
hlkgen.generator @impl<ty: dtype>(%arg0: !meta.scalar<f64>) -> !meta.scalar<f64>
  // This has an explicit constraint saying it must be si32.
  // expected-note @below {{previously constrained "'ty' looks lovely as si32"}}
  constraints <[eq(:dtype ty, si32), "'ty' looks lovely as si32"]>
  implements @itf {
  
// expected-error @+1 {{constraint contradiction detected: "result #0 specifies 'ty' = f64"}}
  kgen.return %arg0: !meta.scalar<f64>
}

// -----

// expected-note @+1 {{interface defined here}}
kgen.generator.interface @itf<ty: dtype>() 

// This implementation infers that the ty argument must be f32.

// expected-error @+1 {{input parameter "ty" has type 'i32' but interface expects '!kgen.dtype'}}
hlkgen.generator @impl<ty: i32>() implements @itf {
  kgen.return
}

// expected-error @+1 {{input parameter "size" is unexpected by interface}}
hlkgen.generator @impl2<ty: dtype, size>() implements @itf {
  kgen.return
}

// expected-error @+1 {{missing interface input parameter "ty" of type '!kgen.dtype'}}
hlkgen.generator @impl3() implements @itf {
  kgen.return
}
