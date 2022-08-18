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

kgen.generator.interface @bufitf<size, ty: dtype -> xyz>(!meta.buffer<size, ty>) -> index

// This implementation infers that the ty argument must be f32.

// expected-error @+3 {{constraint contradiction detected: "argument #0 specifies 'ty' = f32"}}
// expected-note @+2 {{previously constrained "someone told us 'ty' should be f64 dontcha know"}}
// expected-note @+1 {{generator declared here}}
hlkgen.generator @impl<size, ty: dtype -> xyz>(%arg0: !meta.buffer<size, f32>) -> index 
  // This has an explicit constraint saying it must be f64.
  constraints <eq(:dtype ty, f64), "someone told us 'ty' should be f64 dontcha know">
  implements @bufitf {
  %0 = meta.buffer.size %arg0 : !meta.buffer<size, f32>
  kgen.return<xyz = add(size, 2)> %0 : index
}


// -----

// expected-error @+3 {{constraint contradiction detected: "I sez that size should be 12!"}}
// expected-note @+2 {{previously constrained "someone told us size smells like 3"}}
// expected-note @+1 {{generator declared here}}
hlkgen.generator @impl<size>()
  // This has an explicit constraint saying it must be f64.
  constraints <eq(size, 3), "someone told us size smells like 3",
               eq(size, 12), "I sez that size should be 12!"> {
  kgen.return
}

// -----

// expected-error @+3 {{constraint contradiction detected: "someone told us size smells like 3 but can we believe them?"}}
// expected-note @+2 {{previously constrained "three'or'fore"}}
// expected-note @+1 {{generator declared here}}
hlkgen.generator @impl<size>()
  constraints <
    in(size, [3, 4]), "three'or'fore",
    eq(size, 5), "someone told us size smells like 3 but can we believe them?"
   > {
  kgen.return
}

// -----

// expected-error @+3 {{constraint contradiction detected: "three'or'fore"}}
// expected-note @+2 {{previously constrained "someone told us size smells like 3 but can we believe them?"}}
// expected-note @+1 {{generator declared here}}
hlkgen.generator @impl<size>()
  constraints <
    eq(size, 5), "someone told us size smells like 3 but can we believe them?",
    in(size, [3, 4]), "three'or'fore"
   > {
  kgen.return
}

// -----

// expected-error @+3 {{constraint contradiction detected: "three'or'fore"}}
// expected-note @+2 {{previously constrained "seven ate 9"}}
// expected-note @+1 {{generator declared here}}
hlkgen.generator @impl<size>()
  constraints <
    in(size, [7, 8]), "seven ate 9",
    in(size, [3, 4]), "three'or'fore"
   > {
  kgen.return
}

