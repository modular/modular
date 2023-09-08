// RUN: kgen-opt -split-input-file -constraint-reduction -verify-diagnostics %s

// expected-note @below {{generator declared here}}
kgen.generator @impl<size>()
  // This has an explicit constraint saying it must be f64.
  // expected-note @below {{previously constrained "someone told us size smells like 3"}}
  constraints <[eq(size, 3), "someone told us size smells like 3"],
  // expected-error @below {{constraint contradiction detected: "I sez that size should be 12!"}}
               [eq(size, 12), "I sez that size should be 12!"]> {
  kgen.return
}

// -----

// expected-note @below {{generator declared here}}
kgen.generator @impl<size>()
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
kgen.generator @impl<size>()
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
kgen.generator @impl<size>()
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
kgen.generator @equality<a, b>()
  constraints <
    // expected-error @+1 {{constraint contradiction detected: "a is one, and a and b are same"}}
    [eq(a, 1), "a is one"],
    // expected-note @+1 {{previously constrained "b is two"}}
    [eq(b, 2), "b is two"],
    [eq(a, b), "a and b are same"]
   > {
  kgen.return
}
