// RUN: kgen-opt -force-inline -verify-diagnostics

kgen.func @ok_to_inline() always_inline {
  kgen.return
}

// expected-note @below {{to function marked 'always_inline' here}}
// expected-note @below {{back to function here}}
kgen.func @circular.a() always_inline {
  // expected-note @below {{through call here}}
  kgen.call @circular.b() : () always_inline -> ()
  kgen.return
}

// expected-note @below {{to function marked 'always_inline' here}}
kgen.func @circular.b() always_inline {
  // expected-note @below {{through call here}}
  kgen.call @circular.c() : () always_inline -> ()
  kgen.return
}

// expected-note @below {{to function marked 'always_inline' here}}
kgen.func @circular.c() always_inline {
  // expected-note @below {{call here recurses}}
  kgen.call @circular.a() : () always_inline -> ()
  kgen.return
}

// expected-error @below {{function has recursive call to 'always_inline' function}}
kgen.func @top0() {
  // expected-note @below {{through call here}}
  kgen.call @circular.a() : () always_inline -> ()
  kgen.call @ok_to_inline() : () always_inline -> ()
  kgen.return
}
