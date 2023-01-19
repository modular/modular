// RUN: kgen-opt -force-inline -verify-diagnostics

kgen.func @ok_to_inline() force_inline {
  kgen.return
}

// expected-note @below {{to function marked 'force_inline' here}}
// expected-note @below {{back to function here}}
kgen.func @circular.a() force_inline {
  // expected-note @below {{through call here}}
  kgen.call @circular.b() : () force_inline -> ()
  kgen.return
}

// expected-note @below {{to function marked 'force_inline' here}}
kgen.func @circular.b() force_inline {
  // expected-note @below {{through call here}}
  kgen.call @circular.c() : () force_inline -> ()
  kgen.return
}

// expected-note @below {{to function marked 'force_inline' here}}
kgen.func @circular.c() force_inline {
  // expected-note @below {{call here recurses}}
  kgen.call @circular.a() : () force_inline -> ()
  kgen.return
}

// expected-error @below {{function has recursive call to 'force_inline' function}}
kgen.func @top0() {
  // expected-note @below {{through call here}}
  kgen.call @circular.a() : () force_inline -> ()
  kgen.call @ok_to_inline() : () force_inline -> ()
  kgen.return
}
