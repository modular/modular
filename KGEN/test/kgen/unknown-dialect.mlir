// RUN: kgen %s -verify-diagnostics

// expected-error@+1 {{operation has unknown dialect, this is not supported}}
"foo.funclikeop"() ({
  "a.b"() : () -> ()
}) : () -> ()
