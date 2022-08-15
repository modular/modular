// RUN: kgen %s -verify-diagnostics

// expected-error@-3 {{could not sense the contents of this file, cannot proceed}}
"foo.funclikeop"() ({
  "a.b"() : () -> ()
}) : () -> ()
