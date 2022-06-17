// This is the library of kernel generators used by the tests in this directory.
// It is also run as a test, which just verifies that it parses correctly.

// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -o /dev/null

kgen.generator @kernel1(%arg0: si32) -> si32 {
  "someop" () : () -> ()
  kgen.return %arg0 : si32
}

kgen.generator @kernel2(%arg0: si32) -> si32 {
  "someop" () : () -> ()
  kgen.return %arg0 : si32
}

