# MAP dialect parsing tests

Roundtrip tests that validate the `map` dialect IR (the `map.iter` and `imap`
op families) parses and prints back unchanged.

Add `.mlir` files here; they are picked up automatically by the `**/*.mlir`
glob in `GraphCompiler/test/mo-opt/BUILD.bazel`. The usual roundtrip pattern is:

```text
// RUN: mo-opt %s | FileCheck %s
```

There are no `map` ops or attributes defined yet, so there is nothing to
roundtrip. Populate this directory once the first ops land.
