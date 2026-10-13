# Code examples and tests for functions

This directory contains code examples and tests for the
[Functions](../../../manual/functions.mdx) section of the Mojo Manual.

Contents:

- `tests.mojo` is a single standalone Mojo application holding every example
  from the page together with its assertions. Code that appears in the
  documentation is delimited by `# start-<name>` and `# end-<name>` comments;
  `mojo-doc-sync` matches those regions against the fenced blocks in the
  Markdown, so their contents must stay byte-for-byte identical to the page.
- The `BUILD.bazel` file defines:
  - A `mojo_binary` target for each `.mojo` file (using the file name without
    extension).
  - A `modular_run_binary_test` target for each binary (with a `_test`
    suffix).

Two things about `tests.mojo` are load-bearing and easy to break:

- `count_many_things()` must stay the **last** definition in the file, with
  `main()` directly after it. Its documentation block spans the function
  declaration, `def main():`, and main's first statement, so anything
  inserted between them stops the block from matching.
- `main()` is deliberately not `raises`, because the same documentation block
  shows a plain `def main():`. Assertions therefore run through
  `run_tests()`, which catches any failure, reports it, and exits nonzero so
  the test target still fails.

Code blocks in the documentation that are illustrative rather than runnable —
undefined placeholder types, alternative spellings of one signature,
unsupported syntax, and deliberate compile errors — are marked `no-test` in
the Markdown and have no counterpart here.

The `closures` and `lambda` subdirectories hold the examples for the
[Closures](../../../manual/functions/closures.mdx) and
[Lambdas](../../../manual/functions/lambda.mdx) pages.
