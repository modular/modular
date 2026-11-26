# Code examples and tests for calling Mojo from Python

This directory contains code examples and tests for the
[Calling Mojo from Python](../../../../manual/python/mojo-from-python.mdx)
page of the Mojo Manual.

Unlike most directories under `docs/code`, the `.mojo` files here are Python
extension modules rather than standalone programs. Each one defines a
`PyInit_<module>` entry point and no `main()`, so Python imports it.

Page code, verbatim:

- `primes.py`, `fast_primes.mojo`, and `main.py` are the prime-counting
  walkthrough. `fast_primes.mojo` merges the page's two blocks.
- `fast_stats.mojo` and `report.py` are the `RunningStats` example.
  `fast_stats.mojo` merges the page's two blocks.

Tests:

- `test.py` checks the page's claims from a real Python entry point, through
  `mojo.importer`. Its docstring maps each claim to a test and lists what
  isn't tested.
- `tests.mojo` checks the claims that need Mojo-side machinery. It imports
  `fast_primes.mojo` and `fast_stats.mojo` directly, so it tests the page's
  code rather than a copy.
- `edges_module.mojo` holds bindings for the "Sharp edges" and "Worth knowing"
  claims, which the page states without a full example.
- `compile_checks/` holds small modules that `test.py` imports to confirm
  which binding forms compile and which make `import` raise `ImportError`.
  Several of them don't compile on purpose.

The `BUILD.bazel` file defines `main_test`, `report_test`, `interop_test`
(`test.py`), and `tests_test` (`tests.mojo`).
