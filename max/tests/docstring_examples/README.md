# Docstring example testing

Runs `.. code-block:: python` examples in `max/python/max` docstrings as pytest
tests, using [Sybil](https://sybil.readthedocs.io). Coverage is opt-in per Bazel
target: setting `test_docstring_examples = True` on a `modular_py_library`
generates a companion `<name>.docstring_examples` test that runs **every**
docstring code-block example in that target. An example passes if it executes
without raising. Keep the visible example focused on usage and put result checks
in a hidden `.. invisible-code-block: python`, so the rendered docs stay clean
while the example is still verified (no `>>>` REPL blocks). Private API
(underscore-named modules, classes, and functions) is never collected.

## Opting a module in

Collection is per target, all-or-nothing: the generated test runs Sybil over
the target's own `srcs` only (not its dependencies), so every docstring
code-block example in those sources must run. There is no per-file selection. To
opt a module in, in one PR:

1. Make all of its examples self-contained and runnable, or skip the ones that
   can't (see [Skipping an example](#skipping-an-example)). "Self-contained"
   means each example imports the names it uses and opens its own contexts.
2. Set `test_docstring_examples = True` on its `modular_py_library` target. If
   examples import a runtime dependency the library itself can't carry (for
   example `max.engine`, which depends on `graph`), add it to
   `docstring_example_deps` on that target.

Run the generated test the same way as any other, for example
`./bazelw test //max/python/max/<pkg>:<name>.docstring_examples`. The test fails
if the target collects zero examples, so a scoping or dependency bug can't pass
silently.

## Skipping an example

Skip an example that can't run in CI, such as pseudo-code or one that needs
files, weights, or a running server (`max serve` and `generate` are too slow
for CI). Explain the reason in a comment above the skip:

```text
.. Skipped: needs a running server, too slow for CI.
.. skip: next
```

## Write testable examples

Write a plain, copy-pastable example in a visible `code-block`, then put the
result checks in a following `invisible-code-block`. Sybil executes both
blocks, sharing one namespace; Sphinx renders only the first:

```text
.. code-block:: python

    from max.dtype import DType
    from max.engine import InferenceSession
    from max.graph import DeviceRef, Graph, ops

    device = DeviceRef.CPU()
    with Graph("add_example") as graph:
        graph.output(
            ops.add(
                ops.constant([1.0, 2.0], DType.float32, device=device),
                ops.constant([3.0, 4.0], DType.float32, device=device),
            )
        )

    model = InferenceSession().load(graph)
    result = model.execute()[0]

.. invisible-code-block: python

    import numpy as np

    assert np.allclose(result.to_numpy(), [4.0, 6.0])
```

Sybil runs `assert` statements in a visible `code-block` too, so a check the
reader can see is fine. Use `invisible-code-block` only when the check should be
verified but not rendered. Import every name the example uses: the example is
the rendered documentation, so it must run as-is if a reader copies it.
