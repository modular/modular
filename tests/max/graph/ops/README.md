# Python graph API op testing principles

This document outlines some principles for tests we want to write to validate
the behavior of graph API operations.

For each op, we expect tests that:

1. Use hypothesis to check the valid range of inputs to the op, asserting the
   invariants that we expect for any valid input.
2. Use hypothesis to check invalid inputs, asserting that the test raises a
   Python exception (but we don't care about the text of the exception message).
3. Each error message due to an invalid input is tested without hypothesis,
   using specific inputs to validate the full specific error message.

We _do not_ want tests here to execute the graph. These tests should build a
graph only. Any tests that execute a graph belong in the
`SDK/integration-test/python/graph` directory. Tests in this directory should be
fast to run and should probe the valid and invalid inputs to each operation.

We use the [hypothesis](https://hypothesis.readthedocs.io/) library to "fuzz
test" each operation, probing the valid and invalid inputs. We also use specific
test inputs to validate specific error messages.

Quality error messages from the graph API provide a high value to the user. They
should present enough information to the user that they can make a change to
their code to avoid the error message. Validation of error message is essential
to ensure that the user experience of the graph API remains positive across
changes to its implementation.
