This directory contains code examples for the
[Get started with Mojo](../../../../docs/manual/get-started.mdx) tutorial of the
Mojo Manual. These are only "checkpoints" for various steps of the tutorial, to
ensure that the code for these intermediate steps is correct. The complete
reference solution for the tutorial is in
[examples/mojo/life](../../../../../examples/mojo/life/) directory, which is
deployed to the public GitHub repo. Checkpoints aren't needed for steps that
simply "accrete" functionality or for the final result, so there isn't a
checkpoint for each step of the tutorial.

Contents:

- `life_step5.mojo` and `gridv1_step5.mojo`: The result of completing Step 5 of
  the tutorial.

- `life_step7.mojo` and `gridv1_step7.mojo`: The result of completing Step 7 of
  the tutorial.

- `life_step8.mojo` and `gridv1_step8.mojo`: The result of completing Step 8 of
  the tutorial.

- `life_step9.mojo` and `gridv1_step9.mojo`: The result of completing Step 9 of
  the tutorial.

- The `BUILD.bazel` file defines:

  - Runnable binary targets:
    - `life_step5`
    - `life_step7`
    - `life_step8`
    - `life_step9`

  - Test targets:
    - `life_step5_test`
    - `life_step7_test`
    - `life_step8_test`
    - `life_step9_test`
