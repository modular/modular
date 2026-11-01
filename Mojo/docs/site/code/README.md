# Mojo documentation code examples

This directory includes code examples used in the Mojo Manual, the language
reference, and the tools documentation, to ensure those code snippets are
tested. Each subdirectory corresponds to a documentation page, and each example
has a Bazel test that runs it in CI.

## Running with Bazel

Run a single example's test from the Modular internal repo:

```sh
bt //Mojo/docs/site/code/manual/basics:hello_world_test
```

Or run every example under a topic (or the whole tree):

```sh
bt //Mojo/docs/site/code/manual/basics/...
bt //Mojo/docs/site/code/...
```

Use `br` to run an example directly instead of as a test:

```sh
br //Mojo/docs/site/code/manual/basics:hello_world
```

## Examples that are supposed to fail

Some documentation examples teach what the compiler rejects. Those belong in a
file too, so the rejection and the diagnostic the page quotes are both checked
in CI.

Use a distinct file-name suffix for compile-time and runtime errors, and
exclude those files from the glob pattern that selects the regular tests:

| Suffix                 | What it asserts                                                  |
|------------------------|------------------------------------------------------------------|
| `*_compile_error.mojo` | `mojo` exits non-zero and the diagnostic matches                 |
| `*_runtime_error.mojo` | The binary builds, runs, exits non-zero, and the message matches |

Currently `manual/metaprogramming/constraints/` includes examples of
compile-time errors and the associated build rules. No docs tests use the
`*_runtime_error.mojo` suffix. Runtime errors can mostly be handled using the
`assert_` functions from the `std.testing` package. The exception would be error
conditions that abort—such as out-of-bounds access on a collection.

A compile error can't use `mojo_filecheck_test`, because that macro runs the
binary—the program has to build first. Compile errors use `lit_tests`
instead, with a `# RUN:` line driving the compiler directly:

```mojo
# RUN: not %mojo %s 2>&1 | FileCheck %s

# CHECK: error: invalid call to 'pow2': violated constraint
```

Put the `# RUN:` and `# CHECK:` lines above the `# start-` region marker, so
the region stays clean enough to paste into the `.mdx` page.

Capture the expected text from the compiler rather than writing it by
hand—diagnostics go to stderr:

```sh
mojo path/to/example_compile_error.mojo 2>&1
```

The directory's `BUILD.bazel` globs the error files separately and excludes
them from `MOJO_SRCS`, since a `mojo_binary` target for a file that can't
compile would break `bazel build //...`. See
`manual/metaprogramming/constraints/BUILD.bazel` for the shape.

## Contributing

If you see something in the documentation or the code examples that is incorrect
or could be improved, we'd love to accept your contributions.

Be aware that code from this directory is **not** automatically included in the
corresponding documentation file. If you contribute a change to a code example,
please update the code example in the related documentation, as well as any
explanatory text.

Be aware that we don't provide tools to generate a preview of the website,
because the Mojo docs are built along with other content that's not included in
this repo. As such, we recommend you preview your edits in an IDE that can
render Markdown and MDX files, such as VS Code, including the
[VS Code environment in GitHub](https://github.dev/modular/modular/blob/main/).

For more information about how to contribute, see the
[Contributor Guide](../../../CONTRIBUTING.md).
