# Mojo🔥 LSP

## Introduction

The Mojo Language Server is a productivity tool that enhances the authoring
experience of Mojo programs in editors that support the [Language Server
Protocol](https://en.wikipedia.org/wiki/Language_Server_Protocol). It provides
[editing features](https://code.visualstudio.com/api/language-extensions/programmatic-language-features),
such as code completion, diagnostics, quick fixes, hover dialogs, jump to
definition, refactoring utilities, etc.

## Getting Started with VSCode

Just run the `vscode-init` command on your terminal, which will install and
configure the **Mojo** extension on VSCode. The Language Server is part of this
extension and will be automatically launched whenever a `.mojo` or `.🔥` file is
opened.

## Development

### Testing mojo-lsp-server

Our tests live in `KGEN/unittests/mojo-lsp-server/` and are specified using the
C++ GTest framework.

You can read more about GTests in this
[primer](https://github.com/google/googletest/blob/main/docs/primer.md).

#### How to write a test

You can read `KGEN/unittests/mojo-lsp-server/SampleTest.cpp` for a sample
test with some useful explanatory comments.

#### Running the tests

In order to run these tests, you just need to execute
`build check-mojo-lsp-server`, which will run all LSP-related tests.

#### Inspecting the LSP traffic

You can invoke the tests with
`PRESERVE_LSP_IO_FILES=1 build check-mojo-lsp-server`, which will indicate the
test suite to print to stderr the IO files used to communicate with the Language
Server upon failures. In this case, these files are not cleaned up upon
termination, and you can inspect them to debug your issues or even invoke the
Language Server manually with
`cat /path/to/lsp_stdout | mojo-lsp-server -mojo-test`.

### mojo-lsp-simple-client

This little utility can be used to launch an LSP server and simulate some actions
that the user would do on the IDE. This tool can be extremely useful for
debugging issues.

### Debugging

`mojo-lsp-server` offers the `--attach-debugger-on-startup` argument
invocation to start a debug session on VSCode attaching to the Language Server.

There are two main ways to trigger a real debug session that uses this
capability:

- Via the VS Code command `Developer: restart the Mojo LSP Server and Attach the
  debugger to it`. This can be useful for debugging simple issues, but
  relaunching debug sessions requires several manual interactions.
- Via the `mojo-lsp-simple-client`, which offers the `-attach-debugger` option.
  This can be more convenient for automating a LSP session and rerun it
  repeatedly.

### Building

If you need to build the VS Code extension from source, you can use
`vscode-build` to compile the extension and install it locally.
