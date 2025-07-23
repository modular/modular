# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Overview

This is the Mojo programming language repository, containing the Mojo standard
library, examples, and documentation. Mojo is a programming language that
bridges the gap between research and production by combining Python syntax and
ecosystem with systems programming and metaprogramming features.

## Essential Commands

### Building the Standard Library

```bash
./stdlib/scripts/build-stdlib.sh
```

This creates a `build/stdlib.mojopkg` file in the repo root.

### Running Tests

```bash
# Run all tests
./stdlib/scripts/run-tests.sh

# Run specific test file
./stdlib/scripts/run-tests.sh ./stdlib/test/utils/test_span.mojo

# Run tests in specific directory
./stdlib/scripts/run-tests.sh ./stdlib/test/utils

# Run specific test suites with lit directly
lit -sv stdlib/test/builtin stdlib/test/collections
```

Tests are run with `-D ASSERT=all` by default.

### Running Benchmarks

Read the `mojo/stdlib/benchmarks/README.md` for details on how to run benchmarks.

### Code Formatting

```bash
# Format all Mojo files
mojo format ./

# Format is automatically applied via pre-commit hooks
```

### Documentation Validation

```bash
mojo doc --diagnose-missing-doc-strings --validate-doc-strings \
  -o /dev/null stdlib/stdlib/
```

## High-Level Architecture

### Repository Structure

- `stdlib/`: Mojo standard library implementation
  - `stdlib/stdlib/`: Source code organized by module (builtin, collections,
    memory, etc.)
  - `stdlib/test/`: Unit tests mirroring the source structure
  - `stdlib/benchmarks/`: Performance benchmarks
  - `stdlib/scripts/`: Build and test scripts
  - `stdlib/docs/`: Technical documentation
- `docs/`: User-facing documentation and manual
- `examples/`: Mojo example code
- `integration-test/`: Integration tests
- `proposals/`: RFC-style proposal documents

### Key Development Patterns

#### Import System

When developing standard library code that imports other standard library
modules:

```bash
# Build the standard library first
./stdlib/scripts/build-stdlib.sh

# Use the locally built stdlib
MODULAR_MOJO_MAX_IMPORT_PATH=../build mojo main.mojo
```

#### Test Structure

- Tests use the `lit` tool with `FileCheck` for validation
- Migrating to use `testing` module assertions (`assert_equal`, `assert_true`,
  etc.)
- Test files must start with: `# RUN: %mojo %s`

#### Memory Management

- Follow value semantics and ownership conventions
- Use `Reference` types and lifetimes in APIs
- Prefer `AnyType` over `AnyTrivialRegType` (except for MLIR interactions)

## Development Workflow

1. **Branch from `main`**: Always work off the main branch (for nightly builds)
2. **Install nightly Mojo**: Use the nightly build for development
3. **Use nightly VS Code extension**: Install the Mojo nightly extension
4. **Small PRs**: Keep pull requests under 100 lines when possible
5. **Test your changes**: Run relevant tests before submitting
6. **Format code**: Ensure code passes `mojo format`
7. **Document APIs**: Add docstrings following the style guide

## Critical Notes

- **Do NOT** commit secrets or API keys
- **Do NOT** break existing APIs without discussion
- **Do NOT** add dependencies to the stdlib module
- **Always** sign commits with `Signed-off-by` (use `git commit -s`)
- **Always** follow the Apache License v2.0 with LLVM Exceptions
- Prefer using Batch tool for multiple file operations to reduce context usage
- When making multiple bash calls, use Batch to run them in parallel

## Performance Considerations

- Performance improvements must include benchmarks
- Don't sacrifice readability for minor performance gains
- Use the benchmarking infrastructure to track regressions

## Platform Support

- Linux x86_64 and aarch64
- macOS ARM64
- Windows is not currently supported

## Internal APIs

The following are private/internal APIs without backward compatibility
guarantees:

- MLIR dialects (`pop`, `kgen`, `lit`)
- Compiler runtime features (prefixed with `KGEN_CompilerRT_`)

## Contribution Guidelines

- Bug fixes should include reproducing tests
- New features should align with the roadmap
- All code must have corresponding tests
- Follow the coding style guide strictly
- Use pre-commit hooks for automatic formatting
