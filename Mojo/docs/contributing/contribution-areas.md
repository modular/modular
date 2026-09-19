# Contribution areas

We plan to gradually expand the types of contributions we accept. This gives us
time to mature our contribution processes, and it lets the Modular team grow and
adapt to the increased input and the review burden that comes with it.

The sections below describe the contributions we accept today. Before you start
work, check that the part of the codebase you want to improve is covered here,
then follow the [contribution process](contribution-process.md).

Keep each pull request inside a single area. See [keep each pull request to one
contribution](issue-pr-etiquette.md#keep-each-pull-request-to-one-contribution).

## Compiler

The compiler team currently accepts *bug fixes only*.

A change should:

- Fix incorrect or misleading diagnostics, crashes, or mis-compiles where the
  current behavior is clearly not intentional, meaning it deviates from the
  generally perceived semantics of the language, whether explicit or implicit.
- Have a user-observable effect.

A change should not:

- Alter the intended Mojo language semantics.
- Alter the intended IR semantics for our MLIR dialects.
- Alter the compilation pipeline.

For technical details on working in the compiler codebase, see the
[compiler contributor docs](compiler/README.md).

## Standard library

The standard library team currently accepts the following types of changes:

- Well-documented bug fixes submitted with code reproducing the issue in a test
  or benchmark.
- Performance improvements that don't sacrifice code readability or
  maintainability and are accompanied by benchmarks.
- Improvements to standard library documentation.
- Improvements to test coverage.
- Porting of tests from `FileCheck` to using `assert_*` functions from the
  `testing` module.
- Changes that address security vulnerabilities.

The following is a non-exhaustive list of changes we don't accept:

- Changes that don't align with the published roadmap or the core principles of
  the standard library.
- Changes to the `math` module, until more thorough performance benchmarking is
  available.
- Code without tests, especially for core primitives.
- Changes that break existing API or implicit behavior semantics.
- Changes where a contributor's favorite feature or system isn't being used and
  they submit a change unilaterally switching the project to use it.
- Adding support for esoteric platforms.
- Adding dependencies to the codebase.
- Broad formatting or refactoring changes.
- Changes that need broad community consensus.
- Changes where contributors aren't responsive.
- Adding an entire new module without going through the proposal process.

If you're interested in making a more significant change to the standard
library, start with the
[standard library proposal process](stdlib/proposal-process.md).

For technical details on developing for the standard library, see the following
documents:

- [Standard library development](stdlib/stdlib-development.md) covers building,
  testing, and other information you need to work in the standard library.
- [Standard library code style](stdlib/stdlib-code-style.md) provides
  guidelines for writing standard library code.

## Our priorities

For more information on what we're working toward, see the following documents:

- Our [vision document](https://mojolang.org/docs/vision) describes the guiding
  principles behind our efforts.
- Our [roadmap](https://mojolang.org/docs/roadmap/) identifies concrete short-,
  medium-, and longer-term development goals.
