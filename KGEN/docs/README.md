
# Mojo Compiler Dev Manual

This directory contains the Mojo Compiler Developer Manual. This file is the
one-stop-shop entry point to all documentation relevant to the Mojo compiler.

> [!TIP]
>
> The name `KGEN` stands for "kernel generator". When you see KGEN, think Mojo.

## File Overview

- `KGEN/` -- Mojo compiler sources, tests, and documentation
  - `docs/` -- You are here 👋. Main documentation for the Mojo compiler; links
    out to the other docs.
    - `docs/manual/` -- intro docs, written assuming no prior Mojo compiler
      knowledge; for newcomers to the compiler team, or folks making drive-by
      contributions.
    - `docs/overviews/` -- subsystem and cross-cutting behavior overviews, for
      those more familiar with the compiler.
      `docs/arcana/` -- more detailed docs, diving deep into nuanced behavior;
      useful for someone trying to debug the compiler, this has the vital hidden
      clues.
    - `docs/attic/` -- older compiler docs, that capture prior thinking and
      behavior. Occasionally useful to consult when doing code archeology.
  - `lib/` -- C++ sources for the Mojo compiler _libraries_, including parser,
    passes, MLIR dialects, and related tooling (e.g. debugger), etc.
  - `tools/` -- C++ sources for command-line interface _executables_ (CLI), including
    `mojo`, `kgen`, `kgen-opt`, `kgen-translate`, etc.
  - `test/` -- Compiler tests in the form of Mojo source programs that exercise
    features of the language, `mojo` CLI, and related tooling.

## Artifacts

### Command-Line Tools

- `mojo` (_public_) — Main Mojo compiler executable,
  - Input: `.mojo`; Output: executables, `.a`, and `.dylib`
- `kgen` (_internal_) — Compile Mojo code into `kgen` dialect MLIR, for
  consumption by the graph compiler
  - Input: `.mojo`; Output: `.mlir`
- `kgen-opt` (_internal_) — Apply specific optimization passes to KGEN IR.
  - Input: `.mlir`; Output: `.mlir`
- `kgen-translate` (_internal_) — TODO

### Libraries

- `CompilerRT.a` — Runtime library for Mojo, linked in to every compiled
  Mojo programming, and providing facilities to the Mojo standard library.

**TODO:** The Mojo compiler produces dialect libraries(?) that are consumed
by the Graph Compiler so that it has knowledge of the dialects used by the
Mojo compiler.

## Compiler Manual

- [Mojo Design](./manual/MojoNotes.md)
- [Rationale](./manual/Rationale.md)

## Detailed Documentation

See these breakout docs:

- [Generative Kernel Compiler Design Overview](DesignOverview.md)
- [Generative Kernel Compiler Task List](TaskList.md)
- [Design Rationale](Rationale.md) for details.

For more on Mojo see:

- [Mojo🔥 Notes](MojoNotes.md)
