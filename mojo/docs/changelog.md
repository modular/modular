# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language enhancements

- `@parameter for` now works on a broader range of collection types, enabling
  things like `@parameter for i in [1, 2, 3]: ...`.

- Parametric aliases are now supported: Aliases can be specified with an
  optional parameter list (just like functions). Parametric aliases are
  considered first class parameter values, too.

### Language changes

- The `@value` decorator has been formally deprecated with a warning, it will
  be removed in the next release of Mojo.  Please move to the `@fieldwise_init`
  and synthesized `Copyable` and `Movable` trait conformance.

- Implicit trait conformance is removed. All conformances must be explicitly
  declared.

### Standard library changes

- Added support for a wider range of consumer-grade AMD hardware, including:
  - AMD Radeon RX 7xxx GPUs
  - AMD Radeon RX 9xxx GPUs
- Compile-time checks for AMD RDNA3+ GPUs are now provided by the functions:
  - `_is_amd_rdna3()`
  - `_is_amd_rdna4()`
  - `_is_amd_rdna()`
- Added WMMA matrix-multiplication instructions for RDNA3+ GPUs to help support
  running AI models on those GPUs.

- `memory.UnsafePointer` is now implicitly included in all mojo files. Moreover,
  `OpaquePointer` (the equivalent of a `void*` in C) is moved into the `memory`
  module, and is also implicitly included.

- Python interop changes:

  - The `PythonTypeBuilder` utility now allows registering bindings for Python
    static methods, i.e. methods that don't require an instance of the class.

### Tooling changes

- Added progress reporting support to the Mojo language server. This will emit progress
  notifications in your editor when the server is currently parsing a document.

### ❌ Removed

- Various functions from the `sys.info` have been removed.  Use the appropriate method
  on `CompilationTarget` from `sys.info` instead.
  - `is_x86()`
  - `has_sse4()`

### 🛠️ Fixed

- [#4820](https://github.com/modular/modular/issues/4820) - `math.exp2` picks
  the wrong implementation for `float64`.
- [#4121](https://github.com/modular/modular/issues/4121) - better error message
  for `.value()` on empty `Optional`.
