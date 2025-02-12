# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language changes

### Standard library changes

- A new `IntervalTree` data structure has been added to the standard library.
  This is a tree data structure that allows for efficient range queries.

- `StringSlice` now supports several additional methods moved from `String`.
  The existing `String` methods have been updated to instead call the
  corresponding new `StringSlice` methods:

  - `split()`
  - `lower()`
  - `upper()`
  - `is_ascii_digit()`
  - `isupper()`
  - `islower()`
  - `is_ascii_printable()`
  - `rjust()`
  - `ljust()`
  - `center()`

### Tooling changes

#### Mojo Compiler

Mojo compiler now warns about parameter for with large loop unrolling factor
(>1024 by default) which can lead to long compilation time and large generated
code size. Set `--loop-unrolling-warn-threshold` to change default value to
a different threshold or to `0` to disable the warning.

### ❌ Removed

### 🛠️ Fixed
