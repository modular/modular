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

### Language changes

### Standard library changes

- Added `os.path.realpath` to resolve symbolic links to an absolute path and
  remove relative path components (`.`, `..`, etc.). Behaves the same as the
  Python equivalent function.

- `Span` is now `Representable` if its elements implement trait
  `Representable`.

- `Optional` and `OptionalReg` can now be composed with `Bool` in
  expressions, both at comptime and runtime:

  ```mojo
  alias value = Optional[Int](42)

  @parameter
  if CompilationTarget.is_macos() and value:
      print("is macos and value is:", value.value())

- Added `sys.info.platform_map` for specifying types that can have different

  values depending on the platform:

  ```mojo
  from sys.info import platform_map

  alias EDEADLK = platform_map["EDEADLK", linux = 35, macos = 11]()
  ```

- Added support for AMD RX 6900 XT consumer-grade GPU.

### Tooling changes

### ❌ Removed

### 🛠️ Fixed
