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

- Add `Variant.is_type_supported` method. ([PR #4057](https://github.com/modular/max/pull/4057))
  Example:
  
  ```mojo
    def my_function(mut arg: Variant):
        if arg.is_type_supported[Float64]():
            arg = Float64(1.5)

    def main():
        var x = Variant[Int, Float64](1)
        my_function(x)
        if x.isa[Float64]():
            print(x[Float64]) # 1.5
  ```

### Tooling changes

### ❌ Removed

### 🛠️ Fixed
