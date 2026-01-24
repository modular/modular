# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language enhancements
[//]: ### Language changes
[//]: ### Library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language enhancements

### Language changes

### Library changes

- `String.ljust` and `String.rjust` have been renamed to
  `String.ascii_ljust` and `String.ascii_rjust`. Likewise for their
  equivalents on `StringSlice` and `StaticString`
  
- `String.resize` will now panic if the new length would truncate a codepoint.
  Previously it would result in a string with invalid UTF-8.

### Tooling changes

### ❌ Removed

### 🛠️ Fixed
