# Mojo for Visual Studio Code

This VS Code extension from the Modular team adds support for the [Mojo
programming langauge](https://www.modular.com/mojo).

## Features

- Syntax highlighting for `.mojo` and `.🔥` files
- Code completion
- Code diagnostics and quick fixes
- API docs on hover

## Get started

1. Install the Mojo SDK.
2. Install the Mojo VS Code extension.
3. Open any `.mojo` or `.🔥` file.

## Configuration

The extension will attempt to find the path of the Mojo SDK installation using
the `MODULAR_HOME` environment variable. If `MODULAR_HOME` is not set within
the environment, the path can be explicitly set via the `mojo.modularHomePath`
extension setting.

```json
{
    "mojo.modularHomePath": "/absolute/path/to/.modular"
}
```
