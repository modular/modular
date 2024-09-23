# Developer Guidelines

## `null` vs `undefined`

We don't use `null` in the codebase, except for cases in which an external API
expects it.
We do this to simplify handling of optionals and for the conveniences that the
language provides for undefined values.

If you need a way to specify an absence of a value that can't just be described
with `undefined`, then use an enum.

As an convenience, use the `Optional` type to have a unified way to express optionals.

## Building and Debugging

To build and debug the VS Code extension, do the following:

- `cd` to `KGEN/utils/vscode-mojo`
- Run `npm run ci`, which installs NPM packages using the package-lock.json
  versions.

Then there are two paths to running the extension:

1. On MDCM using a terminal within VS Code in an SSH session

- In the Modular repo, run `vscode-build`.
- Then launch VS Code.

1. Debug the extension with a second VS Code window

- In VS Code, open a workspace with the `KGEN/utils/vscode-mojo` directory,
  which picks up the `KGEN/utils/vscode-mojo/.vscode` directory and its
  `launch.json` configuration.
- In VS Code, go to the `Run and Debug` view. The play button to start debugging
  has a dropdown to choose different profiles. Choose the profile for the Mojo
  extension: `Run extension (vscode-mojo)`.
- Push the play button to run the extension. It will open a new window of
  VS Code using the development version extension.
- In the output tab, you can select the `Mojo` output channel to view log
  messages from the Mojo extension.
- In the original VS Code window, you can place breakpoints in the extension
  source code and debug the child window that is running the extension.
