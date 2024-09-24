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

## Publishing

Publishing a new version of the extension is done automatically by the
following CI workflows:

- [Nightly](https://github.com/modularml/modular/actions/workflows/MAXReleaseNightly.yaml)
- [Stable](https://github.com/modularml/modular/actions/workflows/buildAndTestMAX.yaml)

However, if you need to submit one-off fixes of the extension without having
to ship a brand new MAX SDK, you can do it with

- [Nightly](https://github.com/modularml/modular/actions/workflows/promoteVscodeExtensionNightly.yaml)
- [Stable](https://github.com/modularml/modular/actions/workflows/promoteVscodeExtension.yaml)

In both one-off cases, you need to provide both the new extension version and the
version of the matching MAX SDK. You can provide these via the input boxes in the
GitHub UI.

You can find the list of the existing MAX SDKs in:

- [Nightly](https://conda.modular.com/max-nightly/noarch/repodata.json)
- [Stable](https://conda.modular.com/max/noarch/repodata.json)

Look for the `version` entry in the `max` packages.

If you really want to test how a given build of the extension would work before
publishing the one-off update, you can test locally via the `vscode-build`
command, which also asks for the MAX SDK version.
