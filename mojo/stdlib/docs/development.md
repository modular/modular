# Mojo standard library development

This document covers the essentials of developing for the standard library.

If this is your first time contributing, first read everything in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Set up your environment

To get started, you need to do the following:

1. [Fork the repo and create a branch](../../CONTRIBUTING.md#how-to-create-a-pull-request).
2. If you're using VS Code, [Install the nightly Mojo VS Code
  extension](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo-nightly)

    NOTE: You can only have one Mojo extension enabled at a time, remember to
    switch when using the stable release!

3. Install the nightly Mojo compiler:

    We recommend using [`pixi`](https://pixi.sh/latest/), which you can install
    with this command:

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

    Then create a new project environment like this and it will install the
latest nightly version of `mojo` (the CLI compiler) by default:

    ```bash
    pixi init my-project \
      -c https://conda.modular.com/max-nightly/ -c conda-forge \
      && cd my-project
    ```

    ```bash
    pixi add modular
    ```

    Lastly enter the environment shell for `mojo` to be available and
be sure it is installed correctly:

    ```bash
    pixi shell
    ```

    ```bash
    mojo --version
    ```

Now you're ready to start developing.

### Dev container

Or, instead of setting up with `pixi` as shown above, you can use an
externally maintained
[Mojo Dev Container](https://github.com/benz0li/mojo-dev-container) with all
prerequisites installed.

The unit test dependency `lit` is also pre-installed and `pre-commit` is
already set up.

See [Mojo Dev Container &gt; Usage](https://github.com/benz0li/mojo-dev-container#usage)
on how to use with [Github Codespaces](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository)
or [VS Code](https://code.visualstudio.com/docs/devcontainers/containers).

If there is a problem with the Dev Container, please open an issue
[here](https://github.com/benz0li/mojo-dev-container/issues).

## Building the standard library

To build the standard library, you can run:

```bash
./bazelw build //mojo/stdlib/...
```

## Testing the standard library

To run the tests for the standard library, you can run:

```bash
./bazelw test //mojo/stdlib/test/...
```

## Testing only a subset of the standard library

You can run all of the tests within a specific subdirectory by simply
specifying the subdirectory and using `/...`. For example:

```bash
./bazelw test //mojo/stdlib/test/math/...
```

A convenience script exists for executing standard library tests within the mojo
directory:

```bash
pixi run tests ./stdlib/test/bit
```

```bash
pixi run tests ./stdlib/test/bit/test_bit.mojo
```

will automatically execute the equivalent bazelw command

## Formatting changes

Please make sure your changes are formatted before submitting a pull request.
Otherwise, CI will fail in its lint and formatting checks.  `bazel` setup
provides a `format` command.  So, you can format your changes like so:

```bash
./bazelw run format
```

It is advised, to avoid forgetting, to set-up `pre-commit`, which will format
your changes automatically at each commit, and will also ensure that you
always have the latest linting tools applied.

To do so, install pre-commit:

```bash
pixi x pre-commit install
```

and that's it!

If you need to manually apply the `pre-commit`, for example, if you
made a commit with the github UI, you can do `pixi x pre-commit run --all-files`,
and it will apply the formatting to all Mojo files.

You can also consider setting up your editor to automatically format
Mojo files upon saving.

### Raising a PR

Make sure that you've had a look at all the materials from the standard library
[README.md](../README.md). This change wouldn't be accepted because it's missing
tests, and doesn't add useful functionality that warrants new functions. If you
did have a worthwhile change you wanted to raise, follow the steps to
[create a pull request](../../CONTRIBUTING.md#create-a-pull-request).

Congratulations! You've now got an idea on how to contribute to the standard
library, test your changes, and raise a PR.

If you're still having issues, reach out on
[Discord](https://modul.ar/discord).
