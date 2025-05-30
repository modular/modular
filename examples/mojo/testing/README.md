# Modular testing framework examples

This directory contains examples of using the Mojo testing framework for unit
testing. See the [Testing](https://docs.modular.com/mojo/tools/testing) section
of the [Mojo manual](https://docs.modular.com/mojo/manual/) for a complete
discussion of how to use the Mojo testing framework.

## Files

This directory contains the following files:

- `src/my_math/__init__.mojo`: a Mojo package file

- `src/my_math/utils.mojo`: a Mojo module source file

- `src/example.mojo`: a simple Mojo program that uses the functions from the
  `my_math` package

- `test/my_math/test_*.mojo`: Mojo test files containing unit tests for
  functions defined in the `my_math` package

- `pixi.toml`: a [Pixi](https://pixi.sh) project file
  containing the project dependencies and task definitions.

## Run the code

This example project uses the [Pixi](https://pixi.sh/latest/) package
and virtual environment manager.

Once you have installed `pixi`, activate the project's virtual environment by
navigating to the project's root directory and executing:

```bash
pixi shell
```

Run the unit tests contained in the `test` directory by executing:

```bash
mojo test -I src test
```

If desired, you can run the example program by executing:

```bash
mojo src/example.mojo
```

Once you're done, deactivate the project's virtual environment by executing:

```bash
exit
```
