# Modular testing framework examples

This directory contains examples of using the Mojo testing framework. It
demonstrates using the testing framework for both unit testing and testing code
examples in the [documentation
strings](https://docs.modular.com/mojo/manual/basics#code-comments) (also known
as *docstrings*) of Mojo API documentation. See the
[Testing](https://docs.modular.com/mojo/tools/testing) section of the [Mojo
manual](https://docs.modular.com/mojo/manual/) for a complete discussion of how
to use the Mojo testing framework.

## Files

This directory contains the following files:

- `src/my_math/__init__.mojo`: a Mojo package file with package-level docstrings
  containing code examples to test

- `src/my_math/utils.mojo`: a Mojo module source file with both module-level and
  function-level docstrings containing code examples to test

- `src/example.mojo`: a simple Mojo program that uses the functions from the
  `my_math` package

- `test/my_math/test_*.mojo`: Mojo test files containing unit tests for
  functions defined in the `my_math` package

- `mojoproject.toml`: a [Magic](https://docs.modular.com/magic/) project file
  containing the project dependencies and task definitions.

## Run the code

This example project uses the [Magic](https://docs.modular.com/magic/) package
and virtual environment manager. To run the code in this project, you should
follow the instructions in [Install
Magic](https://docs.modular.com/nightly/magic/#install-magic) first.

Once you have installed Magic, activate the project's virtual environment by
navigating to the project's root directory and executing:

```bash
magic shell
```

Run the unit tests contained in the `test` directory by executing:

```bash
mojo test -I src test
```

Run the docstring tests for the API documentation contained in the `src`
directory by executing:

```bash
mojo test src
```

If desired, you can run the example program by executing:

```bash
mojo src/example.mojo
```

Once you're done, deactivate the project's virtual environment by executing:

```bash
exit
```
