# Basic example for MAX Graph API in Python

This is an example of building a model with the MAX Graph API in Python and
execute it with MAX Engine.

## Usage

First, install [Magic](https://docs.modular.com/magic/).

Then run the `magic run addition` command from the terminal:

```sh
magic run addition
```

You should see the following output:

```output
input names are:
name: input0, shape: [1], dtype: DType.float32
name: input1, shape: [1], dtype: DType.float32
result: [2.]
```

## Tests

To run tests, use the following command:

```sh
magic run pytest
```

You should see the following output:

```output
==================== test session starts ====================
platform darwin -- Python 3.12.7, pytest-8.3.3, pluggy-1.5.0
rootdir: /max-repo
configfile: pyproject.toml
collected 6 items                                           

tests/test_addition.py ......                         [100%]

===================== 6 passed in 6.94s =====================
```

## Lint and format

To lint with `mypy` run the following command:

```sh
magic run mypy
```

To format the python files:

```sh
magic run black
magic run isort
```
