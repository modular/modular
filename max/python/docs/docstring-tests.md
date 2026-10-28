# Python docstring testing

This page describes how we check that every public MAX Python API has a
docstring, follows a consistent style, and is included in the generated docs.

## How linting works

We use a set of docstring checks called *pydocstyle*. Ruff is the lint tool
that runs these checks. Bazel runs Ruff as part of the repository lint target,
and CI runs the same checks during presubmit.

- CI checks the Python files changed by a pull request.
- If an API is missing a required docstring or its docstring violates an
  enabled style rule, Ruff reports the file, line, and rule. The lint check
  fails, which blocks the pull request until the problem is fixed.
- Building `//max/python/docs:python-api-docs` does not run the lint checks. A
  successful docs build confirms that Sphinx can generate the API reference,
  but it does not confirm that every docstring follows our style.

Run the checks locally with:

```bash
./bazelw run //oss/modular/bazel/lint:ruff.check
```

To check the full repository instead of only changed files, run:

```bash
./bazelw run //oss/modular/bazel/lint:ruff.check-all
```

## What pydocstyle checks

For files in scope, the enabled checks require:

- Docstrings on public Python modules, classes, functions, and methods.
- Google-style sections such as `Args:`, `Returns:`, and `Raises:`.
- A description for every documented parameter.
- Consistent summaries, capitalization, punctuation, spacing, and docstring
  layout.

We intentionally do not require separate docstrings for magic methods or
`__init__()`. Constructor parameters belong in the class docstring.

Pydocstyle does not validate RST syntax, links between APIs, or whether an API
is included in the generated reference.

You must build the Python API docs to find Sphinx errors, and review the
generated pages to catch rendering or coverage problems:

```bash
./bazelw build //max/python/docs:python-api-docs
```

## Which APIs are covered

We don't enforce docstring coverage for every Python API in the repository.

- The checks apply to selected `.py` and `.pyi` files under
  `max/python/max/`.
- Private implementation modules whose file names begin with `_` are excluded.
- For `.pyi` stub files, only existing docstrings are checked for style.
  Missing docstrings aren't reported.
- Several packages are currently excluded, including `benchmark`, `config`,
  `mlir`, `nn/legacy`, `pipelines/architectures`, `serve`, and `support`.

The source of truth for this scope is the
`[tool.ruff.lint.per-file-ignores]` section of
[`oss/modular/pyproject.toml`](../../../oss/modular/pyproject.toml). Removing an
exclusion expands enforcement to that package; adding an exclusion reduces it.

## Additional docstring checks

The repository lint target also runs a
[custom docstring linter](../../../oss/modular/bazel/lint/lint_docstrings.py)
that enforces a few conventions pydocstyle doesn't cover:

- Class docstrings must not use an `Attributes:` section.
- Parameter entries must not repeat types in parentheses.
- Classes based on `msgspec.Struct` use `Configuration:` instead of `Args:`.

These additional checks apply to `.py` files, but not `.pyi` files. For the
complete writing conventions, see
[`docs/internal/PythonDocstringStyleGuide.md`](../../../docs/internal/PythonDocstringStyleGuide.md).

## How we monitor API reference coverage

Docstring linting checks the documentation in the `.py` files. A separate
[`check_api_coverage.py`](check_api_coverage.py) checks whether the APIs are
actually listed in the RST files that generate the API reference.

The
[`maxPythonApiDocCoverage.yaml`](../../../.github/workflows/maxPythonApiDocCoverage.yaml)
workflow runs this script when a pull request changes MAX Python source or API
reference files. It compares the pull request with its base branch and reports:

- New public APIs that aren't listed in `max/python/docs/`.
- Removed public APIs that are still listed in an RST file.
- New names whose public or private status is unclear.

The script considers a module public when its path has no `_`-prefixed
segments. It considers a symbol public when the package exports it through
`__all__` or a parent `__init__.py`. Some generated architecture APIs and
benchmarking modules are excluded.

When the script finds a problem, the workflow adds or updates a comment on the
pull request with the affected APIs and suggested RST files. This check is
advisory: it doesn't fail CI or block a merge. It also checks only API changes
introduced by the pull request, not the complete existing API surface.

This coverage check does not inspect docstrings. The lint checks described
above enforce whether in-scope APIs have docstrings and whether those
docstrings follow our style.

You can run the coverage check locally with:

```bash
python3 max/python/docs/check_api_coverage.py \
  --base-ref origin/main \
  --head-ref HEAD
```

## Work still to do

As of October 2026, we can't guarantee that every public Python API has a
docstring and appears in the generated reference. The current checks reduce
new gaps, but they don't prove full coverage.

Here's how we can close the gaps:

- **Expand missing-docstring enforcement.** The
  [`per-file-ignores` configuration](../../../oss/modular/pyproject.toml)
  excludes several packages and `_*.py` implementation modules from
  pydocstyle. The following modules are in the docs build, but excluded from
  pydocstyle lint:

  - `max.nn.legacy`
    - `max.nn.legacy.attention`
    - `max.nn.legacy.comm`
    - `max.nn.legacy.graph_groups`
    - `max.nn.legacy.kernels`
    - `max.nn.legacy.kv_cache`
    - `max.nn.legacy.moe`
    - `max.nn.legacy.sampling`
    - `max.nn.legacy.state_space`
    - `max.nn.legacy.state_space.kimi_delta`
  - `max.pipelines.architectures`
  - `max.serve.media`
  - `max.support`
    - `max.support.image`

  Missing docstrings in `.pyi` files are also ignored. We need to remove or
  narrow these exclusions to cover public APIs in those paths, including APIs
  implemented in private modules but re-exported publicly and nanobind APIs
  whose source docstrings live in C++.
- **Define one public API inventory.** The docstring linter, API coverage
  script, and Sphinx configuration don't use exactly the same scope. We need a
  shared definition that includes every supported public API, including
  re-exports and published APIs backed by `max._core`.
- **Close existing coverage gaps.**
  [`check_api_coverage.py`](check_api_coverage.py) checks only changes
  introduced by a pull request. It doesn't audit the complete existing API
  surface, so previously missing RST entries and unclear visibility remain
  undetected. We need a full-tree audit, followed by a clean baseline that CI
  can maintain.
- **Check generated output, not only RST input.**
  [`check_api_coverage.py`](check_api_coverage.py) recognizes entries in RST
  files, but that doesn't prove Sphinx successfully rendered each expected API.
  We need an end-to-end check that compares the canonical public API inventory
  with the generated reference.
- **Make complete coverage a required check.** The API coverage
  workflow posts an advisory comment and always succeeds. After the existing
  gaps are fixed, missing or stale reference entries should fail CI, and the
  Sphinx build should be a required check for relevant changes.
- **Test the coverage tooling.** Add automated tests for public/private
  classification, re-exports, stub files, RST parsing, renamed modules, and
  excluded or generated API families so changes to the checker can't silently
  reduce coverage.
