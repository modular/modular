# Python API Documentation

**Keep this file up to date.** If you change the doc build, update this file.
Add non-obvious Sphinx or build pitfalls here so future sessions don't have to
rediscover them.

## Building

```bash
./bazelw build //max/python/docs:python-api-docs   # Python API docs
./bazelw build //max/python/docs:cli-docs           # CLI docs (separate config)
```

Output: `bazel-bin/max/python/docs/python-api-docs_output/`

`conf.py.in` is a Bazel-processed template (`modular_versioned_expand_template`)
that becomes `conf.py` at build time.

## Markdown output (sphinx-markdown-builder)

Sphinx outputs **Markdown** (not HTML) so the pages can be consumed by
Docusaurus. This is powered by Modular's fork of `sphinx-markdown-builder`
(<https://github.com/modularml/sphinx-markdown-builder>), with a pinned version
in `oss/modular/bazel/common.MODULE.bazel`. The `modular_sphinx_docs`
Bazel rule passes `builder = "markdown"` to Sphinx, and the extension is loaded
in `conf.py.in` as `"sphinx_markdown_builder"`.

After Sphinx generates the Markdown files, `post-process-docs.py` further
adjusts them (heading demotion, empty-section removal, sidebar stub population).

## RST file organization

Each Python module has one flat RST file (for example, `nn.rst`,
`nn.attention.rst`, `pipelines.rst`). There are two documentation patterns:

- **Explicit autosummary** (most files): uses `.. automodule::` with
  `:no-members:`, then groups members into semantic sections with
  `.. autosummary::` directives referencing templates in
  `_templates/autosummary/` (`class.rst`, `function.rst`).
- **Automodule with :members:** (for example, `graph.ops.rst`,
  `nn.kernels.rst`, `experimental.functional.rst`): documents all public
  members automatically without explicit listing.

**Deduplication rule**: when a parent module re-exports members from a
submodule, don't list that member more than once. Decide whether to include
it in the parent module or submodule RST file based on which file offers the
best-matching semantic group of members.

`index.rst` lists all module pages in its toctree. Module pages that have
submodules list them in a "Submodules" toctree at the bottom.

## Adding a new module page

1. Create `{module}.rst` with `.. automodule::` and `.. autosummary::` sections
   (copy an existing file like `nn.rst` as a starting point).
2. Add the filename (without `.rst`) to the toctree in `index.rst`.
3. Add a sidebar entry in `oss/modular/docs/sidebars.js` with
   `"__AUTOGEN:max.module.name__"` as the items stub.
4. Run `./bazelw build //max/python/docs:python-api-docs` and verify.

## Key files

Read these when you need to understand or modify the doc build:

- `conf.py.in` -- Sphinx config: warning suppression (`SuppressionFilter`),
  autodoc hooks (`skip_imported_for_modules`), monkey-patch for `__all__`
  filtering in autosummary.
- `oss/modular/docs/post-process-docs.py` -- Runs after Sphinx generates
  markdown. Demotes headings, removes empty sections, populates
  `__AUTOGEN:prefix__` stubs in `sidebars.js` with generated page paths.
- `oss/modular/docs/sidebars.js` -- Docusaurus sidebar definition. Each
  module's `items` array uses an `__AUTOGEN` stub that `post-process-docs.py`
  fills in.
- `_templates/autosummary/` -- Jinja2 templates for autosummary stub pages
  (`class.rst` for classes and type aliases, `function.rst` for functions,
  `data.rst` for module-level data/constants).
- `docs/internal/PythonDocstringStyleGuide.md` -- Docstring style guide (also
  enforced by `.claude/skills/docstrings`).

## Pydocstyle linting

Docstring style is enforced by ruff `D` rules (Google convention). Run with
`./bazelw run //oss/modular/bazel/lint:ruff.check`. See
`oss/modular/pyproject.toml` (`[tool.ruff.lint.per-file-ignores]`) for which
packages are in scope and which are excluded.

## Common pitfalls

- **RST, not Markdown**: docstrings must use RST. No triple-backtick code
  fences; use `.. code-block:: python` with a blank line after the directive.
- **Blank lines before blocks**: RST requires a blank line before any list,
  code block, or indented content. Without it, Sphinx reports "Unexpected
  indentation."
- **Stale `__all__` entries**: if a name is in `__all__` but not actually
  imported, autosummary will fail with "failed to import." Check the module's
  imports before adding members to an RST file.
- **Type aliases**: Python `Union` types and `TypeAlias` objects have read-only
  `__doc__` attributes, so custom docstrings don't render. Use the `class.rst`
  template for these; autodoc will show the expanded type signature.
- **Attribute docstrings from `.pyi` stubs**: Sphinx's `ModuleAnalyzer`
  can't parse C-extension source to find attribute docstrings (e.g. enum
  member docs on `DType`) because `for_module` raises `PycodeError` when
  `__file__` points to a `.so`. Our `conf.py.in` monkey-patches
  `AttributeDocumenter.get_attribute_comment` to fall back to `.pyi` stub
  files via a separate cache. This is intentionally scoped to attribute-
  docstring lookup only because feeding `.pyi` stubs into the global
  `ModuleAnalyzer` cache breaks Sphinx's handling of `@overload` and
  other stub-only constructs (e.g. dropping overloaded methods on
  `Buffer`).
- **`finfo` on `DType`**: `finfo` is monkey-patched onto `DType` in
  `dtype_extension.py`. There is an explicit skip in `conf.py.in` to prevent
  it from appearing inside `DType`'s member list (it has its own page).
