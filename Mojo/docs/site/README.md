# Mojo documentation

This directory includes most of the documentation at
[mojolang.org/docs](https://mojolang.org/docs).

The only things not here are the `mojo` CLI pages and the Mojo Standard Library
reference. The reference docs are generated from source files, which are located
in the [Mojo/stdlib/std](../../stdlib/std) directory.

## Standard library API doc generation

The standard library docs at
[mojolang.org](https://mojolang.org/docs/std/) are built by Bazel from the
sources in [`stdlib/std`](../../stdlib/std) as follows:

1. **`mojo_library`** (see
   [`stdlib/std/BUILD.bazel`](../../stdlib/std/BUILD.bazel)) wraps the upstream
   rule and attaches a documentation target `std.docs`.
2. **`mojo doc`** runs as part of that target and emits JSON describing public
   APIs.
3. **`mojodoc_json_to_markdown`** (Python) turns that JSON into Markdown using
   templates; see
   [`mojo_doc.bzl`](https://github.com/modular/modular/blob/main/bazel/internal/mojo_doc.bzl)
   and
   [`mojodoc_json_to_markdown.py`](https://github.com/modular/modular/blob/main/bazel/internal/mojodoc_json_to_markdown.py).
4. This package’s [`BUILD.bazel`](BUILD.bazel) pulls
   `//Mojo/stdlib/std:docs` and puts it
   under `docs/std/` inside the site tarball with the manual and other
   generated drops (CLI pages and so on).

**Cross-links in generated Markdown:** ``mojo doc`` emits logical JSON paths
(``/std/...``, ``/kernels/...``).
[`mojodoc_api_href.py`](https://github.com/modular/modular/blob/main/bazel/internal/mojodoc_api_href.py)
is the single place that knows the published site layout and rewrites them:
stdlib → **mojolang.org** ``/docs/std/...``, kernels → **max.modular.com**
``/api/mojo/...``.

## Cheat sheets

The `reference/cheat-sheets/` pages are normal MDX documents published on the
website, like any others, except they use some additional React components
for a unique layout.

The PDF files in `reference/cheat-sheets/assets/` must be explicitly updated
anytime we make a change to the corresponding MDX files, using a Python script
in the `mojosite` project (in the internal `modular-fe` repo).

### To regenerate PDF files (Modular internal only)

Wait until you're completely done editing the cheat sheets, because you must
execute all these commands in order anytime you change an MDX file:

1. In the monorepo root, build the docs:

   ```sh
   br //Mojo/docs/site:install
   ```

2. In the `modular-fe` path, build the static website:

   ```sh
   npm run mojo-build
   ```

   Do _not_ use `mojo-start`.

3. Back in the monorepo root, generate the PDFs using the `build.py` script in
   the `modular-fe` repo, but with the `output` path set to the monorepo's
   `cheat-sheets/assets/` path. For example:

   ```sh
   python3 .derived/modular-fe/mojosite/scripts/cheat-sheets/bin/build.py all --output Mojo/docs/site/reference/cheat-sheets/assets/
   ```

Confirm the PDFs look as expected, and then commit them to the monorepo.

NOTE: Even if the MDX files are unchanged, running the `build.py` script will
probably always create PDFs that look "changed" to git. The PDFs are binary
files and include metadata such as the creation timestamp that will result in a
changed file, even if the visible PDF is exactly the same as it was.

## Contributing

If you see something in the docs that is wrong or could be improved, we'd love
to accept your contributions.

If your change is any one of the following simple changes, please create a pull
request and we will happily accept it as quickly as possible:

- Typo fix
- Markup/rendering fix
- Factual information fix
- New factual information for an existing page

Before embarking on other major changes, please **create an issue** or
**start a discussion**, so we can collaborate and agree on a solution.
For example, adding an entire new page to the documentation is a lot of work
and it might conflict with other work that’s already in progress. We don’t want
you to spend time on something that might require difficult reviews and rework,
or that might get rejected.

Be aware that we don't provide tools to generate a preview of the website,
because the Mojo docs are built along with other content that's not included in
this repo. As such, we recommend you preview your edits in an IDE that can
render Markdown and MDX files, such as VS Code, including the
[VS Code environment in GitHub](https://github.dev/modular/modular/blob/main/).

For more information about how to contribute, see the [Contributor
Guide](../../CONTRIBUTING.md)

## Other docs

- [`/Mojo/docs/contributing`](/Mojo/docs/contributing): Docs for contributors
  to the Mojo compiler and standard library.
- [`/max/docs`](/max/docs): Docs for developers working in the MAX framework
  codebase.
- [`/max/docs/design-docs`](/max/docs/design-docs): Engineering docs that
  describe how core Modular technologies work.
- [max.modular.com](https://max.modular.com): All other developer docs.
