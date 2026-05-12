# Mojo documentation

This directory includes most of the documentation at
[docs.modular.com/mojo](https://docs.modular.com/mojo).

The only things not here are the `mojo` CLI pages and the Mojo Standard Library
reference. The reference docs are generated from source files, which are located
in the [/stdlib/std](../stdlib/std) directory.

## Standard library API doc generation

The published stdlib reference under `/docs/std/` on
[mojolang.org](https://www.mojolang.org/docs/std/) is built by Bazel from the
sources in [`stdlib/std`](../stdlib/std) as follows:

1. **`mojo_library`** (see
   [`stdlib/std/BUILD.bazel`](../stdlib/std/BUILD.bazel)) wraps the upstream
   rule and attaches a documentation target `std.docs`.
2. **`mojo doc`** runs as part of that target and emits JSON describing public
   APIs.
3. **`mojodoc_json_to_markdown`** (Python) turns that JSON into Markdown using
   templates; see [`mojo_doc.bzl`](../../bazel/internal/mojo_doc.bzl) and
   [`mojodoc_json_to_markdown.py`](../../bazel/internal/mojodoc_json_to_markdown.py).
4. This package’s [`BUILD.bazel`](BUILD.bazel) pulls
   `//oss/modular/mojo/stdlib/std:docs` and puts it
   under `docs/std/` inside the site tarball with the manual and other
   generated drops (CLI pages, layout library reference, and so on).

**Cross-links in generated Markdown:** ``mojo doc`` emits logical JSON paths
(``/std/...``, ``/kernels/...``).
[`mojodoc_api_href.py`](../../bazel/internal/mojodoc_api_href.py)
is the single place that knows the published site layout and rewrites them:
stdlib → **mojolang.org** ``/docs/std/...``, ``kernels/layout`` →
**mojolang.org** ``/docs/layout/...``, other kernels → **docs.modular.com**
``/max/api/kernels/...``.

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
[VS Code environment in GitHub](https://github.dev/modular/max/blob/main/).

For more information about how to contribute, see the [Contributor
Guide](../../CONTRIBUTING.md)

## Other docs

- [`/mojo/stdlib/docs`](/mojo/stdlib/docs): Docs for developers working in the
  Mojo standard library.
- [`/max/docs`](/max/docs): Docs for developers working in the MAX framework
  codebase.
- [`/max/docs/design-docs`](/max/docs/design-docs): Engineering docs that
  describe how core Modular technologies work.
- [docs.modular.com](https://docs.modular.com): All other developer docs.
