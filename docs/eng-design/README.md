# Modular Design Docs

As part of our promise to progressively open source Mojo and MAX, and to drive
deeper community engagement, we are publishing engineering design documents to
help educate our developer base about how our core technologies work, how to
best use the Modular platform, and how to become core kernel and model
contributors.

This directory includes Markdown docs, formatted for MKDocs, and basic
scaffolding to build a static docs site with MKDocs.

## Get started

1. Install uv

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Run the server

    ```bash
    uv run mkdocs serve
    ```

3. Open the docs

Point your browser to `http://localhost:8000`

## Adding docs

Each doc will consist of a single markdown file, placed in the `docs`
subdirectory, and a set of image files, placed in the `docs/img/<docname>`
subdirectory. Image files should be small, preferably < 1 MB.

Docs must begin with a YAML header, for example:

```yaml
---
title: "GenAI Tech Talk: Paged Attention"
author: Austin Doolittle
author: Brian Zhang
date: January 30, 2025
---
```

Images can optinally include captions:

```md
[KV Cache](docs/img/genai-paged-attention/img01-kvcache.png)
/// caption
KV Cache Image Caption
///
```

MathJax is supported inline

```md
This results in $O(n^2)$ complexity.
```

Docs must also include a link with a one-line description in the table of
contents located in the `index.md` file/

```md
* [AMD Print Lessons Learned](amd-printf-lessons-learned)

    This document describes the technical challenges and solutions involved in
    implementing print statement debugging for AMD GPU kernels in Mojo by
    porting OpenCL hostcall functionality to avoid dependencies on AMD's
    device-libs and additional LLVM copies.
```
