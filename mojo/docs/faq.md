---
title: Mojo FAQ
sidebar_label: FAQ
description: Answers to questions we expect about Mojo.
---

We tried to anticipate your questions about Mojo on this page. If this page
doesn't answer all your questions, also check out our [community
channels](/community/).

## Motivation

### Why did you build Mojo?

We built Mojo to solve an internal challenge when building the [Modular
Platform](https://www.modular.com)—programming across the entire stack was too
complicated. We wanted a flexible and scalable programming model that could
target CPUs, GPUs, AI accelerators, and other heterogeneous systems that are
pervasive in the AI field. This meant a programming language with powerful
compile-time metaprogramming, integration of adaptive compilation techniques,
caching throughout the compilation flow, and other features that are not
supported by existing languages.

As a result, we're extremely committed to Mojo's long-term success and are
investing heavily in it. Our overall mission is to unify AI software and we
can’t do that without a unified language that can scale across the whole AI
infrastructure stack. Our current focus is to unify CPU and GPU programming
with blazing-fast execution for the Modular Platform. That said, the north star
is for Mojo to support the whole gamut of general-purpose programming over
time.

For more detail, see the [Mojo vision](/docs/vision/).

### Why is it called Mojo?

Mojo means "a magical charm" or "magical powers." We thought this was a fitting
name for a language that brings magical powers to programmers, including
unlocking an innovative programming model for accelerators and other
heterogeneous systems pervasive in AI today.

### What problems does Mojo solve that no other language can?

Mojo combines the usability of Python with the systems programming features it’s
missing. We are guided more by pragmatism than novelty, but Mojo’s use of
[MLIR](https://mlir.llvm.org/) allows it to scale to new exotic hardware types
and domains in a way that other languages haven’t demonstrated. It also has
caching and distributed compilation built into its core. We also believe Mojo
has a good chance of unifying hybrid packages in the broader Python community.

### What kind of developers will benefit the most from Mojo?

Mojo’s initial focus is to bring programmability back to AI, enabling AI
developers to customize and get the most out of their hardware. As such, Mojo
will primarily benefit researchers and other engineers looking to write
high-performance AI operations. Over time, Mojo will expand its focus on general
systems programming. We hope this will help lift the vast Python library
ecosystem and empower more traditional systems developers that use C, C++, Rust,
etc.

### Why build upon Python?

Effectively, all AI research and model development happens in Python today, and
there’s a good reason for this! Python is a powerful high-level language with
clean, simple syntax and a massive ecosystem of libraries. At Modular, one of
our core principles is meeting customers where they are—our goal is not to
further fragment the AI landscape but to unify and simplify AI development
workflows.

Our focus is to innovate in the programmability for AI workloads on
heterogeneous hardware, and we don't see any need to innovate in language
_syntax_ or _community_. We chose to build on Pythonic syntax because it's
simple, familiar, and we believe it's a really nice language. We prioritize
Python interoperability because Python is widely used and loved by the AI
community.

### Why a new language instead of enhancing existing approaches?

For a variety of reasons, Python isn't suitable for systems programming.
Domain-specific languages (DSLs) such as Triton trade maximum efficiency
and generality for ease of use. Other major compiled languages are designed
around CPUs, not heterogenous compute.

For much more on the motivation behind developing Mojo and the Modular Platform,
see our blog series about [How Modular is Democratizing AI
Compute](https://www.modular.com/blog/how-is-modular-democratizing-ai-compute).

## Functionality

### Where can I learn more about Mojo’s features?

The best place to start is the [Mojo Manual](/docs/manual/). And if you want to
see what features are coming in the future, take a look at [the
roadmap](/docs/roadmap).

### What are the benefits of building Mojo with MLIR?

When we realized that no existing language could solve the challenges in
AI compute, we embarked on a first-principles rethinking of how a programming
language should be designed and implemented to solve our problems. Because we
require high-performance support for a wide variety of accelerators,
traditional compiler technologies like LLVM and GCC were not suitable (and any
languages and tools based on them would not suffice). Although they support a
wide range of CPUs and some commonly used GPUs, these compiler technologies
were designed decades ago and are unable to fully support modern chip
architectures. Nowadays, the standard technology for specialized machine
learning accelerators is MLIR.

[MLIR](https://mlir.llvm.org/) provides a flexible infrastructure for building
compilers. It’s based upon layers of intermediate representations (IRs) that
allow for progressive lowering of any code for any hardware, and it has been
widely adopted by the hardware accelerator industry since
[its first release](https://blog.google/technology/ai/mlir-accelerating-ai-open-source-infrastructure/).
Its greatest strength is its ability to build _domain specific_ compilers,
particularly for weird domains that aren’t traditional CPUs and GPUs, such as AI
ASICS, [quantum computing systems](https://github.com/PennyLaneAI/catalyst),
FPGAs, and [custom silicon](https://circt.llvm.org/).

Although you can use MLIR to create a flexible and powerful compiler for any
programming language, Mojo is the world’s first language to be built from the
ground up with MLIR design principles. This means that Mojo not only offers
high-performance compilation for heterogeneous hardware, but it also provides
direct programming support for the MLIR intermediate representations, which
currently isn't possible with any other language.

### Is Mojo only for AI or can it be used for other stuff?

Mojo's initial focus is to solve AI programmability challenges. However,
our goal is to grow Mojo into a general-purpose programming language. We
use Mojo at Modular to develop AI algorithms and GPU kernels, but you can
use it for other things like HPC, data transformations, writing pre/post
processing operations, and much more.

### Is Mojo interpreted or compiled?

Mojo is a compiled language. [`mojo build`](/docs/cli/build/) and [`mojo
run`](/docs/cli/run/) both perform ahead-of-time (AOT) compilation.

### Does Mojo support distributed execution?

Not alone. Mojo is one component of the Modular Platform, which
makes it easier for you to author highly performant, portable CPU and GPU graph
operations, but you’ll also need a runtime (or "OS") that supports graph-level
transformations and heterogeneous compute, which is provided by the
[Modular Platform](https://docs.modular.com/max/intro).

### How do I convert Python programs or libraries to Mojo?

See [Tips for Python devs](/docs/manual/python-to-mojo/) for a quick primer on
important differences between Python and Mojo. The
[Mojo AI skills](/docs/tools/skills/) can help your AI coding assistant
translate Python code into working Mojo code.

You can migrate parts of a Python project to Mojo
by building Mojo bindings for Python. See the documentation about how to [call
Mojo from Python](/docs/manual/python/mojo-from-python).

### What about interoperability with other languages like C/C++?

Mojo code is interoperable with C code. For information, see the docs for the
[`ffi`](/docs/std/ffi/) module, the
[`@export`](/docs/reference/decorators/export/) decorator, and the
[`abi("C")` function effect](/docs/reference/mojo-function-declarations/#abi-c).

Mojo code is also interoperable with C++ code that uses `extern "C"`. We
believe we can deliver better C++ interoperability in the future.

### How does Mojo support hardware lowering?

Mojo leverages LLVM-level dialects for the hardware targets it supports, and it
uses other MLIR-based code-generation backends where applicable. This also
means that Mojo is easily extensible to any hardware backend.

### Who writes the software to add more hardware support for Mojo?

Mojo provides all the language functionality necessary for anyone to extend
hardware support. As such, we expect hardware vendors and community members
will contribute additional hardware support in the future.

## Performance

### Are there any AI-related performance benchmarks for Mojo?

It’s important to remember that Mojo is designed to be a general-purpose
programming language, and any AI-related benchmarks will rely heavily upon other
framework components. For example, our in-house CPU and GPU graph operations
that power the Modular Platform are all written in Mojo. You can learn more
about performance in our blog posts on
[bringing the Modular Platform up on AMD MI355](https://www.modular.com/blog/achieving-state-of-the-art-performance-on-amd-mi355----in-just-14-days)
and
[optimizing matmul performance on the NVIDIA Blackwell GPU](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota).

## Mojo SDK

### How can I get the Mojo SDK?

You can get Mojo and all the developer tools by installing `mojo` with
any Python or Conda package manager. For details, see the
[Mojo installation guide](/install/).

### What's included in the Mojo SDK?

We actually offer two Mojo packages: `mojo` and `mojo-compiler`.

The `mojo` package gives you everything you need for Mojo development.
It includes:

- [`mojo` CLI](/docs/cli/) (includes the Mojo compiler)
- [Mojo standard library](/docs/std/)
- [Layout library](/docs/layout/)
- [`mojo` Python
  package](https://github.com/modular/modular/tree/main/mojo/python/mojo)
- Mojo language server (LSP) for IDE/editor integration
- [Mojo debugger](/docs/tools/debugging/) (includes LLDB)
- [Mojo code formatter](/docs/cli/format/)
- [Mojo REPL](/docs/cli/repl/)

The `mojo-compiler` package is smaller and is useful for environments where you
only need to call or build existing Mojo code. For example, this is good if
you're running Mojo in a production environment or when you're programming in
Python and [calling a Mojo
package](/docs/manual/python/mojo-from-python)—situations where you don't need
the LSP and debugger tools. It includes:

- [`mojo` CLI](/docs/cli/) (includes the Mojo compiler)
- [Mojo standard library](/docs/std/)
- [Layout library](/docs/layout/)
- [`mojo` Python
  package](https://github.com/modular/modular/tree/main/mojo/python/mojo)

### What are the license terms for the SDK?

Please read the [Terms of use](https://www.modular.com/legal/terms).

### What operating systems are supported?

Mojo supports Mac and Linux natively and supports Windows via WSL. For details,
see the [Mojo system requirements](/docs/requirements/).

### Is there IDE integration?

Yes, we've published an official Mojo language extension for
[Visual Studio Code](https://code.visualstudio.com/) and other editors that
support VS Code extensions (such as [Cursor](https://cursor.com/home)). The
extension supports various features including syntax highlighting, code
completion, formatting, hover, etc. It works seamlessly with remote-ssh and dev
containers to enable remote development in Mojo.

You can obtain the extension from either the
[Visual Studio Code Marketplace](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo)
or the
[Open VSX Registry](https://open-vsx.org/extension/modular-mojotools/vscode-mojo).

### Does the Mojo SDK collect telemetry?

Yes, the Mojo SDK collects some basic system information, crash reports, and
some LSP events that enable us to identify, analyze, and prioritize Mojo
issues. v25.6 and earlier versions also collected compiler/runtime events,
but we've since removed them.

Specifically, we collect:

- **Crash reports**: When the Mojo compiler crashes with a stack trace, the
  only information used in the report is the OS version and MAX/Mojo version.
- **LSP performance metrics**: The Mojo LSP reports aggregate data on how long
  it takes to respond to user input (parsing latency). The only information
  used in the report is the milliseconds between user keystrokes and when the
  Mojo LSP is able to show appropriate error or warning messages.

No user information, such as source code, keystrokes, or any other user data,
is ever collected or transmitted.

This telemetry is crucial to help us quickly identify problems and improve our
products. Without this telemetry, we would have to rely on user-submitted bug
reports, and in our decades of experience building developer products, we know
that most people don't do that. The telemetry provides us the insights we need
to build better products for you.

## Versioning & compatibility

### What’s the Mojo versioning strategy?

Mojo is working towards a 1.0 release in Summer 2026. For 1.0, we expect the
core language features to be stabilized. Not all Mojo library APIs will be
stable at 1.0, but stable APIs will be marked.

See our
[roadmap](/docs/roadmap/) to understand where things are headed.

### How often will you be releasing new versions of Mojo?

Mojo development is moving fast and we are regularly releasing updates,
including nightly builds almost every day.

Join the [Mojo Discord channel](http://discord.gg/modular) for notifications and
[sign up for our newsletter](https://www.modular.com/blog#sign-up-for-our-newsletter)
(on the bottom of the Mojo blog page) for more coarse-grained updates.

## Open source

### Will Mojo be open-sourced?

We have committed to open-sourcing Mojo in Fall 2026.

### Why not develop Mojo in the open from the beginning?

Mojo is a big project and has several architectural differences from previous
languages. We believe a tight-knit group of engineers with a common vision can
move faster than a community effort. This development approach is also
well-established from other projects that are now open source (such as LLVM,
Clang, Swift, MLIR, etc.).

## Community

### Where can I ask more questions or share feedback?

If you have questions about upcoming features or have suggestions for the
language, be sure you first read the [Mojo roadmap](/docs/roadmap/), which
provides important information about our current priorities.

To get in touch with the Mojo team and developer community, use the resources
on our [community page](/community/).
