---
title: Mojo FAQ
sidebar_label: FAQ
description: Answers to questions we expect about Mojo.
---

We tried to anticipate your questions about Mojo on this page. If this page
doesn't answer all your questions, also check out our [community
channels](https://www.modular.com/community).

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

As a result, we're extremely committed to Mojo's long term success and are
investing heavily in it. Our overall mission is to unify AI software and we
can’t do that without a unified language that can scale across the whole AI
infrastructure stack. Our current focus is to unify CPU and GPU programming
with blazing-fast execution for the Modular Platform. That said, the north star
is for Mojo to support the whole gamut of general-purpose programming over
time.

For more detail, see the [Mojo vision](/mojo/vision).

### Why is it called Mojo?

Mojo means "a magical charm" or "magical powers." We thought this was a fitting
name for a language that brings magical powers to Python, including unlocking
an innovative programming model for accelerators and other heterogeneous
systems pervasive in AI today.

### Why does Mojo have the 🔥 file extension?

We paired Mojo with fire emoji 🔥 as a fun visual way to impart onto users that
Mojo empowers them to get their Mojo on—to develop faster and more efficiently
than ever before. We also believe that the world can handle a unicode extension
at this point, but you can also just use the `.mojo` extension. :)

### What problems does Mojo solve that no other language can?

Mojo combines the usability of Python with the systems programming features
it’s missing. We are guided more by pragmatism than novelty, but Mojo’s use of
[MLIR](https://mlir.llvm.org/) allows it to scale to new exotic hardware types
and domains in a way that other languages haven’t demonstrated. It also
has caching and distributed compilation built into its
core. We also believe Mojo has a good chance of unifying hybrid packages in the
broader Python community.

### What kind of developers will benefit the most from Mojo?

Mojo’s initial focus is to bring programmability back to AI, enabling AI
developers to customize and get the most out of their hardware. As such, Mojo
will primarily benefit researchers and other engineers looking to write
high-performance AI operations. Over time, Mojo will become much more
interesting to the general Python community as it grows to be a superset of
Python. We hope this will help lift the vast Python library ecosystem and
empower more traditional systems developers that use C, C++, Rust, etc.

### Why build upon Python?

Effectively, all AI research and model development happens in Python today, and
there’s a good reason for this! Python is a powerful high-level language with
clean, simple syntax and a massive ecosystem of libraries. At Modular, one of
our core principles is meeting customers where they are—our goal is not to
further fragment the AI landscape but to unify and simplify AI development
workflows.

Our focus is to innovate in the programmability for AI workloads on
heterogeneous hardware, and we don't see any need to innovate in language
_syntax_ or _community_. So we chose to embrace the Python ecosystem because
it's so widely used, it's loved by the AI ecosystem, and because we believe it
is a really nice language.

### Why not enhance CPython (the major Python implementation) instead?

For a variety of reasons, Python isn't suitable for systems programming.
Python has amazing strengths as a glue layer—it offers low-level
bindings that allow developers to build libraries in C, C++ and many other
languages that have better performance characteristics. This enables
things like NumPy and PyTorch, and a vast number of other libraries in
the AI ecosystem, but it comes with a cost.

Building these hybrid libraries is very complicated. It requires a deep
understanding of CPython and strong C/C++ (or other) programming abilities
(undermining one of the original goals of using Python in the first place).
These hybrid-language libraries also create problems for the library users,
because debuggers generally can't step between Python and C/C++ code.

We’re thrilled to see a big push to improve the performance of
[CPython](https://en.wikipedia.org/wiki/CPython), but our goals for Mojo (such
as to deploy onto GPUs and other accelerators) requires a fundamentally
different architecture and compiler approach. That said, CPython is still a
critical part of our compatibility approach and powers [Mojo's Python
interoperability](/mojo/manual/python).

### Why not enhance another Python implementation (like Codon, PyPy, etc)?

Codon and PyPy aim to improve performance compared to CPython, but Mojo’s goals
are much deeper than this. Our objective isn’t just to create "a faster
Python," but to enable a whole new layer of systems programming that includes
direct access to accelerated hardware.

Many hardware accelerators support very limited dynamic features, or do so with
terrible performance. Furthermore, systems programmers don't seek only
"performance," but also demand a lot of predictability and control over how a
computation happens, so in some cases we cannot accept dynamic features at all.

Furthermore, solving big challenges for the computing industry is hard and
requires a fundamental rethinking of the compiler and runtime infrastructure.
This drove us to build an entirely new approach and we’re willing to put in the
time required to do it properly, rather than tweaking an existing system that
would only solve a small part of the problem. For more detail, see our blog
post about [How Modular is Democratizing AI
Compute](https://www.modular.com/blog/how-is-modular-democratizing-ai-compute).

### Why not make Julia better?

We think [Julia](https://julialang.org/) is a great language and it has a
wonderful community, but Mojo is completely different. While Julia and Mojo
might share some goals and look similar as an easy-to-use and high-performance
alternative to Python, we’re taking a completely different approach to building
Mojo. Notably, Mojo is Python-first and doesn't require existing Python
developers to learn a new syntax.

Mojo also has a bunch of technical advancements compared to Julia, simply
because Mojo is newer and we’ve been able to learn from Julia (and from Swift,
Rust, C++ and many others that came before us). For example, Mojo takes a
different approach to memory ownership and memory management, it scales down to
smaller envelopes, and is designed with AI and MLIR-first principles (though
Mojo is not only for AI).

That said, we also believe there’s plenty of room for many languages and this
isn’t an OR proposition. If you use and love Julia, that's great! We’d love for
you to try Mojo and if you find it useful, then that's great too.

## Functionality

### Where can I learn more about Mojo’s features?

The best place to start is the [Mojo Manual](/mojo/manual). And if you want to
see what features are coming in the future, take a look at [the
roadmap](/mojo/roadmap).

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
widely adopted by the hardware accelerator industry since [its first
release](https://blog.google/technology/ai/mlir-accelerating-ai-open-source-infrastructure/).
Its greatest strength is its ability to build _domain specific_ compilers,
particularly for weird domains that aren’t traditional CPUs and GPUs, such as
AI ASICS, [quantum computing systems](https://github.com/PennyLaneAI/catalyst),
FPGAs, and [custom silicon](https://circt.llvm.org/).

Although you can use MLIR to create a flexible and powerful compiler for any
programming language, Mojo is the world’s first language to be built from the
ground up with MLIR design principles. This means that Mojo not only offers
high-performance compilation for heterogeneous hardware, but it also provides
direct programming support for the MLIR intermediate representations, which
currently isn't possible with any other language.

### Is Mojo only for AI or can it be used for other stuff?

Mojo's initial focus is to solve AI programmability challenges. However, our
goal is to grow Mojo into a general purpose programming language. We use Mojo
at Modular to develop AI algorithms and [GPU
kernels](/max/tutorials/custom-ops-matmul), but you can use it for other things
like HPC, data transformations, writing pre/post processing operations, and
much more.

### Is Mojo interpreted or compiled?

Mojo is a compiled language. [`mojo build`](/mojo/cli/build) performs
ahead-of-time (AOT) compilation to save an executable program. [`mojo
run`](/mojo/cli/run) performs just-in-time (JIT) compilation to execute a Mojo
source file without saving the compiled result.

### How does Mojo compare to Triton Lang?

[Triton Lang](https://triton-lang.org/main/index.html) is a specialized
programming model for one type of accelerator, whereas Mojo is a more general
language that will support more architectures over time and includes a
debugger, a full tool suite, etc.

For more about our thoughts on embedded domain-specific languages (EDSLs) like
Triton, read [Democratizing AI Compute, Part
7](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls).

### Does Mojo support distributed execution?

Not alone. Mojo is one component of the Modular Platform, which
makes it easier for you to author highly performant, portable CPU and GPU graph
operations, but you’ll also need a runtime (or "OS") that supports graph level
transformations and heterogeneous compute, which is provided by
[MAX](/max/intro#components).

### How do I convert Python programs or libraries to Mojo?

You can migrate parts of a Python project to Mojo
by building Mojo bindings for Python. See the documentation about how to [call
Mojo from Python](/mojo/manual/python/mojo-from-python).

### What about interoperability with other languages like C/C++?

Yes, we want to enable developers to port code from languages other than Python
to Mojo as well. We expect that due to Mojo’s similarity to the C/C++ type
systems, migrating code from C/C++ should work well and it’s in [our
roadmap](/mojo/roadmap#cc-interop).

### How does Mojo support hardware lowering?

Mojo leverages LLVM-level dialects for the hardware targets it supports, and it
uses other MLIR-based code-generation backends where applicable. This also
means that Mojo is easily extensible to any hardware backend.

### Who writes the software to add more hardware support for Mojo?

Mojo provides all the language functionality necessary for anyone to extend
hardware support. As such, we expect hardware vendors and community members
will contribute additional hardware support in the future.

## Performance

### Are there any AI related performance benchmarks for Mojo?

It’s important to remember that Mojo is designed to be a general-purpose
programming language, and any AI-related benchmarks will rely heavily upon
other framework components. For example, our in-house CPU and GPU graph
operations that power the Modular Platform are all written in Mojo and you can
learn more about performance in our [matrix multiplication blog
post](https://www.modular.com/blog/the-worlds-fastest-unified-matrix-multiplication).
For details about our end-to-end model performance, read about [how we measure
performance at
Modular](https://www.modular.com/blog/max-gpu-state-of-the-art-throughput-on-a-new-genai-platform).

## Mojo SDK

### How can I get the Mojo SDK?

You can get Mojo and all the developer tools by installing `mojo` with
any Python or Conda package manager. For details, see the
[Mojo installation guide](/mojo/manual/install).

### Is the Mojo Playground still available?

Not for long. The [Mojo Playground](https://developer.modular.com/playground)
is deprecated and will be shut down with the v25.6 release.

Here's the story: When we announced Mojo in May, 2023, Mojo wasn't
available in an SDK; it was available only in web-hosted a JupyterLab
environment. After we made Mojo available for local development, we
shut down the JupyterLab environment and launched a new Mojo Playground
for people to try Mojo on the web. But ever since we made the Mojo SDK
avialable for Linux and Mac, Mojo Playground usage steadily declined.
The trickle of users we get now no longer justifies the maintenance
and hosting costs.

See how to [install Mojo](/mojo/manual/install).

### What are the license terms for the SDK?

Please read the [Terms of use](https://www.modular.com/legal/terms).

### What operating systems are supported?

Mac and Linux. For details, see the
[Mojo system requirements](/mojo/manual/install#system-requirements).

### Is there IDE Integration?

Yes, we've published an official [Mojo language extension](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo)
for VS Code.

The extension supports various features including syntax highlighting, code
completion, formatting, hover, etc. It works seamlessly with remote-ssh and dev
containers to enable remote development in Mojo.

### Does the Mojo SDK collect telemetry?

Yes, the Mojo SDK collects some basic system information, basic
compiler/runtime events, and crash reports that enable us to identify, analyze,
and prioritize Mojo issues.

This telemetry is crucial to help us quickly identify problems and improve our
products. Without this telemetry, we would have to rely on user-submitted bug
reports, and in our decades of experience building developer products, we know
that most people don’t do that. The telemetry provides us the insights we need
to build better products for you.

## Versioning & compatibility

### What’s the Mojo versioning strategy?

Mojo is still in early development and not at a 1.0 version yet. It’s
still missing many foundational features, but please take a look at our
[roadmap](/mojo/roadmap) to understand where things are headed. As such,
the language is evolving rapidly and source stability is not guaranteed.

### How often will you be releasing new versions of Mojo?

Mojo development is moving fast and we are regularly releasing updates,
including nightly builds almost every day.

Join the [Mojo Discord channel](http://discord.gg/modular) for notifications
and [sign up for our newsletter](https://www.modular.com/modverse#signup) for
more coarse-grain updates.

## Open Source

### Will Mojo be open-sourced?

We have committed to open-sourcing Mojo in 2026.
Mojo is still young, so we will continue to incubate it within Modular until
more of its internal architecture is fleshed out.

### Why not develop Mojo in the open from the beginning?

Mojo is a big project and has several architectural differences from previous
languages. We believe a tight-knit group of engineers with a common vision can
move faster than a community effort. This development approach is also
well-established from other projects that are now open source (such as LLVM,
Clang, Swift, MLIR, etc.).

## Community

### Where can I ask more questions or share feedback?

If you have questions about upcoming features or have suggestions
for the language, be sure you first read the [Mojo roadmap](/mojo/roadmap), which
provides important information about our current priorities.

To get in touch with the Mojo team and developer community, use the resources
on our [community page](https://www.modular.com/community).
