---
title: Mojo 🔥 Compiler Dev Manual
markdown-notebook-data-directory: mdnb-data/manual-readme/
---

<!-- markdownlint-disable -->

# Mojo 🔥 Compiler Dev Manual

## Introduction

Welcome to the Mojo Compiler Dev Manual! The main goal of this is to help people
who are just getting started in modifying the Mojo compiler.

The Mojo compiler has a lot of similarities to other compilers, but also a lot
of differences. This doc will cover all of it, and link out to further reading
for the more nuanced topics.

## Passes and Intermediate Representations

See [PassesAndIR.md](PassesAndIR.md) for what our various IR stages look like,
and how our passes transform from one to the next.

## Terminology

See [Terminology.md](Terminology.md) for commonly used terms here.

## Mojo ↔ IR ↔ C++ Correspondence

See [MojoIRCPPCorrespondence.md][MojoIRCPPCorrespondence.md] for how various
given Mojo snippets compile to IR, and what C++ one would use in the compiler to
generate that same IR.

## Debugging

See [Parser Debugging](ParserDebugging.md). A lot of those tricks are applicable
to other stages as well.
