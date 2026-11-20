# Pattern Transformation Principles

This document outlines the principles that should guide the design and
application of pattern transformations in the graph compiler.

## Goals of pattern transformations

The graph compiler should transform input code to canonicalize it.

- It should allow _less_ code and _simpler_ code in pipelines.
- Pattern transformations are _not_ intended to improve run time performance,
  although they might do so. They should _never_ make run time performance
  worse.

## First principle

Transformations are predictable.

## Principles for when to apply patterns

- Transformations should canonicalize code for downstream parts of the compiler.
- Transformations should always be target independent.
- Transformations should be explainable to the end user.
- Transformations should lead to fewer significant operations.
- Transformations should preserve high-level information, rather than attempting
  to re-construct this information after lowering

It is important that we don’t apply a transformation without certainty of its
result. We should strive to publicly document each transformation, and we need a
coarse-grained, target-independent cost model to define “significant”
operations.

## Appendix

This document is an summary of previous
[discussions](https://www.notion.so/modularai/Graph-compiler-pattern-matching-29a1044d37bb8055b5bbe6c929964612).
