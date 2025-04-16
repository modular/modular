---
title: '@parameter'
description: Executes a function or if statement at compile time.
codeTitle: true
---

You can add the `@parameter` decorator on an `if` or `for` statement to run that
code at compile time, or on a nested function to create a [parametric
closure](#parametric-closure).

## Parametric `if` statement

You can add `@parameter` to any `if` condition that's based on a valid
parameter expression (it's an expression that evaluates at compile time). This
ensures that only the live branch of the `if` statement is compiled into the
program, which can reduce your final binary size. For example:

```mojo
@parameter
if True:
    print("this will be included in the binary")
else:
    print("this will be eliminated at compile time")
```

```output
this will be included in the binary
```

## Parametric `for` statement

You can add the `@parameter` decorator to a `for` loop to create a loop that's
"unrolled" at compile time.

The loop sequence and induction values must be valid parameter expressions (that
is, expressions that evaluate at compile time). For example, if you use
`for i in range(LIMIT)`, the expression `range(LIMIT)` defines the loop
sequence. This is a valid parameter expression if `LIMIT` is a parameter, alias,
or integer literal.

The compiler "unrolls" the loop by replacing the `for` loop with
`LIMIT` copies of the loop body with different constant `i` values.

You can use run-time expressions in the body of the loop (for example, in the
following example, the `list`, `threshold`, and `count` variables are all
run-time values).

```mojo
from collections import List
from random import rand

def main():
    alias LIST_SIZE = 128

    var list = List[Float64](length=LIST_SIZE, fill=0)
    rand(list.unsafe_ptr(), LIST_SIZE)

    var threshold = 0.6
    var count = 0

    @parameter
    for i in range(LIST_SIZE):
        if (list[i] > threshold):
            count += 1

    print(String("{} items over 0.6").format(count))
```

The `@parameter for` construct unrolls at the beginning of compilation, which
might explode the size of the program that still needs to be compiled, depending
on the amount of code that's unrolled.

Currently, `@parameter for` requires the sequence's `__iter__` method to
return a `_StridedRangeIterator`, meaning the induction variables must be
`Int`. The intention is to lift this restriction in the future.

## Parametric closure

You can add `@parameter` on a nested function to create a “parametric”
capturing closure. This means you can create a closure function that captures
values from the outer scope (regardless of whether they are variables or
parameters), and then use that closure as a parameter. For example:

```mojo
fn use_closure[func: fn(Int) capturing [_] -> Int](num: Int) -> Int:
    return func(num)

fn create_closure():
    var x = 1

    @parameter
    fn add(i: Int) -> Int:
        return x + i

    var y = use_closure[add](2)
    print(y)

create_closure()
```

```output
3
```

Without the `@parameter` decorator, you'll get a compiler error that says you
"cannot use a dynamic value in call parameter"—referring to the
`use_closure[add](2)` call—because the `add()` closure would still be dynamic.

Note the `[_]` in the function type:

```mojo
fn use_closure[func: fn(Int) capturing [_] -> Int](num: Int) -> Int:
```

This origin specifier represents the set of origins for the values captured by
the parametric closure. This allows the compiler to correctly extend the
lifetimes of those values. For more information on lifetimes and origins, see
[Lifetimes, origins and references](/mojo/manual/values/lifetimes).
