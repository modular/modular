---
markdown-notebook-data-directory: mdnb-data/manual-correspondence/
---

# Mojo ↔ IR ↔ C++ Correspondence

(This page is heavily WIP, check back later)

For any given piece of language semantics, there are three different domains in
which you will view things:

- Mojo source code (`fn main(): ...`)
- MLIR code (`lit.fn @”main()”() -> !kgen.none { ... }`)
- The compiler C++ code that produces that MLIR.

The goal of this section is to give you an intuition for how the same “thing” is
modeled in each of those domains. As a very basic example, consider the question
of how a named function call is represented.

## Setting the stage: The `sprongle` statement

This page will show you how to generate various operations, types, attrs, etc.

We'll generate those in `main`.

The easiest way to do that is to add a `sprongle` statement that we can call
from `main`, like so:

```mojo
fn main():
    sprongle
```

Doing that is pretty straightforward, just copy the `var` statement. Check out
[this PR](https://github.com/modularml/modular/pull/62701) to see it in action.

## Function Calls

```mojo
fn zork(i: Int):
  pass

fn main():
  zork(42)
```

<wolfram-cell ctext="Input17.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
    lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
      %0 = kgen.param.constant: !Int = <{42}>
      %1 = lit.call @main::@"zork(::Int)"(%0) : !lit.generator<("i": !Int) -> !kgen.none>
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_fn
    }
```

Let's see the C++ to generate that `zork(42)` call!

There's two ways to do this: the canonical way, and the easy way.

The canonical way:

- Lookup "zork" from the current scope.
- Assemble an OverloadSet containing the lookup results.
- Ask the OverloadSet to emit a call.

For example, here's how we would parse the `sprongle` statement in...

```mojo
fn zork(i: Int):
  pass

fn main():
  sprongle
```

...to turn it into a `zork(42)`:

```C++
SyntheticNode synthNode(smLoc);

ValueDest dest(EC_Sprongle);
std::string spelling = "zork";
LookupResult lookup = emitter.shared.lookupAndResolveDecl(
    spelling, smLoc, emitter.declScope, /*searchParentScopes=*/true);
if (lookup.isFailure()) {
  emitter.emitError(smLoc, "couldn't find function '") << spelling << "'";
  return failure();
}
ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
auto firstDecl = dyn_cast<FnOp>(decls[0]);
if (!firstDecl) {
  emitter.emitError(smLoc, "found a '")
      << spelling << "' but it wasn't a function";
  return failure();
}
auto result = OverloadSetUValue::create(
    spelling, decls, ParamBindings(emitter.getDeclScope()), &synthNode,
    CallSyntax::kDirectCall);
IntLiteralNode int42("42");
auto operands =
    CallOperands(std::vector<ASTExprAnd<AnyValue>>{ASTExprAnd<AnyValue>{
        emitter.emitExprRValue(&int42, EC_Sprongle),
        &int42,
    }});
result->emitCall(std::move(operands), dest, emitter);
```

(You cant use OverloadSet::lookup because there is no `ASTType` around.)

See this code in context
[here](https://github.com/modularml/modular/pull/62701).

The easier way to do all that is to conjure up some expression nodes, pretend
they came from the user, and emit them:

```c++
ValueDest dest(EC_Sprongle);
IntLiteralNode int42("42");
DeclRefNode zorkDeclRef("zork");
std::vector<Operand> operands = {
    Operand(&int42, smLoc, Operand::kPositional)
};
CallNode callNode(&zorkDeclRef, smLoc, ArrayRef<Operand>(operands), smLoc);
callNode.emitIR(dest, emitter);
```

See this code in context
[here](https://github.com/modularml/modular/pull/62701).

<!-- TODO: automatically extract this from the basics PR -->

<!-- TODO: don't inline code like that, extract it from the PR or preferably
from main -->

## Method Calls

The previous section called a normal top-level function, so let's see a method
call:

```mojo
struct Person:
    var name: String
    var age: Int

    fn __init__(inout self, owned name: String, age: Int):
        self.name = name^
        self.age = age

        self.greet()

    fn greet(self):
        pass


fn main():
    var me = Person("Connor", 25)
```

<wolfram-cell ctext="Input20.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.struct.decl @Person(!AnyType) attributes {sourceName = #Person_name}
      destructor :!lit.signature<[1]("self": !lit.ref<!Person, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @example::@Person::@"__del__(example::Person)" {
      lit.struct.field name : !String
      lit.struct.field age : !Int
      lit.func @"__init__(example::Person=&,stdlib::collections::string::String,::Int)"[mut *"self`2x", mut *"name`2x1"](%self: !lit.ref<!Person, mut *"self`2x"> init_self, %name: !lit.ref<!String, mut *"name`2x1"> owned_in_mem, %age: !Int) -> !kgen.none attributes {sourceName = "__init__", specialFnKind = 2 : i8} {
        lit.ownership.use %name : !lit.ref<!String, mut *"name`2x1">
        %0 = lit.ref.struct.ger %self[name] : <!Person, mut *"self`2x"> -> !String
        %1 = lit.call @stdlib::@collections::@string::@String::@"__moveinit__(stdlib::collections::string::String=&,stdlib::collections::string::String)"[mut *"self`2x"->name, mut *"name`2x1"](%0, %name) : !lit.signature<[2]("self": !lit.ref<!String, mut *[0,0]> init_self, "other": !lit.ref<!String, mut *[0,1]> owned_in_mem, |) -> !kgen.none>
        %2 = lit.ref.struct.ger %self[age] : <!Person, mut *"self`2x"> -> !Int
        lit.ref.store %age, %2 : <!Int, mut *"self`2x"->age>
        %3 = lit.ref.immut %self : <!Person, mut *"self`2x">
        %4 = lit.call @example::@Person::@"greet(example::Person)"[muttoimm *"self`2x"](%3) : !lit.signature<[1]("self": !lit.ref<!Person, imm *[0,0]> borrow_in_mem) -> !kgen.none>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
      }
      lit.func @"greet(example::Person)"[imm *"self`2x"](%self: !lit.ref<!Person, imm *"self`2x"> borrow_in_mem) -> !kgen.none attributes {sourceName = "greet", specialFnKind = 0 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
      }
      lit.func @"__del__(example::Person)"[mut *"self`"](%self: !lit.ref<!Person, mut *"self`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__del__", specialFnKind = 5 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.ownership.mark_destroyed %self : <!Person, mut *"self`">
        lit.return %none : !kgen.none
        lit.end_func
      }
    }
    lit.func @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
      %me = lit.var.decl "me" var : !lit.ref<!Person, mut *"me`">
      %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<!String, mut *"anonymous*`1">
      %0 = kgen.param.constant: !StringLiteral = <{:string "Connor"}>
      %1 = lit.call @stdlib::@collections::@string::@String::@"__init__(stdlib::collections::string::String=&,::StringLiteral)"[mut *"anonymous*`1"](%anonymous2A, %0) : !lit.signature<[1]("self": !lit.ref<!String, mut *[0,0]> init_self, "literal": !StringLiteral) -> !kgen.none>
      %2 = kgen.param.constant: !Int = <{25}>
      %3 = lit.call @example::@Person::@"__init__(example::Person=&,stdlib::collections::string::String,::Int)"[mut *"me`", mut *"anonymous*`1"](%me, %anonymous2A, %2) : !lit.signature<[2]("self": !lit.ref<!Person, mut *[0,0]> init_self, "name": !lit.ref<!String, mut *[0,1]> owned_in_mem, "age": !Int) -> !kgen.none>
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
    lit.func export C @main(%argc: !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>, %argv: !kgen.pointer<pointer<scalar<ui8>>>) -> !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>> attributes {linkageName = "main", sourceName = "__mojo_main_prototype", specialFnKind = 0 : i8} {
      %0 = lit.call @stdlib::@builtin::@_startup::@"__wrap_and_execute_main[fn() -> None](::SIMD[{int32}, {1}],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>)"<:!lit.signature<() -> !kgen.none> @example::@"main()">(%argc, %argv) : !lit.signature<("argc": !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>, "argv": !kgen.pointer<pointer<scalar<ui8>>>) -> !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>>
      lit.return %0 : !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>
      lit.end_func
    }
  }
  lit.package @stdlib { }
}
```

There are a few helpers for generating a method call via C++:

- If calling `__getitem__`, `__setitem__`, `__getattr__`, `__setattr__`, use
   `emitGetterSetterAccess`.
- If calling a constructor, use `IREmitter::emitConstructorCall`
- If calling a method, used `IREmitter::emitNamedMethodCall`

For other cases, use the approach in the above Function Calls section.

([source](https://modular-ai.slack.com/archives/C03GM7S2VMZ/p1748355046451549))

## Declaring a Variable

(WIP)

<!--
# declaring a local variable
fn main():
  var b = True
-->

```mojo
fn foo():
  var x: Int = 5
```

<wolfram-cell ctext="Input16.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.func @"foo()"() -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %x = lit.var.decl "x" var : !lit.ref<!Int, mut *"x`">
      %0 = kgen.param.constant: !Int = <{5}>
      lit.ref.store %0, %x : <!Int, mut *"x`">
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
  }
  lit.package @stdlib { }
}
```

## Reassigning a Local Variable

(WIP)

<!--
- reassigning
fn main():
  var x = 42
  var b = (x < 73)
  b = x > 73
-->

## Overloads

(WIP)

<!--
- overloads (should be included in that)
-->

```python
fn foo() -> Int:
  return 5

fn foo(arg: Int) -> Int:
  return arg + 5

fn main():
  var value = foo()
```

<wolfram-cell ctext="Input21.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.func @"foo()"() -> !Int attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %0 = kgen.param.constant: !Int = <{5}>
      lit.return %0 : !Int
      lit.end_func
    }
    lit.func @"foo(::Int)"(%arg: !Int) -> !Int attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %0 = kgen.param.constant: !Int = <{5}>
      %1 = lit.call @stdlib::@builtin::@int::@Int::@"__add__(::Int,::Int)"(%arg, %0) : !lit.signature<("self": !Int, "rhs": !Int) -> !Int>
      lit.return %1 : !Int
      lit.end_func
    }
    lit.func @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
      %value = lit.var.decl "value" var : !lit.ref<!Int, mut *"value`">
      %0 = lit.call @example::@"foo()"() : !lit.signature<() -> !Int>
      lit.ref.store %0, %value : <!Int, mut *"value`">
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
    lit.func export C @main(%argc: !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>, %argv: !kgen.pointer<pointer<scalar<ui8>>>) -> !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>> attributes {linkageName = "main", sourceName = "__mojo_main_prototype", specialFnKind = 0 : i8} {
      %0 = lit.call @stdlib::@builtin::@_startup::@"__wrap_and_execute_main[fn() -> None](::SIMD[{int32}, {1}],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>)"<:!lit.signature<() -> !kgen.none> @example::@"main()">(%argc, %argv) : !lit.signature<("argc": !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>, "argv": !kgen.pointer<pointer<scalar<ui8>>>) -> !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>>
      lit.return %0 : !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>
      lit.end_func
    }
  }
  lit.package @stdlib { }
}
```

## If Statement

(WIP)

<!--
- if statement
fn main():
  var b = True
  if b:
    print("hello")
  else:
    print("howdy")
-->

## Operators

(WIP)

<!--
- calling an operator
fn main():
  var x = 42
  var b = (x < 73)
  print(b)
-->

## Loops

(WIP)

<!--
- loop
fn main():
  for i in range(0, 3):
    print("hello")
but dont do this like the parser:
  //   var it = iterable.__iter__()
  //   while not it.__isatend__():
  //       var e = it.__next__()
  //       <BODY>
just do it like this:
  //   var i = 0
  //   while i < 3:
  //       print("hello")
  //       i = i + 1
-->

## Combining it all

(WIP)

<!--
- nested for-loop:
fn main():
  for row in range(0, 18):
    for col in range(0, 80):
      print(".")
    print("\n")
-->

<!--
- pulling it all together:
fn main():
  var player_row = 1
  var player_col = 2
  for row in range(0, 18):
    for col in range(0, 80):
      if row == player_row && col == player_col:
        print("@")
      else:
        print(".")
    print("\n")
-->

## Aliases

(WIP)

```python
alias MyInt64 = Scalar[DType.int64]
```

<wolfram-cell ctext="Input15.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.alias.decl *"MyInt64`0x": anystruct<#SIMD <:!DType {:dtype si64}, :!Int {1}>> = <@stdlib::@builtin::@simd::@SIMD<:!DType {:dtype si64}, :!Int {1}>>
  }
  lit.package @stdlib { }
}
```

<!--
- defining an alias:
fn main():
  alias initial_player_row = 1
  alias initial_player_col = 2
  var player_row = initial_player_row
  var player_col = initial_player_col
  for row in range(0, 18):
    for col in range(0, 80):
      if row == player_row && col == player_col:
        print("@")
      else:
        print(".")
    print("\n")
-->

## Defining a Function

(WIP)

<!--
- defining a function:
fn display(player_row: Int, player_col: Int):
  for row in range(0, 18):
    for col in range(0, 80):
      if row == player_row && col == player_col:
        print("@")
      else:
        print(".")
    print("\n")
fn main():
  alias initial_player_row = 1
  alias initial_player_col = 2
  var player_row = initial_player_row
  var player_col = initial_player_col
  display(player_row, player_col)
-->

```mojo
fn foo():
  pass
```

<wolfram-cell ctext="Input09.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.func @"foo()"() -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
  }
}
```

In the parser, you would call `StructEmitter::synthesizeFunction` (yes, even for
non-struct functions. We might soon rename that to `FnEmitter`).

<!--
- use a struct: n/a because weve already been using one
-->

## Structs

(WIP)

### Struct Declaration

(WIP)

```mojo
struct Person:
  var name: String
  var age: Int
```

<wolfram-cell ctext="Input13.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.struct.decl @Person(!AnyType) attributes {sourceName = #Person_name}
      destructor :!lit.signature<[1]("self": !lit.ref<!Person, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @example::@Person::@"__del__(example::Person)" {
      lit.struct.field name : !String
      lit.struct.field age : !Int
      lit.func @"__del__(example::Person)"[mut *"self`"](%self: !lit.ref<!Person, mut *"self`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__del__", specialFnKind = 5 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.ownership.mark_destroyed %self : <!Person, mut *"self`">
        lit.return %none : !kgen.none
        lit.end_func
      }
    }
  }
  lit.package @stdlib { }
}
```

### Struct Type Symbol Reference

(WIP)

`!Person` is used to reference the `@Person` declaration:

```python
struct Person:
  var name: String
  var age: Int

fn foo(x: Person):
  pass
```

<wolfram-cell ctext="Input14.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.struct.decl @Person(!AnyType) attributes {sourceName = #Person_name}
      destructor :!lit.signature<[1]("self": !lit.ref<!Person, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @example::@Person::@"__del__(example::Person)" {
      lit.struct.field name : !String
      lit.struct.field age : !Int
      lit.func @"__del__(example::Person)"[mut *"self`"](%self: !lit.ref<!Person, mut *"self`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__del__", specialFnKind = 5 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.ownership.mark_destroyed %self : <!Person, mut *"self`">
        lit.return %none : !kgen.none
        lit.end_func
      }
    }
    lit.func @"foo(example::Person)"[imm *"x`"](%x: !lit.ref<!Person, imm *"x`"> borrow_in_mem) -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
  }
  lit.package @stdlib { }
}
```

<!--
- structs:
  - declare a struct
      struct Player:
        var row: Int
        var col: Int
  - receive a struct
      fn display(player: Player):
        for row in range(0, 18):
          for col in range(0, 80):
            if row == player_row && col == player_col:
              print("@")
            else:
              print(".")
          print("\n")
  - construct a struct
      fn main():
        alias initial_player_row = 1
        alias initial_player_col = 2
        var player = Player(initial_player_row, initial_player_col)
        display(player)
-->

### Modify a Field

(WIP)

<!--
- modify struct:
    struct Player:
      var row: Int
      var col: Int
    fn display(player: Player):
      for row in range(0, 18):
        for col in range(0, 80):
          if row == player_row && col == player_col:
            print("@")
          else:
            print(".")
        print("\n")
    fn main():
      alias initial_player_row = 1
      alias initial_player_col = 2
      var player = Player(initial_player_row, initial_player_col)
      player.row = player.row + 1
      display(player)
-->

## Argument Conventions

(WIP)

<!-- **TODO:** Present this as a tab-view -->

### Borrowed

(WIP)

```python
fn foo(borrowed arg: String):
  pass
```

<wolfram-cell ctext="Input10.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.func @"foo(stdlib::collections::string::String)"[imm *"arg`"](%arg: !lit.ref<!String, imm *"arg`"> borrow_in_mem) -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
  }
  lit.package @stdlib { }
}
```

### Inout

(WIP)

```python
fn foo(inout arg: String):
  pass
```

<wolfram-cell ctext="Input11.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.func @"foo(stdlib::collections::string::String&)"[mut *"arg`"](%arg: !lit.ref<!String, mut *"arg`"> inout) -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
  }
  lit.package @stdlib { }
}
```

### Owned

(WIP)

```python
fn foo(owned arg: String):
  pass
```

<wolfram-cell ctext="Input12.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.func @"foo(stdlib::collections::string::String)"[mut *"arg`"](%arg: !lit.ref<!String, mut *"arg`"> owned_in_mem) -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
  }
  lit.package @stdlib { }
}
```

### Argument Conventions: Register Passability

(WIP)

<!--
Subtle that args vs ret type convention is different for
register-passable vs register-passable trivial. Write about this in
compiler manual.
-->

<!--
- register-passability
-->

<!--
- argument conventions
  - borrowed
  - inout
  - owned
show inout by adding a move function:
    struct Player:
      var row: Int
      var col: Int
    fn move_player(mut player: Player):
      player.row = player.row + 1
    fn display(player: Player):
      for row in range(0, 18):
        for col in range(0, 80):
          if row == player_row && col == player_col:
            print("@")
          else:
            print(".")
        print("\n")
    fn main():
      alias initial_player_row = 1
      alias initial_player_col = 2
      var player = Player(initial_player_row, initial_player_col)
      move_player(player)
      display(player)
-->

<!--
- reference types
-->

## Methods

(WIP)

<!--
- methods
    make move_player into a method on player.
-->

<!--
- declaration vs uses
- variadics
- raises
- using a generic struct such as tuple
- making a trait
  - conforming a struct to a trait

-->
