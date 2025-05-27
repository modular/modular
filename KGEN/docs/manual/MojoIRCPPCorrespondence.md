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

## Adding a Flag

<!--
# adding a flag
--enable_game
-->

## Adding a Statement

<!--
# adding a statement
fn main():
  game
-->

## Declaring a Variable

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

<!--
- reassigning
fn main():
  var x = 42
  var b = (x < 73)
  b = x > 73
-->

## Function Calls

<!--
- calling a function
fn main():
  var b = True
  print(x)
-->

```python
fn foo(c: String) -> Int:
  return ord(c)
```

<wolfram-cell ctext="Input17.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.func @"foo(stdlib::collections::string::String)"[imm *"c`"](%c: !lit.ref<!String, imm *"c`"> borrow_in_mem) -> !Int attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %0 = lit.call @stdlib::@collections::@string::@"ord(stdlib::collections::string::String)"[imm *"c`"](%c) : !lit.signature<[1]("s": !lit.ref<!String, imm *[0,0]> borrow_in_mem) -> !Int>
      lit.return %0 : !Int
      lit.end_func
    }
  }
  lit.package @stdlib { }
}
```

When we see a “lit” instruction like `lit.call`, that’s not something built into
MLIR, that’s something we define (in
[KGEN/include/KGEN/LITDialect/LITOps.td](https://github.com/modularml/modular/blob/main/KGEN/include/KGEN/LITDialect/LITOps.td#L159)
actually). Same with kgen instructions, which are defined in
[KGEN/include/KGEN/KGENDialect/KGENOps.td](https://github.com/modularml/modular/blob/main/KGEN/include/KGEN/KGENDialect/KGENOps.td#L39).

<wolfram-cell ctext="Input18.wl" />

<wolfram-cell ctext="Input19.wl" />

## Overloads

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

<!--
- calling an operator
fn main():
  var x = 42
  var b = (x < 73)
  print(b)
-->

## Loops

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

### Struct Declaration

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

<!-- **TODO:** Present this as a tab-view -->

### Borrowed

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

<!--
- methods
    make move_player into a method on player.
-->

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

<!--
- declaration vs uses
- variadics
- raises
- using a generic struct such as tuple
- making a trait
  - conforming a struct to a trait

-->
