# Mojo ↔ IR ↔ C++ Correspondence

The goal of this section is to give you an intuition for how the same “thing” is
modeled in each of those domains. As a very basic example, consider the question
of how a named function call is represented.

**In Mojo**, you can call any named function that is in scope, through its
identifier:

```mojo
fn foo():
 bar()

fn bar():
 pass
```

**In MLIR code**, a named function is spelled as a @-prefixed “symbol
reference”:

$ kgen-translate --import-mojo example.mojo

```mlir
module {
    lit.file_module @example {
    lit.func @"foo()"() -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
        %0 = lit.call @example::@"bar()"() : !lit.signature<() -> !kgen.none>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
    }
    lit.func @"bar()"() -> !kgen.none attributes {sourceName = "bar", specialFnKind = 0 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
    }
    }
}
```

## Parsing to IR

### Basic

```mojo
fn foo():
 pass
```

<wolfram-cell ctext="Input09.wl" />

$ kgen-translate --import-mojo example.mojo

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

#### Argument Conventions

##### Borrowed

```python
fn foo(borrowed arg: String):
 pass
```

<wolfram-cell ctext="Input10.wl" />

$ kgen-translate --import-mojo example.mojo

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

##### Inout

```python
fn foo(inout arg: String):
 pass
```

<wolfram-cell ctext="Input11.wl" />

$ kgen-translate --import-mojo example.mojo

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

##### Owned

```python
fn foo(owned arg: String):
 pass
```

<wolfram-cell ctext="Input12.wl" />

$ kgen-translate --import-mojo example.mojo

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

#### Argument Conventions: Register Passability

```c++
// PRECOMMIT: Subtle that args vs ret type convention is different for
// register-passable vs register-passable trivial. Write about this in
// compiler manual.
```

#### Structs

##### Struct Declaration

```mojo
struct Person:
 var name: String
 var age: Int
```

<wolfram-cell ctext="Input13.wl" />

$ kgen-translate --import-mojo example.mojo

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

##### Struct Type Symbol Reference

`!Person` is used to reference the `@Person` declaration:

```python
struct Person:
 var name: String
 var age: Int

fn foo(x: Person):
 pass
```

<wolfram-cell ctext="Input14.wl" />

$ kgen-translate --import-mojo example.mojo

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

#### Aliases

```python
alias MyInt64 = Scalar[DType.int64]
```

<wolfram-cell ctext="Input15.wl" />

$ kgen-translate --import-mojo example.mojo

```mlir
module {
lit.file_module @example {
    lit.alias.decl *"MyInt64`0x": anystruct<#SIMD <:!DType {:dtype si64}, :!Int {1}>> = <@stdlib::@builtin::@simd::@SIMD<:!DType {:dtype si64}, :!Int {1}>>
}
lit.package @stdlib { }
}
```

#### Variable Declarations

```mojo
fn foo():
    var x: Int = 5
```

<wolfram-cell ctext="Input16.wl" />

$ kgen-translate --import-mojo example.mojo

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

#### Function Calls

```python
fn foo(c: String) -> Int:
 return ord(c)
```

<wolfram-cell ctext="Input17.wl" />

$ kgen-translate --import-mojo example.mojo

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

#### Reference Types

#### Methods

```python
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

$ kgen-translate --import-mojo example.mojo

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

#### Overloads

```python
fn foo() -> Int:
 return 5

fn foo(arg: Int) -> Int:
 return arg + 5

fn main():
 var value = foo()
```

<wolfram-cell ctext="Input21.wl" />

$ kgen-translate --import-mojo example.mojo

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

#### Variadics

- `fn foo(*args: Int)`
- `fn foo[T: Stringable](*args: T)`

##### `fn foo(*args: Int)`

```mojo
fn foo(*args: Int):
 pass
```

$ kgen-translate --import-mojo example.mojo

```mlir
module {
    lit.file_module @example {
    lit.func @"foo(::Int*)"(%args: !kgen.variadic<!Int> var) -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
        %args_0 = lit.var.decl "args" arg(0) : !lit.ref<@stdlib::@builtin::@stubs::@VariadicList<:type !Int>, mut *"args`">
        %0 = lit.call @stdlib::@builtin::@stubs::@VariadicList::@"__init__(::VariadicList[$0]=&,Variadic[$0])"[mut *"args`"]<:type !Int>(%args_0, %args) : !lit.signature<[1]("self": !lit.ref<@stdlib::@builtin::@stubs::@VariadicList<:type !Int>, mut *[0,0]> init_self, "value": !kgen.variadic<!Int>) -> !kgen.none>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
    }
    }
    lit.package @stdlib { }
}
```

##### `fn foo[T: Stringable](*args: T)`

```mojo
fn foo[T: AnyType](*args: T):
 pass
```

$ kgen-translate --import-mojo example.mojo

```mlir
module {
    lit.file_module @example {
    lit.func @"foo[::AnyType]($0*)"<T: !AnyType>[imm *"args`"](%args: !kgen.variadic<!lit.ref<:!AnyType T, imm *"args`">, read_mem> var) -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
        %args_0 = lit.var.decl "args" arg(0) : !lit.ref<@stdlib::@builtin::@stubs::@VariadicListMem<:!AnyType T, :i1 0, :origin<0> *"args`">, mut *"args`1">
        %0 = lit.call @stdlib::@builtin::@stubs::@VariadicListMem::@"__init__(::VariadicListMem[$0, $1, $2]=&,Variadic[ref [$2] $0])"[mut *"args`1"]<:!AnyType T, :i1 0, :origin<0> *"args`">(%args_0, %args) : !lit.signature<[1]("self": !lit.ref<@stdlib::@builtin::@stubs::@VariadicListMem<:!AnyType T, :i1 0, :origin<0> *"args`">, mut *[0,0]> init_self, "value": !kgen.variadic<!lit.ref<:!AnyType T, imm *"args`">, read_mem>) -> !kgen.none>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
    }
    }
    lit.package @stdlib { }
}
```

##### `fn foo[T: Stringable](*args: *T)`

```mojo
fn foo[
    *Ts: AnyType
](
    *values: *Ts,
):
 pass

fn bar():
 foo(1)
 foo(1, "two", 3.3)
```

$ kgen-translate --import-mojo example.mojo

```mlir
module {
    lit.file_module @example {
    lit.func @"foo[*::AnyType](*$0)"<Ts: variadic<!AnyType> var>[imm *"values`", imm *"values`1"](%values: !lit.ref<@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> *"values`", :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> Ts>, imm *"values`1"> read_mem|pack) -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
    }
    lit.func @"bar()"() -> !kgen.none attributes {sourceName = "bar", specialFnKind = 0 : i8} {
        %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<!Int, mut *"anonymous*`">
        %0 = kgen.param.constant: !Int = <{1}>
        lit.ref.store %0, %anonymous2A : <!Int, mut *"anonymous*`">
        %1 = lit.ref.immut %anonymous2A : <!Int, mut *"anonymous*`">
        %2 = lit.ref.pack.create(%1) : !lit.ref.pack<:variadic<!AnyType> [#Int1], muttoimm *"anonymous*`">
        %anonymous2A_0 = lit.var.decl "anonymous*" synth : !lit.ref<@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> (mutcast mut *"anonymous*`"), :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1]>, mut *"anonymous*`1">
        %3 = kgen.param.constant: !Bool = <{:i1 0}>
        %4 = lit.call @stdlib::@builtin::@stubs::@VariadicPack::@"__init__(::VariadicPack[$0, $1, $2, $3]=&,__mlir_type.!lit.ref.pack<:variadic<:!lit.anytrait<<_stdlib::_builtin::_stubs::_AnyType>> *(0,2)> *(0,3), mut=#lit.struct.extract<:_stdlib::_builtin::_stubs::_Bool *(0,0), \22value\22>, *(0,1)>,::Bool)"[mut *"anonymous*`1"]<:!Bool {:i1 0}, :origin<0> (mutcast mut *"anonymous*`"), :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1]>(%anonymous2A_0, %2, %3) : !lit.signature<[1]("self": !lit.ref<@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> (mutcast mut *"anonymous*`"), :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1]>, mut *[0,0]> init_self, "value": !lit.ref.pack<:variadic<!AnyType> [#Int1], muttoimm *"anonymous*`">, "is_owned": !Bool) -> !kgen.none>
        %5 = lit.ref.immut %anonymous2A_0 : <@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> (mutcast mut *"anonymous*`"), :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1]>, mut *"anonymous*`1">
        %6 = lit.call @example::@"foo[*::AnyType](*$0)"[muttoimm *"anonymous*`", muttoimm *"anonymous*`1"]<:variadic<!AnyType> [#Int1]>(%5) : !lit.signature<[2]("values": !lit.ref<@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> *[0,0], :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1]>, imm *[0,1]> read_mem|pack) -> !kgen.none>
        %anonymous2A_1 = lit.var.decl "anonymous*" synth : !lit.ref<!Int, mut *"anonymous*`2">
        %7 = kgen.param.constant: !Int = <{1}>
        lit.ref.store %7, %anonymous2A_1 : <!Int, mut *"anonymous*`2">
        %8 = lit.ref.immut %anonymous2A_1 : <!Int, mut *"anonymous*`2">
        %anonymous2A_2 = lit.var.decl "anonymous*" synth : !lit.ref<!StringLiteral, mut *"anonymous*`3">
        %9 = kgen.param.constant: !StringLiteral = <{:string "two"}>
        lit.ref.store %9, %anonymous2A_2 : <!StringLiteral, mut *"anonymous*`3">
        %10 = lit.ref.immut %anonymous2A_2 : <!StringLiteral, mut *"anonymous*`3">
        %anonymous2A_3 = lit.var.decl "anonymous*" synth : !lit.ref<!FloatDyn, mut *"anonymous*`4">
        %11 = kgen.param.constant: !FloatDyn = <{:f64 3.300000e+00}>
        lit.ref.store %11, %anonymous2A_3 : <!FloatDyn, mut *"anonymous*`4">
        %12 = lit.ref.immut %anonymous2A_3 : <!FloatDyn, mut *"anonymous*`4">
        %13 = kgen.rebind %8 : !lit.ref<!Int, muttoimm *"anonymous*`2"> to !lit.ref<!Int, imm {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}>
        %14 = kgen.rebind %10 : !lit.ref<!StringLiteral, muttoimm *"anonymous*`3"> to !lit.ref<!StringLiteral, imm {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}>
        %15 = kgen.rebind %12 : !lit.ref<!FloatDyn, muttoimm *"anonymous*`4"> to !lit.ref<!FloatDyn, imm {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}>
        %16 = lit.ref.pack.create(%13, %14, %15) : !lit.ref.pack<:variadic<!AnyType> [#Int1, #StringLiteral1, #FloatDyn1], imm {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}>
        %anonymous2A_4 = lit.var.decl "anonymous*" synth : !lit.ref<@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}, :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1, #StringLiteral1, #FloatDyn1]>, mut *"anonymous*`5">
        %17 = kgen.param.constant: !Bool = <{:i1 0}>
        %18 = lit.call @stdlib::@builtin::@stubs::@VariadicPack::@"__init__(::VariadicPack[$0, $1, $2, $3]=&,__mlir_type.!lit.ref.pack<:variadic<:!lit.anytrait<<_stdlib::_builtin::_stubs::_AnyType>> *(0,2)> *(0,3), mut=#lit.struct.extract<:_stdlib::_builtin::_stubs::_Bool *(0,0), \22value\22>, *(0,1)>,::Bool)"[mut *"anonymous*`5"]<:!Bool {:i1 0}, :origin<0> {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}, :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1, #StringLiteral1, #FloatDyn1]>(%anonymous2A_4, %16, %17) : !lit.signature<[1]("self": !lit.ref<@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}, :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1, #StringLiteral1, #FloatDyn1]>, mut *[0,0]> init_self, "value": !lit.ref.pack<:variadic<!AnyType> [#Int1, #StringLiteral1, #FloatDyn1], imm {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}>, "is_owned": !Bool) -> !kgen.none>
        %19 = lit.ref.immut %anonymous2A_4 : <@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}, :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1, #StringLiteral1, #FloatDyn1]>, mut *"anonymous*`5">
        %20 = lit.call @example::@"foo[*::AnyType](*$0)"[imm {(mutcast mut *"anonymous*`2"), (mutcast mut *"anonymous*`3"), (mutcast mut *"anonymous*`4")}, muttoimm *"anonymous*`5"]<:variadic<!AnyType> [#Int1, #StringLiteral1, #FloatDyn1]>(%19) : !lit.signature<[2]("values": !lit.ref<@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, :origin<0> *[0,0], :!lit.anytrait<!AnyType> !AnyType, :variadic<!AnyType> [#Int1, #StringLiteral1, #FloatDyn1]>, imm *[0,1]> read_mem|pack) -> !kgen.none>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
    }
    }
    lit.package @stdlib { }
}
```

### KGEN IR vs Generic MLIR

#### MLIR IR By Example

##### Typed Values

Our `lit.struct` is both a type and a value. When prefixed by !, that indicates
an MLIR type of the given name. When prefixed by #, that indicates an MLIR
attribute value of the given name. In this way, separately sigiled names exist
in independent name spaces in MLIR.

```mlir
#lit.struct<{value: i1 = 0}> : !lit.struct<@stdlib::@builtin::@bool::@Bool>
```

#### Inlining Behavior

Guaranteed inlining in Mojo enables a neat trick: you can return a pointer to
stack-allocated data that is valid in the frame of the _caller_, as long as the
function calling `stack_allocation()` is marked with `@always_inline`:

```python
from memory import stack_allocation, UnsafePointer

fn main():
 var mem = foo()
 mem.init_pointee_move(5)
 print(mem[])

@always_inline
fn foo() -> UnsafePointer[Int]:
 var data = stack_allocation[1, Int]()
 return data
```

$ kgen-translate --import-mojo example.mojo

```mlir
module {
    lit.file_module @example {
    lit.func @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
        %mem = lit.var.decl "mem" var : !lit.ref<@stdlib::@memory::@unsafe_pointer::@UnsafePointer<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>, mut *"mem`">
        %0 = lit.call @example::@"foo()"() : !lit.signature<() -> !lit.struct<#UnsafePointer <:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>>>
        lit.ref.store %0, %mem : <@stdlib::@memory::@unsafe_pointer::@UnsafePointer<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>, mut *"mem`">
        %1 = lit.ref.load %mem : <@stdlib::@memory::@unsafe_pointer::@UnsafePointer<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>, mut *"mem`">
        %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<!Int, mut *"anonymous*`1">
        %2 = kgen.param.constant: !Int = <{5}>
        lit.ref.store %2, %anonymous2A : <!Int, mut *"anonymous*`1">
        %3 = lit.call @stdlib::@memory::@unsafe_pointer::@UnsafePointer::@"init_pointee_move[::Movable,::Int,MutableOrigin](stdlib::memory::unsafe_pointer::UnsafePointer[$4, {{0}}, $5, $6],$4)"[mut *"anonymous*`1"]<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin, :!Movable #Int2, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>(%1, %anonymous2A) : !lit.signature<[1]("self": !lit.struct<#UnsafePointer <:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>>, "value": !lit.ref<!Int, mut *[0,0]> owned_in_mem) -> !kgen.none>
        %4 = lit.ref.load %mem : <@stdlib::@memory::@unsafe_pointer::@UnsafePointer<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>, mut *"mem`">
        %5 = lit.call @stdlib::@memory::@unsafe_pointer::@UnsafePointer::@"__getitem__(stdlib::memory::unsafe_pointer::UnsafePointer[$0, $1, $2, $3])"<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>(%4) : !lit.signature<("self": !lit.struct<#UnsafePointer <:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>>) refresult -> !lit.ref<!Int, mut #lit.any.origin>>
        %6 = lit.ref.immut %5 : <!Int, mut #lit.any.origin>
        %7 = lit.ref.pack.create(%6) : !lit.ref.pack<:variadic<!Writable> [#Int3], imm #lit.any.origin>
        %anonymous2A_0 = lit.var.decl "anonymous*" synth : !lit.ref<@stdlib::@builtin::@builtin_list::@VariadicPack<:!Bool {:i1 0}, :origin<0> #lit.any.origin, :!lit.anytrait<!AnyType> !Writable, :variadic<!Writable> [#Int3]>, mut *"anonymous*`2">
        %8 = kgen.param.constant: !Bool = <{:i1 0}>
        %9 = lit.call @stdlib::@builtin::@builtin_list::@VariadicPack::@"__init__(::VariadicPack[$0, $1, $2, $3]=&,__mlir_type.!lit.ref.pack<:variadic<:!lit.anytrait<<_stdlib::_builtin::_anytype::_AnyType>> *(0,2)> *(0,3), mut=#lit.struct.extract<:_stdlib::_builtin::_bool::_Bool *(0,0), \22value\22>, *(0,1)>,::Bool)"[mut *"anonymous*`2"]<:!Bool {:i1 0}, :origin<0> #lit.any.origin, :!lit.anytrait<!AnyType> !Writable, :variadic<!Writable> [#Int3]>(%anonymous2A_0, %7, %8) : !lit.signature<[1]("self": !lit.ref<@stdlib::@builtin::@builtin_list::@VariadicPack<:!Bool {:i1 0}, :origin<0> #lit.any.origin, :!lit.anytrait<!AnyType> !Writable, :variadic<!Writable> [#Int3]>, mut *[0,0]> init_self, "value": !lit.ref.pack<:variadic<!Writable> [#Int3], imm #lit.any.origin>, "is_owned": !Bool) -> !kgen.none>
        %10 = lit.ref.immut %anonymous2A_0 : <@stdlib::@builtin::@builtin_list::@VariadicPack<:!Bool {:i1 0}, :origin<0> #lit.any.origin, :!lit.anytrait<!AnyType> !Writable, :variadic<!Writable> [#Int3]>, mut *"anonymous*`2">
        %anonymous2A_1 = lit.var.decl "anonymous*" synth : !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *"anonymous*`3">
        %11 = kgen.param.materialize: @stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">> = <apply_result_slot(:!lit.signature<[1]("self": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut {}> init_self, "lit": !StringLiteral) -> !kgen.none> rebind(:!lit.signature<[1]("self": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *[0,0]> init_self, "lit": !StringLiteral) -> !kgen.none> @stdlib::@utils::@string_slice::@StringSlice::@"__init__(stdlib::utils::string_slice::StringSlice[{0}, StaticConstantOrigin]=&,::StringLiteral)"<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>), {:string " "})>
        lit.ref.store %11, %anonymous2A_1 : <@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *"anonymous*`3">
        %12 = lit.ref.immut %anonymous2A_1 : <@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *"anonymous*`3">
        %anonymous2A_2 = lit.var.decl "anonymous*" synth : !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *"anonymous*`4">
        %13 = kgen.param.materialize: @stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">> = <apply_result_slot(:!lit.signature<[1]("self": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut {}> init_self, "lit": !StringLiteral) -> !kgen.none> rebind(:!lit.signature<[1]("self": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *[0,0]> init_self, "lit": !StringLiteral) -> !kgen.none> @stdlib::@utils::@string_slice::@StringSlice::@"__init__(stdlib::utils::string_slice::StringSlice[{0}, StaticConstantOrigin]=&,::StringLiteral)"<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>), {:string "\0A"})>
        lit.ref.store %13, %anonymous2A_2 : <@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *"anonymous*`4">
        %14 = lit.ref.immut %anonymous2A_2 : <@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *"anonymous*`4">
        %15 = kgen.param.constant: !Bool = <{:i1 0}>
        %anonymous2A_3 = lit.var.decl "anonymous*" synth : !lit.ref<!FileDescriptor, mut *"anonymous*`5">
        %16 = kgen.param.constant: !FileDescriptor = <apply_result_slot(:!lit.signature<[1]("self": !lit.ref<!FileDescriptor, mut {}> init_self, "value": !Int = {1}) -> !kgen.none> rebind(:!lit.signature<[1]("self": !lit.ref<!FileDescriptor, mut *[0,0]> init_self, "value": !Int = {1}) -> !kgen.none> @stdlib::@builtin::@file_descriptor::@FileDescriptor::@"__init__(::FileDescriptor=&,::Int)"), {1})>
        lit.ref.store %16, %anonymous2A_3 : <!FileDescriptor, mut *"anonymous*`5">
        %17 = lit.call @stdlib::@builtin::@io::@"print[*stdlib::utils::write::Writable](*$0,stdlib::utils::string_slice::StringSlice[{0}, StaticConstantOrigin],stdlib::utils::string_slice::StringSlice[{0}, StaticConstantOrigin],::Bool,::FileDescriptor)"[imm #lit.any.origin, muttoimm *"anonymous*`2", muttoimm *"anonymous*`3", muttoimm *"anonymous*`4", mut *"anonymous*`5"]<:variadic<!Writable> [#Int3]>(%10, %12, %14, %15, %anonymous2A_3) : !lit.signature<[5]("values": !lit.ref<@stdlib::@builtin::@builtin_list::@VariadicPack<:!Bool {:i1 0}, :origin<0> *[0,0], :!lit.anytrait<!AnyType> !Writable, :variadic<!Writable> [#Int3]>, imm *[0,1]> borrow_in_mem|pack, *, "sep": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, imm *[0,2]> borrow_in_mem = apply_result_slot(:!lit.signature<[1]("self": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut {}> init_self, "lit": !StringLiteral) -> !kgen.none> rebind(:!lit.signature<[1]("self": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *[0,0]> init_self, "lit": !StringLiteral) -> !kgen.none> @stdlib::@utils::@string_slice::@StringSlice::@"__init__(stdlib::utils::string_slice::StringSlice[{0}, StaticConstantOrigin]=&,::StringLiteral)"<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>), {:string " "}), "end": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, imm *[0,3]> borrow_in_mem = apply_result_slot(:!lit.signature<[1]("self": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut {}> init_self, "lit": !StringLiteral) -> !kgen.none> rebind(:!lit.signature<[1]("self": !lit.ref<@stdlib::@utils::@string_slice::@StringSlice<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>, mut *[0,0]> init_self, "lit": !StringLiteral) -> !kgen.none> @stdlib::@utils::@string_slice::@StringSlice::@"__init__(stdlib::utils::string_slice::StringSlice[{0}, StaticConstantOrigin]=&,::StringLiteral)"<:!Bool {:i1 0}, :origin<0> #lit.origin.field<#lit.static.origin : !lit.origin<0>, "__constants__">>), {:string "\0A"}), "flush": !Bool = {:i1 0}, "file": !lit.ref<!FileDescriptor, mut *[0,4]> owned_in_mem = apply_result_slot(:!lit.signature<[1]("self": !lit.ref<!FileDescriptor, mut {}> init_self, "value": !Int = {1}) -> !kgen.none> rebind(:!lit.signature<[1]("self": !lit.ref<!FileDescriptor, mut *[0,0]> init_self, "value": !Int = {1}) -> !kgen.none> @stdlib::@builtin::@file_descriptor::@FileDescriptor::@"__init__(::FileDescriptor=&,::Int)"), {1})) -> !kgen.none>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
    }
    lit.func @"foo()"() -> !lit.struct<#UnsafePointer <:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>> always_inline attributes {sourceName = "foo", specialFnKind = 0 : i8} {
        %data = lit.var.decl "data" var : !lit.ref<@stdlib::@memory::@unsafe_pointer::@UnsafePointer<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>, mut *"data`">
        %0 = lit.call @stdlib::@memory::@memory::@"stack_allocation[::Int,::AnyType,stdlib::collections::optional::Optional[::StringLiteral],::Int,stdlib::memory::pointer::AddressSpace]()"<:!Int {1}, :!AnyType #Int1, :@stdlib::@collections::@optional::@Optional<:!CollectionElement #StringLiteral1> apply_result_slot(:!lit.signature<[1]("self": !lit.ref<@stdlib::@collections::@optional::@Optional<:!CollectionElement #StringLiteral1>, mut {}> init_self, "value": !kgen.none) -> !kgen.none> rebind(:!lit.signature<[1]("self": !lit.ref<@stdlib::@collections::@optional::@Optional<:!CollectionElement #StringLiteral1>, mut *[0,0]> init_self, "value": !kgen.none) -> !kgen.none> @stdlib::@collections::@optional::@Optional::@"__init__(stdlib::collections::optional::Optional[$0]=&,None)"<:!CollectionElement #StringLiteral1>), #kgen.none), :!Int apply_result_slot(:!lit.signature<[1]("self": !lit.ref<!Int, mut {}> init_self, "value": !IntLiteral) -> !kgen.none> rebind(:!lit.signature<[1]("self": !lit.ref<!Int, mut *[0,0]> init_self, "value": !IntLiteral) -> !kgen.none> @stdlib::@builtin::@int::@Int::@"__init__(::Int=&,::IntLiteral)"), cond(apply(:!lit.signature<("self": !Bool) -> i1> @stdlib::@builtin::@bool::@Bool::@"__mlir_i1__(::Bool)", apply(:!lit.signature<() -> !Bool> @stdlib::@sys::@info::@"is_nvidia_gpu()")), apply(:!lit.signature<() -> !IntLiteral> @stdlib::@sys::@info::@"alignof[::AnyType,__mlir_type.!kgen.target]()"<:!AnyType #Int1, :target current_target()>), {:!kgen.int_literal 1})), :!AddressSpace {_value: !Int = {0}}>() : !lit.signature<() -> !lit.struct<#UnsafePointer <:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>>>
        lit.ref.store %0, %data : <@stdlib::@memory::@unsafe_pointer::@UnsafePointer<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>, mut *"data`">
        %1 = lit.ref.load %data : <@stdlib::@memory::@unsafe_pointer::@UnsafePointer<:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>, mut *"data`">
        lit.return %1 : !lit.struct<#UnsafePointer <:!AnyType #Int1, :!AddressSpace {_value: !Int = {0}}, :!Int apply(:!lit.signature<() -> !Int> @stdlib::@memory::@unsafe_pointer::@"_default_alignment[::AnyType]()"<:!AnyType #Int1>), :origin<1> #lit.any.origin>>
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

```wolfram,cell:Input
BuildSourceCode["Mojo"]
```

<wolfram-cell ctext="Input25.wl" />

### IR Navigation Techniques

Reading MLIR is often difficult, due to the verbose nature of IR. This section
documents a few low-tech techniques for navigating IR.

#### Parsed IR

- Grepping for a struct definition: `lit.struct.decl @Foo`

- Grepping for a function definition: `lit.func @”foo(`

#### Elaborated IR

#### How method signatures vary based on value category

The examples below show how the signature of the `copy()` method changes
depending on whether a type is memory-only, register-passable, or
register-passable trivial.

```mojo
trait MyExplicitlyCopyable:
 fn my_copy(self) -> Self:
  ...

@value
@register_passable("trivial")
struct Foo(MyExplicitlyCopyable):
 var data: Int

 fn my_copy(self) -> Self:
  return self
```

##### Register Passable (Trivial)

$ kgen-translate --import-mojo example.mojo

```mlir
module {
    lit.file_module @example {
    lit.trait.decl @MyExplicitlyCopyable<?, *"_Self`": !MyExplicitlyCopyable>(!UnknownDestructibility, !AnyType)  unspecified attributes {dtorSig = !kgen.signature<!lit.signature<[1]<!MyExplicitlyCopyable, |>("self": !lit.ref<:!MyExplicitlyCopyable *(0,0), mut *[0,0]> owned_in_mem, |) -> !kgen.none>>} {
        lit.func @"my_copy($0)"[imm *"self`2x", mut *"__result__`2x1"](%self: !lit.ref<:!MyExplicitlyCopyable *"_Self`", imm *"self`2x"> read_mem, ?, %__result__: !lit.ref<:!MyExplicitlyCopyable *"_Self`", mut *"__result__`2x1"> byref_result) -> !kgen.none attributes {sourceName = "my_copy", specialFnKind = 0 : i8} {
        lit.trait_func
        }
        lit.func @"__del__($0)"[mut *"self`2x"](%self: !lit.ref<:!MyExplicitlyCopyable *"_Self`", mut *"self`2x"> owned_in_mem, |) -> !kgen.none attributes {isInherited, sourceName = "__del__", specialFnKind = 5 : i8} {
        lit.trait_func
        }
    }
    lit.struct.decl @Foo(!MyExplicitlyCopyable, !UnknownDestructibility[!MyExplicitlyCopyable], !AnyType[!MyExplicitlyCopyable], !Copyable, !Movable) register_passable_trivial attributes {sourceName = #Foo_name} {
        lit.struct.field data : !Int
        lit.func @"my_copy(example::Foo)"(%self: !Foo) -> !Foo attributes {sourceName = "my_copy", specialFnKind = 0 : i8} {
        lit.return %self : !Foo
        lit.end_func
        }
        lit.func @"__init__(example::Foo=&,::Int)"[mut *"self`"](%0[*""]: !lit.ref<!Foo, mut *"self`"> init_self, |, %data: !Int) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__init__", specialFnKind = 2 : i8} {
        %1 = lit.ref.struct.ger %0[data] : <!Foo, mut *"self`"> -> !Int
        lit.ref.store %data, %1 : <!Int, mut *"self`"->data>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
        }
        lit.func @"__del__(example::Foo)_thunk"[mut *"self`"](%0[*""]: !lit.ref<!Foo, mut *"self`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__del__", specialFnKind = 5 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        kgen.return %none : !kgen.none
        }
        lit.func @"__copyinit__(example::Foo=&,example::Foo)_thunk"[mut *"self`", imm *"existing`"](%0[*""]: !lit.ref<!Foo, mut *"self`"> init_self, %1[*""]: !lit.ref<!Foo, imm *"existing`"> read_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__copyinit__", specialFnKind = 3 : i8} {
        %2 = lit.ref.load %1 : <!Foo, imm *"existing`">
        lit.ref.store %2, %0 : <!Foo, mut *"self`">
        %none = kgen.param.constant: none = <#kgen.none>
        kgen.return %none : !kgen.none
        }
        lit.func @"__moveinit__(example::Foo=&,example::Foo)_thunk"[mut *"self`", mut *"existing`"](%0[*""]: !lit.ref<!Foo, mut *"self`"> init_self, %1[*""]: !lit.ref<!Foo, mut *"existing`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__moveinit__", specialFnKind = 4 : i8} {
        %2 = lit.load.consume %1 : !lit.ref<!Foo, mut *"existing`">
        lit.ref.store %2, %0 : <!Foo, mut *"self`">
        %none = kgen.param.constant: none = <#kgen.none>
        kgen.return %none : !kgen.none
        }
    }
    }
    lit.package @stdlib { }
}
```

##### Register Passable (Non-Trivial)

$ kgen-translate --import-mojo example.mojo

```mlir
module {
    lit.file_module @example {
    lit.struct.decl @Foo(!UnknownDestructibility, !Copyable, !AnyType[!Copyable], !Movable) register_passable attributes {sourceName = #Foo_name}
        destructor :!lit.signature<[1]("self": !lit.ref<!Foo, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @example::@Foo::@"__del__(example::Foo)"
    copy :!lit.signature<[2]("self": !lit.ref<!Foo, mut *[0,0]> init_self, "other": !lit.ref<!Foo, imm *[0,1]> read_mem, |) -> !kgen.none> @example::@Foo::@"__copyinit__(example::Foo=&,example::Foo)" {
        lit.struct.field data : !Int
        lit.func @"copy(example::Foo)"[imm *"self`2x"](%self: !lit.ref<!Foo, imm *"self`2x"> read_mem) -> !Foo attributes {sourceName = "copy", specialFnKind = 0 : i8} {
        %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<!Foo, mut *"anonymous*`2x1">
        %0 = lit.call @example::@Foo::@"__copyinit__(example::Foo=&,example::Foo)"[mut *"anonymous*`2x1", imm *"self`2x"](%anonymous2A, %self) : !lit.signature<[2]("self": !lit.ref<!Foo, mut *[0,0]> init_self, "other": !lit.ref<!Foo, imm *[0,1]> read_mem, |) -> !kgen.none>
        %1 = lit.load.consume %anonymous2A : !lit.ref<!Foo, mut *"anonymous*`2x1">
        lit.return %1 : !Foo
        lit.end_func
        }
        lit.func @"__del__(example::Foo)"[mut *"self`"](%self: !lit.ref<!Foo, mut *"self`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__del__", specialFnKind = 5 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.ownership.mark_destroyed %self : <!Foo, mut *"self`">
        lit.return %none : !kgen.none
        lit.end_func
        }
        lit.func @"__copyinit__(example::Foo=&,example::Foo)"[mut *"self`", imm *"existing`"](%self: !lit.ref<!Foo, mut *"self`"> init_self, %other: !lit.ref<!Foo, imm *"existing`"> read_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__copyinit__", specialFnKind = 3 : i8} {
        %0 = lit.ref.struct.ger %self[data] : <!Foo, mut *"self`"> -> !Int
        %1 = lit.ref.struct.ger %other[data] : <!Foo, imm *"existing`"> -> !Int
        %2 = lit.ref.load %1 : <!Int, imm *"existing`"->data>
        lit.ref.store %2, %0 : <!Int, mut *"self`"->data>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
        }
        lit.func @"__init__(example::Foo=&,::Int)"[mut *"self`"](%0[*""]: !lit.ref<!Foo, mut *"self`"> init_self, |, %data: !Int) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__init__", specialFnKind = 2 : i8} {
        %1 = lit.ref.struct.ger %0[data] : <!Foo, mut *"self`"> -> !Int
        lit.ref.store %data, %1 : <!Int, mut *"self`"->data>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
        }
        lit.func @"__moveinit__(example::Foo=&,example::Foo)_thunk"[mut *"self`", mut *"existing`"](%0[*""]: !lit.ref<!Foo, mut *"self`"> init_self, %1[*""]: !lit.ref<!Foo, mut *"existing`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__moveinit__", specialFnKind = 4 : i8} {
        %2 = lit.load.consume %1 : !lit.ref<!Foo, mut *"existing`">
        lit.ref.store %2, %0 : <!Foo, mut *"self`">
        %none = kgen.param.constant: none = <#kgen.none>
        kgen.return %none : !kgen.none
        }
    }
    }
    lit.package @stdlib { }
}
```

##### Memory Only

$ kgen-translate --import-mojo example.mojo

```mlir
module {
    lit.file_module @example {
    lit.struct.decl @Foo(!UnknownDestructibility, !Copyable, !AnyType[!Copyable], !Movable) attributes {sourceName = #Foo_name}
        destructor :!lit.signature<[1]("self": !lit.ref<!Foo, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @example::@Foo::@"__del__(example::Foo)"
    move :!lit.signature<[2]("self": !lit.ref<!Foo, mut *[0,0]> init_self, "other": !lit.ref<!Foo, mut *[0,1]> owned_in_mem, |) -> !kgen.none> @example::@Foo::@"__moveinit__(example::Foo=&,example::Foo)"
    copy :!lit.signature<[2]("self": !lit.ref<!Foo, mut *[0,0]> init_self, "other": !lit.ref<!Foo, imm *[0,1]> read_mem, |) -> !kgen.none> @example::@Foo::@"__copyinit__(example::Foo=&,example::Foo)" {
        lit.struct.field data : !Int
        lit.func @"copy(example::Foo)"[imm *"self`2x", mut *"__result__`2x1"](%self: !lit.ref<!Foo, imm *"self`2x"> read_mem, ?, %__result__: !lit.ref<!Foo, mut *"__result__`2x1"> byref_result) -> !kgen.none attributes {sourceName = "copy", specialFnKind = 0 : i8} {
        %0 = lit.call @example::@Foo::@"__copyinit__(example::Foo=&,example::Foo)"[mut *"__result__`2x1", imm *"self`2x"](%__result__, %self) : !lit.signature<[2]("self": !lit.ref<!Foo, mut *[0,0]> init_self, "other": !lit.ref<!Foo, imm *[0,1]> read_mem, |) -> !kgen.none>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
        }
        lit.func @"__del__(example::Foo)"[mut *"self`"](%self: !lit.ref<!Foo, mut *"self`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__del__", specialFnKind = 5 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.ownership.mark_destroyed %self : <!Foo, mut *"self`">
        lit.return %none : !kgen.none
        lit.end_func
        }
        lit.func @"__moveinit__(example::Foo=&,example::Foo)"[mut *"self`", mut *"existing`"](%self: !lit.ref<!Foo, mut *"self`"> init_self, %other: !lit.ref<!Foo, mut *"existing`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__moveinit__", specialFnKind = 4 : i8} {
        %0 = lit.ref.struct.ger %self[data] : <!Foo, mut *"self`"> -> !Int
        %1 = lit.ref.struct.ger %other[data] : <!Foo, mut *"existing`"> -> !Int
        %2 = lit.load.consume %1 : !lit.ref<!Int, mut *"existing`"->data>
        lit.ref.store %2, %0 : <!Int, mut *"self`"->data>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.ownership.mark_destroyed %other : <!Foo, mut *"existing`">
        lit.return %none : !kgen.none
        lit.end_func
        }
        lit.func @"__copyinit__(example::Foo=&,example::Foo)"[mut *"self`", imm *"existing`"](%self: !lit.ref<!Foo, mut *"self`"> init_self, %other: !lit.ref<!Foo, imm *"existing`"> read_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__copyinit__", specialFnKind = 3 : i8} {
        %0 = lit.ref.struct.ger %self[data] : <!Foo, mut *"self`"> -> !Int
        %1 = lit.ref.struct.ger %other[data] : <!Foo, imm *"existing`"> -> !Int
        %2 = lit.ref.load %1 : <!Int, imm *"existing`"->data>
        lit.ref.store %2, %0 : <!Int, mut *"self`"->data>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
        }
        lit.func @"__init__(example::Foo=&,::Int)"[mut *"self`"](%0[*""]: !lit.ref<!Foo, mut *"self`"> init_self, |, %data: !Int) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__init__", specialFnKind = 2 : i8} {
        %1 = lit.ref.struct.ger %0[data] : <!Foo, mut *"self`"> -> !Int
        lit.ref.store %data, %1 : <!Int, mut *"self`"->data>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
        }
    }
    }
    lit.package @stdlib { }
}
```
