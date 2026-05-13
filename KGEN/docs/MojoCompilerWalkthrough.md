# Mojo Compiler Walkthrough

A comprehensive guide for compiler engineers new to the KGEN/Mojo codebase.

**Resources:**
[Video walkthrough](https://drive.google.com/file/d/1NAh7RpJDXbWMlapHPZce8CAVSgv8D_oX/view?usp=drive_link)
|
[Slides](https://docs.google.com/presentation/d/1cc9dN-7u9dS661jqIGuvGRheWrzfLb6eyMgquRWHrzs/edit?usp=drive_link)

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Phase 1: Parsing and Type Checking](#phase-1-parsing-and-type-checking)
4. [Phase 2: Semantic Checking and LIT Lowering](#phase-2-semantic-checking-and-lit-lowering)
5. [Phase 3: Pre-Elaboration Optimization](#phase-3-pre-elaboration-optimization)
6. [Phase 4: Elaboration (Monomorphization)](#phase-4-elaboration-monomorphization)
7. [Phase 5: Post-Elaboration Lowering & Optimization](#phase-5-post-elaboration-lowering--optimization)
8. [Phase 6: Lowering to LLVM](#phase-6-lowering-to-llvm)
9. [Mojo Packages and Precompiled Files](#mojo-packages-and-precompiled-files)
10. [Debug Information](#debug-information)
    - [Parametric Debug Info](#parametric-debug-info)
    - [How Passes Preserve Debug Info](#how-passes-preserve-debug-info)
11. [MLIR Dialects Reference](#mlir-dialects-reference)
12. [Key Passes Summary](#key-passes-summary)
13. [Developer Tools](#developer-tools)
14. [Additional Resources](#additional-resources)

---

## Overview

The Mojo compiler (KGEN - "Kernel Generator") is built entirely on top of
**MLIR** (Multi-Level Intermediate Representation). Unlike traditional
compilers with distinct AST → IR → machine code phases, Mojo uses MLIR
dialects to represent code at different abstraction levels, all within the
same framework.

### Build & Run Commands

```bash
# Parse Mojo to LIT IR
br //KGEN/tools/kgen-translate -- -import-mojo main.mojo

# Run specific passes
kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit input.mlir

# Full compilation
mojo build main.mojo

# Elaborate without codegen
kgen --elaborate main.mojo

# See all passes that run
kgen --mlir-print-ir-before-all -elaborate main.mojo 2>&1 | grep 'IR Dump Before'
```

---

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MOJO SOURCE CODE                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PARSING & TYPE CHECKING                                           │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────────┐  │
│  │    Lexer    │───▶│  Three-Phase     │───▶│  IR Emission               │  │
│  │             │    │  Parser          │    │  (ExprNodes)               │  │
│  └─────────────┘    └──────────────────┘    └────────────────────────────┘  │
│                                                                             │
│  Output: LIT Dialect (parametric, source-level IR)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: SEMANTIC CHECKING & LIT LOWERING                                  │
│  ┌────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐   │
│  │ LowerSemanticCF│───▶│ CheckLifetimes  │───▶│ LowerLIT                │   │
│  │                │    │ (Borrow Check)  │    │ (lit → kgen)            │   │
│  └────────────────┘    └─────────────────┘    └─────────────────────────┘   │
│                                                                             │
│  Output: KGEN Dialect (parametric IR, no lit ops)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: PRE-ELABORATION OPTIMIZATION                                      │
│  ┌──────────┐ ┌─────────┐ ┌────────────────────────┐                        │
│  │   SROA   │ │ Mem2Reg │ │ RemoveUnusedParams     │                        │
│  └──────────┘ └─────────┘ └────────────────────────┘                        │
│  ┌─────────────────┐ ┌──────────────┐                                       │
│  │ InlineParametric│ │ ApplyInliner │                                       │
│  └─────────────────┘ └──────────────┘                                       │
│  Purpose: Reduce IR size and complexity before elaboration                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: ELABORATION (MONOMORPHIZATION)                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ElaborateGenerators Pass                                              │ │
│  │  • Instantiates all generators (functions and structs) with params     │ │
│  │  • Evaluates compile-time code via Interpreter                         │ │
│  │  • Runs in parallel                                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Output: kgen.func + kgen.struct.instance (concrete, no parameters)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: POST-ELABORATION LOWERING & OPTIMIZATION                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Lowering:  Arg Conventions, Calling Conventions                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────┐ ┌─────────┐ ┌─────────┐ ┌────────────┐ ┌──────────────────┐   │
│  │   SROA   │ │ Mem2Reg │ │  SCCP   │ │ Inlining   │ │ Loop Unrolling   │   │
│  └──────────┘ └─────────┘ └─────────┘ └────────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: LOWERING TO LLVM                                                  │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────────┐    │
│  │ LowerKGENToLLVM  │─▶│ LowerPOPToLLVM      │─▶│ LowerControlFlow     │    │
│  │                  │  │                     │  │                      │    │
│  └──────────────────┘  └─────────────────────┘  └──────────────────────┘    │
│                                                                             │
│  Output: LLVM Dialect → LLVM IR → Machine Code                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Parsing and Type Checking

> **Key Insight**: Mojo does not have a traditional AST. The parser directly
> emits MLIR operations in the `lit` dialect, which serves as the
> "source-level IR."

### Location in Codebase

- `KGEN/lib/MojoParser/` - Parser implementation
- `KGEN/include/KGEN/MojoParser/` - Parser headers

### Three-Phase Parsing

Mojo uses a **lazy, three-phase parser** to handle forward references without
requiring explicit forward declarations:

#### Phase 1a: Name Resolution

The parser first scans for declarations, registering names without parsing
anything else:

```mojo
struct Foo[T: Stringable]:    # Register name "Foo"
    var field: T              # SKIPPED

def bar(x: Foo[Int]):          # Register name "bar"
    pass                      # SKIPPED
```

#### Phase 1b: Signature Resolution

Now that a name is known, resolve the type of a name:

```mojo
struct Foo[T: Stringable]:    # Resolved: "Foo" has one parameter T: Stringable
    var field: T              # Body still skipped

def bar(x: Foo[Int]):          # Resolved: "bar" has one argument: x: Foo[Int] and no results
    SKIPPED                   # Body still skipped
```

#### Phase 1c: Body Resolution

Finally, parse the body of a declaration. This only performs name resolution
on nested declarations.

### Key Parser Components

| File                  | Purpose                                               |
|-----------------------|-------------------------------------------------------|
| `Lexer.cpp`           | Tokenization with Python-style significant whitespace |
| `ParserExprs.cpp`     | Expression parsing                                    |
| `ParserStmts.cpp`     | Statement parsing                                     |
| `IREmitter.cpp`       | MLIR generation (emits LIT dialect ops)               |
| `DeclResolver.cpp`    | Name resolution and overload resolution               |
| `OverloadFitness.cpp` | Function overload selection                           |
| `ParamInf.cpp`        | Parameter inference                                   |
| `Signatures.cpp`      | Function signature handling                           |
| `CallEmission.cpp`    | Function call emission logic                          |
| `ClosureEmitter.cpp`  | Closure code generation                               |
| `StructEmitter.cpp`   | Struct definition emission                            |
| `Traits.cpp`          | Trait handling and conformance                        |
| `ExprNodes.cpp`       | Expression node representation (see below)            |
| `ASTDecl.cpp`         | Declaration representation                            |
| `ASTType.cpp`         | Type representation for AST nodes                     |
| `ASTPrinter.cpp`      | Pretty-printing of AST structures                     |

### Generated IR: LIT Dialect

For this Mojo code:

```mojo
def foo(arg: Int):
    pass

def main():
    foo(5)
```

The parser generates:

```mlir
lit.fn @"foo(::Int)"(%arg: !Int) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

lit.fn @"main()"() -> !kgen.none {
  %0 = kgen.param.constant: !Int = <{5}>
  %1 = lit.call @main::@"foo(::Int)"(%0) : !lit.generator<("arg": !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

### Expression Nodes (Not AST)

Instead of a full AST, the parser uses a lightweight **expression node**
system (`ExprNodes.h`) for deferred expression parsing. This allows:

1. Lexical use before definition (e.g., list comprehensions)
2. Out-of-order code generation (e.g., conditional expressions)
3. Context-dependent resolution (type vs. value)

---

## Phase 2: Semantic Checking and LIT Lowering

This phase performs semantic analysis on the LIT IR and lowers it to the KGEN
dialect.

### Location in Codebase

- `KGEN/lib/LowerLIT/`

### The LIT Dialect

**LIT** is the source-level IR, closely reflecting Mojo semantics:

| Operation         | Purpose              |
|-------------------|----------------------|
| `lit.fn`          | Function definition  |
| `lit.call`        | Function call        |
| `lit.var.decl`    | Variable declaration |
| `lit.ref.store`   | Store to reference   |
| `lit.ref.load`    | Load from reference  |
| `lit.struct.decl` | Struct definition    |
| `lit.trait.decl`  | Trait definition     |
| `lit.return`      | Return statement     |

### Key LIT Types

| Type                   | Description                           |
|------------------------|---------------------------------------|
| `!lit.struct<@Symbol>` | User-defined struct type              |
| `!lit.trait<@Symbol>`  | User-defined trait type               |
| `!lit.ref<T, origin>`  | Reference type with lifetime tracking |
| `!lit.generator<sig>`  | Generator type (with metadata)        |
| `!lit.fn<sig>`         | Function type                         |
| `!lit.meta<@S>`        | Metatype for struct                   |
| `!lit.anytrait<@T>`    | Metatype for trait                    |

### Semantic Checking Passes

#### LowerSemanticCF

Lowers semantic control flow (e.g. `lit.return`) to terminators and diagnoses
unreachable code.

#### CheckLifetimes (Borrow Checker)

- Inserts destructor calls
- Rejects use-after-free
- Performs borrow checking for references

#### LowerLIT

Converts LIT dialect to KGEN dialect, e.g.

- `lit.fn` → `kgen.generator`
- `lit.call` → `kgen.call`
- `lit.ref` → `!kgen.pointer`

---

## Phase 3: Pre-Elaboration Optimization

After lowering to KGEN dialect, several optimization passes run **before**
elaboration.

### Why Pre-Elaboration Optimization?

Fewer, smaller, simpler generators are faster to instantiate (e.g. better for
elaborator caching).

### The KGEN Dialect

**KGEN** is the "canonical" parametric IR after semantic checking:

| Operation               | Purpose                                 |
|-------------------------|-----------------------------------------|
| `kgen.generator`        | Parametric function template            |
| `kgen.struct.generator` | Parametric struct/type template         |
| `kgen.func`             | Concrete function (post-elaboration)    |
| `kgen.struct.instance`  | Concrete struct type (post-elaboration) |
| `kgen.call`             | Function call                           |
| `kgen.param.constant`   | Parameter value materialization         |
| `kgen.param.if`         | Compile-time conditional                |

#### Generators: Function and Struct

> **Key Terminology**: A "generator" is a **parametric template** that may
> have compile-time parameters. The term applies to both function generators
> (`kgen.generator`) and struct generators (`kgen.struct.generator`). After
> elaboration, these become concrete, fully-instantiated operations:
> `kgen.func` for functions and `kgen.struct.instance` for types. This is
> analogous to C++ templates vs instantiated templates, but don't confuse
> "generator" with Python's yield-based generators—they are unrelated
> concepts.

Both `kgen.generator` and `kgen.struct.generator` implement the
`GeneratorOpInterface`, which makes elaboration **general over all
generators**.

**Function generator example:**

```mlir
# Before elaboration: parametric function generator
kgen.generator @add<rhs>(%lhs: index) -> index {
  %0 = kgen.param.constant = <rhs>
  %1 = index.add %lhs, %0
  kgen.return %1 : index
}

# After elaboration: concrete function
kgen.func @"add<42>"(%lhs: index) -> index {
  %0 = index.constant 42
  %1 = index.add %lhs, %0
  kgen.return %1 : index
}
```

**Struct generator example:**

```mlir
# Before elaboration: parametric struct generator
kgen.struct.generator @LinkedList<T: type> =
    struct_inst<"LinkedList"[T]<:type T>(
      data: typevalue<T>,
      next: pointer<typevalue<#kgen.genref<@LinkedList<:type T>>>>
    )> {
  # Contains conformance tables.
}

# After elaboration: concrete struct instance
kgen.struct.instance @"LinkedList,T=index" =
    struct_inst<"LinkedList"[T]<:type index>(
      data: index,
      next: pointer<typevalue<#kgen.instref<@"LinkedList,T=index">>>
    )>
```

Struct generators are used for user-defined structs, and also internally for
closures (the parser generates `kgen.struct.generator` operations to represent
closure types).

### The POP Dialect

**POP** (Parametric Operations) provides parametric operations for common LLVM
instructions:

| Operation                  | Purpose                    |
|----------------------------|----------------------------|
| `pop.add`, `pop.mul`, etc. | Arithmetic on SIMD types   |
| `pop.load`, `pop.store`    | Memory operations          |
| `pop.bitcast`              | Type reinterpretation      |
| `pop.simd.splat`           | Broadcast scalar to vector |

### The HLCF Dialect

**HLCF** (High-Level Control Flow) represents structured control flow:

| Operation                     | Purpose                      |
|-------------------------------|------------------------------|
| `hlcf.if`                     | Conditional branch           |
| `hlcf.for`                    | For loop                     |
| `hlcf.loop`                   | Loop (while, do-while, etc.) |
| `hlcf.break`, `hlcf.continue` | Loop control                 |

### Key Pre-Elaboration Passes

| Pass                   | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| `SROA`                 | Scalar Replacement of Aggregates (on generators)               |
| `Mem2Reg`              | Promote memory to registers (on generators)                    |
| `Canonicalizer`        | Apply rewrite patterns                                         |
| `InlineParametric`     | Inline `nodebug` functions and small functions pre-elaboration |
| `SCCP`                 | Sparse Conditional Constant Propagation                        |
| `ApplyInliner`         | Handle `apply` operator inlining                               |
| `EliminateDeadSymbols` | Remove unreferenced generators                                 |
| `RemoveUnusedParams`   | Clean up unused parameters                                     |

> **Note**: Pre-elaboration inlining is restricted to `always_inline_no_debug`
> functions and certain "small" functions. Too much inlining before
> elaboration increases pressure on the elaborator and reduces cache
> granularity.

---

## Phase 4: Elaboration (Monomorphization)

### Location in Codebase

- `KGEN/lib/Elaborator/Elaborator.cpp`
- `KGEN/lib/Elaborator/IREvaluator.cpp`
- `KGEN/lib/Elaborator/ParametricElaborator.cpp` (new)
- `KGEN/lib/Elaborator/ParametricIREvaluator.cpp` (new)

### What Elaboration Does

Elaboration is analogous to **template instantiation** in C++, but more
powerful. It operates on all generators (both `kgen.generator` for functions
and `kgen.struct.generator` for types) through the common
`GeneratorOpInterface`:

1. **Parameter Substitution**: Replace generic parameters with concrete values
2. **Compile-Time Evaluation**: Evaluate parameter expressions
3. **Constraint Checking**: Verify static assertions

The output of elaboration transforms:

- `kgen.generator` → `kgen.func` (concrete functions)
- `kgen.struct.generator` → `kgen.struct.instance` (concrete types)
- Parametric ops from any dialect → concrete ops

### The Expansion Graph

The elaborator builds an **expansion graph** where:

- **ParamNode**: A generator instantiation (generator reference + input
  parameter values)
- **ImplNode**: Contains the concrete function being elaborated for a given
  ParamNode (exists as a separate struct for legacy reasons, but is 1:1 with
  ParamNodes)

```text
                    main()
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    foo<1>        foo<2>         bar<1>
```

### Parallel Elaboration

Elaboration runs in parallel:

- Independent generator instantiations are processed concurrently
- Evaluation synchronization points force serialization

### The Interpreter

For compile-time code evaluation, KGEN includes an **interpreter**:

```mlir
# The `apply` operator triggers interpretation
kgen.func @call_it() {
  %result = kgen.param.constant = <apply(:() -> index @call_me)>
  kgen.return
}
```

The interpreter:

- **Uses fold hooks**: MLIR operations with constant inputs are evaluated via
  their dedicate interpreter hooks, or their fold hooks.
- **"Bytecode" compilation**: Functions are "compiled" to
  `FunctionIRBytecode` for efficient evaluation
- **Emulated memory model**: Maintains a virtual address space for loads/stores
- **Supports control flow**: If/for/while/function calls all work at compile
  time
- **No actual JIT required**: Works in environments that forbid actual JIT,
  using interpretation instead

---

## Phase 5: Post-Elaboration Lowering & Optimization

After elaboration produces concrete functions (`kgen.func`), this phase
performs two major tasks:

1. **Lower Remaining KGEN Semantics**: Transform high-level function
   signatures and call sites
2. **Optimization**: Apply aggressive optimizations on the now-concrete IR

### Phase Structure

This phase begins with **lowering passes** that transform calling conventions,
followed by **optimization passes** that improve the generated code.

#### Step 1: Lowering Passes

| Pass                      | Purpose                                                                         |
|---------------------------|---------------------------------------------------------------------------------|
| `LowerArgConventions`     | Lowers KGEN arg passing conventions (e.g. `byref_result`, `byref_error`, packs) |
| `LowerCallingConventions` | Lowers high-level KGEN types (pack, variant, none) to concrete representations  |

These lowering passes must run **before** optimization because they affect
function signatures and call sites. Once calling conventions are lowered, the
IR is ready for aggressive optimization.

#### Step 2: Optimization Passes

| Pass                      | Purpose                                 |
|---------------------------|-----------------------------------------|
| `SROA`                    | Scalar Replacement of Aggregates        |
| `Mem2Reg`                 | Promote memory to registers             |
| `Canonicalizer`           | Apply rewrite patterns                  |
| `SCCP`                    | Sparse Conditional Constant Propagation |
| `AutomaticInline`         | Aggressive inlining with heuristics     |
| `LoopUnrolling`           | Unroll loops based on hints             |
| `DeadArgumentElimination` | Remove unused function arguments        |

## Phase 6: Lowering to LLVM

### Location in Codebase

- `KGEN/lib/KGENToLLVM/`
- `KGEN/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp`

### KGEN to LLVM Type Mapping

| KGEN/POP Type        | LLVM Type                        |
|----------------------|----------------------------------|
| `!kgen.scalar<f32>`  | `f32`                            |
| `!kgen.simd<4, f32>` | `<4 x f32>`                      |
| `!kgen.pointer<T>`   | `ptr` (opaque pointer, LLVM 15+) |
| `!pop.array<4, T>`   | `[4 x T]`                        |
| Struct types         | LLVM struct types                |

### LLVM to Machine Code

After lowering to `llvm` dialect:

1. **MLIR-to-LLVM Translation**: Convert LLVM dialect to LLVM IR
2. **LLVM Optimization**: Run LLVM's optimization pipeline
3. **Code Generation**: Generate target-specific machine code

---

## Mojo Packages and Precompiled Files

> **Key Insight**: Mojo precompiled files (`.mojoc` files) are precompiled MLIR
> bytecode that contains the post-parse IR. When imported, ops from these
> files are loaded lazily.

### What is a Mojo Package?

A Mojo **source package** is a directory containing Mojo source files with an
`__init__.mojo` file that marks it as a package:

```text
my_package/
├── __init__.mojo      # Required - marks directory as a package
├── module1.mojo       # A module within the package
├── module2.mojo       # Another module
└── subpackage/        # Nested package
    ├── __init__.mojo  # Required for nested packages too
    └── utils.mojo
```

The `__init__.mojo` file is the package's entry point and typically
re-exports the public API:

```mojo
# my_package/__init__.mojo
from .module1 import MyStruct, my_function
from .module2 import helper
```

A **precompiled file** or **binary package** (`.mojoc` file) is the
precompiled form of a source package, created by the `mojo package` command.
Binary packages are tied to the specific version of the compiler that produced
them so it is not advised to use these as a distributable form; they act as a
sort of cache file.

### Location in Codebase

- `KGEN/tools/mojo/Precompile/` - Precompile command implementation
- `KGEN/lib/MojoParser/SharedState.cpp` - Import logic for both source and
  binary packages
- `KGEN/lib/MojoParser/EntryPoint.cpp` - Package parsing entry points

### Package Creation Flow

The `mojo package` command follows the first two phases of the
[High-Level Architecture](#high-level-architecture) (parsing and semantic
checking), then stops before elaboration:

1. **Phase 1: Parsing** - Parses the source package directory
   (`__init__.mojo` and submodules) and generates LIT dialect IR, just like
   normal compilation
2. **Phase 2: Semantic Checking** - Runs `runCheckLITPipeline`
   (LowerSemanticCF, CheckLifetimes) to verify the package is semantically
   valid
3. **Package Building** - Calls `buildPackageModule` to:
   - Create a `lit.package` op with **stubs only** (function signatures and
     type declarations, no bodies)
   - Serialize the full post-semantic-checking IR into a
     `postParseModuleAttr` (bytecode blob)
   - Record package dependencies and attach any external bitcode libraries
4. **Output** - Emits a `.mojoc` file containing MLIR bytecode with the
   stripped `lit.package` stub and the full IR in the attribute

**Key Difference**: Package creation stops after semantic checking (Phase 2)
and emits the IR as bytecode. The full IR (including function bodies) is
preserved in the `postParseModuleAttr` so that elaboration can happen later
when the package is imported and compiled with concrete parameter values from
the importing code.

### Package vs. Normal Compilation

| Aspect          | Normal Compilation            | Package Creation             |
|-----------------|-------------------------------|------------------------------|
| **Input**       | `.mojo` file                  | Source directory             |
| **Output**      | Precompiled file file         | `.mojoc` bytecode            |
| **Pipeline**    | Full pipeline to machine code | Parse + semantics check only |
| **Elaboration** | Yes                           | No                           |

### The lit.package Operation

```mlir
lit.package @my_package {
  // Stripped declarations (stubs only)
  lit.fn @"my_func(::Int)"(%arg: !Int) -> !Int

  lit.struct.decl @MyStruct {
    lit.struct.field @x : !Int
    lit.fn @"__init__(inout::Self,::Int)"(...)
  }

  // Nested modules
  lit.file_module @submodule { ... }
} {
  // Attributes
  postParseModule = dense_resource<...> : vector<12345xi8>,
  dependencies = [@other_package]
}
```

The `postParseModuleAttr` contains the full post-parse IR serialized as
bytecode. When the package is compiled, this bytecode is deserialized and
merged into the compilation.

### Importing Packages in the Parser

When the parser encounters an import statement, it resolves the module through
`SharedState::importModuleState`:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  IMPORT RESOLUTION (importModuleState)                                      │
│  • Search import paths for the module name                                  │
│  • Find: foo.mojo, foo/ (source package), or foo.mojoc (binary)             │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                      │                      │
        Source File │           Source Pkg │           Binary Pkg │
                    ▼                      ▼                      ▼
┌─────────────────────┐  ┌───────────────────────┐  ┌──────────────────────┐
│  createModuleState  │  │ createPackageState    │  │ createBinaryPkgState │
│  • Lex and parse    │  │ • Create lit.package  │  │ • Read bytecode      │
│  • Register decls   │  │ • Parse __init__.mojo │  │ • Lazy load ops      │
│  • Import builtins  │  │ • Lazy resolution     │  │ • Register stubs     │
└─────────────────────┘  └───────────────────────┘  └──────────────────────┘
```

### Binary Package Import Details

When importing a `.mojoc` file (`createBinaryPackageState`):

1. **Lazy Loading**: The bytecode is read using `mlir::BytecodeReader` with
   lazy loading enabled
2. **Stub Registration**: The `lit.package` stub is inserted into the current
   module
3. **Thunk Deduplication**: Function conversion thunks are moved to the
   top-level module (deduplicated if already present)
4. **Decl Registration**: Declarations are registered as "loaded from
   bytecode" with signatures already resolved
5. **On-Demand Materialization**: Operations are only fully materialized when
   needed (similar to when parsing from source files)

### Command-Line Usage

```bash
# Create a .mojoc from a source directory
mojo package my_package/ -o my_package.mojoc

# Create a kgen module (full pre-elaboration IR)
mojo package my_package/ --kgen-module -o my_package.mlirbc

# Compile code that imports the precompiled package
mojo build main.mojo -I .  # Finds my_package.mojoc in search paths
```

### Key Implementation Details

1. **Stubs Only**: Package files contain only function signatures and type
   declarations, not bodies
2. **Deferred Elaboration**: The full IR is stored but not elaborated until
   compilation time
3. **Lazy Loading**: Binary packages use bytecode lazy loading to minimize
   memory usage
4. **Thunk Deduplication**: Conversion function thunks are shared across
   packages
5. **Dependency Tracking**: Packages include their dependencies for recursive
   resolution

---

## Debug Information

> **Key Insight**: Mojo uses a dedicated `debuginfo` dialect to track source
> location information alongside IR. Debug info "lowers" progressively
> alongside the abstraction level of the code, allowing types and scopes to
> be refined as parameters become concretized and more target-specific
> information becomes available.

### Parametric Debug Info

One of the elegant aspects of the interaction between the `debuginfo` dialect
and KGEN's parametric IR is that **debug info is naturally parametric
pre-elaboration**. DebugInfo types and attributes can contain parameter
references, meaning:

1. **Debug info contains parameters**: A local variable's debug type might be
   `!debuginfo.unresolved<!kgen.param<T>>` before elaboration
2. **Multi-instantiation includes debug info**: When elaboration instantiates
   a generator multiple times with different concrete types, the debug info is
   also instantiated (concretized) for each version
3. **Progressive resolution**: The `!debuginfo.unresolved<T>` wrapper allows
   debug types to be gradually resolved as the IR is lowered and more
   target-specific information becomes available

```mlir
# Pre-elaboration: Debug info contains a parameter reference
#local_var = #debuginfo.local_variable<
  scope = #subprogram, name = "value"
> : !debuginfo.unresolved<!kgen.param<T>>  # T is a parameter!

# After elaboration with T=Int: Debug info is concretized
#local_var_int = #debuginfo.local_variable<
  scope = #subprogram_int, name = "value"
> : !debuginfo.unresolved<!Int>
```

This design means debug information "just works" with the parametric system —
there's no special machinery needed to handle debug info during elaboration.
The same parameter substitution that concretizes types and values also
concretizes debug info.

### Location in Codebase

- `Support/include/Support/DebugInfoDialect/` - Dialect definition (TableGen +
  headers)
- `Support/lib/DebugInfoDialect/` - Dialect implementation
- `Support/lib/DebugInfoDialect/DebugInfoToLLVM/` - LLVM lowering

### How Source Locations are Tracked

Debug information is attached to operations using MLIR's **fused location**
mechanism. A `DIScopeAttr` (such as `DISubprogramAttr` or
`DILexicalBlockAttr`) is fused with a `FileLineColLoc`:

```mlir
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit,
  scope = #file,
  sourceName = <"foo">,
  linkageName = "foo()",
  file = #file,
  line = 10,
  scopeLine = 10,
  subprogramFlags = Definition
> : !debuginfo.subroutine<() -> (): DW_CC_normal>

kgen.func @foo() {
  kgen.return loc(fused<#subprogram>["foo.mojo":12:5])
} loc(fused<#subprogram>["foo.mojo":10:1])
```

Every operation within a function carries a location with the subprogram
scope fused in. Nested scopes (like `if` blocks or loops) use
`DILexicalBlockAttr` to create a scope hierarchy.

### The DebugInfo Dialect

The `debuginfo` dialect provides types, attributes, and operations for
representing debug information:

#### Debug Info Types

| Type                                         | Description                               |
|----------------------------------------------|-------------------------------------------|
| `!debuginfo.basic<name {...}>`               | Primitive types (int, float, etc.)        |
| `!debuginfo.struct<Name(members)>`           | Composite struct types                    |
| `!debuginfo.member<name: type>`              | Struct member description                 |
| `!debuginfo.ptr<element {...}>`              | Pointer type with size/align              |
| `!debuginfo.array<N x element>`              | Fixed-size array                          |
| `!debuginfo.subroutine<(args) -> (results)>` | Function signature                        |
| `!debuginfo.unresolved<T>`                   | Wrapper for not-yet-lowered MLIR types    |
| `!debuginfo.ti.ptr<element>`                 | Target-independent pointer (pre-lowering) |
| `!debuginfo.variant<Name(...)>`              | Variant/union type                        |

The `!debuginfo.unresolved<T>` type is crucial for **progressive resolution**:
it wraps MLIR types (like `index` or `!kgen.pointer`) that don't have a
native debug representation yet. As the IR is lowered, these are resolved to
concrete debug types.

#### Debug Info Scope Attributes

| Attribute                        | Description                                |
|----------------------------------|--------------------------------------------|
| `#debuginfo.compile_unit<...>`   | Top-level compilation unit                 |
| `#debuginfo.file<name in dir>`   | Source file reference                      |
| `#debuginfo.subprogram<...>`     | Function/method description                |
| `#debuginfo.lexical_block<...>`  | Nested lexical scope (if/for/while blocks) |
| `#debuginfo.local_variable<...>` | Local variable description                 |
| `#debuginfo.source_name<...>`    | Mangled source name with lineage           |

#### Debug Info Operations

| Operation                           | Description                                 |
|-------------------------------------|---------------------------------------------|
| `debuginfo.value #var #expr = %val` | Declares a variable has a new value         |
| `debuginfo.kill #var`               | Indicates variable is no longer valid       |
| `debuginfo.line_table_loc`          | Forces a line-table entry (for breakpoints) |

Example of variable tracking:

```mlir
#local_var = #debuginfo.local_variable<
  scope = #subprogram,
  name = "x",
  file = #file,
  line = 15
> : !debuginfo.unresolved<index>

#expr = #debuginfo.expr.irvalue : !debuginfo.unresolved<index>

%x = index.constant 42
debuginfo.value #local_var #expr = %x : index
```

#### Debug Info Expression Attributes

Expressions describe how an IR value relates to a source-level variable:

| Expression                        | Description                                   |
|-----------------------------------|-----------------------------------------------|
| `#debuginfo.expr.irvalue`         | The IR value directly represents the variable |
| `#debuginfo.expr.deref<inner>`    | Dereference a pointer to get the value        |
| `#debuginfo.expr.refof<inner>`    | Take reference of a value                     |
| `#debuginfo.expr.agg<inner, idx>` | Extract field from aggregate                  |

### Debug Info Flow Through the Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  PARSING                                                                    │
│  • DIBuilder initialized with compile unit                                  │
│  • Subprogram scopes created for each function                              │
│  • Lexical blocks pushed for if/for/while bodies                            │
│  • Locations fused with scope: loc(fused<#scope>["file":line:col])          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PRE-ELABORATION / ELABORATION                                              │
│  • Scopes preserved through transformations                                 │
│  • debuginfo.value ops track variable assignments                           │
│  • Types remain as !debuginfo.unresolved<T> until lowering                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LLVM LOWERING (DebugInfoToLLVM)                                            │
│  • debuginfo.subprogram → llvm.di_subprogram                                │
│  • debuginfo.value → llvm.dbg.value                                         │
│  • !debuginfo.unresolved<T> resolved to concrete LLVM debug types           │
│  • Target-specific adaptations (NVPTX, AMDGPU)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How the Parser builds DebugInfo

The `DIBuilder` class
(`Support/include/Support/DebugInfoDialect/IR/DIBuilder.h`) provides a
high-level API for constructing debug info:

```cpp
DebugInfo::DIBuilder diBuilder(context);

// Initialize compile unit (once per file)
diBuilder.initializeCompileUnit(
    llvm::dwarf::DW_LANG_Mojo, file, "Mojo", /*isOptimized=*/true,
    EmissionKind::Full);

// Push/pop scopes for lexical blocks
{
  auto guard = diBuilder.pushScopeGuard(subprogramAttr);
  // Operations created here get this scope attached

  {
    auto blockGuard = diBuilder.pushNestedLexicalBlock(file, line, col);
    // Nested scope for if/for/while body
  }
}

// Create scoped locations
Location scopedLoc = diBuilder.createScopedLoc(originalLoc);
```

The DIBuilder maintains a **scope stack** that automatically attaches the
correct scope to locations. The parser uses this to create proper debug scopes
for functions, loops, and conditional blocks.

### How Passes Preserve Debug Info

Debug information must be maintained through compiler transformations, which
requires special handling in several passes. This section covers the key
mechanisms.

#### Inlining and Call Site Locations

When a function is inlined, all operations from the inlined function body must
have their locations updated to include the **call site location**. This
creates a nested location structure that preserves the call stack for
debugging:

```mlir
// Before inlining foo() into bar():

#calleeSp = #debuginfo.subprogram<...> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#calleeSrcLoc = loc("foo.mojo":5:3)
#calleeLoc = loc(fused<#calleeSp>[#calleeSrcLoc])
kgen.func @foo() {
  %x = index.constant 42 loc(#calleeLoc)
}

// After inlining - ops get CallSiteLoc wrapping original location:
#callerLoc = loc(fused<#callerSp>[#callerSrcLoc])
kgen.func @bar() {
  %x = index.constant 42 loc(callsite(#calleeLoc at #callerLoc))
}
```

**Cost considerations**: Updating debug info for every inlined operation is
expensive. The compiler handles this differently at different optimization
levels via `InlinerDebugInfoUpdateTime`:

| Mode         | When                                            | Use Case                                                           |
|--------------|-------------------------------------------------|--------------------------------------------------------------------|
| `kImmediate` | Update right after each function is inlined     | Higher optimization levels (O1+) where more transformations follow |
| `kDeferred`  | Tag inlined scopes, batch update at end of pass | O0 where fewer optimizations follow inlining                       |
| `kNever`     | Don't update debug info                         | When compiling without debug info                                  |

```cpp
// From Pipeline.cpp - optimization level affects debug info update strategy
pm.addPass(createAutomaticInline(
    {options.debugLevel == CompilationOptions::kNoDebug
         ? InlinerDebugInfoUpdateTime::kNever
         : (options.optimizationLevel == 0
                ? InlinerDebugInfoUpdateTime::kDeferred
                : InlinerDebugInfoUpdateTime::kImmediate),
     options.optimizationLevel}, ...));
```

The deferred mode works by tagging inlined scopes with an attribute during
inlining, then walking the IR at the end to perform all updates in a single
pass—reducing the overhead of repeated location rewrites during aggressive
inlining.

#### Debug Info Expressions (SROA/Mem2Reg)

When passes like **SROA** (Scalar Replacement of Aggregates) or **Mem2Reg**
transform how a variable is stored, the debug info must track how to
reconstruct the original variable from the transformed representation. This is
done using **debug info expressions** (similar to DWARF expressions) that
describe the relationship between IR values and source-level variables.

The key expression types are:

| Expression                        | Description                                         |
|-----------------------------------|-----------------------------------------------------|
| `#debuginfo.expr.irvalue`         | The IR value directly represents the variable       |
| `#debuginfo.expr.deref<inner>`    | Dereference a pointer to get the value              |
| `#debuginfo.expr.refof<inner>`    | Take reference of a value                           |
| `#debuginfo.expr.agg<inner, idx>` | Extract field `idx` from aggregate to get the value |

#### Example: Mem2Reg transformation

When Mem2Reg promotes a stack allocation to a register, it must update the
debug expression to indicate that the variable is no longer behind a pointer:

```mlir
// Before Mem2Reg: variable 'x' is on the stack
%alloc = pop.stack_allocation : !kgen.pointer<!Int>
debuginfo.value #x_var #debuginfo.expr.irvalue = %alloc : !kgen.pointer<!Int>
pop.store %value, %alloc

// After Mem2Reg: variable 'x' is directly in a register
// Expression changes from "irvalue" (ptr) to "refof(irvalue)" (the value itself)
debuginfo.value #x_var #debuginfo.expr.refof<#debuginfo.expr.irvalue> = %value : !Int
```

The `DIExprLeafReplacer` class handles this transformation by replacing the
leaf of existing expressions with new sub-expressions that describe the
opposite of the transformation.

#### Example: SROA transformation

When SROA splits a struct into individual fields, each field gets its own
debug expression:

```mlir
// Before SROA: struct with fields 'a' and 'b'
%alloc = pop.stack_allocation : !kgen.pointer<!MyStruct>
debuginfo.value #struct_var #debuginfo.expr.irvalue = %alloc

// After SROA: split into separate values
%a_val = ... : !Int
%b_val = ... : !Float
// Field 'a' (index 0) - expression describes reconstruction
debuginfo.value #struct_var #debuginfo.expr.agg<#debuginfo.expr.refof<#irvalue>, 0> = %a_val
// Field 'b' (index 1)
debuginfo.value #struct_var #debuginfo.expr.agg<#debuginfo.expr.refof<#irvalue>, 1> = %b_val
```

This mechanism is crucial for debugger experience: When you inspect a variable
in the debugger, the DWARF expressions (lowered from debuginfo expressions)
tell the debugger how to reconstruct the original source-level variable from
potentially scattered IR values.

### Debug Levels

The compiler supports different debug info levels controlled by `-debug-level`:

| Level              | Description                                  |
|--------------------|----------------------------------------------|
| `none`             | No debug info generated                      |
| `line-tables-only` | Line numbers only (smaller output)           |
| `full`             | Complete debug info with variables and types |

Functions marked `@always_inline("nodebug")` suppress debug info, reducing
overhead for zero-cost abstractions.

### Key Interfaces

Operations that contain debug-scoped code implement these interfaces:

| Interface                 | Description                                       |
|---------------------------|---------------------------------------------------|
| `SubprogramScoped`        | Operations representing functions with debug info |
| `InlinedSubprogramScoped` | Inlined function calls with callsite info         |

Helper functions for working with debug info:

```cpp
// Extract subprogram scope from a function
DISubprogramAttr scope = DebugInfo::extractScope(funcOp);

// Extract scope from any operation's location
DIScopeAttr scope = DebugInfo::extractScope(op);

// Extract the original source location from a potentially nested location
FileLineColLoc srcLoc = DebugInfo::extractSourceLoc(loc);
```

### Example: Full Debug Info

For this Mojo code:

```mojo
def add(a: Int, b: Int) -> Int:
    var result = a + b
    return result
```

The IR with full debug info looks like:

```mlir
#file = #debuginfo.file<"example.mojo" in "">
#compile_unit = #debuginfo.compile_unit<
  sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo",
  isOptimized = true, emissionKind = Full
>
#add_name = #debuginfo.source_name<(fn)"add"(#Int_name, #Int_name) from <(module)"example">>
#subprogram = #debuginfo.subprogram<
  compileUnit = #compile_unit, scope = #file, sourceName = #add_name,
  linkageName = "add(::Int,::Int)", file = #file, line = 1, scopeLine = 1,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<(!Int, !Int) -> (!Int): DW_CC_normal>

#result_var = #debuginfo.local_variable<
  scope = #subprogram, name = "result", file = #file, line = 2
> : !debuginfo.unresolved<!Int>

kgen.func @"add(::Int,::Int)"(%a: !Int, %b: !Int) -> !Int {
  %sum = ...  // a + b computation
  debuginfo.value #result_var = %sum : !Int loc(fused<#subprogram>["example.mojo":2:5])
  kgen.return %sum : !Int loc(fused<#subprogram>["example.mojo":3:5])
} loc(fused<#subprogram>["example.mojo":1:1])
```

---

## MLIR Dialects Reference

### Dialect Hierarchy

> **Note**: The Mojo compiler mixes multiple dialects at each stage. The
> dialects below (`kgen`, `pop`, `hlcf`) can appear alongside `lit` even
> before LowerLIT. Upstream dialects like `index`, `llvm`, `nvvm`, and
> `rocdl` can appear anywhere in the flow.

```text
Source Level (before LowerLIT)
├── lit    - Source-level Mojo constructs (lit.fn, lit.call, lit.ref, etc.)
├── kgen   - Parameter operations (kgen.param.constant, etc.)
├── pop    - Parametric SIMD/memory operations
└── hlcf   - High-level control flow

Pre-Elaboration (after LowerLIT, before ElaborateGenerators)
├── kgen   - Parametric generators (kgen.generator, kgen.struct.generator) - NO lit ops
├── pop    - Parametric SIMD/memory operations
└── hlcf   - High-level control flow

Post-Elaboration (after ElaborateGenerators)
├── kgen   - Concrete ops only (kgen.func, kgen.struct.instance, NO generators)
├── pop    - Concrete SIMD/memory operations (no parametric types)
└── hlcf   - High-level control flow

Target Level (after LowerToLLVM)
└── llvm   - LLVM dialect (all other dialects lowered away)
```

### Dialect Summary Table

**Mojo-specific dialects:**

| Dialect     | Purpose                         | Parametric? | When Lowered                 |
|-------------|---------------------------------|-------------|------------------------------|
| `lit`       | Source-level IR                 | Yes         | By LowerLIT (Phase 2)        |
| `kgen`      | Canonical Mojo IR               | Yes → No    | During elaboration (Phase 4) |
| `pop`       | SIMD/memory ops                 | Yes → No    | During elaboration (Phase 4) |
| `hlcf`      | Structured control flow         | No          | To LLVM (Phase 6)            |
| `co`        | Coroutines                      | No          | To LLVM (Phase 6)            |
| `debuginfo` | Debug information               | No          | To LLVM (Phase 6)            |
| `interp`    | Interpreter operations and data | No          | To LLVM (Phase 6)            |

**Upstream/third-party dialects** (can appear at any stage):

| Dialect | Purpose                          |
|---------|----------------------------------|
| `index` | Index arithmetic (upstream MLIR) |
| `llvm`  | LLVM IR bridge (target dialect)  |
| `nvvm`  | NVIDIA GPU intrinsics            |
| `rocdl` | AMD GPU intrinsics               |

---

## Key Passes Summary

### Semantic Checking & LIT Lowering (Phase 2)

| Pass               | Description                              |
|--------------------|------------------------------------------|
| `LowerSemanticCF`  | Lower `lit.return` to terminators        |
| `VerifyParameters` | Check parameter usage validity           |
| `CheckLifetimes`   | Borrow checking and destructor insertion |
| `LowerLIT`         | Convert LIT → KGEN                       |

### Pre-Elaboration Optimization (Phase 3)

| Pass                   | Description                                |
|------------------------|--------------------------------------------|
| `OutlineClosures`      | Extract closure bodies                     |
| `SROA`                 | Scalar Replacement of Aggregates           |
| `Mem2Reg`              | Promote memory to registers                |
| `Canonicalizer`        | Apply rewrite patterns                     |
| `InlineParametric`     | Inline `nodebug` functions pre-elaboration |
| `SCCP`                 | Sparse Conditional Constant Propagation    |
| `ApplyInliner`         | Handle `apply` operator inlining           |
| `EliminateDeadSymbols` | Remove unreferenced generators             |

### Elaboration (Phase 4)

| Pass                  | Description                                                          |
|-----------------------|----------------------------------------------------------------------|
| `LiftAndFoldApply`    | Hoist `apply` operators for elaboration (prep for Elaboration)       |
| `ReorderParamOps`     | Reorder parameter declaration & assertion ops (prep for Elaboration) |
| `ElaborateGenerators` | Main monomorphization pass                                           |

### Post-Elaboration Optimization (Phase 5)

| Pass                          | Description                        |
|-------------------------------|------------------------------------|
| `EliminateDuplicateFunctions` | Deduplicate identical functions    |
| `ResolveCompilerPromises`     | Resolve deferred type computations |
| `LowerArgConventions`         | Handle arg conventions             |
| `LowerCallingConventions`     | Handle calling conventions         |
| `AutomaticInline`             | Aggressive inlining                |
| `RaiseForLoops`               | Recognize loop patterns            |
| `LoopUnrolling`               | Unroll decorated loops             |
| `LowerLoops`                  | Lower HLCF loops to CFG            |
| `LowerClosures`               | Lower closure representations      |
| `LowerAsyncFunctions`         | Lower async/coroutines             |

### LLVM Lowering (Phase 6)

| Pass                                | Description                                    |
|-------------------------------------|------------------------------------------------|
| `LowerKGENToLLVM`, `LowerPOPToLLVM` | Convert KGEN/POP ops to LLVM                   |
| `LowerControlFlow`                  | Convert HLCF to branches                       |
| `DebugInfoToLLVM`                   | Lower debuginfo to LLVM dialect representation |

---

## Developer Tools

### Command-Line Tools

| Tool             | Purpose                          |
|------------------|----------------------------------|
| `mojo`           | Main compiler (public)           |
| `kgen`           | Internal compiler driver         |
| `kgen-translate` | Parser-only (produces LIT IR)    |
| `kgen-opt`       | Run specific optimization passes |

### Debugging Tips

```bash
# Print IR before each pass
kgen --mlir-print-ir-before-all -elaborate main.mojo

# Print IR after each pass
kgen --mlir-print-ir-after-all -elaborate main.mojo

# Print IR at a specific pass (useful for large outputs)
kgen --mlir-print-ir-after=lower-lit main.mojo 2>&1 | less

# Debug specific subsystems (requires debug build)
kgen -debug-only=elaborator -elaborate main.mojo

# Dump IR to file for easier inspection
kgen --mlir-print-ir-after-all -elaborate main.mojo 2> ir-dump.mlir
```

### Testing

The compiler tests live in `KGEN/test/` and use LLVM's **lit** test framework
with **FileCheck**:

```bash
# Run all KGEN tests
bt //KGEN/test/...

# Run a specific test file
bt //KGEN/test/mojo-integration:my_test.mojo

# Or just use the file name
bt KGEN/test/mojo-integration/my_test.mojo
```

Test files use FileCheck directives to verify output:

```mojo
# RUN: kgen-translate -import-mojo %s | FileCheck %s

def foo():
    pass

# CHECK: lit.fn @"foo()"
```

### Source Files Quick Reference

| Directory                                   | Contents                                          |
|---------------------------------------------|---------------------------------------------------|
| `KGEN/lib/MojoParser/`                      | Parser and type checker                           |
| `KGEN/lib/LITDialect/`                      | LIT dialect implementation                        |
| `KGEN/lib/KGENDialect/`                     | KGEN dialect implementation                       |
| `KGEN/lib/POPDialect/`                      | POP dialect implementation                        |
| `KGEN/lib/HLCFDialect/`                     | HLCF dialect implementation                       |
| `KGEN/lib/CODialect/`                       | Coroutine dialect implementation                  |
| `KGEN/lib/Elaborator/`                      | Elaboration/monomorphization                      |
| `KGEN/lib/Interpreter/`                     | Compile-time interpreter (bytecode, memory model) |
| `KGEN/lib/LowerLIT/`                        | LIT lowering passes                               |
| `KGEN/lib/KGENToLLVM/`                      | LLVM lowering passes                              |
| `KGEN/lib/Transforms/`                      | Optimization passes                               |
| `KGEN/lib/MOGGPreElab/`                     | GPU kernel annotation passes                      |
| `KGEN/lib/Compiler/Pipeline/`               | Pass pipeline construction                        |
| `KGEN/tools/mojo/Precompile/`               | Precompile command implementation                 |
| `Support/lib/DebugInfoDialect/`             | Debug info dialect implementation                 |
| `Support/include/Support/DebugInfoDialect/` | Debug info dialect headers and TableGen           |
| `KGEN/include/KGEN/*/`                      | Headers and TableGen definitions                  |
| `KGEN/test/`                                | Compiler tests (lit + FileCheck)                  |
| `KGEN/docs/`                                | Documentation                                     |

---

## Additional Resources

- [Region-Based Control Flow in MLIR (starts at 5:24)](https://youtu.be/vvVR3FyU9TE?si=0j2MgCRJXNX-1enj&t=324)
- [Mojo Compiler Overview (first 18 mins)](https://www.youtube.com/watch?v=Invd_dxC2RU)
- [Mojo DebugInfo (starts at 10:47)](https://youtu.be/9jfukpjCPIg?si=lSz9ZN_AbzsnVcm8&t=647)
