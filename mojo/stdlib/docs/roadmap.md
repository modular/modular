# Overview (Q2 2025)

We're excited to share the latest **Mojo :mojo: roadmap**, outlining our
priorities for the first half of 2025. This update is organized into three
focus areas: **compiler**, **libraries**, and **tooling**.

If you have any questions, comments, or concerns, please feel free to reply on
the [forum post](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395)
discussing the roadmap.

Our teams are focused on expanding platform capabilities and increasing
developer productivity. Highlights for the first half of 2025 include:

- Adding more type system features and improving compiler performance/error
  messages
- Open sourcing remaining components of Mojo standard library and MAX AI
  Kernels (done! :white_check_mark:)
- Simplifying tensor types and improving GPU kernel programming
- Enhancing Python → Mojo interoperability
- Improving LSP, documentation, and developer tooling

Our top priority this year is making Mojo the best language for GPU
programming. That means continuous investment in the language, its libraries,
and the surrounding tooling ecosystem.

## Compiler Roadmap

Our current focus is on boosting productivity for in-house developers. To
date, they've authored nearly **500,000 lines of Mojo code**—over 30% of
Modular's codebase—and that number keeps growing!

### What's in that code?

The AI space evolves rapidly. New algorithms and levels of abstraction must
support an expanding array of GPU hardware targets. This creates a perfect
storm of complexity, with developers pushing against every dimension of the
Cartesian product of algorithms × data types × hardware targets.

### Developer needs

- Add language features to help developers manage complexity through concise,
  highly parameterized code.
- Reduce development overhead by making the iterative development process
  faster and smoother.

### Key compiler initiatives

#### 1. Main focus: add more expressivity to the Mojo type system

- Add trait unions: `alias CopyableAndMovable = Copyable & Movable`
- Add parametric aliases: `alias Scalar[DT: DType] = SIMD[1, DT]`
- Default methods implementation in traits:

``` mojo
# Support all comparison operators by implementing just one method
trait Comparable:
  fn compare(s: Self, other: Self) -> Int
     ...
  fn __eq__(s: Self, other: Self): return compare(s, other) == 0
  fn __ne__(s: Self, other: Self): return compare(s, other) != 0
```

- New `requires` keyword for adding constraints to structs and functions

```mojo
struct SIMD[dtype: DType, size: Int]
  requires size.is_power_of_two(), "simd width must be power of 2"
  requires dtype is not DType.invalid, "simd type cannot be DType.invalid"
```

#### 2. Improve error messages

When compilation fails, the Mojo compiler prints detailed diagnostic
messages. These messages have sometimes included compiler implementation
details, such as MLIR representation and type system internals. This forces
users to spend extra time parsing messages to find the relevant information.

We plan to make compiler errors more concise and helpful by:

- Shortening error messages from the `constrained()` function and improving
  their source location accuracy
- Investigating implementing stack traces for runtime errors and exceptions

#### 3. Speed up the Mojo compiler

It has always been our goal to make Mojo lighting fast. :high_voltage:

In the past, our focus was on the LLVM and code generation phase. Now that
the code generation pass runs in parallel
[[details](https://youtu.be/yuSBEXkjfEA?si=SJG0rqyN7JMggRfL)], we want to
focus on the front-end, to speed up the parser, package importer, and
elaborator.

We will continue improving the quality of generated code while expanding
support for additional hardware targets.

## Libraries Roadmap

### 1. **Open sourcing the remaining Mojo standard library modules and the MAX AI Kernels**

We recently open sourced the final components of the Mojo Standard Library,
along with the MAX AI Kernels, which contains over 200k lines of Mojo code!
:partying_face: This is a major milestone that unlocks more of Mojo's
capabilities for community exploration, contribution, and extension.

We aren't yet accepting contributions to the MAX AI Kernels, but will share
an update soon on when that will change. We are accepting contributions to
the recently open sourced standard library modules today! Be sure to check
out [the standard library contribution guidelines](https://github.com/modular/max/blob/main/mojo/CONTRIBUTING.md)
if you're planning to submit a pull request.

### 2. **Core standard library evolution**

We continue to invest in foundational types and APIs. For example, `String`
recently got a major overhaul and now includes Small String Optimization for
better performance. We welcome and appreciate your contributions as we evolve
the core standard library together. :handshake:

### 3. **Simplifying tensor types**

We're streamlining the tensor type system to make it easier to understand and
work with. This includes:

- Deprecating `max.Tensor` in favor of a cleaner type hierarchy.
- Removing `NDBuffer` from the kernel library APIs.
- Standardizing on `ManagedTensorSlice` for host-side operations and
  `LayoutTensor` for device-side (GPU) operations.

These changes aim to reduce friction and make tensor-based programming more
intuitive.

### 4. **Python → Mojo interoperability**

One of our most exciting initiatives: enabling progressive adoption of Mojo
in existing Python applications. This work lets you integrate Mojo where it
matters most—unlocking performance without needing to rewrite everything.

We're actively collaborating with other internal teams on real-world use
cases like model serving, by writing "manual bindings" — something akin to
nanobind or pybind11 for calling into Mojo functions from Python. Relatedly,
we also plan to release enhancements to the Mojo compiler for compiling Mojo
to dynamic libraries.

### 5. **B200 support + kernel intrinsics**

We're bringing up support for [NVIDIA Blackwell GPUs](https://en.wikipedia.org/wiki/Blackwell_(microarchitecture)),
along with foundational intrinsics needed to write performant kernels for
operations like matrix multiplication and attention.

### 6. **`LayoutTensor` enhancements & GPU kernel ergonomics**

We're evolving the `LayoutTensor` APIs and refining the kernel library to
make GPU kernel programming more ergonomic and composable. Expect
improvements that make it easier to express high-performance code with less
boilerplate.

## Tooling Roadmap

### LSP Improvements

- **Reducing parsed code**
The language server needs only the signatures of code dependencies, not
their implementation details. We can dramatically reduce its workload by
parsing just the signatures, since function bodies are irrelevant for our
purposes. When computing completions, we can avoid parsing most of the
document—we only need to fully parse the function body where the cursor is
located.

- **Progress Reporting**
Even with extensive optimization, larger documents may still exceed our
performance targets. To address this, we'll add a progress indicator to show
users that their input is being processed. Instead of waiting to collect all
results, we'll stream them to the client as they become available. Our target
response times are under 200ms for code editing operations and under 2
seconds for non-editing operations like symbol definition requests.

- General bug fixes to improve stability and performance while reducing
community-reported crashes.

### Mojo REPL

We will invest in fixing critical Mojo REPL bugs.

### Doc tooling

Improving public documentation on `docs.modular.com`:

- Fixing DocString section handling to properly publish sections that contain
  blank lines or mismatched indentation.
- Enhancing type names in documentation by shortening built-in Standard
  Library types with long namespaces.
- Adding associated aliases to trait documentation for better completeness.
