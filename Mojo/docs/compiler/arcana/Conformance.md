# Conformance

This doc goes over how struct-trait conformances are checked and materialized
in the parser.

## Key Concepts

- StructDeclOp: A struct declaration containing aliases, fields, methods, etc.
and conformances.

- TraitDeclOp: A trait declaration containing associated aliases & methods.

- ConformanceOp: A table that tracks how a struct conforms to a trait. For
every struct, and for every trait that the struct conforms to, there is a
ConformanceOp. Its `sym_name` is the flattened name of the trait, which is what
a `#kgen.get_witness` looks the table up by. It also carries `immediateParents`
(the first level of inherited conformance tables, sorted by flattened name) and
a `constraint`.

- WitnessOp: A specific entry in a ConformanceOp that maps a name & type pair
(required by the trait) to an implementation (provided by the struct). For
example, if the trait requires a particular function (with a particular name &
type), there will be a WitnessOp in the ConformanceOp that has that name, and
has a symbol reference to the actual function declared in the struct that
satisfies that requirement.

### Conditional Conformances

A conformance is not necessarily unconditional. A ConformanceOp's `constraint`
records the condition under which the conformance applies, so a struct can
conform to a trait only when its parameters satisfy that condition (for
example, `List[T]` is `Copyable` only when `T` is). An unconditional
conformance carries a trivially true constraint and prints without a `where`
clause, so the common case is invisible; do not assume the presence of a
ConformanceOp alone means the struct conforms.

## Lazy Resolution of Conformances (CALROC)

Like most other things in the parser, conformances are also lazily resolved.

ConformanceOps are modeled as ASTDecls inside StructDeclOps:

- When a StructDeclOp is fully resolved, a nested ConformanceOp is created for
each explicit conformance that the struct declares. At this point, they have
empty bodies (marked as signature-resolved).

- Body resolving a conformance registers one member decl per trait
requirement, under the witness name, and checks nothing itself.

- Body resolving a member checks that one requirement and installs its
WitnessOp, so a `get_witness` resolves only the entry it names. The first
member found unsatisfied checks the rest and reports every unsatisfied
requirement in one error, then marks the conformance erroneous so later
members fail silently.

- At the end of parsing, all reachable decls that are signature resolved will
be automatically fully resolved. Members are reachable through their
conformance, so every explicit conformance declared by a struct in the "main"
module is still checked in full.

Closure-wrapper conformances are created already filled, so they never get
member decls. Bytecode conformances materialize their witnesses instead.

Note that ConformanceOps are always created as signature-resolved because there
is no less-resolved state for it (it is a synthesized entity after all).

When creating packages, all conformance tables for every trait a struct
conforms to are created in it (as usual). When loading a StructDeclOp from
bytecode, we follow a similar process:

- When a bytecode StructDeclOp is fully materialized, its inner ConformanceOps
are created as signature resolved, and have empty bodies (non-materialized
yet).

- When a bytecode ConformanceOp is fully materialized, its inner WitnessOps are
materialized and any references are recursively resolved as usual. This loads
in any witnessing functions for example.
