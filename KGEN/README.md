## KGEN dialect layering

KGEN has several dialects: `hlkgen`, `kgen`, `meta`, `pop`, and `zap`. KGEN IR
also uses the `index`, `scf`, and `llvm` dialects.

`hlkgen` is a high-level dialect for building kernel libraries. It is lowered
to `kgen` before elaboration. The `kgen` dialect is the canonical dialect for
describing parametric IR. The dialect defines the parameter system and the
types, attributes, and operations for interacting with parameters.

`scf` and `index` are non-parameterized, target-independent dialects that exist
in "KGEN IR" pre-elaboration and post-elaboration. `llvm` is a target-dependent
dialect that can exist at all levels of KGEN IR. However, it locks the
particular kernel to the LLVM target.

`meta`, `pop`, and `zap` are parameterized, target-independent dialects used to
build parametric kernels. The `zap` dialect only exists pre-elaboration.

In summary:

- `zap` and `hlkgen` exist pre-elaboration. They are lowered to `kgen`, `pop`,
and `meta` before elaboration.
- `pop` and `meta` exist pre and post elaboration. Operations in the dialect
become non-parametric post-elaboration. They are lowered to `llvm` when
executing kernels.
- `index` and `scf` exist pre and post elaboration. They are lowered to `llvm`
when executing kernels.
- `llvm` can exist at all levels of KGEN IR to describe target-specific
operations, but then the kernel can only target LLVM.

![KGEN dialect hierarchy](docs/img/dialects.svg)
