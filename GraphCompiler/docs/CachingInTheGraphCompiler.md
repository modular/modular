# Graph Compiler Caching

Graph compilation is a potentially expensive process depending on the size of
the input graph. The latency of compiling even small graphs can make iterative
development in the Graph API unpleasant.

The Graph Compiler also drives more interactive use-cases like PyTorch interop,
where compilation may occur as part of execution of a PyTorch model. Such use
cases require we spend as little time compiling as possible.

To remedy this, the Graph Compiler uses several layers of caching, built on the
observation that the code for a given graph does not change much between runs.
The Graph Compiler relies on 3 levels of caching to detect changes and avoid
unnecessary recompilation when possible:

- **The MEF cache**: sits in front of the model compilation process. This cache
  maps RMO/MO graphs to compiled MEF objects. The goal of the MEF cache is to
  entirely avoid the MO->MEF compilation process.

- **The MOGG cache**: sits between the Graph Compiler and the most of the Mojo
  compiler. This cache maps `mogg.routine.async` operations to
  executable code. The MOGG cache is implemented inside the `JitCompileKernels`
  transform and functions as a thin layer over the Mojo cache. Entries in the
  MOGG cache actually point to the compilation artifacts that live in the Mojo
  cache.

- **The Mojo cache**: functions as part of the Mojo compiler and is only
  relevant to the Graph Compiler in that the MOGG cache uses the Mojo cache's
  internal structure.

This document is mostly concerned with describing the functioning of the MOGG
cache and its strategy for achieving incremental compilation.

## MOGG Caching

The structure of the MOGG cache is indexed by a stable hash of the IR fragment
we are performing a lookup on. The IR fragments are those flow into or built
during the `JitCompileKernels` transform.

These IR fragments are composed of:

- Miscellaneous KGEN, MOGG, MGP, and Index dialect operations
  (none of these are super important)
- `kgen.call` operations to functions defined either in the input module or to
  a module living in `MojoLibraryAnalysis`.

```mlir
kgen.generator @__gen_mogg_harness_a654c6b11ab78c1b6569ce83bd1fe1d8(...) {
  %0 = pop.stack_allocation 1 x index
  pop.store %arg1, %0 align<1> : !kgen.pointer<index>
  %1 = pop.pointer.bitcast %0 : !kgen.pointer<index> to !kgen.pointer<none>
  %2 = kgen.call @"MOGGKernelAPILib::MOGGKernelAPI::to_managed_tensor_slice[...]"(...)
  %3 = kgen.call @"testing_kernelsLib::testing_kernels::ParameterizedOpaqueType::execute[...]"(...)
  kgen.return
}
```

The MOGG cache is driven by a hash representing this IR. The cache itself is
essentially the mapping `cache :: Hash -> CompiledObject`. Thus, the hash
encodes the full details of the lookup, and so needs to satisfy a few
requirements:

1. **Stability**: the hash function needs to be stable between runs of the
   compiler. This is relevant due to the fact that MLIR's builtin hashing
   routines do not typically guarantee stability.
2. **Semantic Encoding**: the hash function is the only thing ensuring the
   correct compiled artifact is associated with a piece of IR.
3. **Incremental**: The hash should avoid changing when code not semantically
   relevant to the IR under consideration changes.

## The Old Implementation

Previous implementations of the MOGG cache's hash function worked by computing
the full bytecode hash of the IR plus external dependencies. In this case, the
external dependencies were the contents of the `MojoLibraryAnalysis`, which
holds all the backing Mojo kernels. This function meets two of the requirements:

1. **Stability**: bytecode hashing is stable between runs of the compiler.
2. **Semantic Encoding**: the hash function encode all information relevant to
   a given IR fragment and the underlying hash is robust enough that collisions
   are unlikely.

The major downside of the "hash the full `MojoLibraryAnalysis`" approach is that
any change to any kernel library invalidates the entire MOGG cache. That imposes
a significant compile time cost when iterating on kernel implementations.

## The New Implementation

### The Algorithm

Below is pseudo-code for the new algorithm:

```python
def semantic_hash(scc: SCC, node: Node):
    # Incorporate the inputs to the SCC,
    # The node's position in the SCC, and
    # The bytecodes of the Node's backing function.
    return hash(scc.inputs, scc.position(node), node.get_bytecode())


def add_semantic_hash(module):
    # Compute the SCC of the module's callgraph (needed to handle recursive functions)
    SCCs = compute_strongly_connected_components(module)

    symbol_table = SymbolTable(module)

    for SCC in dfs_postorder(SCCs):
        for node in SCC:
            # Compute the bytecode hash of all functions in the SCC
            # All functions referred to by the SCC are either in the
            # SCC or have been processed already.
            hash = str(semantic_hash(SCC, node))

            # Append the hash to all functions in the SCC and update
            # any callers
            for func in SCC:
                func.set_attr("semantic_hash", hash)
```

The high-level idea is to treat the compilation process in `JitCompileKernels`
the same way a modern build system does. Compilation artifacts are stored in
a content-addressed store (CAS) where objects are identified by:

1. Their internal contents (in this case a slightly modified hash of
   a function's bytecodes).
2. The "inputs" to each function. In a semantic hash, the "inputs" are the
   functions which can be called from the function being hashed.

This has the effect of transitively encoding the semantics of a function and its
callees into the generated hash. The generated structure is similar to a
[Merkle Tree](https://en.wikipedia.org/wiki/Merkle_tree). This transitive
encoding gets us the **Incremental** requirement listed above.

The hash computation proceeds as a traversal of the strongly-connected
components in the callgraph due potential cycles (i.e. recursive functions).
Recursive functions require special handling as a cycle would require finding an
assignment of `x` for `x = hash(x)` (it is the same reason you cannot have
a cycle in your Git commit graph).

### The Business Logic

With that hashing scheme in mind, the MOGG cache itself the logic required to
perform cache lookups based on the hash and optionally compile the IR under
consideration.

The full workflow looks like the following in `JitCompileKernels` for a given
input graph:

1. Create a full stub module from: (1) the KGEN IR from the input graph, (2) all
   the KGEN code from each registered kernel library (i.e. the contents of the
   `MojoLibraryAnalysis`).
2. Generate the semantic hash for the full stub module.
3. For each entry point in the stub module, use the generated semantic hash to
   perform a lookup in the MOGG cache.
4. If the IR has not been cached previously, perform elaboration and LLVM
   compilation.
