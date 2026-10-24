//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This defines operations in the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_TD

include "GraphCompiler/MAPDialect/Dialect/MAPDialect.td"
include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.td"
include "GraphCompiler/MAPDialect/Dialect/MAPTypes.td"
include "GraphCompiler/MODialect/MOTypes.td"
include "GraphCompiler/MODialect/MOInterfaces.td"
include "GraphCompiler/MOGGDialect/MOGGAttrs.td"
include "GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "GraphCompiler/Interfaces/LowerableToMOGGInterface.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for ops in this dialect.
class MAP_Op<string mnemonic, list<Trait> traits = []>
    : Op<MAPDialect, mnemonic, traits>;

// Type constraint matching the `map.index<N>` type.
def MAP_AnyIndex : Type<
    CPred<"::llvm::isa<::M::MAP::IndexType>($_self)">,
    "map.index type">;

//===----------------------------------------------------------------------===//
// map.iter.opaque
//===----------------------------------------------------------------------===//

def MAP_IterOpaqueOp : MAP_Op<"iter.opaque", [
    AttrSizedOperandSegments,
    IsolatedFromAbove,
    MAP_IterOpInterface,
    SingleBlock,
    NoTerminator,
    MO_ExplicitDevice,
    MO_HasDeviceInterface,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  ]> {
  let summary = "An opaque (hand-written) kernel with fusible load/store paths.";
  let description = [{
    Wraps a call to an opaque Mojo kernel (e.g. a hand-tuned GEMM) that exposes
    named load/store callbacks. The body holds the load/store callbacks
    (`map.iter.prologue` / `map.iter.epilogue`) plus the `map.call.opaque` that
    invokes the kernel; it is not a sequence of computation ops.

    The region is isolated from above: `inputs` and `outputs` are shadowed as
    body block arguments. Outputs are destination-passing, so the body needs no
    terminator unless a `map.call.opaque` returns a by-value result, which
    `map.yield` carries out.

    ```mlir
    %mm = map.iter.opaque
        inputs (%A : !mo.tensor<[M, K], bf16>, %B : !mo.tensor<[K, N], bf16>)
        outputs (%out : !mo.tensor<[M, N], bf16>) {
      %epi = map.iter.epilogue (%val : bf16, %idx : !map.index<2>) capturing {%cap = %out : !mo.tensor<[M, N], bf16>} {
        map.store %cap[%idx] = %val : !mo.tensor<[M, N], bf16>
      } : !mo.tensor<[M, N], bf16>
      map.call.opaque @builtin_kernels.linalg.Matmul[transpose_b = false]
        inputs (%A, %B) outputs (%epi) : (...) -> (...)
    }
    ```

    A mutating kernel takes an `!mo.chain` through `$inputChain` and returns
    one as the trailing result. The chain is pure sequencing: it has no block
    argument and never reaches the `map.call.opaque`.

    `hasReadEffect`/`hasWriteEffect` carry forward the original op's memory
    effects instead of the conservative "unknown effects" default.

    `fusionInfo` records the kernel's fusion-tree provenance (a `mogg.fusion`
    attribute), typed `AnyAttr` so `map` need not depend on `mogg`.

    The execution device is an explicit `#M.device_ref` (`MO_ExplicitDevice`)
    rather than inferred from types as on `map.iter.elementwise`: inference cannot
    tell "runs on the host" from "don't know" for unannotated tensors, which
    matters for passes that must refuse to act on device kernels.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer, MO_Opaque]>>:$inputs,
    Optional<MO_Chain>:$inputChain,
    OptionalAttr<AnyAttr>:$fusionInfo,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasReadEffect,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasWriteEffect,
    DefaultValuedOptionalAttr<MOGG_DispatchKindAttr,
                              "::M::MOGG::DispatchKind::Deferred">:$dispatch
  );
  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Opaque, MO_Chain]>>:$results);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// The number of results that have a body block argument. `$results` is
    /// ordered [DPS results..., by-value opaque?, chain?]; only the DPS ones
    /// have one.
    unsigned getNumNonChainResults();

    /// The by-value `!mo.opaque` result, or null when there is none.
    ::mlir::Value getOpaqueResult();

    /// The trailing `getNumNonChainResults()` body block arguments (block
    /// arguments are ordered [non-chain inputs..., DPS outputs...]).
    ::mlir::Block::BlockArgListType getOutputBlockArgs();

    /// The body block arguments shadowing `$inputs`, one-to-one.
    ::mlir::Block::BlockArgListType getInputBlockArgs();

    /// Returns the `$inputs` operand `arg` shadows, or null when `arg` shadows
    /// none (e.g. an output block arg). The inverse of `getInputBlockArgs()`.
    ::mlir::Value getInputFor(::mlir::BlockArgument arg);

    /// The `map.iter.epilogue` wrapping DPS result `resultIdx` -- the one that
    /// captures its output block arg -- or null when that result has none.
    IterEpilogueOp getEpilogueForResult(unsigned resultIdx);

    /// Inserts `vals` as new (non-chain) inputs starting at `idx`, updating
    /// both operands and the corresponding shadowed body block arguments.
    void insertInputsAt(unsigned idx, ::mlir::ValueRange vals);

    /// Erases a range of (non-chain) inputs at `[idx, idx + len)`, updating
    /// both operands and the corresponding shadowed body block arguments.
    void eraseInputsAt(unsigned idx, size_t len = 1);

    /// Returns the output chain (always the last result), or null.
    ::mlir::Value getOutputChain();

    /// Sets the input chain operand. No-op when `chain` is null. Requires a
    /// chain input to already exist.
    void setChain(::mlir::Value chain);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// map.yield
//===----------------------------------------------------------------------===//

def MAP_YieldOp : MAP_Op<"yield", [Terminator]> {
  let summary = "Terminates a map.iter.opaque body, yielding its results.";
  let description = [{
    Terminates a `map.iter.prologue`, a `map.iter.reduce` (yielding its
    contribution), or a `map.iter.opaque` body whose `map.call.opaque` returns
    a by-value result.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// map.iter.epilogue
//===----------------------------------------------------------------------===//

def MAP_IterEpilogueOp : MAP_Op<"iter.epilogue", [
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    AttrSizedOperandSegments,
    MAP_CapturingFusionClosure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
  ]> {
  let summary = "A store-path callback inlined into an opaque kernel's output.";
  let description = [{
    An SSA callback the enclosing `map.call.opaque` consumes on its `outputs`:
    it is handed a value and an index and stores (or transforms then stores)
    it. The leading block arguments are the value and the index; the rest
    shadow the captured operands. The `capturing {...}` clause is print/parse
    sugar over those operands.

    `$intermediateTensor` (not a capture, so no block argument) is the tensor
    whose base pointer the kernel-facing epilogue slice is built over. Set by
    `AllocateAccumulatingBuffers`.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer]>>:$captures,
    Optional<AnyType>:$intermediateTensor
  );
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// map.iter.prologue
//===----------------------------------------------------------------------===//

def MAP_IterPrologueOp : MAP_Op<"iter.prologue", [
    IsolatedFromAbove,
    SingleBlock,
    MAP_CapturingFusionClosure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
  ]> {
  let summary = "A load-path callback inlined into an opaque kernel's input.";
  let description = [{
    An SSA callback the enclosing `map.call.opaque` consumes on its `inputs`:
    it is handed an index and yields the value the kernel reads there. The
    leading block argument is the index; the rest shadow the captured operands.
    The `capturing {...}` clause is print/parse sugar over those operands.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer]>>:$captures);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// map.call
//===----------------------------------------------------------------------===//

def MAP_ConstantOp : MAP_Op<"constant", [
    ConstantLike,
    Pure,
    DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
  ]> {
  let summary = "A constant folded into a kernel body.";
  let description = [{
    Carries an `mo.constant` that `FuseConstantIterOpaqueInputs` folded into a
    `map.iter.opaque` body. A clone of the `mo.constant` cannot be used: the
    body must hold only `LowerableToMOGGInterface` ops, and `mo` cannot
    implement it without depending on `mogg`.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{ $value attr-dict `:` type($result) }];
  // `ConstantLike` requires a folder returning the constant's value.
  let hasFolder = 1;
}

def MAP_CallOpaqueOp : MAP_Op<"call.opaque", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
  ]> {
  let summary = "Invoke an opaque Mojo kernel by symbol.";
  let description = [{
    Calls the kernel declared under `callee` (a `mogg.kernel.decl` symbol, e.g.
    `@builtin_kernels.linalg.Matmul`) with the given `inputs` and `outputs`. The
    kernel parameters (e.g. `packed_b`, `transpose_b`) are carried in brackets
    immediately after the callee symbol, e.g.
    `@...Matmul[packed_b = false, transpose_b = false]`.

    `kernelName` is the callee's `registration` (e.g. `mo.matmul`). Several
    structs may register under one name, so it cannot be derived from
    `callee`. Passes that need the implemented MO op match on it, so renaming
    a Mojo struct cannot change their behavior.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    StrAttr:$kernelName,
    DictionaryAttr:$params,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs,
    // The source op's `MO_SameVariadicOperandSize` trait, so `mogg.call.opaque`
    // can split the flat operands into equal-length variadic groups.
    DefaultValuedAttr<BoolAttr, "false">:$hasSameVariadicOperandSize
  );

  // A by-value result, with no `$outputs` entry; carried out by `map.yield`.
  let results = (outs Optional<MO_Opaque>:$opaqueResult);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $callee `[` custom<KernelParamsBody>($params) `]`
    `inputs` `(` $inputs `)` `outputs` `(` $outputs `)`
    (`returns` type($opaqueResult)^)?
    `kernelName` `=` $kernelName
    (`hasSameVariadicOperandSize` `=` $hasSameVariadicOperandSize^)?
    attr-dict `:` functional-type($inputs, $outputs)
  }];
}

//===----------------------------------------------------------------------===//
// map.iter.elementwise
//===----------------------------------------------------------------------===//

def MAP_IterElementwiseOp : MAP_Op<"iter.elementwise", [
  AttrSizedOperandSegments,
  IsolatedFromAbove,
  MAP_IterOpInterface,
  DeclareOpInterfaceMethods<MAP_StructuredIterOpInterface>,
  SingleBlock,
  NoTerminator,
  DeclareOpInterfaceMethods<MO_HasDeviceInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Structured iteration over a tensor shape";
  let description = [{
    Iterates over every point in an N-dimensional shape, executing the body
    region once per index. The shape is a `MOSH::ShapeAttr` and may contain
    static integer dims, named symbolic parameters (D0, D1, …), or parametric
    expressions.

    Inputs (tensors or, for in-place kernels, buffers) are shadowed as body
    block arguments. Outputs are extra block arguments matching the non-chain
    results. Block arguments are ordered [inputs..., outputs..., index].

    The optional `chain` operand/result models mutating kernels (e.g.
    `mo.mutable.store`). It has no block argument, and the output chain is
    always the last result.

    `hasReadEffect`/`hasWriteEffect` carry forward the original op's memory
    effects instead of the conservative "unknown effects" default.

    Example:
    ```mlir
    %result = map.iter.elementwise<[8, 16, 128]>
        inputs  (%x: !mo.tensor<[8, 16, 128], f32>,
                 %y: !mo.tensor<[8, 16, 128], f32>)
        outputs (%out: !mo.tensor<[8, 16, 128], f32>)
        index   (%idx: !map.index<3>)
        -> !mo.tensor<[8, 16, 128], f32>
    { ... }
    ```
  }];

  let arguments = (ins
    MOSH_ShapeAttr:$shape,
    Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer]>>:$inputs,
    Optional<MO_Chain>:$chain,
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasReadEffect,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasWriteEffect,
    DefaultValuedOptionalAttr<MOGG_DispatchKindAttr,
                              "::M::MOGG::DispatchKind::Deferred">:$dispatch
  );

  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Chain]>>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::Block::BlockArgListType getInputBlockArgs();
    ::mlir::Block::BlockArgListType getOutputBlockArgs();

    // Inserts/erases inputs along with their body block arguments.
    void insertInputsAt(unsigned idx, ValueRange vals);
    void eraseInputsAt(unsigned idx, size_t len = 1);

    Value getInputChain() { return getChain(); }
    // Null if the last result isn't a chain.
    Value getOutputChain();
    // No-op when `chain` is null.
    void setChain(Value chain);
  }];
}

//===----------------------------------------------------------------------===//
// map.iter.elementwise.body
//===----------------------------------------------------------------------===//

def MAP_IterElementwiseBodyOp : MAP_Op<"iter.elementwise.body", [
  HasParent<"IterOpaqueOp">,
  IsolatedFromAbove,
  SingleBlock,
  NoTerminator,
  MAP_CapturingFusionClosure,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "The per-index body of a structured MAP loop, wrapped for outlining.";
  let description = [{
    Holds the per-index computation of a `map.iter.elementwise` after
    `OutlineStructuredKernelBodies` wraps it in a `map.iter.opaque`. The
    opaque's input and output block arguments are its `$captures` (see
    `CapturingFusionClosure`); the leading block argument is the index.
    `shape` is copied from the wrapped `map.iter.elementwise`.

    Example:
    ```mlir
    %r = map.iter.opaque
        inputs (%x: !mo.tensor<[8, 16, 128], f32>, %y: !mo.tensor<[8, 16, 128], f32>)
        outputs (%out: !mo.tensor<[8, 16, 128], f32>) {
      map.iter.elementwise.body<[8, 16, 128]>
          index (%idx: !map.index<3>)
          capturing {%x0 = %x, %y0 = %y, %out0 = %out} {
        %xv = map.load %x0[%idx] : !mo.tensor<[8, 16, 128], f32>
        %yv = map.load %y0[%idx] : !mo.tensor<[8, 16, 128], f32>
        %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                  : !map.index<3>, f32, f32 -> f32
        map.store %out0[%idx] = %sum : !mo.tensor<[8, 16, 128], f32>
      }
    }
    ```
  }];

  let arguments = (ins MOSH_ShapeAttr:$shape,
                       Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer]>>:$captures);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.iter.rowwise
//===----------------------------------------------------------------------===//

def MAP_IterRowwiseOp : MAP_Op<"iter.rowwise", [
  AttrSizedOperandSegments,
  IsolatedFromAbove,
  MAP_IterOpInterface,
  DeclareOpInterfaceMethods<MAP_StructuredIterOpInterface>,
  SingleBlock,
  NoTerminator,
  DeclareOpInterfaceMethods<MO_HasDeviceInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "A kernel that runs its body once per row of a shape";
  let description = [{
    Iterates over the rows of `shape` along `axis`. The body runs once per row
    and holds the row's phases in order: folds (`map.iter.reduce`), row-scalar
    ops on their results, and stores.

    The row index `%row` indexes the per-row space, `shape` with the `axis`
    coordinate pinned to 0, so it addresses a per-row (`axis` size 1) output
    directly. Each result is per-row or has the full `shape`.

    Inputs (tensors or, for in-place kernels, buffers) are shadowed as body
    block arguments. Outputs are extra block arguments matching the non-chain
    results. Block arguments are ordered [inputs..., outputs..., row].

    As on `map.iter.elementwise`, the optional `chain` operand/result models
    mutating kernels; it has no block argument, and the output chain is always
    the last result. `hasReadEffect`/`hasWriteEffect` carry forward the
    original op's memory effects instead of the conservative "unknown effects"
    default, and `dispatch` prints only when it isn't `deferred`.

    Example:
    ```mlir
    %sum = map.iter.rowwise<[M, N], axis = 1>
        inputs  (%x: !mo.tensor<[M, N], f32>)
        outputs (%out: !mo.tensor<[M, 1], f32>)
        row     (%row: !map.index<2>) -> !mo.tensor<[M, 1], f32> {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
          index (%i: !map.index<2>) -> f32 {
        %v = map.load %x[%i] : !mo.tensor<[M, N], f32>
        map.yield %v : f32
      }
      map.store %out[%row] = %acc : !mo.tensor<[M, 1], f32>
    }
    ```
  }];

  let arguments = (ins
    MOSH_ShapeAttr:$shape,
    I64Attr:$axis,
    Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer]>>:$inputs,
    Optional<MO_Chain>:$chain,
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasReadEffect,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasWriteEffect,
    DefaultValuedOptionalAttr<MOGG_DispatchKindAttr,
                              "::M::MOGG::DispatchKind::Deferred">:$dispatch
  );

  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Chain]>>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::Block::BlockArgListType getInputBlockArgs();
    ::mlir::Block::BlockArgListType getOutputBlockArgs();
    ::mlir::BlockArgument getRowArg();

    // Inserts/erases inputs along with their body block arguments.
    void insertInputsAt(unsigned idx, ::mlir::ValueRange vals);
    void eraseInputsAt(unsigned idx, size_t len = 1);

    ::mlir::Value getInputChain() { return getChain(); }
    // Null if the last result isn't a chain.
    ::mlir::Value getOutputChain();
    // No-op when `chain` is null.
    void setChain(::mlir::Value chain);
  }];
}

//===----------------------------------------------------------------------===//
// map.iter.rowwise.body
//===----------------------------------------------------------------------===//

def MAP_IterRowwiseBodyOp : MAP_Op<"iter.rowwise.body", [
  HasParent<"IterOpaqueOp">,
  IsolatedFromAbove,
  SingleBlock,
  NoTerminator,
  MAP_CapturingFusionClosure,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "The per-row body of a map.iter.rowwise, wrapped for outlining.";
  let description = [{
    Holds the per-row computation of a `map.iter.rowwise` after
    `OutlineStructuredKernelBodies` wraps it in a `map.iter.opaque`, as
    `map.iter.elementwise.body` does for `map.iter.elementwise`. The opaque's input and
    output block arguments are its `$captures` (see `CapturingFusionClosure`);
    the leading block argument is the row index. `shape` and `axis` are copied
    from the wrapped `map.iter.rowwise`.

    Example:
    ```mlir
    %r = map.iter.opaque
        inputs (%x: !mo.tensor<[M, N], f32>)
        outputs (%out: !mo.tensor<[M, 1], f32>) {
      map.iter.rowwise.body<[M, N], axis = 1>
          row (%row: !map.index<2>)
          capturing (%x0 = %x : !mo.tensor<[M, N], f32>,
                     %out0 = %out : !mo.tensor<[M, 1], f32>) {
        %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum
            index (%i: !map.index<2>) -> f32 {
          %v = map.load %x0[%i] : !mo.tensor<[M, N], f32>
          map.yield %v : f32
        }
        map.store %out0[%row] = %acc : !mo.tensor<[M, 1], f32>
      }
    }
    ```
  }];

  let arguments = (ins MOSH_ShapeAttr:$shape,
                       I64Attr:$axis,
                       Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer]>>:$captures);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.iter.reduce
//===----------------------------------------------------------------------===//

def MAP_IterReduceOp : MAP_Op<"iter.reduce", [
  ParentOneOf<["IterRowwiseOp", "IterRowwiseBodyOp"]>,
  SingleBlock,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "A fold over the current row with a monoid";
  let description = [{
    Folds every element of the enclosing rowwise's current row with `monoid`,
    a `mogg.kernel.decl` with a `reduction`. The region runs once per element
    at index `%i` and yields the element's contribution. The results are the
    monoid's output fields, in order, one value per row, and print with the
    fields' names.

    The region reads the row and must not store.

    Example:
    ```mlir
    %acc_values, %acc_indices = map.iter.reduce @builtin_kernels.monoids.ArgMax
        index (%i: !map.index<2>) -> (f32, si64) {
      %v = map.load %x[%i] : !mo.tensor<[M, N], f32>
      map.yield %v : f32
    }
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$monoid);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::BlockArgument getIndexArg() {
      return getBody().front().getArgument(0);
    }
    YieldOp getYield();
  }];
}

//===----------------------------------------------------------------------===//
// map.constant.scalar
//===----------------------------------------------------------------------===//

def MAP_ConstantScalarOp : MAP_Op<"constant.scalar", [
  ConstantLike,
  Pure,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "A compile-time scalar constant, as a value not a tensor";
  let description = [{
    The single element of a compile-time constant, replacing the `map.load`
    that would read it (see `LoadOp`'s canonicalizer). The result has the
    loaded tensor's element type, so it is interchangeable with `map.load`.

    Example:
    ```mlir
    %cst = map.constant.scalar 2.000000e+00 : f32 : f32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyTypeOf<[AnyFloat, AnyInteger]>:$result);

  let assemblyFormat = "$value attr-dict `:` qualified(type($result))";
  // `ConstantLike` requires a folder for `m_Constant` to recover the value.
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.load
//===----------------------------------------------------------------------===//

def MAP_LoadOp : MAP_Op<"load", [
  TypesMatchWith<"result type matches tensor element type",
                 "tensor", "result",
                 "::M::MO::castFromTensorLike($_self).getElementType()">,
  TypesMatchWith<"index type matches tensor rank",
                 "tensor", "index",
                 "::M::MAP::IndexType::get($_self.getContext(), "
                 "::M::MO::castFromTensorLike($_self).getRank())">,
  Pure,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>
]> {
  let summary = "Load a value from a tensor at an N-D index";
  let description = [{
    Loads the element at an N-dimensional index from a tensor (or, fused into
    an in-place kernel, a buffer). Logically one scalar per index; the lowering
    may load a wider vector or tile.

    Example:
    ```mlir
    %xv = map.load %x[%idx] : !mo.tensor<[8, 16, 128], f32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor, MAP_AnyIndex:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{ $tensor `[` $index `]` `:` qualified(type($tensor)) attr-dict }];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// map.store
//===----------------------------------------------------------------------===//

def MAP_StoreOp : MAP_Op<"store", [
  TypesMatchWith<"value type matches tensor/buffer element type",
                 "tensor", "value",
                 "::M::MO::castFromTensorLike($_self).getElementType()">,
  TypesMatchWith<"index type matches tensor/buffer rank",
                 "tensor", "index",
                 "::M::MAP::IndexType::get($_self.getContext(), "
                 "::M::MO::castFromTensorLike($_self).getRank())">,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "Store a value into a tensor (or buffer) at an N-D index";
  let description = [{
    Stores an element at an N-dimensional index into an output tensor or, for
    in-place kernels like `mo.mutable.store`, an input buffer. Logically one
    scalar per index; the lowering may store a wider vector or tile.

    Example:
    ```mlir
    map.store %out[%idx] = %outv : !mo.tensor<[8, 16, 128], f32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor, MAP_AnyIndex:$index, AnyType:$value);

  let assemblyFormat = [{ $tensor `[` $index `]` `=` $value `:` qualified(type($tensor)) attr-dict }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.call.elementwise
//===----------------------------------------------------------------------===//

def MAP_CallElementwiseOp : MAP_Op<"call.elementwise", [
  Pure,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "Call a named pointwise kernel on the current index and scalar arguments";
  let description = [{
    Calls a named pointwise kernel with the loop's current index and scalar
    operands, returning one scalar. `params` carries the kernel's
    compile-time parameters (e.g. a custom kernel's `bias`), absent when there
    are none.

    Example:
    ```mlir
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<3>, f32, f32 -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       OptionalAttr<DictionaryAttr>:$params,
                       MAP_AnyIndex:$index, Variadic<AnyType>:$args);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $callee (`[` custom<KernelParamsBody>($params)^ `]`)?
    `` `(` $index `,` $args `)` attr-dict
    `:` type($index) `,` type($args) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// map.index.slice
//===----------------------------------------------------------------------===//

def MAP_SliceIndexOp : MAP_Op<"index.slice", [
  Pure,
  DeclareOpInterfaceMethods<MAP_IndexTransformOpInterface, ["createStridedView"]>,
]> {
  let summary = "Project an output-shape index to the input-shape index for a slice view";
  let description = [{
    Projects the loop's output-shape index to the input-shape index of a
    `mo.slice`: `result[d] = index[d] * steps[d] + starts[d]`, reading each
    bound from `spec` when constant and from its operand tensor otherwise.
    `stops` is carried for verification and later passes.

    Example:
    ```mlir
    %idx_ = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>>
        (%idx, %starts, %stops, %steps)
        : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
    ```
  }];

  let arguments = (ins
    MAP_SliceSpecAttr:$spec,
    MAP_AnyIndex:$index,
    MO_Tensor:$starts,
    MO_Tensor:$stops,
    MO_Tensor:$steps
  );
  let results = (outs MAP_AnyIndex:$result);

  let assemblyFormat = [{
    `<` qualified($spec) `>` `(` $index `,` $starts `,` $stops `,` $steps `)` attr-dict
    `:` `(` qualified(type($index)) `,` qualified(type($starts)) `,`
        qualified(type($stops)) `,` qualified(type($steps)) `)` `->`
    qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.index.slice.select
//===----------------------------------------------------------------------===//

def MAP_SliceSelectOp : MAP_Op<"index.slice.select", [
  ParentOneOf<["IterEpilogueOp", "IterElementwiseOp", "IterElementwiseBodyOp",
               "SliceSelectOp"]>,
  SingleBlock,
  NoTerminator,
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<MAP_DynamicShapeIndexOpInterface>,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "Run a body at a slice's output index, where the slice selects an index";
  let description = [{
    The store-side inverse of `map.index.slice`: `index` addresses the
    slice's *input* space (the spec's in-shape), and the body runs only where
    the slice selects `index`, with its one block argument bound to the
    matching index in the slice's *output* space (the spec's out-shape). A
    slice is not a bijection, so where it does not select `index` there is no
    output index, and the body does not run. Nothing in the body needs to
    know about the slice: it may load and store at its argument like any loop
    body at its own index.

    For each dimension `d`, with the start resolved the numpy way,
    `s = starts[d] < 0 ? max(starts[d] + dim(in, d), 0) : starts[d]`, and
    `off = index[d] - s`, the slice selects `index` iff every
    `off % steps[d] == 0` and `0 <= off / steps[d] < dim(out, d)`, and then
    the output index is `off / steps[d]`. `stops` is not read: the out-shape
    already bounds the slice.

    Starts and steps must be static and steps positive. The innermost
    dimension must be unsliced: the body runs once per stored vector, which
    spans that dimension. For the same reason `index` must be one the vector
    runs along: an epilogue's or a foreach's own index, or an enclosing
    select's argument, never a transformed one. The spec's in-shape is the
    shape that index addresses (the epilogue's, the foreach's loop shape, or
    the enclosing select's out-shape), and a load or store at the body's
    argument addresses a tensor of the out-shape. A slice of a slice is two
    nested selects.

    Example:
    ```mlir
    map.index.slice.select<#map.index.slice.spec<[2, 9, 16] -> [2, 8, 16], starts = [0, 1, 0], stops = [2, 9, 16], steps = [1, 1, 1]>> (%idx) -> (%iout) : !map.index<3> {
      map.store %window[%iout] = %val : !mo.tensor<[2, 8, 16], f32>
    }
    ```
  }];

  let arguments = (ins
    MAP_SliceSpecAttr:$spec,
    MAP_AnyIndex:$index
  );
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Checks the rules a select of `spec` at `index` must meet, all but its
    /// body's: the ones above on the steps, starts, innermost dim and index.
    /// Reports the first broken rule through `emitError` when given; the
    /// escaping-reader merge asks without one before building a select.
    static ::mlir::LogicalResult verifySpecAt(
        SliceSpecAttr spec, ::mlir::Value index,
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError = {});
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.index.slice.stride
//===----------------------------------------------------------------------===//

def MAP_SliceStrideOp : MAP_Op<"index.slice.stride", [
  ParentOneOf<["IterOpaqueOp"]>,
  Pure,
  MAP_StrideViewOpInterface,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "Materialize a slice view of `input`, addressable directly by the loop's own index";
  let description = [{
    A view of `input` with the slice's *output* shape, which a `map.load`
    addresses with the loop's own index. Nothing is copied.

    Sits in a `map.iter.opaque`'s body, before the `map.iter.elementwise.body`
    whose loads it feeds. `input` is a buffer when the view feeds an in-place
    kernel's fused load.

    Example:
    ```mlir
    %strided = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>>
        (%input, %starts, %stops, %steps)
        : (!mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 32], f32>
    ```
  }];

  let arguments = (ins
    MAP_SliceSpecAttr:$spec,
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$input,
    MO_Tensor:$starts,
    MO_Tensor:$stops,
    MO_Tensor:$steps
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `<` qualified($spec) `>` `(` $input `,` $starts `,` $stops `,` $steps `)` attr-dict
    `:` `(` qualified(type($input)) `,` qualified(type($starts)) `,`
        qualified(type($stops)) `,` qualified(type($steps)) `)` `->`
    qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.index.broadcast
//===----------------------------------------------------------------------===//

def MAP_BroadcastIndexOp : MAP_Op<"index.broadcast", [
  Pure,
  DeclareOpInterfaceMethods<MAP_IndexTransformOpInterface, ["createStridedView"]>,
]> {
  let summary = "Project an output-shape index to the input-shape index for a broadcast view";
  let description = [{
    Projects the loop's output-shape index to the input-shape index of a
    `mo.static.broadcast_to`, with numpy semantics: leading dimensions `in`
    lacks are dropped and size-1 dimensions project to 0. Which dimensions
    those are is decided by `createStridedView`'s primitive.

    Example:
    ```mlir
    %idx_ = map.index.broadcast<#map.index.broadcast.spec<[1, 64] -> [8, 64]>>(%idx)
        : !map.index<2> -> !map.index<2>
    ```
  }];

  let arguments = (ins
    MAP_BroadcastSpecAttr:$spec,
    MAP_AnyIndex:$index
  );
  let results = (outs MAP_AnyIndex:$result);

  let assemblyFormat = [{
    `<` qualified($spec) `>` `(` $index `)` attr-dict
    `:` qualified(type($index)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.index.broadcast.stride
//===----------------------------------------------------------------------===//

def MAP_BroadcastStrideOp : MAP_Op<"index.broadcast.stride", [
  ParentOneOf<["IterOpaqueOp"]>,
  Pure,
  MAP_StrideViewOpInterface,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "Materialize a broadcast view of `input`, addressable directly by the loop's own index";
  let description = [{
    A view of `input` with the broadcast's *output* shape, which a `map.load`
    addresses with the loop's own index. Nothing is copied.

    Sits in a `map.iter.opaque`'s body, before the `map.iter.elementwise.body`
    whose loads it feeds. `input` is a buffer when the view feeds an in-place
    kernel's fused load.

    Example:
    ```mlir
    %strided = map.index.broadcast.stride<#map.index.broadcast.spec<[1, 64] -> [8, 64]>>(%input)
        : !mo.tensor<[1, 64], f32> -> !mo.tensor<[8, 64], f32>
    ```
  }];

  let arguments = (ins
    MAP_BroadcastSpecAttr:$spec,
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$input
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `<` qualified($spec) `>` `(` $input `)` attr-dict
    `:` qualified(type($input)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.index.transpose
//===----------------------------------------------------------------------===//

def MAP_TransposeIndexOp : MAP_Op<"index.transpose", [
  Pure,
  DeclareOpInterfaceMethods<MAP_IndexTransformOpInterface, ["createStridedView"]>,
]> {
  let summary = "Project an output-shape index to the input-shape index for a transpose view";
  let description = [{
    Projects the loop's output-shape index to the input-shape index of a
    `mo.transpose`. Since `out.dim[i] == in.dim[perm[i]]`, this applies
    `perm`'s inverse.

    `permutations` is the runtime permutation, read only where `spec` is
    dynamic (the eager interpreter's one-graph-per-op transpose). It rides
    along for the hoisted `map.index.transpose.stride`; this op is never
    lowered to MOGG directly.

    Example:
    ```mlir
    %idx_ = map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>>(%idx, %permutations)
        : (!map.index<2>, !mo.tensor<[2], si64>) -> !map.index<2>
    ```
  }];

  let arguments = (ins
    MAP_TransposeSpecAttr:$spec,
    MAP_AnyIndex:$index,
    MO_Tensor:$permutations
  );
  let results = (outs MAP_AnyIndex:$result);

  let assemblyFormat = [{
    `<` qualified($spec) `>` `(` $index `,` $permutations `)` attr-dict
    `:` `(` qualified(type($index)) `,` qualified(type($permutations)) `)` `->`
    qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.index.transpose.stride
//===----------------------------------------------------------------------===//

def MAP_TransposeStrideOp : MAP_Op<"index.transpose.stride", [
  ParentOneOf<["IterOpaqueOp"]>,
  Pure,
  MAP_StrideViewOpInterface,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "Materialize a transpose view of `input`, addressable directly by the loop's own index";
  let description = [{
    A view of `input` with the transpose's *output* shape, which a `map.load`
    addresses with the loop's own index. Nothing is copied; the strides are
    `input`'s reordered by `perm`.

    Sits in a `map.iter.opaque`'s body, before the `map.iter.elementwise.body`
    whose loads it feeds. `input` is a buffer when the view feeds an in-place
    kernel's fused load.

    `permutations` is read only where `spec` is dynamic; emission elides the
    read behind a `comptime if` for a static perm.

    Example:
    ```mlir
    %strided = map.index.transpose.stride<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>>(%input, %permutations)
        : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
    ```
  }];

  let arguments = (ins
    MAP_TransposeSpecAttr:$spec,
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$input,
    MO_Tensor:$permutations
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `<` qualified($spec) `>` `(` $input `,` $permutations `)` attr-dict
    `:` `(` qualified(type($input)) `,` qualified(type($permutations)) `)` `->`
    qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.index.reshape
//===----------------------------------------------------------------------===//

def MAP_ReshapeIndexOp : MAP_Op<"index.reshape", [
  Pure,
  DeclareOpInterfaceMethods<MAP_IndexTransformOpInterface, ["createStridedView"]>,
  DeclareOpInterfaceMethods<MAP_DynamicShapeIndexOpInterface>,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "Project an output-shape index to the input-shape index for a reshape view";
  let description = [{
    Projects the loop's output-shape index to the input-shape index of a
    `mo.static.reshape` through the shared row-major linear index.

    In a `map.iter.epilogue` it transforms the STORE index instead, with the
    spec's `in`/`out` swapped; `LoopInvariantViewMotion` does not hoist it
    there.

    Example:
    ```mlir
    %idx_ = map.index.reshape<#map.index.reshape.spec<[2, 3] -> [3, 2]>>(%idx)
        : !map.index<2> -> !map.index<2>
    ```
  }];

  let arguments = (ins
    MAP_ReshapeSpecAttr:$spec,
    MAP_AnyIndex:$index
  );
  let results = (outs MAP_AnyIndex:$result);

  let assemblyFormat = [{
    `<` qualified($spec) `>` `(` $index `)` attr-dict
    `:` qualified(type($index)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.index.reshape.stride
//===----------------------------------------------------------------------===//

def MAP_ReshapeStrideOp : MAP_Op<"index.reshape.stride", [
  ParentOneOf<["IterOpaqueOp"]>,
  Pure,
  MAP_StrideViewOpInterface,
  DeclareOpInterfaceMethods<LowerableToMOGGInterface>,
]> {
  let summary = "Materialize a reshape view of `input`, addressable directly by the loop's own index";
  let description = [{
    A view of `input` with the reshape's *output* shape, which a `map.load`
    addresses with the loop's own index. Nothing is copied.

    Sits in a `map.iter.opaque`'s body, before the `map.iter.elementwise.body`
    whose loads it feeds. `input` is a buffer when the view feeds an in-place
    kernel's fused load.

    Example:
    ```mlir
    %strided = map.index.reshape.stride<#map.index.reshape.spec<[2, 3] -> [3, 2]>>(%input)
        : !mo.tensor<[2, 3], f32> -> !mo.tensor<[3, 2], f32>
    ```
  }];

  let arguments = (ins
    MAP_ReshapeSpecAttr:$spec,
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$input
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `<` qualified($spec) `>` `(` $input `)` attr-dict
    `:` qualified(type($input)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_TD
