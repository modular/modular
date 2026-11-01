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
// This defines attributes in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GraphCompiler/MOGGDialect/MOGGEnums.td"
include "Mojo/LITDialect/LITTypes.td"

/// Base class for MOGG attributes usable anywhere an `mlir::Location` is.
class MOGGLocationAttr<string attrName, string attrMnemonic,
                       list<Trait> traits = []>
    : LocationAttrDef<MOGGDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class MOGGAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MOGGDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// ProfileScopeLocationAttr
//===----------------------------------------------------------------------===//

def MOGG_ProfileScopeLocationAttr
    : MOGGLocationAttr<"ProfileScopeLocation", "profile_scope"> {
  let summary = "A `Graph.profile_scope()` scope-name annotation on a Location";
  let description = [{
    `Graph.profile_scope` wraps the location of each op in its `with` block in
    one of these per active scope; the innermost scope is the outermost layer.
    `color` is the optional NVTX color name.
  }];

  let parameters = (ins "mlir::StringAttr":$name,
                        "mlir::LocationAttr":$childLoc,
                        OptionalParameter<"mlir::StringAttr">:$color);
  let assemblyFormat = "`<` $name `,` $childLoc (`,` `color` `=` $color^)? `>`";
}

//===----------------------------------------------------------------------===//
// VariadicABIAttr -- how a `mogg.tuple.create` run must be packed
//===----------------------------------------------------------------------===//

def MOGG_VariadicABIAttr
    : EnumAttr<MOGGDialect, MOGG_VariadicABI, "variadic_abi"> {
  let summary = "Packing ABI for a variadic tensor run (see MOGG_VariadicABI)";
}

//===----------------------------------------------------------------------===//
// ABIKindAttr -- the calling convention of a `mogg.func`
//===----------------------------------------------------------------------===//

def MOGG_ABIKindAttr : EnumAttr<MOGGDialect, MOGG_ABIKind, "abi"> {
  let summary = "Calling convention of a kernel stub (see MOGG_ABIKind)";
}

//===----------------------------------------------------------------------===//
// DispatchKindAttr -- how the runtime schedules a routine
//===----------------------------------------------------------------------===//

def MOGG_DispatchKindAttr
    : EnumAttr<MOGGDialect, MOGG_DispatchKind, "dispatch"> {
  let summary = "How the runtime schedules a routine (see MOGG_DispatchKind)";
}

//===----------------------------------------------------------------------===//
// TileFusionLaneAttr -- the tile epilogue's lane
//===----------------------------------------------------------------------===//

def MOGG_TileFusionLaneAttr
    : EnumAttr<MOGGDialect, MOGG_TileFusionLane, "tile_lane"> {
  let summary = "Compute or store lane of a tile epilogue (see MOGG_TileFusionLane)";
}

//===----------------------------------------------------------------------===//
// TensorArgContractAttr -- the trait-lane tensor-argument contract
//===----------------------------------------------------------------------===//

def MOGG_TensorArgContractAttr : MOGGAttr<"TensorArgContract", "tensor_arg"> {
  let summary = "Distilled trait contract for a trait-bound tensor argument";
  let description = [{
    Carried on `mogg.tensor_arg.__init__`. Mirrors the per-argument dict in
    `KernelFunctionAttr.tensor_arg_contracts` (`tensorArgContractToDict` /
    `dictToTensorArgContract` in `KernelDeclAdaptor.cpp`). This is the only
    home for role/access information; the `!mo.tensor` type does not carry it.

      - `role`         : `input` / `output` / `mutable_input`. An argument is
                         mutable iff its role is not `input`.
      - `fused`        : whether the operand is functor-backed.
      - `access`       : set of `load` / `store` / `transform` (`none` if
                         empty). A non-fused dense argument may carry several
                         (e.g. `load|store`); a fused argument exactly one.
      - `capabilities` : the subset of `access` that required the callable
                         capability (e.g. `Loadable`), not just the marker
                         (e.g. `LoadAccess`).
      - `bound`        : the deepest trait bound on a shape-function tensor
                         argument (`tensor`, `dense`, `tile`). Execute-lane
                         arguments carry `unspecified`, which printing elides.
      - `stable`       : whether the argument's storage is a stable address
                         slot (the `Stable` marker): emit hands the kernel a
                         `StableAddr`-backed wrapper read through
                         `StableAddrEngine`. Printing elides `stable` at
                         `false`.

  }];

  let parameters = (ins
    EnumParameter<MOGG_TensorArgRole>:$role,
    "bool":$fused,
    EnumParameter<MOGG_TensorArgAccess>:$access,
    EnumParameter<MOGG_TensorArgAccess>:$capabilities,
    DefaultValuedParameter<"::M::MOGG::TensorArgBoundKind",
        "::M::MOGG::TensorArgBoundKind::unspecified">:$bound,
    DefaultValuedParameter<"bool", "false">:$stable
  );

  // `struct` prints the parameters in declaration order and elides `bound` /
  // `stable` at their defaults, so every pre-existing contract keeps its
  // spelling; it also parses the keys in any order.
  let assemblyFormat = [{
    `<` `{` struct(params) `}` `>`
  }];

  let extraClassDeclaration = [{
    /// Mirrors `TensorArgContract::isMutable`.
    bool isMutable() const {
      return getRole() != ::M::MOGG::TensorArgRoleKind::input;
    }
  }];
}

//===----------------------------------------------------------------------===//
// KernelFunctionAttr
//===----------------------------------------------------------------------===//

def MOGG_KernelFunctionAttr : MOGGAttr<"KernelFunction", "kernel_function"> {
  let summary = "Bundles a Mojo function signature with its fn-level "
                "parameter declarations and cached slot-name mappings";
  let description = [{
    Holds, for one function variant of a Mojo kernel
    (`execute` / `shape` / `elementwise` / `build_graph`):

      - `signature`     : the LIT `FnTypeGeneratorType`.
      - `params`        : the def-level `ParamDeclAttr`s. Kept because the
                          signature's metadata lacks declared types.
      - `arg_params`    : per-argument `{struct-slot-name -> bound-value}`
                          dicts, or `UnitAttr` when the struct decl is not in
                          the loaded package (e.g. tensors, device contexts).
      - `result_params` : the same for result types, needed only by `shape`
                          and `elementwise`. Import errors unless each result
                          type is non-parameterised, in the loaded package, or
                          in the hardcoded stdlib table.
      - `tensor_arg_contracts` : per-argument dict with `role`, `fused`,
                          `access`, `capabilities` (bitmasks), plus `bound`
                          for shape-function arguments; `UnitAttr` for
                          non-trait arguments. Empty when no argument is
                          trait-bound.

    `arg_params` drives opaque-parameter inference in MO-to-MOGG: for an
    `MO::OpaqueType` operand `{S0 = 4, S1 = 8}` and an argument typed
    `SIMDPair[S0 = P0, S1 = P1]`, the slot dict
    `{S0 -> ParamDeclRef<P0>, S1 -> ParamDeclRef<P1>}` threads
    `P0 = 4, P1 = 8` into the call.
  }];

  let parameters = (ins
    "::M::KGEN::LIT::FnTypeGeneratorType":$signature,
    "::mlir::ArrayAttr":$params,
    "::mlir::ArrayAttr":$arg_params,
    "::mlir::ArrayAttr":$result_params,
    DefaultValuedParameter<"::mlir::ArrayAttr",
                           "::mlir::ArrayAttr::get($_ctxt, {})">:
        $tensor_arg_contracts
  );

  let assemblyFormat = [{
    `<` $signature `,` $params `,` $arg_params `,` $result_params
        (`,` $tensor_arg_contracts^)?
    `>`
  }];
}

def MOGG_KernelReductionOutputAttr
    : MOGGAttr<"KernelReductionOutput", "kernel_reduction_output"> {
  let summary = "One output field of a reduction monoid";
  let description = [{
    `name` is the field's name and `dtype` its `SIMD` element type as declared:
    a `#kgen.param.decl.ref` to a struct parameter (inferred from the
    reduction's result type) or a fixed dtype. The field's full type isn't
    kept: it refers to struct parameters by bare name, which only resolve in
    the struct declaration's scope.
  }];

  let parameters = (ins
    "::mlir::StringAttr":$name,
    "::mlir::Attribute":$dtype
  );

  let assemblyFormat = [{
    `<` $name `,` $dtype `>`
  }];
}

def MOGG_KernelReductionAttr
    : MOGGAttr<"KernelReduction", "kernel_reduction"> {
  let summary = "Describes a registered `ReduceOp` monoid";
  let description = [{
    - `signature`    : the monoid struct's parameter signature, with defaults.
    - `width_param`  : the struct parameter the trait's `width` alias names;
                       empty when `width` is not a bare struct parameter.
    - `outputs`      : `#mogg.kernel_reduction_output`s, the `SIMD` fields
                       whose width is `width_param`.
  }];

  let parameters = (ins
    "::M::KGEN::LIT::TypeSignatureType":$signature,
    "::mlir::StringAttr":$width_param,
    "::mlir::ArrayAttr":$outputs
  );

  let assemblyFormat = [{
    `<` $signature `,` $width_param `,` $outputs `>`
  }];
}

class MOGGFusionGroupAttr<string attrName, string attrMnemonic,
                          string name>
    : MOGGAttr<attrName, attrMnemonic> {
  let extraClassDeclaration = [{
    static constexpr llvm::StringLiteral getFusionName() { return "}] # name # [{"; }
  }];
}

//===----------------------------------------------------------------------===//
// FusionInfo: Leaf Attributes
//===----------------------------------------------------------------------===//

def MOGG_FusionLeafAttr : MOGGAttr<"FusionLeaf", "fusion_leaf"> {
  let summary = "A single op in a fusion info tree.";
  let description = [{
    Stores the op's location so the tree carries full provenance, separately
    from the outer kernel's FusedLoc.
  }];

  let parameters = (ins "mlir::StringAttr":$opName,
                        OptionalParameter<"mlir::LocationAttr">:$loc,
                        OptionalParameter<"mlir::TypeAttr">:$signature);
  let assemblyFormat = "`<` $opName (`,` `loc` `=` $loc^)? (`,` `sig` `=` $signature^)? `>`";
}

//===----------------------------------------------------------------------===//
// FusionInfo: Group Attributes
//===----------------------------------------------------------------------===//

def MOGG_FusionSequenceAttr : MOGGFusionGroupAttr<"FusionSequence", "fusion_sequence", "Sequence"> {
  let summary = "Sequence fusion info.";

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionHorizontalAttr : MOGGFusionGroupAttr<"FusionHorizontal", "fusion_horizontal", "Horizontal"> {
  let summary = "An attribute that represents horizontal fusion info.";
  let description = [{
    This attribute holds information for horizontal fusion: sibling kernels
    over one index space merged into one kernel, so an operand they share is
    loaded once.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionElementwiseAttr : MOGGFusionGroupAttr<"FusionElementwise", "fusion_elementwise", "Elementwise"> {
  let summary = "Elementwise fusion info.";

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionEpilogueAttr : MOGGFusionGroupAttr<"FusionEpilogue", "fusion_epilogue", "Epilogue"> {
  let summary = "Epilogue fusion info.";
  let description = [{
    `operandIndices` tracks which outputs were fused.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children,
                        ArrayRefParameter<"uint32_t">:$operandIndices);
  let assemblyFormat = "`<` `[` $children `]` `,` `operandIndices` `=` `[` $operandIndices `]` `>`";
}

def MOGG_FusionPrologueAttr : MOGGFusionGroupAttr<"FusionPrologue", "fusion_prologue", "Prologue"> {
  let summary = "Prologue fusion info.";
  let description = [{
    `operandIndices` tracks which inputs were fused.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children,
                        ArrayRefParameter<"uint32_t">:$operandIndices);
  let assemblyFormat = "`<` `[` $children `]` `,` `operandIndices` `=` `[` $operandIndices `]` `>`";
}

def MOGG_FusionViewAttr : MOGGFusionGroupAttr<"FusionView", "fusion_view", "View"> {
  let summary = "View fusion info.";

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionMutableLoadAttr : MOGGFusionGroupAttr<"FusionMutableLoad", "fusion_mutable_load", "MutableLoad"> {
  let summary = "Mutable load fusion info.";

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

def MOGG_FusionRowwiseAttr : MOGGFusionGroupAttr<"FusionRowwise", "fusion_rowwise", "Rowwise"> {
  let summary = "Rowwise (reduction) fusion info.";
  let description = [{
    Kernels fused into a `map.iter.rowwise`.
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$children);
  let assemblyFormat = "`<` `[` $children `]` `>`";
}

//===----------------------------------------------------------------------===//
// FusionInfoAttr - the root in the fusion tree info stored on ops
//===----------------------------------------------------------------------===//

def MOGG_FusionInfoAttr : MOGGAttr<"FusionInfo", "fusion"> {
  let summary = "An attribute that holds the fusion info tree.";
  let description = [{
    Ops print it as an alias named after the fused ops (see
    `MOGGOpAsmDialectInterface`), with the tree spelled out once at the top:

    ```mlir
    #slice_matmul_relu_add = #mogg.fusion<#mogg.fusion_sequence<[
        #mogg.fusion_view<[#mogg.fusion_leaf<"mo.slice">]>,
        #mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul">,
            #mogg.fusion_elementwise<[#mogg.fusion_leaf<"mo.relu">,
                                      #mogg.fusion_leaf<"mo.add">]>],
            operandIndices = [0]>]>>
    ... {fusionInfo = #slice_matmul_relu_add}
    ```

    `toString` renders the function-call form used in trace labels, e.g.
    `Sequence(View(mo.slice), Epilogue(mo.matmul, ...))`.

    The `root` parameter can be any of:
    - FusionLeafAttr: single operation
    - FusionSequenceAttr: soft fusion (multiple kernels)
    - FusionHorizontalAttr: horizontal fusion of sibling kernels
    - FusionElementwiseAttr: hard elementwise fusion
    - FusionEpilogueAttr: epilogue fusion (with optional operand index)
    - FusionPrologueAttr: prologue fusion (with optional operand index)
    - FusionViewAttr: view fusion
    - FusionMutableLoadAttr: mutable load fusion
  }];

  let parameters = (ins "mlir::Attribute":$root);

  let assemblyFormat = "`<` $root `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    std::string toString(bool includeLineNumbers = false, bool quote = false,
                         bool stripDialectPrefixes = false) const;

    /// Returns the fused ops' names in tree order, without their dialect
    /// prefix (`mo.matmul` -> `matmul`, `shapeof(mo.range)` ->
    /// `shapeof(range)`), skipping the group nodes.
    llvm::SmallVector<std::string> getLeafNames() const;

    /// Returns a multi-line tree. Leaves in \p highlightLeaves (matched by
    /// attribute identity, not op name) get "<-- error". \p indent prefixes
    /// each line with two spaces.
    std::string toPrettyTree(
        const llvm::DenseSet<mlir::Attribute> &highlightLeaves = {},
        bool indent = false) const;
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGATTRS_TD
