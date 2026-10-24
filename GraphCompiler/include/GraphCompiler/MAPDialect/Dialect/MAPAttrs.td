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
// This defines attributes in the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_TD

include "GraphCompiler/MAPDialect/Dialect/MAPDialect.td"
include "GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for attributes in this dialect.
class MAP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MAPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def MAP_ViewSignatureAttr : MAP_Attr<"ViewSignature", "view.signature"> {
  let summary = "Input/output shape signature shared by all MAP views";
  let description = [{
    The static input/output `MOSH::ShapeAttr`s of a view op, embedded by
    every view kind. Printed as `in -> out`, e.g.:

    ```mlir
    #map.view.signature<[?, 64] -> [?, 32]>
    ```
  }];

  let parameters = (ins
    "MOSH::ShapeAttr":$in,
    "MOSH::ShapeAttr":$out
  );

  let hasCustomAssemblyFormat = 1;
}

def MAP_SliceSpecAttr : MAP_Attr<"SliceSpec", "index.slice.spec"> {
  let summary = "Static shape and per-dimension bounds for a slice view";
  let description = [{
    Static shape/bounds metadata for a `mo.slice` view. `signature` keeps the
    rank. `starts`/`stops`/`steps` hold one entry per dimension: the constant
    bound, or `?` (`ShapedType::kDynamic`) when it must be read at runtime
    from the `map.index.slice`/`map.index.slice.stride` tensor operand.

    Example:
    ```mlir
    #sspec = #map.index.slice.spec<[?, 64] -> [?, 32], starts = [?, 32], stops = [?, 64], steps = [1, 1]>
    ```
  }];

  let parameters = (ins
    "MAP::ViewSignatureAttr":$signature,
    "mlir::DenseI64ArrayAttr":$starts,
    "mlir::DenseI64ArrayAttr":$stops,
    "mlir::DenseI64ArrayAttr":$steps
  );

  // Custom so `starts`/`stops`/`steps` print `?` instead of the raw
  // `ShapedType::kDynamic` sentinel.
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins
      "MOSH::ShapeAttr":$in,
      "MOSH::ShapeAttr":$out,
      "llvm::ArrayRef<int64_t>":$starts,
      "llvm::ArrayRef<int64_t>":$stops,
      CArg<"llvm::ArrayRef<int64_t>", "{}">:$steps
    ), [{
      // Omitted steps are inferred; see `inferSliceSteps`.
      llvm::SmallVector<int64_t> inferredSteps;
      if (steps.empty()) {
        inferredSteps = MAP::inferSliceSteps(out, starts, stops);
        steps = inferredSteps;
      }
      MLIRContext *ctx = in.getContext();
      return $_get(ctx, MAP::ViewSignatureAttr::get(ctx, in, out),
                   mlir::DenseI64ArrayAttr::get(ctx, starts),
                   mlir::DenseI64ArrayAttr::get(ctx, stops),
                   mlir::DenseI64ArrayAttr::get(ctx, steps));
    }]>
  ];
}

def MAP_BroadcastSpecAttr : MAP_Attr<"BroadcastSpec", "index.broadcast.spec"> {
  let summary = "Static input/output shape signature for a broadcast view";
  let description = [{
    Static shape metadata for a `mo.static.broadcast_to` view, with numpy
    semantics: `in` aligns with `out`'s trailing dimensions, and leading `out`
    dimensions and aligned size-1 `in` dimensions are replicated. Whether a
    dynamic dimension replicates is left to the primitive `MAPToMOGG` lowers
    into.

    Example:
    ```mlir
    #bspec = #map.index.broadcast.spec<[1, 64] -> [8, 64]>
    ```
  }];

  let parameters = (ins
    "MAP::ViewSignatureAttr":$signature
  );

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

def MAP_ReshapeSpecAttr : MAP_Attr<"ReshapeSpec", "index.reshape.spec"> {
  let summary = "Static input/output shape signature for a reshape view";
  let description = [{
    Static shape metadata for a `mo.static.reshape` view. Rank may change in
    either direction, and the projection is a function of `signature` alone.

    Example:
    ```mlir
    #rspec = #map.index.reshape.spec<[2, 3] -> [3, 2]>
    ```
  }];

  let parameters = (ins
    "MAP::ViewSignatureAttr":$signature
  );

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

def MAP_TransposeSpecAttr : MAP_Attr<"TransposeSpec", "index.transpose.spec"> {
  let summary = "Static shape signature and permutation for a transpose view";
  let description = [{
    Static shape/permutation metadata for a `mo.transpose` view with a
    constant `perm`. `signature` keeps the rank, and
    `out.dim[i] == in.dim[perm[i]]`.

    Example:
    ```mlir
    #tspec = #map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>
    ```
  }];

  let parameters = (ins
    "MAP::ViewSignatureAttr":$signature,
    "mlir::DenseI64ArrayAttr":$perm
  );

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_TD
