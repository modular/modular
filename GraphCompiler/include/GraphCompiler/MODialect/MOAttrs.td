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
// This defines attributes in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD

include "GraphCompiler/MODialect/MODialect.td"
include "Mojo/KGENDialect/KGENAttrs.td"

/// Helps declare optional parameter declaration arguments.
def MO_ParamDecls : DefaultValuedAttr<KGEN_ParamDeclArrayAttr,
                                      "ArrayRef<ParamDeclAttr>({})">;

class MOAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MODialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// MO_LayoutAttr
//===----------------------------------------------------------------------===//

def MO_LayoutAttr : MOAttr<"Layout", "layout"> {
  let summary = "An attribute that represents tensor memory layout.";
  let description = [{
    This attribute holds the memory layout information for some tensor value.
  }];

  let parameters = (ins "mlir::StringAttr":$format);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilder<(ins "mlir::StringRef" : $formatStr)>,
  ];
}

def MO_DTypeAttr : MOAttr<"DType", "dtype"> {
  let summary = "An attribute that represents tensor data type.";
  let description = [{
    This attribute holds the data type of a tensor.
  }];

  let parameters = (ins "DType":$dtype);
  let assemblyFormat = [{
    `<` $dtype `>`
  }];
}

def MO_ChainAttr : KGEN_ExprAttr<"Chain", "chain"> {
  let summary = "Chain expression attribute";

  let description = [{
    Represents non-error chain values. The type of this attribute is always
    `!mo.chain`.

    Example:

    ```mlir
    #mo<chain> : !mo.chain
    ```
  }];

  let dialect = MODialect;
  let parameters = (ins
    AttributeSelfTypeParameter<"", "MO::ChainType">:$type);

  let builders = [
    TypeBuilderWithInferredContext<(ins "MO::ChainType":$type),
      [{ return get(type.getContext(), type); }]>
  ];

  let assemblyFormat = "";
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
